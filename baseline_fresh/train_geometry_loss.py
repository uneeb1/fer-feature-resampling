#!/usr/bin/env python3
"""
Geometry-loss experiment: M2 + center loss + bounded inter-class separation.

The separation loss pushes class centers apart (cosine similarity → 0) via a
margin hinge, while center loss compacts samples toward their class center.
Together they produce a better-separated 512-d feature space for downstream
feature-space resampling.

NO backbone/head/logit changes — the loss acts ONLY on the 512-d post-avgpool
embedding and its class centers.

Usage:
  Stage A — CE baseline (revised recipe):
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0 --lam-sep 0

  Stage A — center-only:
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0

  Stage A — geometry sweep (seed 42):
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.001 --extract-features
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.01  --extract-features
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.1   --extract-features

  Stage B — best config, 3 seeds:
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep <best> --seeds 42 123 456 --extract-features
    python train_geometry_loss.py --config config_m2.yaml --lam-center 0 --lam-sep 0 --seeds 42 123 456 --extract-features
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score, silhouette_samples
from torch.utils.data import DataLoader

from src.dataset import load_fer2013_csv, load_ferplus_csv, verify_splits, FER2013Dataset, CLASSES
from src.model import FERResNet18
from src.transforms import get_train_transform, get_val_transform
from src.train import evaluate, evaluate_tta, get_lr, mixup_data, mixup_criterion
from src.features import extract_features, save_features


# ---------------------------------------------------------------------------
# Center loss (unchanged from Phase 4)
# ---------------------------------------------------------------------------
class CenterLoss(nn.Module):
    def __init__(self, num_classes=7, feat_dim=512):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers = self.centers.index_select(0, labels)
        return 0.5 * (features - centers).pow(2).sum(dim=1).mean()

    def forward_mixup(self, features, y_a, y_b, lam):
        da = (features - self.centers.index_select(0, y_a)).pow(2).sum(dim=1)
        db = (features - self.centers.index_select(0, y_b)).pow(2).sum(dim=1)
        return 0.5 * (lam * da + (1.0 - lam) * db).mean()


# ---------------------------------------------------------------------------
# Separation loss: bounded, margin-hinged inter-class separation
# ---------------------------------------------------------------------------
class SeparationLoss(nn.Module):
    """Push class centers apart until orthogonal, then stop (bounded, hinged).
    Operates on the SAME centers maintained by CenterLoss."""
    def __init__(self, sep_target=0.0):
        super().__init__()
        self.tau = sep_target

    def forward(self, centers):
        c = F.normalize(centers, dim=1)
        sim = c @ c.t()
        K = c.size(0)
        iu = torch.triu_indices(K, K, offset=1)
        pair = sim[iu[0], iu[1]]
        return torch.clamp(pair - self.tau, min=0.0).mean()


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------
def compute_geometry(features, labels, centers_np, class_names):
    """Compute geometry diagnostics on a feature set."""
    geo = {}
    n_classes = len(class_names)

    # Overall silhouette
    try:
        geo["silhouette_overall"] = float(silhouette_score(features, labels))
    except Exception:
        geo["silhouette_overall"] = None

    # Pair silhouettes
    entangled_pairs = [("fear", "angry"), ("sad", "neutral")]
    geo["silhouette_pairs"] = {}
    for a, b in entangled_pairs:
        ia, ib = class_names.index(a), class_names.index(b)
        mask = (labels == ia) | (labels == ib)
        if mask.sum() > 2:
            try:
                geo["silhouette_pairs"][f"{a}_vs_{b}"] = float(
                    silhouette_score(features[mask], labels[mask]))
            except Exception:
                geo["silhouette_pairs"][f"{a}_vs_{b}"] = None

    # Center cosine similarity matrix
    c_norm = centers_np / (np.linalg.norm(centers_np, axis=1, keepdims=True) + 1e-8)
    cos_sim = (c_norm @ c_norm.T).tolist()
    geo["center_cosine_similarity"] = cos_sim

    # Per-class compactness (mean sample-to-center distance)
    geo["per_class_compactness"] = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            dists = np.linalg.norm(features[mask] - centers_np[c], axis=1)
            geo["per_class_compactness"][class_names[c]] = float(dists.mean())

    # Separation-to-compaction ratio
    mean_compact = np.mean(list(geo["per_class_compactness"].values()))
    iu = np.triu_indices(n_classes, k=1)
    cos_sim_arr = np.array(geo["center_cosine_similarity"])
    mean_cos_sep = float(cos_sim_arr[iu].mean())
    geo["mean_compactness"] = float(mean_compact)
    geo["mean_center_cosine"] = mean_cos_sep
    geo["separation_to_compaction_ratio"] = float((1.0 - mean_cos_sep) / (mean_compact + 1e-8))

    return geo


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "angry": "#E64B35", "disgust": "#4DBBD5", "fear": "#00A087",
    "happy": "#F39B7F", "sad": "#3C5488", "surprise": "#8491B4",
    "neutral": "#91D1C2"
}
ENTANGLED_CLASSES = {"angry", "fear", "sad", "neutral"}


def make_figures(history, test_features, test_labels, centers_np,
                 test_preds, class_names, best_epoch, out_dir, geo_data):
    """Generate all presentation-quality figures. Each independently guarded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    colors = [CLASS_COLORS[c] for c in class_names]
    fig_paths = []

    # 1. Val accuracy curve
    try:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
        ax.plot(range(1, len(history["val_acc"])+1), history["val_acc"], "b-", lw=1.5)
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"Best epoch {best_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy"); ax.set_title("Validation Accuracy")
        ax.legend(); fig.savefig(f"{out_dir}/val_accuracy_curve.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/val_accuracy_curve.png")
    except Exception as e:
        print(f"[WARN] val_accuracy_curve failed: {e}")

    # 2. Loss curve
    try:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
        epochs = range(1, len(history["train_loss"])+1)
        ax.plot(epochs, history["train_loss"], label="Train", lw=1.5)
        ax.plot(epochs, history["val_loss"], label="Val", lw=1.5)
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"Best epoch {best_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Train vs Val Loss")
        ax.legend(); fig.savefig(f"{out_dir}/loss_curve.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/loss_curve.png")
    except Exception as e:
        print(f"[WARN] loss_curve failed: {e}")

    # 3. Train vs Val Macro-F1
    try:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
        epochs = range(1, len(history["val_f1"])+1)
        ax.plot(epochs, history["train_f1"], label="Train F1", lw=1.5)
        ax.plot(epochs, history["val_f1"], label="Val F1", lw=1.5)
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"Best epoch {best_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1"); ax.set_title("Train vs Val Macro-F1 (Overfitting Monitor)")
        ax.legend(); fig.savefig(f"{out_dir}/trainval_macroF1_curve.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/trainval_macroF1_curve.png")
    except Exception as e:
        print(f"[WARN] trainval_macroF1_curve failed: {e}")

    # 4. t-SNE of test features
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        emb = tsne.fit_transform(test_features)
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True, dpi=300)
        for i, name in enumerate(class_names):
            mask = test_labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1], c=CLASS_COLORS[name],
                       label=name, s=8, alpha=0.6)
        ax.set_title("t-SNE of Test Features (512-d)"); ax.legend(markerscale=3)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        fig.savefig(f"{out_dir}/tsne_test_features.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/tsne_test_features.png")
    except Exception as e:
        print(f"[WARN] tsne_test_features failed: {e}")

    # 5. Center cosine heatmap
    try:
        cos_sim = np.array(geo_data["center_cosine_similarity"])
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True, dpi=300)
        im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(7)); ax.set_yticklabels(class_names)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cos_sim[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im); ax.set_title("Center Cosine Similarity")
        fig.savefig(f"{out_dir}/center_cosine_heatmap.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/center_cosine_heatmap.png")
    except Exception as e:
        print(f"[WARN] center_cosine_heatmap failed: {e}")

    # 6. Confusion matrix (row-normalized + counts)
    try:
        cm = confusion_matrix(test_labels, test_preds, labels=range(7))
        for norm_mode, fname in [("true", "confusion_matrix.png"), (None, "confusion_matrix_counts.png")]:
            cm_plot = cm.astype(float)
            if norm_mode == "true":
                cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + 1e-8)
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True, dpi=300)
            im = ax.imshow(cm_plot, cmap="Blues")
            ax.set_xticks(range(7)); ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticks(range(7)); ax.set_yticklabels(class_names)
            fmt_str = ".2f" if norm_mode else "d"
            for i in range(7):
                for j in range(7):
                    val = cm_plot[i, j] if norm_mode else cm[i, j]
                    ax.text(j, i, f"{val:{fmt_str}}", ha="center", va="center", fontsize=8)
            fig.colorbar(im)
            title = "Confusion Matrix (Row-Normalized)" if norm_mode else "Confusion Matrix (Counts)"
            ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            fig.savefig(f"{out_dir}/{fname}"); plt.close(fig)
            fig_paths.append(f"{out_dir}/{fname}")
    except Exception as e:
        print(f"[WARN] confusion_matrix failed: {e}")

    # 7. Per-class F1 bar
    try:
        pcf1 = f1_score(test_labels, test_preds, average=None, labels=range(7))
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
        bar_colors = [CLASS_COLORS[c] for c in class_names]
        edge_colors = ["black" if c in ENTANGLED_CLASSES else "none" for c in class_names]
        lw = [2.0 if c in ENTANGLED_CLASSES else 0 for c in class_names]
        bars = ax.bar(class_names, pcf1, color=bar_colors, edgecolor=edge_colors, linewidth=lw)
        ax.set_ylabel("F1 Score"); ax.set_title("Per-Class F1 (entangled classes highlighted)")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, pcf1):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        fig.savefig(f"{out_dir}/per_class_f1_bar.png"); plt.close(fig)
        fig_paths.append(f"{out_dir}/per_class_f1_bar.png")
    except Exception as e:
        print(f"[WARN] per_class_f1_bar failed: {e}")

    # 8. Lambda sweep (if multiple dirs exist)
    try:
        parent = os.path.dirname(out_dir) or "."
        sweep_dirs = sorted([d for d in os.listdir(parent) if d.startswith("M2geo_")])
        if len(sweep_dirs) > 1:
            lams, val_f1s, test_f1s = [], [], []
            for d in sweep_dirs:
                mf = os.path.join(parent, d, "metrics.json")
                if os.path.exists(mf):
                    with open(mf) as f:
                        m = json.load(f)
                    lam = m.get("lam_sep", 0)
                    lams.append(lam)
                    seeds_data = m["per_seed"]
                    first = list(seeds_data.values())[0]
                    val_f1s.append(first["val_f1"])
                    test_f1s.append(first["test_f1"])
            if lams:
                fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
                ax.plot(lams, val_f1s, "o-", label="Val Macro-F1")
                ax.plot(lams, test_f1s, "s--", label="Test Macro-F1")
                ax.set_xlabel("λ_sep"); ax.set_ylabel("Macro-F1")
                ax.set_title("Geometry Loss: λ_sep Sweep")
                ax.set_xscale("log"); ax.legend()
                fig.savefig(f"{out_dir}/lambda_sweep.png"); plt.close(fig)
                fig_paths.append(f"{out_dir}/lambda_sweep.png")
        else:
            print("[INFO] Only one M2geo_ dir found; skipping lambda_sweep.png")
    except Exception as e:
        print(f"[WARN] lambda_sweep failed: {e}")

    # 9. Silhouette pairs bar
    try:
        pairs = geo_data.get("silhouette_pairs", {})
        if pairs:
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True, dpi=300)
            names = list(pairs.keys())
            vals = [pairs[n] for n in names]
            ax.bar(names, vals, color=["#00A087", "#3C5488"])
            ax.set_ylabel("Silhouette Score"); ax.set_title("Silhouette: Entangled Pairs")
            for i, (n, v) in enumerate(zip(names, vals)):
                if v is not None:
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
            fig.savefig(f"{out_dir}/silhouette_pairs_bar.png"); plt.close(fig)
            fig_paths.append(f"{out_dir}/silhouette_pairs_bar.png")
    except Exception as e:
        print(f"[WARN] silhouette_pairs_bar failed: {e}")

    # Contact sheet
    try:
        pngs = [p for p in fig_paths if os.path.exists(p)]
        if pngs:
            from PIL import Image
            imgs = [Image.open(p) for p in pngs]
            ncols = 3
            nrows = math.ceil(len(imgs) / ncols)
            thumb_w, thumb_h = 800, 600
            sheet = Image.new("RGB", (thumb_w * ncols, thumb_h * nrows), (255, 255, 255))
            for idx, img in enumerate(imgs):
                img.thumbnail((thumb_w, thumb_h))
                r, c = divmod(idx, ncols)
                sheet.paste(img, (c * thumb_w, r * thumb_h))
            sheet.save(f"{out_dir}/figures.png")
            print(f"Saved contact sheet: {out_dir}/figures.png")
    except Exception as e:
        print(f"[WARN] contact sheet failed: {e}")

    plt.close("all")


# ---------------------------------------------------------------------------
# Training loop — center + separation
# ---------------------------------------------------------------------------
def train_one_epoch_geometry(model, loader, criterion, optimizer_main,
                             center_loss, sep_loss, optimizer_center,
                             lam_center, lam_sep,
                             device, mixup_alpha=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if mixup_alpha > 0:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
            logits, feats = model(mixed_images, return_features=True)
            loss_ce = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss_ctr = center_loss.forward_mixup(feats, y_a, y_b, lam) if lam_center > 0 else torch.tensor(0.0, device=device)
        else:
            logits, feats = model(images, return_features=True)
            loss_ce = criterion(logits, labels)
            loss_ctr = center_loss(feats, labels) if lam_center > 0 else torch.tensor(0.0, device=device)

        loss_sep = sep_loss(center_loss.centers) if lam_sep > 0 else torch.tensor(0.0, device=device)

        loss = loss_ce + lam_center * loss_ctr + (lam_center * lam_sep) * loss_sep

        optimizer_main.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()

        if lam_center > 0:
            for p in center_loss.parameters():
                if p.grad is not None:
                    p.grad.data *= (1.0 / lam_center)

        optimizer_main.step()
        optimizer_center.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    train_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, correct / total, train_f1


def train_model_geometry(model, train_loader, val_loader, cfg, device, save_path,
                         center_loss_module, sep_loss_module,
                         lam_center, lam_sep, alpha_center,
                         param_groups=None, log_fn=print):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])

    if param_groups is None:
        param_groups = [{"params": model.parameters(), "lr": cfg["training"]["lr"]}]
    optimizer_main = torch.optim.SGD(
        param_groups,
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    optimizer_center = torch.optim.SGD(
        center_loss_module.parameters(), lr=alpha_center
    )

    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_epochs"]
    base_lr = cfg["training"]["lr"]
    mixup_alpha = cfg.get("augmentation", {}).get("mixup_alpha", 0.0)
    best_f1, best_epoch = 0.0, 0
    patience = cfg["training"].get("early_stop_patience", 20)
    min_delta = cfg["training"].get("early_stop_min_delta", 0.0)
    epochs_no_improve = 0
    stop_epoch = epochs
    initial_lrs = [pg["lr"] for pg in optimizer_main.param_groups]
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
               "train_f1": [], "lr": [], "train_acc": []}

    for epoch in range(epochs):
        lr_scale = get_lr(epoch, 1.0, warmup, epochs)
        for pg, init_lr in zip(optimizer_main.param_groups, initial_lrs):
            pg["lr"] = init_lr * lr_scale
        history["lr"].append(base_lr * lr_scale)

        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch_geometry(
            model, train_loader, criterion, optimizer_main,
            center_loss_module, sep_loss_module, optimizer_center,
            lam_center, lam_sep,
            device, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc, val_f1, val_pcf1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["train_f1"].append(train_f1)

        lrs = [pg["lr"] for pg in optimizer_main.param_groups]
        lr_str = "/".join(f"{x:.2e}" for x in lrs)
        log_fn(f"Epoch {epoch+1:3d}/{epochs} | LR {lr_str} | "
               f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
               f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f} | "
               f"{elapsed:.1f}s")

        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "center_loss_state_dict": center_loss_module.state_dict(),
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
            }, save_path)
            log_fn(f"  -> New best val F1: {val_f1:.4f} (saved)")
        else:
            epochs_no_improve += 1

        if epoch >= warmup and epochs_no_improve >= patience:
            stop_epoch = epoch + 1
            log_fn(f"Early stop at epoch {stop_epoch} (best val F1 {best_f1:.4f} @ epoch {best_epoch})")
            break

    if stop_epoch == epochs:
        stop_epoch = epochs
    log_fn(f"Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    return history, best_epoch, best_f1, stop_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="M2 + Geometry Loss (Center + Separation)")
    p.add_argument("--config", default="config_m2.yaml")
    p.add_argument("--csv", default=None, help="Override CSV path")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--lam-center", type=float, default=0.001,
                   help="Center loss weight (0 = disable)")
    p.add_argument("--lam-sep", type=float, default=0.01,
                   help="Separation loss weight (0 = disable)")
    p.add_argument("--sep-target", type=float, default=0.0,
                   help="Cosine similarity target for hinge (0.0 = orthogonal)")
    p.add_argument("--alpha-center", type=float, default=0.5,
                   help="Center optimizer LR")
    p.add_argument("--epochs", type=int, default=60,
                   help="Total epochs (overrides config)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stop patience (overrides config)")
    p.add_argument("--min-delta", type=float, default=0.001,
                   help="Early stop min delta (overrides config)")
    p.add_argument("--weight-decay", type=float, default=0.002,
                   help="Weight decay (overrides config)")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: M2geo_lamc{lam}_lams{lam})")
    p.add_argument("--no-tta", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--extract-features", action="store_true", default=False,
                   help="Extract 512-d features from best-val checkpoint (seed 42)")
    p.add_argument("--dataset", choices=["fer2013", "ferplus"], default="fer2013",
                   help="Dataset to use (default: fer2013)")
    p.add_argument("--ferplus-csv", default=None,
                   help="Path to fer2013new.csv (required when --dataset ferplus)")
    return p.parse_args()


def main():
    args = parse_args()

    lam_center = args.lam_center
    lam_sep = args.lam_sep
    sep_target = args.sep_target
    alpha_center = args.alpha_center

    if args.dataset == "ferplus" and not args.ferplus_csv:
        raise ValueError("--ferplus-csv is required when --dataset ferplus")

    ds_prefix = "ferplus_" if args.dataset == "ferplus" else ""
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = f"{ds_prefix}M2geo_lamc{lam_center}_lams{lam_sep}"
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_dir}/features", exist_ok=True)
    os.makedirs(f"{base_dir}/predictions", exist_ok=True)
    os.makedirs(f"{base_dir}/geometry", exist_ok=True)
    os.makedirs(f"{base_dir}/graphs", exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.csv:
        cfg["data"]["csv_path"] = args.csv
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["early_stop_patience"] = args.patience
    cfg["training"]["early_stop_min_delta"] = args.min_delta
    cfg["training"]["weight_decay"] = args.weight_decay

    res = cfg["data"]["resolution"]
    seeds = cfg["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"M2 + GEOMETRY LOSS (Center + Separation)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"lam_center: {lam_center}  lam_sep: {lam_sep}  sep_target: {sep_target}")
    print(f"alpha_center: {alpha_center}")
    print(f"Epochs: {args.epochs}  Patience: {args.patience}  Min-delta: {args.min_delta}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seeds: {seeds}")
    print(f"Output: {base_dir}")

    # Save resolved config
    run_config = {
        "base_config": args.config,
        "dataset": args.dataset,
        "lam_center": lam_center,
        "lam_sep": lam_sep,
        "sep_target": sep_target,
        "alpha_center": alpha_center,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "weight_decay": args.weight_decay,
        "lr": cfg["training"]["lr"],
        "momentum": cfg["training"]["momentum"],
        "label_smoothing": cfg["training"]["label_smoothing"],
        "warmup_epochs": cfg["training"]["warmup_epochs"],
        "dropout": cfg.get("model", {}).get("dropout", 0.4),
        "mixup_alpha": cfg.get("augmentation", {}).get("mixup_alpha", 0.0),
        "seeds": seeds,
        "output_dir": base_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{base_dir}/config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Load data
    csv_path = cfg["data"]["csv_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(args.config), csv_path)
    leakage_filter = cfg["data"].get("leakage_filter", True)
    if args.dataset == "ferplus":
        splits = load_ferplus_csv(csv_path, args.ferplus_csv, leakage_filter=leakage_filter)
    else:
        splits = load_fer2013_csv(csv_path, leakage_filter=leakage_filter)

    counts = verify_splits(splits)
    print("\n=== Split Counts ===")
    for c in range(7):
        row = f"{CLASSES[c]:<10}" + "".join(f"{counts[s].get(c, 0):<10}" for s in counts.keys())
        print(row)

    train_tf = get_train_transform(res, cfg["augmentation"]["rotation_degrees"],
                                    cfg["augmentation"]["random_crop_pad"])
    val_tf = get_val_transform(res)
    clahe = cfg["data"]["clahe"]
    nw = cfg["data"]["num_workers"]

    all_results = {}
    best_overall_f1, best_seed = 0.0, None

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        seed_everything(seed)

        train_ds = FER2013Dataset(splits["train"], transform=train_tf, clahe=clahe)
        val_ds = FER2013Dataset(splits["val"], transform=val_tf, clahe=clahe)

        train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                                  shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

        dropout = cfg.get("model", {}).get("dropout", 0.4)
        freeze_stages = cfg.get("model", {}).get("freeze_stages", None)
        model = FERResNet18(num_classes=7, freeze_stages=freeze_stages, dropout=dropout).to(device)

        center_loss_module = CenterLoss(num_classes=7, feat_dim=512).to(device)
        sep_loss_module = SeparationLoss(sep_target=sep_target).to(device)

        lr_mults = cfg.get("training", {}).get("lr_mults", None)
        param_groups = model.param_groups(cfg["training"]["lr"], lr_mults)
        ckpt_path = f"{base_dir}/checkpoints/best_seed{seed}.pt"

        history, best_epoch, best_f1, stop_epoch = train_model_geometry(
            model, train_loader, val_loader, cfg, device, ckpt_path,
            center_loss_module, sep_loss_module,
            lam_center, lam_sep, alpha_center,
            param_groups=param_groups,
        )

        # Save history
        with open(f"{base_dir}/history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Load best checkpoint for test eval
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        center_loss_module.load_state_dict(ckpt["center_loss_state_dict"])

        test_ds_eval = FER2013Dataset(splits["test"], transform=val_tf, clahe=clahe)
        test_loader = DataLoader(test_ds_eval, batch_size=cfg["training"]["batch_size"],
                                 shuffle=False, num_workers=nw)

        if cfg["evaluation"]["tta"] and not args.no_tta:
            test_ds_raw = FER2013Dataset(splits["test"], transform=None, clahe=clahe)
            test_acc, test_f1, test_pcf1, test_preds, test_labels = evaluate_tta(
                model, test_ds_raw, device, resolution=res
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
            _, test_acc, test_f1, test_pcf1, test_preds, test_labels = evaluate(
                model, test_loader, criterion, device
            )

        # Val preds
        criterion_val = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
        _, val_acc_final, val_f1_final, val_pcf1_final, val_preds, val_labels = evaluate(
            model, val_loader, criterion_val, device
        )

        print(f"\nSeed {seed} TEST: Acc={test_acc:.4f} Macro-F1={test_f1:.4f}")
        for i, name in enumerate(CLASSES):
            print(f"  {name:<10}: F1={test_pcf1[i]:.4f}")

        all_results[seed] = {
            "val_f1": float(best_f1),
            "val_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "test_per_class_f1": {CLASSES[i]: float(test_pcf1[i]) for i in range(7)},
        }

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_seed = seed

        # Save predictions
        np.save(f"{base_dir}/predictions/test_preds.npy", np.array(test_preds))
        np.save(f"{base_dir}/predictions/test_labels.npy", np.array(test_labels))
        np.save(f"{base_dir}/predictions/val_preds.npy", np.array(val_preds))
        np.save(f"{base_dir}/predictions/val_labels.npy", np.array(val_labels))

    # Feature extraction
    feat_seed = 42 if 42 in seeds else best_seed
    if args.extract_features and feat_seed in seeds:
        print(f"\nExtracting 512-d features (seed {feat_seed})...")
        ckpt = torch.load(f"{base_dir}/checkpoints/best_seed{feat_seed}.pt",
                          map_location=device, weights_only=True)
        model = FERResNet18(num_classes=7, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        center_loss_module = CenterLoss(num_classes=7, feat_dim=512).to(device)
        center_loss_module.load_state_dict(ckpt["center_loss_state_dict"])

        all_split_features = {}
        for split_name in ["train", "val", "test"]:
            ds = FER2013Dataset(splits[split_name], transform=val_tf, clahe=clahe)
            feats, lbls = extract_features(model, ds, device,
                                           batch_size=cfg["training"]["batch_size"])
            save_features(feats, lbls, f"{base_dir}/features/{split_name}")
            all_split_features[split_name] = (feats, lbls)

        # Geometry diagnostics
        centers_np = center_loss_module.centers.detach().cpu().numpy()
        geo_test = compute_geometry(all_split_features["test"][0], all_split_features["test"][1],
                                    centers_np, list(CLASSES))
        geo_train = compute_geometry(all_split_features["train"][0], all_split_features["train"][1],
                                     centers_np, list(CLASSES))
        geo_combined = {"test": geo_test, "train": geo_train}
        with open(f"{base_dir}/geometry/geometry.json", "w") as f:
            json.dump(geo_combined, f, indent=2)
        print(f"Saved geometry diagnostics: {base_dir}/geometry/geometry.json")

        # Figures (seed 42 only)
        test_feats = all_split_features["test"][0]
        test_lbls = all_split_features["test"][1]
        make_figures(history, test_feats, test_lbls, centers_np,
                     test_preds, list(CLASSES), best_epoch,
                     f"{base_dir}/graphs", geo_test)
    elif not args.extract_features:
        print("\n[INFO] Skipping feature extraction (use --extract-features to enable)")

    # Save metrics
    test_f1s = [all_results[s]["test_f1"] for s in seeds]
    test_accs = [all_results[s]["test_acc"] for s in seeds]
    metrics = {
        "experiment": "M2_geometry_loss",
        "dataset": args.dataset,
        "lam_center": lam_center,
        "lam_sep": lam_sep,
        "sep_target": sep_target,
        "alpha_center": alpha_center,
        "epochs": args.epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "seeds": seeds,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_seed": {str(s): all_results[s] for s in seeds},
        "aggregate": {
            "test_macro_f1_mean": float(np.mean(test_f1s)),
            "test_macro_f1_std": float(np.std(test_f1s)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
        },
        "best_seed": best_seed,
    }
    with open(f"{base_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {base_dir}/metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — lam_center={lam_center} lam_sep={lam_sep}")
    print(f"{'='*60}")
    print(f"Test Macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    for seed in seeds:
        r = all_results[seed]
        print(f"  Seed {seed}: val_f1={r['val_f1']:.4f} test_f1={r['test_f1']:.4f}")


if __name__ == "__main__":
    main()
