#!/usr/bin/env python3
"""
Center-loss experiment: M2 + joint softmax + center loss (Wen et al., ECCV 2016).
Surgical diff from main.py — identical training except the added center-loss term.

Usage:
  Stage A (lam sweep, seed 42 only):
    python train_center_loss.py --config config_m2.yaml --lam-center 0.001
    python train_center_loss.py --config config_m2.yaml --lam-center 0.01
    python train_center_loss.py --config config_m2.yaml --lam-center 0.1

  Stage B (best lam, 3 seeds):
    python train_center_loss.py --config config_m2.yaml --lam-center <best> --seeds 42 123 456
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
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.dataset import load_fer2013_csv, verify_splits, FER2013Dataset, CLASSES
from src.model import FERResNet18
from src.transforms import get_train_transform, get_val_transform
from src.train import evaluate, evaluate_tta, get_lr, mixup_data, mixup_criterion
from src.features import extract_features, save_features


# ---------------------------------------------------------------------------
# Center loss (Wen et al., ECCV 2016)
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
# Training loop — identical to src/train.py train_one_epoch + train_model
# except center loss is added to the backward pass.
# ---------------------------------------------------------------------------
def train_one_epoch_center(model, loader, criterion, optimizer_main,
                           center_loss, optimizer_center, lam_center,
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
            loss_ctr = center_loss.forward_mixup(feats, y_a, y_b, lam)
        else:
            logits, feats = model(images, return_features=True)
            loss_ce = criterion(logits, labels)
            loss_ctr = center_loss(feats, labels)

        loss = loss_ce + lam_center * loss_ctr

        optimizer_main.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()

        # Scale center grads so alpha stays independent of lam_center
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


def train_model_center(model, train_loader, val_loader, cfg, device, save_path,
                       center_loss_module, lam_center, alpha_center,
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
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "train_f1": [], "lr": []}

    for epoch in range(epochs):
        lr_scale = get_lr(epoch, 1.0, warmup, epochs)
        for pg, init_lr in zip(optimizer_main.param_groups, initial_lrs):
            pg["lr"] = init_lr * lr_scale
        history["lr"].append(base_lr * lr_scale)

        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch_center(
            model, train_loader, criterion, optimizer_main,
            center_loss_module, optimizer_center, lam_center,
            device, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc, val_f1, val_pcf1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
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
    p = argparse.ArgumentParser(description="M2 + Center Loss Training")
    p.add_argument("--config", default="config_m2.yaml")
    p.add_argument("--csv", default=None, help="Override CSV path")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--lam-center", type=float, required=True,
                   help="Center loss weight (sweep: 0.001, 0.01, 0.1)")
    p.add_argument("--alpha-center", type=float, default=0.5,
                   help="Center optimizer LR (fixed)")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: M2center_lam{lam})")
    p.add_argument("--no-tta", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--extract-features", action="store_true", default=False,
                   help="Extract 512-d features from best-val checkpoint (seed 42)")
    return p.parse_args()


def main():
    args = parse_args()

    lam_center = args.lam_center
    alpha_center = args.alpha_center

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = f"M2center_lam{lam_center}"
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_dir}/features", exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.csv:
        cfg["data"]["csv_path"] = args.csv
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    res = cfg["data"]["resolution"]
    seeds = cfg["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"M2 + CENTER LOSS")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"lam_center: {lam_center}")
    print(f"alpha_center: {alpha_center}")
    print(f"Seeds: {seeds}")
    print(f"Output: {base_dir}")
    print(f"Config: {args.config}")

    # Save config
    run_config = {
        "base_config": args.config,
        "lam_center": lam_center,
        "alpha_center": alpha_center,
        "seeds": seeds,
        "output_dir": base_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{base_dir}/config_center_loss.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Load data
    csv_path = cfg["data"]["csv_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(args.config), csv_path)
    leakage_filter = cfg["data"].get("leakage_filter", True)
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

        lr_mults = cfg.get("training", {}).get("lr_mults", None)
        param_groups = model.param_groups(cfg["training"]["lr"], lr_mults)
        ckpt_path = f"{base_dir}/checkpoints/best_seed{seed}.pt"

        history, best_epoch, best_f1, stop_epoch = train_model_center(
            model, train_loader, val_loader, cfg, device, ckpt_path,
            center_loss_module, lam_center, alpha_center,
            param_groups=param_groups,
        )

        # Load best checkpoint for test eval
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

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

        print(f"\nSeed {seed} TEST: Acc={test_acc:.4f} Macro-F1={test_f1:.4f}")
        for i, name in enumerate(CLASSES):
            print(f"  {name:<10}: F1={test_pcf1[i]:.4f}")

        all_results[seed] = {
            "val_f1": best_f1,
            "val_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_per_class_f1": {CLASSES[i]: float(test_pcf1[i]) for i in range(7)},
        }

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_seed = seed

    # Feature extraction (best seed, or seed 42 if requested)
    feat_seed = 42 if 42 in seeds else best_seed
    if args.extract_features or feat_seed in seeds:
        print(f"\nExtracting 512-d features (seed {feat_seed})...")
        ckpt = torch.load(f"{base_dir}/checkpoints/best_seed{feat_seed}.pt",
                          map_location=device, weights_only=True)
        model = FERResNet18(num_classes=7, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        for split_name in ["train", "val", "test"]:
            ds = FER2013Dataset(splits[split_name], transform=val_tf, clahe=clahe)
            feats, lbls = extract_features(model, ds, device,
                                           batch_size=cfg["training"]["batch_size"])
            save_features(feats, lbls, f"{base_dir}/features/{split_name}")

    # Save all results to JSON
    test_f1s = [all_results[s]["test_f1"] for s in seeds]
    test_accs = [all_results[s]["test_acc"] for s in seeds]
    metrics = {
        "experiment": "M2_center_loss",
        "lam_center": lam_center,
        "alpha_center": alpha_center,
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
    print(f"FINAL SUMMARY — lam_center={lam_center}")
    print(f"{'='*60}")
    print(f"Test Macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    for seed in seeds:
        r = all_results[seed]
        print(f"  Seed {seed}: val_f1={r['val_f1']:.4f} test_f1={r['test_f1']:.4f}")


if __name__ == "__main__":
    main()
