"""
KMeansSMOTE on frozen ImageNet ResNet-18 features (FER2013).
Self-contained experiment: extract 512-d features, apply KMeansSMOTE,
train linear head, evaluate with imbalanced + balanced protocol,
generate all figures and comparison tables.
"""

import os, sys, copy, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import (f1_score, confusion_matrix, classification_report)
from collections import Counter
from imblearn.over_sampling import KMeansSMOTE
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BATCH_SIZE = 64
DROPOUT = 0.3
SEED = 42
N_DRAWS = 100
SAMPLES_PER_CLASS = 55

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXP_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
CHECKPOINT  = os.path.join(BASE_DIR, "experiments", "exp_03", "best_model.pth")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Transforms ────────────────────────────────────────────────────────────────

img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def extract_features(model, loader):
    hook_output = {}
    def hook_fn(module, inp, out):
        hook_output["feat"] = out
    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            _ = model(inputs)
            f = hook_output["feat"].squeeze(-1).squeeze(-1)
            feats.append(f.cpu().numpy())
            labs.append(labels.numpy())
    handle.remove()
    return np.concatenate(feats), np.concatenate(labs)


def train_head(train_X, train_y, val_X, val_y, num_epochs=50):
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds   = TensorDataset(torch.FloatTensor(val_X),   torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, best_acc = 0.0, None, 0, 0.0

    for epoch in range(num_epochs):
        head.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(head(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = head(X_batch.to(DEVICE))
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head, best_f1, best_acc, best_epoch


def predict(head, features):
    ds = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = head(X_batch.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)


def balanced_eval(preds, labels):
    rng = np.random.RandomState(0)
    macro_scores = []
    class_scores = {c: [] for c in CLASS_NAMES}

    for _ in range(N_DRAWS):
        indices = []
        for c in range(NUM_CLASSES):
            cls_idx = np.where(labels == c)[0]
            chosen = rng.choice(cls_idx, size=min(SAMPLES_PER_CLASS, len(cls_idx)), replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        y_true = labels[indices]
        y_pred = preds[indices]

        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_scores.append(macro)
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        for ci, cname in enumerate(CLASS_NAMES):
            class_scores[cname].append(per_class[ci])

    macro_scores = np.array(macro_scores)
    class_scores = {k: np.array(v) for k, v in class_scores.items()}
    return macro_scores, class_scores


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Counts
    im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title(f"{title} — Counts", fontsize=12)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Row-normalized
    im1 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f"{title} — Row-Normalized", fontsize=12)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                         color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1_comparison(baseline_pc, svmsmote_pc, kmeans_pc, save_path):
    x = np.arange(NUM_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, [baseline_pc[c]["mean"] for c in CLASS_NAMES], width,
                   yerr=[baseline_pc[c]["ci95"] for c in CLASS_NAMES],
                   label="Baseline", capsize=3, color="#4e79a7")
    bars2 = ax.bar(x, [svmsmote_pc[c]["mean"] for c in CLASS_NAMES], width,
                   yerr=[svmsmote_pc[c]["ci95"] for c in CLASS_NAMES],
                   label="SVMSMOTE", capsize=3, color="#f28e2b")
    bars3 = ax.bar(x + width, [kmeans_pc[c]["mean"] for c in CLASS_NAMES], width,
                   yerr=[kmeans_pc[c]["ci95"] for c in CLASS_NAMES],
                   label="KMeansSMOTE", capsize=3, color="#59a14f")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Balanced F1 (100 draws, 55/class)", fontsize=12)
    ax.set_title("Per-Class F1 Comparison — Balanced Eval", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_macro_f1_comparison(configs, save_path):
    names = list(configs.keys())
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    imbal_vals = [configs[n]["imbal"] for n in names]
    bal_vals = [configs[n]["bal_mean"] for n in names]
    bal_errs = [configs[n].get("bal_ci", 0) for n in names]

    bars1 = ax.bar(x - width/2, imbal_vals, width, label="Imbalanced F1", color="#4e79a7")
    bars2 = ax.bar(x + width/2, bal_vals, width, yerr=bal_errs,
                   label="Balanced F1 (95% CI)", capsize=4, color="#f28e2b")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Macro-F1 Comparison — Imbalanced vs Balanced Eval", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.55, 0.75)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.grid(axis='y', alpha=0.3)

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ── KMeansSMOTE with robustness ──────────────────────────────────────────────

def try_kmeans_smote(X, y, seed=42):
    counts = Counter(y.tolist())
    majority = max(counts.values())
    targets = {c: majority for c in counts if counts[c] < majority}
    min_class_size = min(counts.values())

    fit_log = {"attempts": [], "success": False, "final_settings": None}

    configs_to_try = [
        {"cluster_balance_threshold": "auto", "k_neighbors": min(5, min_class_size - 1)},
        {"cluster_balance_threshold": 0.1,    "k_neighbors": min(5, min_class_size - 1)},
        {"cluster_balance_threshold": 0.0,    "k_neighbors": min(5, min_class_size - 1)},
        {"cluster_balance_threshold": "auto", "k_neighbors": min(3, min_class_size - 1)},
        {"cluster_balance_threshold": 0.1,    "k_neighbors": min(3, min_class_size - 1)},
        {"cluster_balance_threshold": 0.0,    "k_neighbors": min(3, min_class_size - 1)},
        {"cluster_balance_threshold": 0.0,    "k_neighbors": min(2, min_class_size - 1)},
        {"cluster_balance_threshold": 0.0,    "k_neighbors": 1},
    ]

    for cfg in configs_to_try:
        attempt = {"settings": cfg.copy(), "status": None, "error": None}
        try:
            sampler = KMeansSMOTE(
                sampling_strategy=targets,
                random_state=seed,
                cluster_balance_threshold=cfg["cluster_balance_threshold"],
                k_neighbors=cfg["k_neighbors"],
            )
            X_res, y_res = sampler.fit_resample(X, y)
            attempt["status"] = "success"
            attempt["resampled_size"] = int(len(y_res))
            attempt["resampled_dist"] = {str(k): int(v) for k, v in Counter(y_res.tolist()).items()}
            fit_log["attempts"].append(attempt)
            fit_log["success"] = True
            fit_log["final_settings"] = cfg.copy()
            print(f"  KMeansSMOTE succeeded with: {cfg}")
            print(f"  Resampled: {len(y_res)} samples")
            print(f"  New dist: {dict(Counter(y_res.tolist()))}")
            return X_res, y_res, fit_log
        except Exception as e:
            attempt["status"] = "failed"
            attempt["error"] = str(e)
            fit_log["attempts"].append(attempt)
            print(f"  KMeansSMOTE failed with {cfg}: {e}")

    fit_log["success"] = False
    print("\n  WARNING: All KMeansSMOTE configs failed!")
    print("  FINDING: KMeansSMOTE cannot fit on this dataset with any tested config.")
    return None, None, fit_log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("START — KMeansSMOTE on Frozen ImageNet ResNet-18 Features (FER2013)")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Seed       : {SEED}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    # Load frozen model
    model = build_model()
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Model loaded and frozen.\n")

    # Extract features
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=img_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=img_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader)
    print(f"  Train: {train_feats.shape}, dist: {dict(Counter(train_labels.tolist()))}")

    print("Extracting val features...")
    val_feats, val_labels = extract_features(model, val_loader)
    print(f"  Val: {val_feats.shape}")

    # ── Baseline (no SMOTE) ──────────────────────────────────────────────────

    print("\n--- Baseline: linear head on original features (no SMOTE) ---")
    head_base, base_f1, base_acc, base_epoch = train_head(
        train_feats, train_labels, val_feats, val_labels)
    print(f"  Best epoch: {base_epoch}, F1={base_f1:.4f}, Acc={base_acc:.4f}")

    base_preds = predict(head_base, val_feats)
    base_imbal_f1 = f1_score(val_labels, base_preds, average="macro", zero_division=0)
    base_imbal_acc = (base_preds == val_labels).mean()
    base_imbal_pc = f1_score(val_labels, base_preds, average=None,
                              labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"  Imbalanced: F1={base_imbal_f1:.4f}, Acc={base_imbal_acc:.4f}")

    base_bal_macros, base_bal_class = balanced_eval(base_preds, val_labels)
    base_bal_mean = base_bal_macros.mean()
    base_bal_ci = 1.96 * base_bal_macros.std() / np.sqrt(len(base_bal_macros))
    print(f"  Balanced F1: {base_bal_mean:.4f} +/- {base_bal_ci:.4f}")

    # Baseline confusion matrix
    plot_confusion_matrix(val_labels, base_preds, "Baseline (Frozen ImageNet)",
                          os.path.join(FIGURES_DIR, "confusion_matrix_baseline.png"))
    print("  Saved: confusion_matrix_baseline.png")

    # ── KMeansSMOTE ──────────────────────────────────────────────────────────

    print("\n--- KMeansSMOTE ---")
    print("Attempting KMeansSMOTE with various configs...")
    X_res, y_res, fit_log = try_kmeans_smote(train_feats, train_labels, seed=SEED)

    # Save fit log
    with open(os.path.join(RESULTS_DIR, "kmeans_fit_log.json"), "w") as f:
        json.dump(fit_log, f, indent=2)
    print(f"  Saved: kmeans_fit_log.json")

    if X_res is None:
        print("\nFINDING: KMeansSMOTE could not be fitted. No resampled results.")
        print("Saving partial results and exiting.")
        partial = {
            "method": "KMeansSMOTE_frozen",
            "status": "FAILED_TO_FIT",
            "fit_log": fit_log,
            "baseline_imbal_f1": round(float(base_imbal_f1), 4),
            "baseline_bal_f1": round(float(base_bal_mean), 4),
        }
        with open(os.path.join(RESULTS_DIR, "kmeans_smote_results.json"), "w") as f:
            json.dump(partial, f, indent=2)
        print("\n" + "=" * 70)
        print("END — KMeansSMOTE (FAILED TO FIT)")
        print("=" * 70)
        return

    # Train linear head on resampled features
    print("\nTraining linear head on KMeansSMOTE-resampled features...")
    head_km, km_f1, km_acc, km_epoch = train_head(X_res, y_res, val_feats, val_labels)
    print(f"  Best epoch: {km_epoch}, F1={km_f1:.4f}, Acc={km_acc:.4f}")

    # Imbalanced eval
    km_preds = predict(head_km, val_feats)
    km_imbal_f1 = f1_score(val_labels, km_preds, average="macro", zero_division=0)
    km_imbal_acc = (km_preds == val_labels).mean()
    km_imbal_pc = f1_score(val_labels, km_preds, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"\n  Imbalanced: F1={km_imbal_f1:.4f}, Acc={km_imbal_acc:.4f}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {km_imbal_pc[i]:.4f}")

    # Balanced eval
    print(f"\n  Balanced eval ({N_DRAWS} draws, {SAMPLES_PER_CLASS}/class)...")
    km_bal_macros, km_bal_class = balanced_eval(km_preds, val_labels)
    km_bal_mean = km_bal_macros.mean()
    km_bal_ci = 1.96 * km_bal_macros.std() / np.sqrt(len(km_bal_macros))
    print(f"  Balanced F1: {km_bal_mean:.4f} +/- {km_bal_ci:.4f}")

    # ── Paired significance test vs baseline ─────────────────────────────────

    print("\n--- Paired significance test (KMeansSMOTE vs Baseline) ---")
    diff = km_bal_macros - base_bal_macros
    t_stat, t_p = stats.ttest_rel(km_bal_macros, base_bal_macros)
    w_stat, w_p = stats.wilcoxon(km_bal_macros, base_bal_macros)
    wins = int((km_bal_macros > base_bal_macros).sum())
    print(f"  Mean diff: {diff.mean():.4f} +/- {1.96*diff.std()/np.sqrt(len(diff)):.4f}")
    print(f"  t-test p={t_p:.4e}, Wilcoxon p={w_p:.4e}, wins={wins}/100")

    # ── Save results ─────────────────────────────────────────────────────────

    km_results = {
        "method": "KMeansSMOTE_frozen",
        "seed": SEED,
        "kmeans_settings": fit_log["final_settings"],
        "head_best_epoch": km_epoch,
        "imbalanced_macro_f1": round(float(km_imbal_f1), 4),
        "imbalanced_accuracy": round(float(km_imbal_acc), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(km_imbal_pc[i]), 4) for i in range(NUM_CLASSES)},
        "resampled_size": int(len(y_res)),
        "resampled_dist": {str(k): int(v) for k, v in Counter(y_res.tolist()).items()},
    }
    with open(os.path.join(RESULTS_DIR, "kmeans_smote_results.json"), "w") as f:
        json.dump(km_results, f, indent=2)

    bal_detail = {
        "n_draws": N_DRAWS,
        "samples_per_class": SAMPLES_PER_CLASS,
        "macro_f1_mean": round(float(km_bal_mean), 4),
        "macro_f1_ci95": round(float(km_bal_ci), 4),
        "per_class": {
            c: {
                "mean": round(float(km_bal_class[c].mean()), 4),
                "ci95": round(float(1.96 * km_bal_class[c].std() / np.sqrt(len(km_bal_class[c]))), 4),
            }
            for c in CLASS_NAMES
        },
        "paired_test_vs_baseline": {
            "mean_diff": round(float(diff.mean()), 4),
            "ci95_diff": round(float(1.96*diff.std()/np.sqrt(len(diff))), 4),
            "ttest_p": float(t_p),
            "wilcoxon_p": float(w_p),
            "wins": wins,
        },
    }
    with open(os.path.join(RESULTS_DIR, "balanced_eval.json"), "w") as f:
        json.dump(bal_detail, f, indent=2)

    print("  Saved: kmeans_smote_results.json, balanced_eval.json")

    # ── Figures ───────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    # Confusion matrices
    plot_confusion_matrix(val_labels, km_preds, "KMeansSMOTE (Frozen ImageNet)",
                          os.path.join(FIGURES_DIR, "confusion_matrix_kmeans_smote.png"))
    print("  Saved: confusion_matrix_kmeans_smote.png")

    # Per-class F1 comparison (balanced eval)
    # Reference data from exp_a_balanced_eval
    baseline_pc = {
        "angry":    {"mean": 0.5489, "ci95": 0.0085},
        "disgust":  {"mean": 0.5947, "ci95": 0.0100},
        "fear":     {"mean": 0.4824, "ci95": 0.0114},
        "happy":    {"mean": 0.8209, "ci95": 0.0068},
        "neutral":  {"mean": 0.6340, "ci95": 0.0102},
        "sad":      {"mean": 0.5212, "ci95": 0.0089},
        "surprise": {"mean": 0.8081, "ci95": 0.0080},
    }
    svmsmote_pc = {
        "angry":    {"mean": 0.5657, "ci95": 0.0084},
        "disgust":  {"mean": 0.7654, "ci95": 0.0072},
        "fear":     {"mean": 0.4869, "ci95": 0.0110},
        "happy":    {"mean": 0.8416, "ci95": 0.0069},
        "neutral":  {"mean": 0.6364, "ci95": 0.0093},
        "sad":      {"mean": 0.5247, "ci95": 0.0094},
        "surprise": {"mean": 0.8237, "ci95": 0.0074},
    }
    kmeans_pc = {
        c: {
            "mean": round(float(km_bal_class[c].mean()), 4),
            "ci95": round(float(1.96 * km_bal_class[c].std() / np.sqrt(len(km_bal_class[c]))), 4),
        }
        for c in CLASS_NAMES
    }

    plot_per_class_f1_comparison(baseline_pc, svmsmote_pc, kmeans_pc,
                                  os.path.join(FIGURES_DIR, "per_class_f1_comparison.png"))
    print("  Saved: per_class_f1_comparison.png")

    # Macro F1 comparison
    macro_configs = {
        "Baseline": {"imbal": 0.6440, "bal_mean": 0.6301, "bal_ci": 0.0044},
        "SVMSMOTE": {"imbal": 0.6367, "bal_mean": 0.6647, "bal_ci": 0.0025},
        "KMeansSMOTE": {"imbal": round(float(km_imbal_f1), 4),
                        "bal_mean": round(float(km_bal_mean), 4),
                        "bal_ci": round(float(km_bal_ci), 4)},
    }
    # Recalculate baseline bal CI from the reference data
    # baseline bal CI: from results_v2 paired_tests SVMSMOTE mean_diff=0.0334 ci95=0.0025
    # baseline bal mean=0.6301, so CI ~= 0.0044 (using 1.96*std/sqrt(100))
    # Actually let me use our freshly computed baseline
    macro_configs["Baseline"]["bal_mean"] = round(float(base_bal_mean), 4)
    macro_configs["Baseline"]["bal_ci"] = round(float(base_bal_ci), 4)
    macro_configs["Baseline"]["imbal"] = round(float(base_imbal_f1), 4)

    plot_macro_f1_comparison(macro_configs,
                              os.path.join(FIGURES_DIR, "macro_f1_comparison.png"))
    print("  Saved: macro_f1_comparison.png")

    # ── Comparison table ─────────────────────────────────────────────────────

    print("\n" + "=" * 140)
    print("COMPARISON TABLE")
    print("=" * 140)
    header = f"{'Config':<22s} {'Imbal F1':>9s} {'Bal F1':>16s}"
    for c in CLASS_NAMES:
        header += f" {c[:7]:>9s}"
    print(header)
    print("-" * len(header))

    # Baseline row
    row = f"{'Baseline':<22s} {0.6440:>9.4f} {0.6301:>9.4f}        "
    ref_pc = [0.6194, 0.5758, 0.4887, 0.8820, 0.5771, 0.5377, 0.7305]
    for v in ref_pc:
        row += f" {v:>9.4f}"
    print(row)

    # SVMSMOTE row
    row = f"{'SVMSMOTE frozen':<22s} {0.6367:>9.4f} {0.6647:>9.4f}        "
    svm_pc = [0.5918, 0.5079, 0.4904, 0.8756, 0.6421, 0.5615, 0.7936]
    for v in svm_pc:
        row += f" {v:>9.4f}"
    print(row)

    # KMeansSMOTE row
    row = f"{'KMeansSMOTE frozen':<22s} {km_imbal_f1:>9.4f} {km_bal_mean:.4f}+/-{km_bal_ci:.4f}"
    for c in CLASS_NAMES:
        row += f" {km_imbal_pc[CLASS_NAMES.index(c)]:>9.4f}"
    print(row)

    print("-" * len(header))

    # Balanced per-class detail
    print(f"\nBALANCED PER-CLASS F1 (100 draws, 55/class, mean +/- 95% CI):")
    print(f"{'Class':<10s} {'Baseline':>18s} {'SVMSMOTE':>18s} {'KMeansSMOTE':>18s}")
    print("-" * 70)
    for c in CLASS_NAMES:
        b = baseline_pc[c]
        s = svmsmote_pc[c]
        k = kmeans_pc[c]
        print(f"{c:<10s} {b['mean']:.4f}+/-{b['ci95']:.4f}   {s['mean']:.4f}+/-{s['ci95']:.4f}   {k['mean']:.4f}+/-{k['ci95']:.4f}")
    print(f"{'MACRO':<10s} {0.6301:.4f}+/-0.0044   {0.6647:.4f}+/-0.0025   {km_bal_mean:.4f}+/-{km_bal_ci:.4f}")

    # ── List saved files ─────────────────────────────────────────────────────

    print("\n--- Saved files ---")
    for d, label in [(RESULTS_DIR, "results"), (FIGURES_DIR, "figures")]:
        for f in sorted(os.listdir(d)):
            print(f"  {label}/{f}")

    print("\n" + "=" * 70)
    print("END — KMeansSMOTE on Frozen ImageNet ResNet-18 Features")
    print("=" * 70)


if __name__ == "__main__":
    main()
