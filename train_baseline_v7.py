"""
Baseline v7 — Two-stage transfer learning with relaxed regularization.
Stage A: frozen backbone, train head only (3 epochs).
Stage B: unfreeze all, differential LR with cosine annealing.
Relaxed vs v6: dropout 0.4, WD 5e-4, backbone LR 2e-5, head LR 1e-4, patience 4.
Goal: recover F1 performance while keeping clean training curve.
"""

import os
import sys
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────

SEED            = 42
DATA_DIR        = "./data"
RESULTS_DIR     = "./results/baseline_v7"
FIGURES_DIR     = "./figures/baseline_v7"
LOGS_DIR        = "./logs"
NUM_CLASSES     = 7
BATCH_SIZE      = 64
MAX_EPOCHS      = 30
STAGE_A_EPOCHS  = 3
HEAD_LR_A       = 1e-3
BACKBONE_LR_B   = 2e-5
HEAD_LR_B       = 1e-4
WEIGHT_DECAY    = 5e-4
DROPOUT         = 0.4
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM  = 1.0
PATIENCE        = 4
VAL_SPLIT       = 0.2
NUM_WORKERS     = 4
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ── Dirs ─────────────────────────────────────────────────────────────────────

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Transforms ───────────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Datasets ─────────────────────────────────────────────────────────────────

full_train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=train_transforms)
full_train_eval = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=eval_transforms)
test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"), transform=eval_transforms)

targets = np.array(full_train_dataset.targets)
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(targets, targets))

train_subset = Subset(full_train_dataset, train_idx)
val_subset = Subset(full_train_eval, val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True)

print(f"Device       : {DEVICE}")
print(f"Train samples: {len(train_subset)}")
print(f"Val samples  : {len(val_subset)}")
print(f"Test samples : {len(test_dataset)}")

# ── Model ────────────────────────────────────────────────────────────────────

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# ── History ──────────────────────────────────────────────────────────────────

history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

# ── Training functions ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    return (running_loss / n, correct / total,
            f1_score(all_labels, all_preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    return (running_loss / n, correct / total,
            f1_score(all_labels, all_preds, average="macro", zero_division=0),
            np.array(all_preds), np.array(all_labels))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE A — Frozen backbone, train head only
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE A — Warmup (backbone FROZEN, head only)")
print("=" * 60)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer_a = optim.Adam(model.fc.parameters(), lr=HEAD_LR_A,
                         weight_decay=WEIGHT_DECAY)

for epoch in range(1, STAGE_A_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc, tr_f1 = train_one_epoch(
        model, train_loader, optimizer_a, criterion, DEVICE, GRAD_CLIP_NORM)
    vl_loss, vl_acc, vl_f1, _, _ = evaluate(
        model, val_loader, criterion, DEVICE)
    elapsed = time.time() - t0

    rec = {"epoch": epoch, "phase": "A",
           "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
           "val_loss": vl_loss, "val_acc": vl_acc, "val_macro_f1": vl_f1}
    history.append(rec)

    print(f"  [{epoch}/{STAGE_A_EPOCHS}] "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
          f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f} "
          f"({elapsed:.1f}s)")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())

# ═══════════════════════════════════════════════════════════════════════════
# STAGE B — Unfreeze all, differential LR + cosine annealing
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE B — Fine-tune (all layers UNFROZEN)")
print("=" * 60)

for param in model.parameters():
    param.requires_grad = True

backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
head_params = [p for n, p in model.named_parameters() if "fc" in n]

optimizer_b = optim.Adam([
    {"params": backbone_params, "lr": BACKBONE_LR_B},
    {"params": head_params,     "lr": HEAD_LR_B},
], weight_decay=WEIGHT_DECAY)

remaining_epochs = MAX_EPOCHS - STAGE_A_EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=remaining_epochs)

for epoch_b in range(1, remaining_epochs + 1):
    global_epoch = STAGE_A_EPOCHS + epoch_b
    t0 = time.time()
    tr_loss, tr_acc, tr_f1 = train_one_epoch(
        model, train_loader, optimizer_b, criterion, DEVICE, GRAD_CLIP_NORM)
    vl_loss, vl_acc, vl_f1, _, _ = evaluate(
        model, val_loader, criterion, DEVICE)
    scheduler.step()
    elapsed = time.time() - t0

    rec = {"epoch": global_epoch, "phase": "B",
           "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
           "val_loss": vl_loss, "val_acc": vl_acc, "val_macro_f1": vl_f1}
    history.append(rec)

    print(f"  [{global_epoch}/{MAX_EPOCHS}] "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
          f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f} "
          f"({elapsed:.1f}s)")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_epoch = global_epoch
        patience_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(RESULTS_DIR, "best_model.pth"))
        print(f"    ✓ New best (val macro-F1={best_val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"    No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n⚡ Early stopping at epoch {global_epoch}")
            break

# ── Save history ─────────────────────────────────────────────────────────────

with open(os.path.join(LOGS_DIR, "baseline_v7_history_s42.json"), "w") as f:
    json.dump(history, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION — Val + Test (separate)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EVALUATION — Best model")
print("=" * 60)

model.load_state_dict(best_model_wts)

_, val_acc, val_f1, val_preds, val_labels = evaluate(
    model, val_loader, criterion, DEVICE)
_, test_acc, test_f1, test_preds, test_labels = evaluate(
    model, test_loader, criterion, DEVICE)

val_report = classification_report(
    val_labels, val_preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True)
test_report = classification_report(
    test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True)
val_report_str = classification_report(
    val_labels, val_preds, target_names=CLASS_NAMES, zero_division=0)
test_report_str = classification_report(
    test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0)

# Train/val gap at best epoch
best_rec = history[best_epoch - 1]
train_acc_best = best_rec["train_acc"]
gap = (train_acc_best - val_acc) * 100

print(f"\nBest epoch       : {best_epoch}")
print(f"Val  macro-F1    : {val_f1:.4f}")
print(f"Val  accuracy    : {val_acc:.4f}")
print(f"Test macro-F1    : {test_f1:.4f}")
print(f"Test accuracy    : {test_acc:.4f}")
print(f"Train/val gap    : {gap:.1f}%")
print(f"\n--- Val Classification Report ---\n{val_report_str}")
print(f"--- Test Classification Report ---\n{test_report_str}")

# Per-class test F1
per_class_f1 = {c: test_report[c]["f1-score"] for c in CLASS_NAMES}
print("Per-class Test F1:")
for c, f in per_class_f1.items():
    print(f"  {c:10s}: {f:.4f}")

# ── Save results ─────────────────────────────────────────────────────────────

results = {
    "best_epoch": best_epoch,
    "val_macro_f1": val_f1,
    "val_accuracy": val_acc,
    "test_macro_f1": test_f1,
    "test_accuracy": test_acc,
    "train_val_gap_pct": round(gap, 2),
    "per_class_test_f1": per_class_f1,
    "config": {
        "stage_a_epochs": STAGE_A_EPOCHS, "head_lr_a": HEAD_LR_A,
        "backbone_lr_b": BACKBONE_LR_B, "head_lr_b": HEAD_LR_B,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING, "grad_clip": GRAD_CLIP_NORM,
        "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "seed": SEED,
    }
}
with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
    f.write(f"Baseline v7 — Two-stage Transfer Learning\n")
    f.write(f"{'='*50}\n")
    f.write(f"Best epoch       : {best_epoch}\n")
    f.write(f"Val  macro-F1    : {val_f1:.4f}\n")
    f.write(f"Val  accuracy    : {val_acc:.4f}\n")
    f.write(f"Test macro-F1    : {test_f1:.4f}\n")
    f.write(f"Test accuracy    : {test_acc:.4f}\n")
    f.write(f"Train/val gap    : {gap:.1f}%\n\n")
    f.write(f"--- Val Report ---\n{val_report_str}\n")
    f.write(f"--- Test Report ---\n{test_report_str}\n")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

epochs_all = [r["epoch"] for r in history]
tr_losses = [r["train_loss"] for r in history]
vl_losses = [r["val_loss"] for r in history]
tr_accs = [r["train_acc"] * 100 for r in history]
vl_accs = [r["val_acc"] * 100 for r in history]
tr_f1s = [r["train_f1"] for r in history]
vl_f1s = [r["val_macro_f1"] for r in history]

stage_a_end = STAGE_A_EPOCHS

def plot_training_curves(epochs, tr, vl, ylabel, title, fname, best_ep, cutoff=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    n = cutoff if cutoff else len(epochs)
    e, t, v = epochs[:n], tr[:n], vl[:n]
    ax.plot(e, t, "b-o", ms=4, label=f"Train {ylabel}")
    ax.plot(e, v, "r-o", ms=4, label=f"Val {ylabel}")
    ax.axvline(x=stage_a_end + 0.5, color="gray", ls="--", lw=1.5,
               label="Freeze→Unfreeze")
    if best_ep <= e[-1]:
        ax.axvline(x=best_ep, color="green", ls="--", lw=1.5,
                   label=f"Best epoch ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close()
    print(f"✓ Saved: {fname}")

# 1. Clean curves (up to best epoch)
plot_training_curves(epochs_all, tr_losses, vl_losses, "Loss",
                     "Baseline v7 — Loss (to best epoch)", "training_curves_loss.png",
                     best_epoch, cutoff=best_epoch)
plot_training_curves(epochs_all, tr_accs, vl_accs, "Accuracy (%)",
                     "Baseline v7 — Accuracy (to best epoch)", "training_curves_acc.png",
                     best_epoch, cutoff=best_epoch)
plot_training_curves(epochs_all, tr_f1s, vl_f1s, "Macro-F1",
                     "Baseline v7 — Macro-F1 (to best epoch)", "training_curves.png",
                     best_epoch, cutoff=best_epoch)

# 2. Full curves (appendix)
plot_training_curves(epochs_all, tr_losses, vl_losses, "Loss",
                     "Baseline v7 — Loss (full run)", "training_curves_full_loss.png",
                     best_epoch)
plot_training_curves(epochs_all, tr_accs, vl_accs, "Accuracy (%)",
                     "Baseline v7 — Accuracy (full run)", "training_curves_full_acc.png",
                     best_epoch)
plot_training_curves(epochs_all, tr_f1s, vl_f1s, "Macro-F1",
                     "Baseline v7 — Macro-F1 (full run)", "training_curves_full.png",
                     best_epoch)

# 3. Confusion matrix (test)
fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title("Baseline v7 — Test Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("✓ Saved: confusion_matrix.png")

# 4. Per-class F1 bar chart (test)
fig, ax = plt.subplots(figsize=(9, 5))
f1_vals = [per_class_f1[c] for c in CLASS_NAMES]
colors = sns.color_palette("Set2", NUM_CLASSES)
bars = ax.bar(CLASS_NAMES, f1_vals, color=colors, edgecolor="black", lw=0.5)
for bar, v in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(y=test_f1, color="red", ls="--", lw=1.5, label=f"Macro-F1={test_f1:.4f}")
ax.set_ylabel("F1 Score")
ax.set_title("Baseline v7 — Per-class Test F1", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "per_class_f1.png"), dpi=150)
plt.close()
print("✓ Saved: per_class_f1.png")

# 5. Class distribution
fig, ax = plt.subplots(figsize=(9, 5))
train_counts = np.bincount(targets, minlength=NUM_CLASSES)
test_targets = np.array(test_dataset.targets)
test_counts = np.bincount(test_targets, minlength=NUM_CLASSES)
x = np.arange(NUM_CLASSES)
w = 0.35
ax.bar(x - w/2, train_counts, w, label="Train", color="steelblue", edgecolor="black", lw=0.5)
ax.bar(x + w/2, test_counts, w, label="Test", color="coral", edgecolor="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Class Distribution — Train vs Test", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("✓ Saved: class_distribution.png")

# 6. t-SNE of test features
print("Computing t-SNE (this takes a moment)...")
feature_model = nn.Sequential(*list(model.children())[:-1])
feature_model.eval()
features_list, labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        feats = feature_model(inputs).squeeze()
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
features_np = np.concatenate(features_list)
labels_np = np.concatenate(labels_list)

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
embedded = tsne.fit_transform(features_np)

fig, ax = plt.subplots(figsize=(10, 8))
for i, name in enumerate(CLASS_NAMES):
    mask = labels_np == i
    ax.scatter(embedded[mask, 0], embedded[mask, 1], s=8, alpha=0.6, label=name)
ax.set_title("Baseline v7 — t-SNE of Test Features (512-d)", fontsize=13, fontweight="bold")
ax.legend(markerscale=3, fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne.png"), dpi=150)
plt.close()
print("✓ Saved: tsne.png")

# ── Final summary ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BASELINE v7 COMPLETE")
print("=" * 60)
print(f"Best epoch       : {best_epoch}")
print(f"Val  macro-F1    : {val_f1:.4f}")
print(f"Test macro-F1    : {test_f1:.4f}")
print(f"Test accuracy    : {test_acc:.4f}")
print(f"Train/val gap    : {gap:.1f}%")
print(f"Results          : {RESULTS_DIR}/")
print(f"Figures          : {FIGURES_DIR}/")
print(f"History          : {LOGS_DIR}/baseline_v7_history_s42.json")
