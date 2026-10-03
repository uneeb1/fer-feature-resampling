#!/usr/bin/env python3
"""Generate diagnostic figures from the original M2 features (the ones behind results_resampling_M2_original.csv)."""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEEDS = [42, 123, 456]
OUT = Path("m2_original_figures")
os.makedirs(OUT, exist_ok=True)

CLASS_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#3C5488", "#B09C85", "#8491B4"]

# ── Load ──
Xtr = np.load(FEAT_DIR / "train_features.npy"); ytr = np.load(FEAT_DIR / "train_labels.npy")
Xva = np.load(FEAT_DIR / "val_features.npy");   yva = np.load(FEAT_DIR / "val_labels.npy")
Xte = np.load(FEAT_DIR / "test_features.npy");  yte = np.load(FEAT_DIR / "test_labels.npy")
print(f"Feature dir: {FEAT_DIR}")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
print(f"Train counts: {dict(zip(CLASSES, np.bincount(ytr)))}")
print(f"Val   counts: {dict(zip(CLASSES, np.bincount(yva)))}")
print(f"Test  counts: {dict(zip(CLASSES, np.bincount(yte)))}")

# ── Step 1: Reproduce baseline ──
print("\n--- Reproducing baseline (mean over 3 seeds) ---")
macros, per_classes = [], []
last_pred = None
for s in SEEDS:
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=s).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    m = f1_score(yte, pred, average="macro")
    pc = f1_score(yte, pred, average=None, labels=range(len(CLASSES)))
    macros.append(m); per_classes.append(pc)
    last_pred = pred
    print(f"  seed {s}: macro-F1 = {m:.4f}")

macro_mean = np.mean(macros)
pc_mean = np.mean(per_classes, 0)
print(f"\nMean macro-F1 = {macro_mean:.10f}")
print(f"Per-class F1:  {', '.join(f'{CLASSES[i]}={pc_mean[i]:.4f}' for i in range(len(CLASSES)))}")

expected_macro = 0.6874413728725074
if abs(macro_mean - expected_macro) > 1e-6:
    print(f"\n*** MISMATCH: expected {expected_macro}, got {macro_mean} ***")
    exit(1)
else:
    print(f"\nBaseline CONFIRMED: {macro_mean:.4f} matches CSV.")

# Use seed-456 predictions (last seed) for confusion matrix, matching the resampling script
pred_baseline = last_pred

plt.rcParams.update({"font.size": 12, "font.family": "sans-serif",
                      "axes.spines.top": False, "axes.spines.right": False})

# ── Figure 1: t-SNE of test features ──
print("\nComputing t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
Xte_2d = tsne.fit_transform(Xte)

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True, dpi=200)
for c in range(len(CLASSES)):
    mask = yte == c
    ax.scatter(Xte_2d[mask, 0], Xte_2d[mask, 1], s=8, alpha=0.6,
               color=CLASS_COLORS[c], label=CLASSES[c].capitalize())
ax.legend(fontsize=10, markerscale=3, loc="best")
ax.set_title("t-SNE of M2 Original Test Features (512-d)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
fig.savefig(OUT / "tsne_test_features.png")
plt.close(fig)
print(f"  saved {OUT}/tsne_test_features.png")

# ── Figure 2: Normalized confusion matrix ──
cm = confusion_matrix(yte, pred_baseline, labels=range(len(CLASSES)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True, dpi=200)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels([c.capitalize() for c in CLASSES])
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=10, color=color)
fig.colorbar(im, ax=ax, shrink=0.8, label="Proportion")
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
ax.set_title("M2 Baseline — Normalized Confusion Matrix (seed 456)", fontsize=13, fontweight="bold")
fig.savefig(OUT / "confusion_matrix_norm.png")
plt.close(fig)
print(f"  saved {OUT}/confusion_matrix_norm.png")

# ── Figure 3: Raw count confusion matrix ──
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True, dpi=200)
max_val = cm.max()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels([c.capitalize() for c in CLASSES])
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        color = "white" if cm[i, j] > max_val * 0.5 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10, color=color)
fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
ax.set_title("M2 Baseline — Confusion Matrix Counts (seed 456)", fontsize=13, fontweight="bold")
fig.savefig(OUT / "confusion_matrix_counts.png")
plt.close(fig)
print(f"  saved {OUT}/confusion_matrix_counts.png")

# ── Figure 4: Per-class F1 bar chart ──
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, dpi=200)
x = np.arange(len(CLASSES))
bars = ax.bar(x, pc_mean, color=CLASS_COLORS, edgecolor="white", linewidth=1.2, width=0.6)
for bar, val in zip(bars, pc_mean):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(macro_mean, ls="--", color="grey", lw=1, alpha=0.7, label=f"Macro-F1 = {macro_mean:.4f}")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
ax.set_ylabel("Test F1", fontsize=12)
ax.set_ylim(0.4, 0.96)
ax.legend(fontsize=10)
ax.set_title("M2 Baseline — Per-Class F1 (mean over 3 seeds)", fontsize=13, fontweight="bold")
fig.savefig(OUT / "per_class_f1.png")
plt.close(fig)
print(f"  saved {OUT}/per_class_f1.png")

# ── Figure 5: Class distribution (train / val / test) ──
tr_counts = np.bincount(ytr)
va_counts = np.bincount(yva)
te_counts = np.bincount(yte)

fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True, dpi=200)
w = 0.25
b1 = ax.bar(x - w, tr_counts, w, label="Train", color="#3C5488", edgecolor="white")
b2 = ax.bar(x,     va_counts, w, label="Val",   color="#00A087", edgecolor="white")
b3 = ax.bar(x + w, te_counts, w, label="Test",  color="#E64B35", edgecolor="white")
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 30,
                str(int(h)), ha="center", va="bottom", fontsize=7, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
ax.set_ylabel("Sample Count", fontsize=12)
ax.legend(fontsize=10)
ax.set_title("FER2013 — Class Distribution (Train / Val / Test)", fontsize=13, fontweight="bold")
fig.savefig(OUT / "class_distribution.png")
plt.close(fig)
print(f"  saved {OUT}/class_distribution.png")

print("\nDone.")
