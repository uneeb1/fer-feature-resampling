#!/usr/bin/env python3
"""Mixup experiment figures: t-SNE (real vs synthetic), confusion matrices, per-class F1, delta chart."""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEEDS = [42, 123, 456]
OUT = Path("mixup_experiment_figures")
os.makedirs(OUT, exist_ok=True)

CLASS_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#3C5488", "#B09C85", "#8491B4"]
BL_COL, MX_COL = "#3C5488", "#E64B35"

plt.rcParams.update({"font.size": 12, "font.family": "sans-serif",
                      "axes.spines.top": False, "axes.spines.right": False})

Xtr = np.load(FEAT_DIR / "train_features.npy"); ytr = np.load(FEAT_DIR / "train_labels.npy")
Xte = np.load(FEAT_DIR / "test_features.npy");  yte = np.load(FEAT_DIR / "test_labels.npy")
print(f"Loaded: train {Xtr.shape}, test {Xte.shape}")


def mixup_resample(Xtr, ytr, seed, alpha, target):
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr)
    tgt = counts.max() if target == "full" else int(target)
    Xout, yout = [], []
    for c in range(len(CLASSES)):
        Xc = Xtr[ytr == c]
        n_needed = tgt - len(Xc)
        if n_needed <= 0:
            continue
        n = len(Xc)
        idx_i = rng.integers(0, n, size=n_needed)
        idx_j = rng.integers(0, n, size=n_needed)
        lam = rng.beta(alpha, alpha, size=(n_needed, 1))
        synth = lam * Xc[idx_i] + (1.0 - lam) * Xc[idx_j]
        Xout.append(synth); yout.append(np.full(n_needed, c))
    return np.vstack(Xout), np.concatenate(yout)


# Generate Mixup synthetic samples (val-tuned config: alpha=2.0, target=full)
Xsynth, ysynth = mixup_resample(Xtr, ytr, seed=42, alpha=2.0, target="full")
print(f"Synthetic samples generated: {Xsynth.shape[0]}")
print(f"Synth per class: {dict(zip(CLASSES, np.bincount(ysynth, minlength=7)))}")

# Train baseline and Mixup classifiers (seed 42 for confusion matrices)
clf_bl = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(Xtr, ytr)
Xtr_mix = np.vstack([Xtr, Xsynth]); ytr_mix = np.concatenate([ytr, ysynth])
clf_mx = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(Xtr_mix, ytr_mix)

pred_bl = clf_bl.predict(Xte)
pred_mx = clf_mx.predict(Xte)
print(f"Baseline macro-F1: {f1_score(yte, pred_bl, average='macro'):.4f}")
print(f"Mixup   macro-F1: {f1_score(yte, pred_mx, average='macro'):.4f}")


# ── 1. t-SNE: Real train vs Mixup synthetic (minority classes only) ──
print("\nComputing t-SNE (real + synthetic, minority classes)...")
minority_classes = [c for c in range(len(CLASSES)) if np.bincount(ytr)[c] < np.bincount(ytr).max()]
minority_names = [CLASSES[c] for c in minority_classes]

# Subsample for speed: up to 1000 real + 1000 synth per class
real_idx, synth_idx = [], []
for c in minority_classes:
    rc = np.where(ytr == c)[0]
    sc = np.where(ysynth == c)[0]
    real_idx.append(rc[np.random.default_rng(42).choice(len(rc), min(800, len(rc)), replace=False)])
    synth_idx.append(sc[np.random.default_rng(42).choice(len(sc), min(800, len(sc)), replace=False)])

real_idx = np.concatenate(real_idx); synth_idx = np.concatenate(synth_idx)
X_combined = np.vstack([Xtr[real_idx], Xsynth[synth_idx]])
y_combined = np.concatenate([ytr[real_idx], ysynth[synth_idx]])
is_synth = np.concatenate([np.zeros(len(real_idx)), np.ones(len(synth_idx))])

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb = tsne.fit_transform(X_combined)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True, dpi=200)
for ax, flag, title in zip(axes, [0, 1], ["Real Train Samples", "Mixup Synthetic Samples"]):
    mask = is_synth == flag
    for c in minority_classes:
        cmask = mask & (y_combined == c)
        ax.scatter(emb[cmask, 0], emb[cmask, 1], s=10, alpha=0.6,
                   color=CLASS_COLORS[c], label=CLASSES[c].capitalize())
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
fig.suptitle("t-SNE: Real vs Mixup Synthetic Features (Minority Classes)", fontsize=15, fontweight="bold", y=1.02)
fig.savefig(OUT / "tsne_real_vs_mixup_synthetic.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/tsne_real_vs_mixup_synthetic.png")

# ── 1b. t-SNE overlay: real + synthetic together ──
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True, dpi=200)
# Plot real first (faded), then synthetic on top
for c in minority_classes:
    rmask = (is_synth == 0) & (y_combined == c)
    ax.scatter(emb[rmask, 0], emb[rmask, 1], s=12, alpha=0.3,
               color=CLASS_COLORS[c], marker="o")
for c in minority_classes:
    smask = (is_synth == 1) & (y_combined == c)
    ax.scatter(emb[smask, 0], emb[smask, 1], s=12, alpha=0.6,
               color=CLASS_COLORS[c], marker="x", label=f"{CLASSES[c].capitalize()} (synth)")
ax.legend(fontsize=9, markerscale=2, loc="best")
ax.set_title("t-SNE Overlay: Real (circles) + Mixup Synthetic (×)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
fig.savefig(OUT / "tsne_overlay_real_and_mixup.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/tsne_overlay_real_and_mixup.png")

# ── 2. Confusion matrix: Baseline vs Mixup side by side ──
cm_bl = confusion_matrix(yte, pred_bl, labels=range(len(CLASSES)))
cm_mx = confusion_matrix(yte, pred_mx, labels=range(len(CLASSES)))
cm_bl_n = cm_bl.astype(float) / cm_bl.sum(axis=1, keepdims=True)
cm_mx_n = cm_mx.astype(float) / cm_mx.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True, dpi=200)
for ax, cm_n, title in zip(axes, [cm_bl_n, cm_mx_n],
    ["Baseline (no resampling)", "Mixup (α=2.0, target=full)"]):
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels([c.capitalize() for c in CLASSES])
    for i in range(7):
        for j in range(7):
            color = "white" if cm_n[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=13, fontweight="bold")
fig.colorbar(im, ax=axes, shrink=0.7, label="Proportion")
fig.suptitle("Normalized Confusion Matrix — Baseline vs Mixup (seed 42)", fontsize=14, fontweight="bold", y=1.02)
fig.savefig(OUT / "confusion_matrix_baseline_vs_mixup.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/confusion_matrix_baseline_vs_mixup.png")

# ── 3. Per-class F1: Baseline vs Mixup (mean over 3 seeds) ──
bl_pcs, mx_pcs = [], []
for s in SEEDS:
    clf_b = LogisticRegression(max_iter=2000, C=1.0, random_state=s).fit(Xtr, ytr)
    Xs, ys = mixup_resample(Xtr, ytr, s, 2.0, "full")
    clf_m = LogisticRegression(max_iter=2000, C=1.0, random_state=s).fit(
        np.vstack([Xtr, Xs]), np.concatenate([ytr, ys]))
    bl_pcs.append(f1_score(yte, clf_b.predict(Xte), average=None, labels=range(7)))
    mx_pcs.append(f1_score(yte, clf_m.predict(Xte), average=None, labels=range(7)))

bl_pc = np.mean(bl_pcs, 0); mx_pc = np.mean(mx_pcs, 0)

x = np.arange(7); w = 0.35
fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True, dpi=200)
b1 = ax.bar(x - w/2, bl_pc, w, label="Baseline", color=BL_COL, edgecolor="white", linewidth=1.2)
b2 = ax.bar(x + w/2, mx_pc, w, label="Mixup", color=MX_COL, edgecolor="white", linewidth=1.2)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=12)
ax.set_ylabel("Test F1", fontsize=13); ax.set_ylim(0.45, 0.95)
ax.legend(fontsize=12, loc="upper left")
ax.set_title("M2 Features — Per-Class F1: Baseline vs Mixup (mean over 3 seeds)", fontsize=14, fontweight="bold")
fig.savefig(OUT / "per_class_f1_baseline_vs_mixup.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/per_class_f1_baseline_vs_mixup.png")

# ── 4. Delta per-class (horizontal bar) ──
delta = mx_pc - bl_pc
colors = [MX_COL if d >= 0 else BL_COL for d in delta]
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True, dpi=200)
ax.barh(x, delta, color=colors, height=0.5, edgecolor="white", linewidth=1.2)
ax.axvline(0, color="grey", lw=0.8)
for i, d in enumerate(delta):
    ax.text(d + (0.001 if d >= 0 else -0.001), i,
            f"{d:+.4f}", ha="left" if d >= 0 else "right", va="center",
            fontsize=11, fontweight="bold")
ax.set_yticks(x); ax.set_yticklabels([c.capitalize() for c in CLASSES], fontsize=12)
ax.set_xlabel("Δ F1 (Mixup − Baseline)", fontsize=13)
ax.set_title("Mixup Effect — Per-Class F1 Change vs Baseline", fontsize=14, fontweight="bold")
fig.savefig(OUT / "delta_per_class_mixup.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/delta_per_class_mixup.png")

# ── 5. Class count before/after Mixup ──
counts_before = np.bincount(ytr)
counts_after = np.bincount(ytr_mix)
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True, dpi=200)
b1 = ax.bar(x - w/2, counts_before, w, label="Before (original)", color=BL_COL, edgecolor="white")
b2 = ax.bar(x + w/2, counts_after,  w, label="After Mixup", color=MX_COL, edgecolor="white")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
ax.set_ylabel("Sample Count", fontsize=12)
ax.legend(fontsize=11)
ax.set_title("Training Set — Class Counts Before vs After Mixup Oversampling", fontsize=14, fontweight="bold")
fig.savefig(OUT / "class_counts_before_after_mixup.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/class_counts_before_after_mixup.png")

print("\nDone — all Mixup experiment figures saved.")
