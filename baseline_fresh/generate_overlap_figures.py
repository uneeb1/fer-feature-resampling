#!/usr/bin/env python3
"""Figures showing geometric class overlap in M2 feature space — the key thesis argument."""

import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
OUT = Path("mixup_experiment_figures")
os.makedirs(OUT, exist_ok=True)

CLASS_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#3C5488", "#B09C85", "#8491B4"]

plt.rcParams.update({"font.size": 12, "font.family": "sans-serif",
                      "axes.spines.top": False, "axes.spines.right": False})

Xte = np.load(FEAT_DIR / "test_features.npy"); yte = np.load(FEAT_DIR / "test_labels.npy")
print(f"Test: {Xte.shape}")

# ── t-SNE (reuse same embedding for all plots) ──
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb = tsne.fit_transform(Xte)

# ── 1. Separable vs Entangled: two-panel t-SNE ──
separable = [1, 3, 5]  # disgust, happy, surprise
entangled = [0, 2, 4, 6]  # angry, fear, sad, neutral

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True, dpi=200)

# Left: separable classes (others greyed out)
ax = axes[0]
for c in range(7):
    mask = yte == c
    if c in separable:
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7,
                   color=CLASS_COLORS[c], label=CLASSES[c].capitalize(), zorder=3)
    else:
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.08, color="#cccccc", zorder=1)
ax.legend(fontsize=11, markerscale=2.5, loc="upper right")
ax.set_title("Separable Classes\n(distinct clusters → resampling helps)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

# Right: entangled classes (others greyed out)
ax = axes[1]
for c in range(7):
    mask = yte == c
    if c in entangled:
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7,
                   color=CLASS_COLORS[c], label=CLASSES[c].capitalize(), zorder=3)
    else:
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.08, color="#cccccc", zorder=1)
ax.legend(fontsize=11, markerscale=2.5, loc="upper right")
ax.set_title("Entangled Classes\n(overlapping clusters → resampling cannot fix)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

fig.suptitle("M2 Feature Space — Why Resampling Has Class-Specific Effects",
             fontsize=16, fontweight="bold", y=1.03)
fig.savefig(OUT / "tsne_separable_vs_entangled.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/tsne_separable_vs_entangled.png")


# ── 2. Per-class silhouette scores (quantifies overlap) ──
print("Computing silhouette scores...")
sil = silhouette_samples(Xte, yte)
sil_means = [sil[yte == c].mean() for c in range(7)]

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, dpi=200)
x = np.arange(7)
bars = ax.bar(x, sil_means, color=CLASS_COLORS, edgecolor="white", linewidth=1.2, width=0.6)
for bar, val in zip(bars, sil_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
ax.set_ylabel("Mean Silhouette Score", fontsize=12)
ax.set_title("Per-Class Silhouette Score — Cluster Separability in 512-d Feature Space",
             fontsize=13, fontweight="bold")
# Annotate
ax.annotate("well-separated", xy=(3, sil_means[3]), xytext=(3, sil_means[3]+0.06),
            fontsize=10, ha="center", color="#00A087", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#00A087"))
ax.annotate("entangled", xy=(2, sil_means[2]), xytext=(2, sil_means[2]-0.07),
            fontsize=10, ha="center", color="#E64B35", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E64B35"))
fig.savefig(OUT / "silhouette_per_class.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/silhouette_per_class.png")


# ── 3. Pairwise overlap: confusion-style heatmap of misclassified neighbors ──
# For each test sample, find its 10 nearest neighbors and count how many are a different class
from sklearn.neighbors import NearestNeighbors
print("Computing neighbor overlap...")
nn = NearestNeighbors(n_neighbors=11, metric="euclidean").fit(Xte)
_, indices = nn.kneighbors(Xte)

overlap_matrix = np.zeros((7, 7))
for i in range(len(Xte)):
    true_c = yte[i]
    neighbor_labels = yte[indices[i, 1:]]  # skip self
    for nl in neighbor_labels:
        overlap_matrix[true_c, nl] += 1

# Normalize per row
row_sums = overlap_matrix.sum(axis=1, keepdims=True)
overlap_norm = overlap_matrix / row_sums

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True, dpi=200)
# Zero out diagonal for clarity
diag = np.diag(overlap_norm).copy()
overlap_display = overlap_norm.copy()
np.fill_diagonal(overlap_display, 0)

im = ax.imshow(overlap_display, cmap="Reds", vmin=0)
ax.set_xticks(range(7)); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=45, ha="right")
ax.set_yticks(range(7)); ax.set_yticklabels([c.capitalize() for c in CLASSES])
for i in range(7):
    for j in range(7):
        if i == j:
            ax.text(j, i, f"{diag[i]:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#3C5488")
        else:
            val = overlap_display[i, j]
            color = "white" if val > 0.15 else "black"
            ax.text(j, i, f"{overlap_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
fig.colorbar(im, ax=ax, shrink=0.8, label="Proportion of off-diagonal neighbors")
ax.set_xlabel("Neighbor Class", fontsize=12); ax.set_ylabel("True Class", fontsize=12)
ax.set_title("Feature-Space Neighbor Overlap (k=10 NN)\nDiagonal = same-class neighbors (blue), Off-diagonal = overlap (red)",
             fontsize=12, fontweight="bold")
fig.savefig(OUT / "neighbor_overlap_heatmap.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/neighbor_overlap_heatmap.png")


# ── 4. Combined argument figure: silhouette vs Mixup Δ F1 ──
mixup_delta = [0.0011, 0.0283, 0.0026, 0.0003, 0.0003, 0.0060, -0.0030]

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=200)
for c in range(7):
    ax.scatter(sil_means[c], mixup_delta[c], s=200, color=CLASS_COLORS[c],
               edgecolor="black", linewidth=1, zorder=3)
    ax.annotate(CLASSES[c].capitalize(), (sil_means[c], mixup_delta[c]),
                textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Silhouette Score (cluster separability)", fontsize=13)
ax.set_ylabel("Δ F1 from Mixup", fontsize=13)
ax.set_title("Resampling Gain vs Class Separability\n(separable classes benefit; entangled ones don't)",
             fontsize=14, fontweight="bold")
fig.savefig(OUT / "separability_vs_mixup_gain.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/separability_vs_mixup_gain.png")

print("\nDone — overlap/entanglement figures saved.")
