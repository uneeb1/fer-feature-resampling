"""
Generate DeepSMOTE comparison figures.
Reads results/deepsmote/deepsmote_dual_results.json + results/smote_compare/comparison_results.json.
"""
print("=" * 60)
print("START — make_figures_deepsmote.py")
print("=" * 60)

import os
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "./figures/deepsmote"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/deepsmote/deepsmote_dual_results.json") as f:
    ds_data = json.load(f)
with open("results/smote_compare/comparison_results.json") as f:
    svm_data = json.load(f)["summary"]

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── 1. 6-way macro-F1 bar chart ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

conditions = [
    ("v5\nbaseline", 0.678, 0, "#4878CF"),
    ("v5 +\nSVMSMOTE", svm_data["v5_smote"]["macro_f1_mean"], svm_data["v5_smote"]["macro_f1_std"], "#6ACC65"),
    ("v5 +\nDeepSMOTE", ds_data["v5_deepsmote"]["macro_f1_mean"], ds_data["v5_deepsmote"]["macro_f1_std"], "#B5CF6B"),
    ("v7\nbaseline", 0.648, 0, "#D65F5F"),
    ("v7 +\nSVMSMOTE", svm_data["v7_smote"]["macro_f1_mean"], svm_data["v7_smote"]["macro_f1_std"], "#B47CC7"),
    ("v7 +\nDeepSMOTE", ds_data["v7_deepsmote"]["macro_f1_mean"], ds_data["v7_deepsmote"]["macro_f1_std"], "#C49C94"),
]

labels = [c[0] for c in conditions]
means = [c[1] for c in conditions]
stds = [c[2] for c in conditions]
colors = [c[3] for c in conditions]

bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", lw=0.5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axvline(x=2.5, color="gray", ls=":", lw=1)
ax.set_ylabel("Test Macro-F1")
ax.set_title("Macro-F1: Baseline vs SVMSMOTE vs DeepSMOTE — v5 and v7",
             fontsize=13, fontweight="bold")
ax.set_ylim(0.55, 0.72)
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "deepsmote_vs_svmsmote_macrof1.png"), dpi=300)
plt.close()
print("✓ Saved: deepsmote_vs_svmsmote_macrof1.png")

# ── 2. t-SNE: original vs DeepSMOTE synthetic ───────────────────────────────

for bname in ["v5", "v7"]:
    print(f"  Computing t-SNE for {bname}...")
    td = np.load(f"results/deepsmote/tsne_data_{bname}.npz")
    X_orig, y_orig = td["X_original"], td["y_original"]
    X_synth, y_synth = td["X_synthetic"], td["y_synthetic"]

    # Subsample for speed
    n_orig_sample = min(3000, len(X_orig))
    n_synth_sample = min(2000, len(X_synth))
    rng = np.random.RandomState(42)
    orig_idx = rng.choice(len(X_orig), n_orig_sample, replace=False)
    synth_idx = rng.choice(len(X_synth), n_synth_sample, replace=False)

    X_all = np.concatenate([X_orig[orig_idx], X_synth[synth_idx]])
    y_all = np.concatenate([y_orig[orig_idx], y_synth[synth_idx]])
    is_synth = np.concatenate([np.zeros(n_orig_sample), np.ones(n_synth_sample)])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb = tsne.fit_transform(X_all)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by class
    for i, cls in enumerate(CLASSES):
        mask = (y_all == i) & (is_synth == 0)
        axes[0].scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5, label=cls)
        mask_s = (y_all == i) & (is_synth == 1)
        axes[0].scatter(emb[mask_s, 0], emb[mask_s, 1], s=8, alpha=0.3, marker="x")
    axes[0].set_title(f"{bname} — by class (x = synthetic)", fontsize=12, fontweight="bold")
    axes[0].legend(markerscale=3, fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Right: original vs synthetic
    axes[1].scatter(emb[is_synth == 0, 0], emb[is_synth == 0, 1],
                    s=8, alpha=0.4, c="steelblue", label="Original")
    axes[1].scatter(emb[is_synth == 1, 0], emb[is_synth == 1, 1],
                    s=8, alpha=0.4, c="coral", label="DeepSMOTE synthetic")
    axes[1].set_title(f"{bname} — original vs decoded synthetic", fontsize=12, fontweight="bold")
    axes[1].legend(markerscale=3, fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.suptitle(f"DeepSMOTE t-SNE — {bname} features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"tsne_deepsmote_before_after_{bname}.png"), dpi=300)
    plt.close()
    print(f"✓ Saved: tsne_deepsmote_before_after_{bname}.png")

# ── 3. AE reconstruction loss curves ────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, bname in enumerate(["v5", "v7"]):
    hist = ds_data[f"{bname}_deepsmote"]["ae_histories"][0]  # seed 42
    epochs = [h["epoch"] for h in hist]
    losses = [h["recon_loss"] for h in hist]
    axes[idx].plot(epochs, losses, "b-", lw=1.5)
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("MSE Reconstruction Loss")
    axes[idx].set_title(f"{bname} Autoencoder Convergence", fontsize=12, fontweight="bold")
    axes[idx].grid(True, ls="--", alpha=0.4)
    axes[idx].set_yscale("log")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reconstruction_loss.png"), dpi=300)
plt.close()
print("✓ Saved: reconstruction_loss.png")

print("\n" + "=" * 60)
print("END — make_figures_deepsmote.py")
print("=" * 60)
