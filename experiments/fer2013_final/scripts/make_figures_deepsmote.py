"""Stage 4: DeepSMOTE figures."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DS = os.path.join(BASE, "results", "deepsmote")
RESULTS_SMOTE = os.path.join(BASE, "results", "smote")
LOGS_DS = os.path.join(BASE, "logs", "deepsmote")
FIGURES = os.path.join(BASE, "figures", "deepsmote")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
SVMSMOTE_FULL_F1 = {
    "angry": 0.604, "disgust": 0.626, "fear": 0.542,
    "happy": 0.880, "neutral": 0.688, "sad": 0.569, "surprise": 0.811,
}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]


def main():
    print("=" * 60)
    print("STAGE 4 — DeepSMOTE Figures")
    print("=" * 60)

    os.makedirs(FIGURES, exist_ok=True)

    with open(os.path.join(RESULTS_DS, "deepsmote_results.json")) as f:
        results = json.load(f)

    # 1. t-SNE before/after DeepSMOTE
    print("Computing t-SNE (original vs DeepSMOTE-resampled)...")
    orig_data = np.load(os.path.join(RESULTS_SMOTE, "features_train.npz"))
    ds_data = np.load(os.path.join(RESULTS_DS, "features_train_deepsmote.npz"))

    X_orig, y_orig = orig_data["features"], orig_data["labels"]
    X_ds, y_ds = ds_data["features"], ds_data["labels"]
    n_original = int(ds_data["n_original"])

    rng = np.random.RandomState(42)

    if len(X_orig) > 5000:
        idx = rng.choice(len(X_orig), 5000, replace=False)
        X_orig_sub, y_orig_sub = X_orig[idx], y_orig[idx]
    else:
        X_orig_sub, y_orig_sub = X_orig, y_orig

    if len(X_ds) > 5000:
        idx = rng.choice(len(X_ds), 5000, replace=False)
        X_ds_sub, y_ds_sub = X_ds[idx], y_ds[idx]
        is_synthetic = idx >= n_original
    else:
        X_ds_sub, y_ds_sub = X_ds, y_ds
        is_synthetic = np.arange(len(X_ds)) >= n_original

    tsne_orig = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_orig_sub)
    tsne_ds = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_ds_sub)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, emb, labels, title in [
        (axes[0], tsne_orig, y_orig_sub, "Original Train Features"),
        (axes[1], tsne_ds, y_ds_sub, "After DeepSMOTE (decoded)"),
    ]:
        for i, cls in enumerate(CLASSES):
            mask = labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1], c=COLORS[i], label=cls,
                      s=6, alpha=0.5, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=3, loc="best")

    synth_mask = is_synthetic
    if synth_mask.sum() > 0:
        axes[1].scatter(tsne_ds[synth_mask, 0], tsne_ds[synth_mask, 1],
                       facecolors="none", edgecolors="black", s=20, alpha=0.3,
                       linewidths=0.5)

    fig.suptitle("t-SNE: Train Features Before/After DeepSMOTE", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES, "tsne_deepsmote_before_after.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 2. Per-class F1 comparison: baseline vs SVMSMOTE vs DeepSMOTE
    ds_full = results["full"]["test_per_class_f1"]
    ds_means = [ds_full[c]["mean"] for c in CLASSES]
    ds_stds = [ds_full[c]["std"] for c in CLASSES]
    svmsmote_vals = [SVMSMOTE_FULL_F1[c] for c in CLASSES]
    baseline_vals = [BASELINE_TEST_F1[c] for c in CLASSES]

    x = np.arange(7)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, baseline_vals, width, label="Baseline v5", color="#4363d8", alpha=0.8)
    ax.bar(x, svmsmote_vals, width, label="SVMSMOTE (full)", color="#f58231", alpha=0.8)
    ax.bar(x + width, ds_means, width, label="DeepSMOTE (full)", color="#e6194b", alpha=0.8,
           yerr=ds_stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Test F1")
    ax.set_title("Per-Class Test F1: Baseline vs SVMSMOTE vs DeepSMOTE")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "per_class_f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 3. Reconstruction loss curve
    with open(os.path.join(LOGS_DS, "ae_history.json")) as f:
        ae_hist = json.load(f)

    epochs = [h["epoch"] for h in ae_hist]
    losses = [h["recon_loss"] for h in ae_hist]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, linewidth=1.5, color="#e6194b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Reconstruction Loss")
    ax.set_title("Autoencoder Training: Reconstruction Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "reconstruction_loss.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\n{'='*60}")
    print(f"All DeepSMOTE figures saved to: {FIGURES}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
