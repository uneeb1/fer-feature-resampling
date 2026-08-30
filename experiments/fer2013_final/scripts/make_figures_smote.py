"""Generate SMOTE-specific figures (Stage 2)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_SMOTE = os.path.join(BASE, "results", "smote")
FIGURES = os.path.join(BASE, "figures", "smote")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_SHORT = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
BASELINE_MACRO_F1 = 0.680

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]


def main():
    print("=" * 60)
    print("STAGE 2 — Step 3: SMOTE Figures")
    print("=" * 60)

    os.makedirs(FIGURES, exist_ok=True)

    with open(os.path.join(RESULTS_SMOTE, "smote_results.json")) as f:
        results = json.load(f)

    # ---- 1. Confusion matrix (seed 42) ----
    cm = np.array(results["per_seed"][0]["test"]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in [
        (axes[0], cm, "SVMSMOTE — Test Confusion (Raw)", "d"),
        (axes[1], cm_norm, "SVMSMOTE — Test Confusion (Row-Normalized)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CLASS_SHORT, fontsize=9)
        ax.set_yticklabels(CLASS_SHORT, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        for i in range(7):
            for j in range(7):
                val = data[i, j]
                text = f"{val:{fmt}}" if fmt == "d" else f"{val:{fmt}}"
                color = "white" if data[i, j] > data.max() * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(FIGURES, "confusion_matrix_smote.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- 2. Per-class F1 comparison ----
    smote_per_class = results["mean_std"]["test"]["per_class_f1"]
    smote_means = [smote_per_class[c]["mean"] for c in CLASSES]
    smote_stds = [smote_per_class[c]["std"] for c in CLASSES]
    baseline_vals = [BASELINE_TEST_F1[c] for c in CLASSES]

    x = np.arange(7)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, baseline_vals, width, label="Baseline v5", color="#4363d8", alpha=0.8)
    ax.bar(x + width/2, smote_means, width, label="SVMSMOTE", color="#e6194b", alpha=0.8,
           yerr=smote_stds, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Test F1"); ax.set_title("Per-Class Test F1: Baseline v5 vs SVMSMOTE")
    ax.legend(); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "per_class_f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- 3. Macro-F1 comparison ----
    sm = results["mean_std"]["test"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Baseline v5", "SVMSMOTE"],
           [BASELINE_MACRO_F1, sm["macro_f1_mean"]],
           yerr=[0, sm["macro_f1_std"]],
           color=["#4363d8", "#e6194b"], alpha=0.8, capsize=6, width=0.5)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Macro-F1: Baseline v5 vs SVMSMOTE")
    ax.set_ylim(0.5, 0.8)
    ax.grid(axis="y", alpha=0.3)
    for i, (v, s) in enumerate([(BASELINE_MACRO_F1, 0), (sm["macro_f1_mean"], sm["macro_f1_std"])]):
        ax.text(i, v + s + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES, "macro_f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- 4. t-SNE before/after SMOTE ----
    print("\nComputing t-SNE (train, before/after SMOTE)...")
    train_orig = np.load(os.path.join(RESULTS_SMOTE, "features_train.npz"))
    train_resamp = np.load(os.path.join(RESULTS_SMOTE, "features_train_resampled.npz"))

    X_orig, y_orig = train_orig["features"], train_orig["labels"]
    X_resamp, y_resamp = train_resamp["features"], train_resamp["labels"]
    n_original = int(train_resamp["n_original"])

    # Subsample for speed (max 5000 per panel)
    rng = np.random.RandomState(42)
    n_orig = len(X_orig)
    if n_orig > 5000:
        idx = rng.choice(n_orig, 5000, replace=False)
        X_orig_sub, y_orig_sub = X_orig[idx], y_orig[idx]
    else:
        X_orig_sub, y_orig_sub = X_orig, y_orig

    n_resamp = len(X_resamp)
    if n_resamp > 5000:
        idx = rng.choice(n_resamp, 5000, replace=False)
        X_resamp_sub, y_resamp_sub = X_resamp[idx], y_resamp[idx]
        is_synthetic_sub = idx >= n_original
    else:
        X_resamp_sub, y_resamp_sub = X_resamp, y_resamp
        is_synthetic_sub = np.arange(n_resamp) >= n_original

    tsne_orig = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_orig_sub)
    tsne_resamp = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_resamp_sub)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, emb, labels, title in [
        (axes[0], tsne_orig, y_orig_sub, "Original Train Features"),
        (axes[1], tsne_resamp, y_resamp_sub, "After SVMSMOTE"),
    ]:
        for i, cls in enumerate(CLASSES):
            mask = labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1], c=COLORS[i], label=cls,
                      s=6, alpha=0.5, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=3, loc="best")

    # Mark synthetic points on right panel
    synth_mask = is_synthetic_sub
    if synth_mask.sum() > 0:
        axes[1].scatter(tsne_resamp[synth_mask, 0], tsne_resamp[synth_mask, 1],
                       facecolors="none", edgecolors="black", s=20, alpha=0.3,
                       linewidths=0.5, label="synthetic")

    fig.suptitle("t-SNE: Train Features Before/After SVMSMOTE", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES, "tsne_before_after.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- 5. t-SNE test features ----
    print("Computing t-SNE (test)...")
    test_data = np.load(os.path.join(RESULTS_SMOTE, "features_test.npz"))
    X_test, y_test = test_data["features"], test_data["labels"]
    if len(X_test) > 5000:
        idx = rng.choice(len(X_test), 5000, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    tsne_test = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_test)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(CLASSES):
        mask = y_test == i
        ax.scatter(tsne_test[mask, 0], tsne_test[mask, 1], c=COLORS[i], label=cls,
                  s=8, alpha=0.5, edgecolors="none")
    ax.set_title("t-SNE: Test Features (v5 Backbone)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, markerscale=3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "tsne_test_smote.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\n{'='*60}")
    print("All SMOTE figures saved to:", FIGURES)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
