"""Stage 3: Figures for SVMSMOTE ratio sweep."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results", "smote_ratio")
FIGURES = os.path.join(BASE, "figures", "smote_ratio")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
BASELINE_MACRO_F1 = 0.680


def main():
    print("=" * 60)
    print("STAGE 3 — Ratio Sweep Figures")
    print("=" * 60)

    os.makedirs(FIGURES, exist_ok=True)

    with open(os.path.join(RESULTS, "ratio_sweep_results.json")) as f:
        strategies = json.load(f)

    targets = [s["target"] for s in strategies]
    macro_means = [s["test_macro_f1_mean"] for s in strategies]
    macro_stds = [s["test_macro_f1_std"] for s in strategies]
    disgust_means = [s["test_per_class_f1"]["disgust"]["mean"] for s in strategies]
    disgust_stds = [s["test_per_class_f1"]["disgust"]["std"] for s in strategies]

    # 1. ratio_vs_macrof1.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(targets, macro_means, yerr=macro_stds, marker="o", capsize=5,
                linewidth=2, markersize=8, color="#e6194b", label="SVMSMOTE")
    ax.axhline(y=BASELINE_MACRO_F1, color="#4363d8", linestyle="--", linewidth=1.5,
               label=f"Baseline v5 ({BASELINE_MACRO_F1:.3f})")
    ax.set_xlabel("Disgust Target Count", fontsize=11)
    ax.set_ylabel("Test Macro-F1", fontsize=11)
    ax.set_title("SVMSMOTE Ratio Sweep: Test Macro-F1 vs Target Count", fontsize=12)
    ax.set_xticks(targets)
    ax.set_xticklabels([str(t) for t in targets])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "ratio_vs_macrof1.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 2. ratio_vs_disgust_f1.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(targets, disgust_means, yerr=disgust_stds, marker="s", capsize=5,
                linewidth=2, markersize=8, color="#3cb44b", label="SVMSMOTE disgust F1")
    ax.axhline(y=BASELINE_TEST_F1["disgust"], color="#4363d8", linestyle="--", linewidth=1.5,
               label=f"Baseline v5 disgust ({BASELINE_TEST_F1['disgust']:.3f})")
    ax.set_xlabel("Disgust Target Count", fontsize=11)
    ax.set_ylabel("Disgust Test F1", fontsize=11)
    ax.set_title("Disgust F1 vs Oversampling Target", fontsize=12)
    ax.set_xticks(targets)
    ax.set_xticklabels([str(t) for t in targets])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "ratio_vs_disgust_f1.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 3. per_class_f1_best_ratio.png — best strategy vs baseline
    best_idx = int(np.argmax(macro_means))
    best = strategies[best_idx]
    best_label = f"SVMSMOTE target={best['target']}"

    smote_vals = [best["test_per_class_f1"][c]["mean"] for c in CLASSES]
    smote_errs = [best["test_per_class_f1"][c]["std"] for c in CLASSES]
    baseline_vals = [BASELINE_TEST_F1[c] for c in CLASSES]

    x = np.arange(7)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, baseline_vals, width, label="Baseline v5", color="#4363d8", alpha=0.8)
    ax.bar(x + width/2, smote_vals, width, label=best_label, color="#e6194b", alpha=0.8,
           yerr=smote_errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Test F1")
    ax.set_title(f"Per-Class Test F1: Baseline v5 vs Best Ratio ({best_label})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES, "per_class_f1_best_ratio.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\n{'='*60}")
    print(f"All ratio sweep figures saved to: {FIGURES}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
