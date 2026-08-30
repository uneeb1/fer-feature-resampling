"""
Generate comparison figures for SVMSMOTE v5 vs v7 experiment.
Reads results/smote_compare/comparison_results.json.
"""
print("=" * 60)
print("START — make_figures_smote_compare.py")
print("=" * 60)

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "./figures/smote_compare"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/smote_compare/comparison_results.json") as f:
    data = json.load(f)
summary = data["summary"]

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CONDS = ["v5_baseline", "v5_smote", "v7_baseline", "v7_smote"]
LABELS = ["v5 baseline", "v5 + SVMSMOTE", "v7 baseline", "v7 + SVMSMOTE"]
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

# ── 1. Macro-F1 4-way bar chart ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
means = [summary[c]["macro_f1_mean"] for c in CONDS]
stds = [summary[c]["macro_f1_std"] for c in CONDS]
bars = ax.bar(LABELS, means, yerr=stds, capsize=5, color=COLORS, edgecolor="black", lw=0.5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Test Macro-F1")
ax.set_title("SVMSMOTE Effect on Macro-F1 — v5 vs v7", fontsize=13, fontweight="bold")
ax.set_ylim(0.55, 0.75)
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "macrof1_4way.png"), dpi=300)
plt.close()
print("✓ Saved: macrof1_4way.png")

# ── 2. Disgust F1 4-way bar chart ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
means = [summary[c]["per_class_mean"]["disgust"] for c in CONDS]
stds = [summary[c]["per_class_std"]["disgust"] for c in CONDS]
bars = ax.bar(LABELS, means, yerr=stds, capsize=5, color=COLORS, edgecolor="black", lw=0.5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Disgust F1")
ax.set_title("SVMSMOTE Effect on Disgust F1 — v5 vs v7", fontsize=13, fontweight="bold")
ax.set_ylim(0.0, 1.0)
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "disgust_f1_4way.png"), dpi=300)
plt.close()
print("✓ Saved: disgust_f1_4way.png")

# ── 3. Per-class grouped bar chart ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(CLASS_NAMES))
width = 0.2

for i, (cond, label, color) in enumerate(zip(CONDS, LABELS, COLORS)):
    means = [summary[cond]["per_class_mean"][c] for c in CLASS_NAMES]
    stds = [summary[cond]["per_class_std"][c] for c in CLASS_NAMES]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                  label=label, color=color, edgecolor="black", lw=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("F1 Score")
ax.set_title("Per-class F1 — v5 vs v7 × Baseline vs SVMSMOTE", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_4way.png"), dpi=300)
plt.close()
print("✓ Saved: per_class_4way.png")

print("\n" + "=" * 60)
print("END — make_figures_smote_compare.py")
print("=" * 60)
