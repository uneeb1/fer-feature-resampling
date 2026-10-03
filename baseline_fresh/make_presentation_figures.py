#!/usr/bin/env python3
"""Presentation-ready figures: M2 Baseline vs Mixup."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, os

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
OUT = "presentation_figures"
os.makedirs(OUT, exist_ok=True)

def load_csv(path):
    with open(path) as f:
        return {r["method"].strip(): r for r in csv.DictReader(f)}

acc_data = load_csv("results_resampling_M2_with_accuracy.csv")
m2_data = {**load_csv("results_resampling_M2_original.csv"),
           **{k: v for k, v in load_csv("results_resampling_mixup_M2.csv").items() if k != "Baseline"}}

BL = "#3C5488"
MX = "#E64B35"

plt.rcParams.update({"font.size": 13, "font.family": "sans-serif",
                      "axes.spines.top": False, "axes.spines.right": False})


# ── 1. Macro-F1 + Accuracy side by side ──
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, dpi=300)

for ax, metric, title in zip(axes,
    [("test_macro_mean", "test_macro_std"), ("test_acc_mean", "test_acc_std")],
    ["Test Macro-F1", "Test Accuracy"]):
    bl_m = float(acc_data["Baseline"][metric[0]])
    bl_s = float(acc_data["Baseline"][metric[1]])
    mx_m = float(acc_data["Mixup*"][metric[0]])
    mx_s = float(acc_data["Mixup*"][metric[1]])
    bars = ax.bar([0, 1], [bl_m, mx_m], yerr=[bl_s, mx_s],
                  color=[BL, MX], width=0.55, capsize=6, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [bl_m, mx_m]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Mixup"], fontsize=13)
    ax.set_ylabel(title, fontsize=13)
    ymin = min(bl_m, mx_m) - 0.012
    ymax = max(bl_m, mx_m) + 0.008
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=15, fontweight="bold")

fig.suptitle("M2 Features — Baseline vs Feature-Space Mixup", fontsize=16, fontweight="bold", y=1.02)
fig.savefig(f"{OUT}/m2_baseline_vs_mixup_macro_acc.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/m2_baseline_vs_mixup_macro_acc.png")


# ── 2. Per-class F1 grouped bars ──
bl_pc = np.array([float(m2_data["Baseline"][c]) for c in CLASSES])
mx_pc = np.array([float(m2_data["Mixup*"][c]) for c in CLASSES])

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True, dpi=300)
x = np.arange(len(CLASSES))
w = 0.35
b1 = ax.bar(x - w/2, bl_pc, w, label="Baseline", color=BL, edgecolor="white", linewidth=1.2)
b2 = ax.bar(x + w/2, mx_pc, w, label="Mixup", color=MX, edgecolor="white", linewidth=1.2)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=12)
ax.set_ylabel("Test F1", fontsize=13)
ax.set_ylim(0.45, 0.95)
ax.legend(fontsize=12, loc="upper left")
ax.set_title("M2 Features — Per-Class F1: Baseline vs Mixup", fontsize=15, fontweight="bold")
fig.savefig(f"{OUT}/m2_baseline_vs_mixup_per_class.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/m2_baseline_vs_mixup_per_class.png")


# ── 3. Delta per-class (lollipop chart) ──
delta = mx_pc - bl_pc
colors = [MX if d >= 0 else BL for d in delta]

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True, dpi=300)
ax.barh(x, delta, color=colors, height=0.5, edgecolor="white", linewidth=1.2)
ax.axvline(0, color="grey", lw=0.8)
for i, d in enumerate(delta):
    ax.text(d + (0.001 if d >= 0 else -0.001), i,
            f"{d:+.4f}", ha="left" if d >= 0 else "right", va="center",
            fontsize=11, fontweight="bold")
ax.set_yticks(x); ax.set_yticklabels([c.capitalize() for c in CLASSES], fontsize=12)
ax.set_xlabel("Δ F1 (Mixup − Baseline)", fontsize=13)
ax.set_title("M2 Features — Per-Class F1 Change with Mixup", fontsize=15, fontweight="bold")
fig.savefig(f"{OUT}/m2_mixup_delta_per_class.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/m2_mixup_delta_per_class.png")


# ── 4. Class distribution (test set) with disgust highlighted ──
test_counts = [491, 55, 528, 879, 594, 416, 626]
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, dpi=300)
bar_colors = ["#8491B4"] * 7
bar_colors[1] = MX  # highlight disgust
bars = ax.bar(x, test_counts, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.6)
for bar, cnt in zip(bars, test_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=12)
ax.set_ylabel("Test Samples", fontsize=13)
ax.set_title("FER2013 Test Set — Class Distribution", fontsize=15, fontweight="bold")
fig.savefig(f"{OUT}/fer2013_test_class_distribution.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/fer2013_test_class_distribution.png")


# ── 5. All 4 methods macro-F1 bar chart (presentation-clean) ──
methods = ["Baseline", "SMOTE*", "Gaussian*", "Mixup*"]
labels  = ["Baseline", "SMOTE", "Gaussian", "Mixup"]
macro_vals = [float(m2_data[m]["test_macro_mean"]) for m in methods]
method_colors = [BL, "#00A087", "#8491B4", MX]

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, dpi=300)
bars = ax.bar(range(len(methods)), macro_vals, color=method_colors, width=0.55,
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, macro_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
            f"{val:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.axhline(macro_vals[0], ls="--", color=BL, alpha=0.4, lw=1)
ax.set_xticks(range(len(methods))); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Test Macro-F1", fontsize=13)
ymin = min(macro_vals) - 0.006; ymax = max(macro_vals) + 0.005
ax.set_ylim(ymin, ymax)
ax.set_title("M2 Features — Macro-F1 by Resampling Method", fontsize=15, fontweight="bold")
fig.savefig(f"{OUT}/m2_all_methods_macro_f1.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT}/m2_all_methods_macro_f1.png")

print("\nDone — all figures in presentation_figures/")
