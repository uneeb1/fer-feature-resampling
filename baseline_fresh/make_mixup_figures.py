#!/usr/bin/env python3
"""Generate comparison figures: Baseline / SMOTE / Gaussian / Mixup / DC on M2 and Geometry."""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
OUT_DIR = "mixup_figures"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Baseline":  "#3C5488",
    "SMOTE*":    "#00A087",
    "Gaussian*": "#E64B35",
    "Mixup*":    "#F39B7F",
    "DC*":       "#8491B4",
}


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = {r["method"].strip(): r for r in reader}
    return rows


def get_per_class(row):
    return np.array([float(row[c]) for c in CLASSES])


def get_macro(row):
    return float(row["test_macro_mean"]), float(row["test_macro_std"])


# ── Load data ──
m2_main = load_csv("results_resampling_M2_original.csv")
m2_mixup = load_csv("results_resampling_mixup_M2.csv")
geo_main = load_csv("results_resampling_geometry.csv")
geo_mixup = load_csv("results_resampling_mixup_geometry.csv")

# Merge Mixup into the main dicts
m2 = {**m2_main, **{k: v for k, v in m2_mixup.items() if k != "Baseline"}}
geo = {**geo_main, **{k: v for k, v in geo_mixup.items() if k != "Baseline"}}

METHOD_ORDER = ["Baseline", "SMOTE*", "Gaussian*", "Mixup*", "DC*"]


def fig_macro_f1_comparison(data, title, filename):
    """Grouped bar chart: macro-F1 per method."""
    methods = [m for m in METHOD_ORDER if m in data]
    means = [get_macro(data[m])[0] for m in methods]
    stds  = [get_macro(data[m])[1] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, dpi=300)
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  capsize=4, width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Test Macro-F1", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    bl = means[0]
    ax.axhline(bl, ls="--", color=COLORS["Baseline"], alpha=0.5, lw=1)
    ymin = min(means) - 0.008
    ymax = max(means) + 0.006
    ax.set_ylim(ymin, ymax)
    fig.savefig(f"{OUT_DIR}/{filename}")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{filename}")


def fig_per_class(data, title, filename):
    """Grouped bar chart: per-class F1 for all methods."""
    methods = [m for m in METHOD_ORDER if m in data]
    n_methods = len(methods)
    n_classes = len(CLASSES)
    x = np.arange(n_classes)
    w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True, dpi=300)
    for i, m in enumerate(methods):
        vals = get_per_class(data[m])
        offset = (i - n_methods/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=m, color=COLORS[m], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Test F1", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.4, 0.96)
    fig.savefig(f"{OUT_DIR}/{filename}")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{filename}")


def fig_delta_heatmap(data, title, filename):
    """Heatmap: Δ F1 vs baseline per class per method."""
    methods = [m for m in METHOD_ORDER if m in data and m != "Baseline"]
    bl = get_per_class(data["Baseline"])
    deltas = np.array([get_per_class(data[m]) - bl for m in methods])

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True, dpi=300)
    vmax = max(abs(deltas.min()), abs(deltas.max()), 0.02)
    im = ax.imshow(deltas, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    for i in range(len(methods)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{deltas[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Δ F1 vs Baseline", shrink=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.savefig(f"{OUT_DIR}/{filename}")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{filename}")


def fig_substrate_comparison():
    """Side-by-side macro-F1: M2 vs Geometry for each method."""
    methods = [m for m in METHOD_ORDER if m in m2 and m in geo]
    m2_vals = [get_macro(m2[m])[0] for m in methods]
    geo_vals = [get_macro(geo[m])[0] for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True, dpi=300)
    b1 = ax.bar(x - w/2, m2_vals, w, label="M2 Original", color="#3C5488")
    b2 = ax.bar(x + w/2, geo_vals, w, label="Geometry", color="#E64B35")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Test Macro-F1", fontsize=11)
    ax.set_title("Macro-F1 by Method: M2 Original vs Geometry Features", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ymin = min(min(m2_vals), min(geo_vals)) - 0.008
    ymax = max(max(m2_vals), max(geo_vals)) + 0.006
    ax.set_ylim(ymin, ymax)
    fig.savefig(f"{OUT_DIR}/substrate_comparison_macro_f1.png")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/substrate_comparison_macro_f1.png")


print("Generating figures...")
fig_macro_f1_comparison(m2, "M2 Original Features — Macro-F1 by Resampling Method", "m2_macro_f1.png")
fig_macro_f1_comparison(geo, "Geometry Features — Macro-F1 by Resampling Method", "geo_macro_f1.png")
fig_per_class(m2, "M2 Original Features — Per-Class F1 by Method", "m2_per_class_f1.png")
fig_per_class(geo, "Geometry Features — Per-Class F1 by Method", "geo_per_class_f1.png")
fig_delta_heatmap(m2, "M2 Original — Δ F1 vs Baseline", "m2_delta_heatmap.png")
fig_delta_heatmap(geo, "Geometry — Δ F1 vs Baseline", "geo_delta_heatmap.png")
fig_substrate_comparison()
print("Done.")
