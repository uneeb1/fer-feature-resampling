"""
Experiment A v2: Per-class F1 under balanced eval + paired significance test.
Reuses saved heads from v1, adds:
  1. Per-class F1 breakdown (balanced eval, 100 draws)
  2. Paired per-draw difference CI + Wilcoxon signed-rank p-value
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from scipy import stats
from collections import Counter

FEATURES_DIR = "./experiments/features"
HEAD_DIR = "./experiments/exp_a_balanced_eval"
OUT_DIR = "./experiments/exp_a_balanced_eval"
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

N_DRAWS = 100
SAMPLES_PER_CLASS = 55

CONFIGS = {
    "baseline": "baseline_seed42.pth",
    "SVMSMOTE_full": "SVMSMOTE_full_seed456.pth",
    "ADASYN_minority_only": "ADASYN_minority_only_seed123.pth",
}


def load_head(path):
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    head.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    head.eval()
    return head


def predict(head, features):
    ds = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = head(X_batch.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)


def balanced_subsample_indices(labels, n_per_class, rng):
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)
    return np.array(indices)


def main():
    print(f"Device: {DEVICE}")
    val_features = np.load(os.path.join(FEATURES_DIR, "val_features.npy"))
    val_labels = np.load(os.path.join(FEATURES_DIR, "val_labels.npy"))

    heads = {}
    for name, fname in CONFIGS.items():
        path = os.path.join(HEAD_DIR, fname)
        heads[name] = load_head(path)
        print(f"Loaded {name} from {path}")

    # Precompute predictions on full val set (deterministic per head)
    full_preds = {name: predict(head, val_features) for name, head in heads.items()}

    # Run paired balanced draws
    rng = np.random.RandomState(0)
    per_draw_macro = {name: [] for name in CONFIGS}
    per_draw_class = {name: {c: [] for c in CLASS_NAMES} for name in CONFIGS}

    for draw in range(N_DRAWS):
        idx = balanced_subsample_indices(val_labels, SAMPLES_PER_CLASS, rng)
        y_true = val_labels[idx]

        for name in CONFIGS:
            y_pred = full_preds[name][idx]
            macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            per_draw_macro[name].append(macro)
            per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
            for ci, cname in enumerate(CLASS_NAMES):
                per_draw_class[name][cname].append(per_class[ci])

    # Convert to arrays
    for name in CONFIGS:
        per_draw_macro[name] = np.array(per_draw_macro[name])
        for cname in CLASS_NAMES:
            per_draw_class[name][cname] = np.array(per_draw_class[name][cname])

    # === Per-class F1 table (balanced eval) ===
    print(f"\n{'='*80}")
    print("PER-CLASS F1 UNDER BALANCED EVAL (100 draws, 55/class)")
    print(f"{'='*80}")
    print(f"{'Class':<12}", end="")
    for name in CONFIGS:
        print(f"  {name:>22}", end="")
    print(f"  {'Delta (SMOTE-BL)':>18}")
    print("-" * 98)

    results_perclass = {}
    for cname in CLASS_NAMES:
        bl = per_draw_class["baseline"][cname]
        print(f"{cname:<12}", end="")
        row = {}
        for name in CONFIGS:
            arr = per_draw_class[name][cname]
            m, ci = arr.mean(), 1.96 * arr.std() / np.sqrt(len(arr))
            print(f"  {m:>7.4f} ±{ci:.4f}     ", end="")
            row[name] = {"mean": round(float(m), 4), "ci95": round(float(ci), 4)}
        # Delta: SVMSMOTE - baseline
        diff = per_draw_class["SVMSMOTE_full"][cname] - bl
        dm, dci = diff.mean(), 1.96 * diff.std() / np.sqrt(len(diff))
        sign = "+" if dm >= 0 else ""
        print(f"  {sign}{dm:>.4f} ±{dci:.4f}")
        row["delta_svmsmote"] = {"mean": round(float(dm), 4), "ci95": round(float(dci), 4)}
        results_perclass[cname] = row
    print()

    # Macro row
    print(f"{'MACRO':<12}", end="")
    for name in CONFIGS:
        arr = per_draw_macro[name]
        m, ci = arr.mean(), 1.96 * arr.std() / np.sqrt(len(arr))
        print(f"  {m:>7.4f} ±{ci:.4f}     ", end="")
    diff_macro = per_draw_macro["SVMSMOTE_full"] - per_draw_macro["baseline"]
    dm, dci = diff_macro.mean(), 1.96 * diff_macro.std() / np.sqrt(len(diff_macro))
    print(f"  +{dm:.4f} ±{dci:.4f}")

    # === Paired significance tests ===
    print(f"\n{'='*80}")
    print("PAIRED TESTS (same 100 balanced draws)")
    print(f"{'='*80}")

    for smote_name in ["SVMSMOTE_full", "ADASYN_minority_only"]:
        diff = per_draw_macro[smote_name] - per_draw_macro["baseline"]
        mean_diff = diff.mean()
        ci_diff = 1.96 * diff.std() / np.sqrt(len(diff))
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(per_draw_macro[smote_name], per_draw_macro["baseline"])
        ttest_stat, ttest_p = stats.ttest_rel(per_draw_macro[smote_name], per_draw_macro["baseline"])

        print(f"\n{smote_name} vs baseline:")
        print(f"  Mean F1 diff:     +{mean_diff:.4f}")
        print(f"  95% CI of diff:   [{mean_diff - ci_diff:.4f}, {mean_diff + ci_diff:.4f}]")
        print(f"  Paired t-test:    t={ttest_stat:.3f}, p={ttest_p:.2e}")
        print(f"  Wilcoxon signed:  W={wilcoxon_stat:.1f}, p={wilcoxon_p:.2e}")
        print(f"  Draws where SMOTE > baseline: {(diff > 0).sum()}/{N_DRAWS}")

    # Save full results
    save_data = {
        "per_class": results_perclass,
        "paired_tests": {},
    }
    for smote_name in ["SVMSMOTE_full", "ADASYN_minority_only"]:
        diff = per_draw_macro[smote_name] - per_draw_macro["baseline"]
        _, wp = stats.wilcoxon(per_draw_macro[smote_name], per_draw_macro["baseline"])
        _, tp = stats.ttest_rel(per_draw_macro[smote_name], per_draw_macro["baseline"])
        save_data["paired_tests"][smote_name] = {
            "mean_diff": round(float(diff.mean()), 4),
            "ci95_diff": round(float(1.96 * diff.std() / np.sqrt(len(diff))), 4),
            "wilcoxon_p": float(wp),
            "ttest_p": float(tp),
            "wins": int((diff > 0).sum()),
        }

    with open(os.path.join(OUT_DIR, "results_v2.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/results_v2.json")


if __name__ == "__main__":
    main()
