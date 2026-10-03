#!/usr/bin/env python3
"""
Feature-Space Mixup Resampling — mirrors resampling_experiment.py protocol exactly.

Mixup (within-class, random pairs, Beta(α,α) interpolation) is added as a new
method alongside SMOTE/Gaussian/DC. Uses the identical val-tune→test-once→bootstrap
protocol on frozen M2 512-d features.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import csv
import time
import warnings
warnings.filterwarnings("ignore")

FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES  = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEEDS    = [42, 123, 456]
N_BOOTSTRAP = 1000

MIXUP_ALPHA_GRID  = [0.2, 0.4, 1.0, 2.0]
MIXUP_TARGET_GRID = [2000, 3000, "full"]


def load(feat_dir):
    d = feat_dir
    Xtr = np.load(d / "train_features.npy"); ytr = np.load(d / "train_labels.npy")
    Xva = np.load(d / "val_features.npy");   yva = np.load(d / "val_labels.npy")
    Xte = np.load(d / "test_features.npy");  yte = np.load(d / "test_labels.npy")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    print(f"train counts: {dict(zip(CLASSES, np.bincount(ytr)))}")
    print(f"feature min = {Xtr.min():.4f} (>=0 expected)\n")
    return Xtr, ytr, Xva, yva, Xte, yte


def mixup_resample(Xtr, ytr, seed, alpha, target):
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr)
    tgt = counts.max() if target == "full" else int(target)
    Xout, yout = [Xtr], [ytr]
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
        Xout.append(synth)
        yout.append(np.full(n_needed, c))
    return np.vstack(Xout), np.concatenate(yout)


def fit_head(Xtr, ytr, seed):
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xtr, ytr)

def score_macro(clf, X, y):
    return f1_score(y, clf.predict(X), average="macro")

def scores_full(clf, X, y):
    pred = clf.predict(X)
    return (f1_score(y, pred, average="macro"),
            f1_score(y, pred, average=None, labels=range(len(CLASSES))))


def tune_mixup(Xtr, ytr, Xva, yva):
    print("--- Tuning Mixup on VALIDATION (alpha × target) ---")
    best = {"val_macro": -1}
    for alpha in MIXUP_ALPHA_GRID:
        for target in MIXUP_TARGET_GRID:
            macros = []
            for s in SEEDS:
                Xr, yr = mixup_resample(Xtr, ytr, s, alpha, target)
                m = score_macro(fit_head(Xr, yr, s), Xva, yva)
                macros.append(m)
            vm = float(np.mean(macros))
            print(f"  alpha={alpha}, target={target}: val macro-F1={vm:.4f}")
            if vm > best["val_macro"]:
                best = {"val_macro": vm, "alpha": alpha, "target": target}
    print(f"  -> Best: alpha={best['alpha']}, target={best['target']}, "
          f"val macro-F1={best['val_macro']:.4f}\n")
    return best


def bootstrap_delta(pred_a, pred_b, y, metric="macro", cls=None, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y); deltas = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if metric == "macro":
            a = f1_score(y[idx], pred_a[idx], average="macro")
            b = f1_score(y[idx], pred_b[idx], average="macro")
        else:
            a = f1_score(y[idx], pred_a[idx], average=None, labels=range(len(CLASSES)))[cls]
            b = f1_score(y[idx], pred_b[idx], average=None, labels=range(len(CLASSES)))[cls]
        deltas.append(b - a)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float((deltas <= 0).mean())
    return deltas.mean(), lo, hi, p


def run():
    t0 = time.time()
    out_path = Path("results_resampling_mixup_M2.csv")
    Xtr, ytr, Xva, yva, Xte, yte = load(FEAT_DIR)

    mixup_cfg = tune_mixup(Xtr, ytr, Xva, yva)

    methods = {
        "Baseline": {
            "fn": lambda s: (Xtr, ytr),
            "val_macro": None,
            "config_str": "none",
        },
        "Mixup*": {
            "fn": lambda s: mixup_resample(Xtr, ytr, s,
                                            mixup_cfg["alpha"],
                                            mixup_cfg["target"]),
            "val_macro": mixup_cfg["val_macro"],
            "config_str": f"α={mixup_cfg['alpha']}, target={mixup_cfg['target']}",
        },
    }

    results = {}
    preds_by_method = {}
    for name, spec in methods.items():
        ms, ps = [], []
        last_pred = None
        for s in SEEDS:
            Xr, yr = spec["fn"](s)
            clf = fit_head(Xr, yr, s)
            m, per = scores_full(clf, Xte, yte)
            ms.append(m); ps.append(per)
            last_pred = clf.predict(Xte)
        results[name] = {
            "macro_mean": np.mean(ms), "macro_std": np.std(ms),
            "per_mean": np.mean(ps, 0),
            "val_macro": spec["val_macro"],
            "config_str": spec["config_str"],
        }
        preds_by_method[name] = last_pred

    print("=" * 100)
    print("TEST RESULTS (mean over seeds) — * = val-tuned")
    print("=" * 100)
    hdr = f"{'Method':<14} {'MacroF1':<15} " + " ".join(f"{c[:4]:>6}" for c in CLASSES)
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<14} {r['macro_mean']:.4f}±{r['macro_std']:.4f}  "
              + " ".join(f"{v:6.3f}" for v in r['per_mean']))

    # Bootstrap
    di = CLASSES.index("disgust")
    base_pred = preds_by_method["Baseline"]
    pred = preds_by_method["Mixup*"]
    print()
    print("=" * 100)
    print("SIGNIFICANCE (bootstrap, seed-456 predictions, Mixup* vs Baseline)")
    print("=" * 100)
    print(f"{'Method':<14} {'Metric':<12} {'Delta':>8} {'95% CI':>20} {'p(no-imp)':>10}  Verdict")
    print("-" * 84)
    for metric, cls, lbl in [("macro", None, "macro-F1"), ("perclass", di, "disgust F1")]:
        d, lo, hi, p = bootstrap_delta(base_pred, pred, yte, metric=metric, cls=cls)
        sig = "SIGNIFICANT" if lo > 0 else "not significant"
        print(f"{'Mixup*':<14} {lbl:<12} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {p:10.3f}  {sig}")

    # Save CSV
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "val_macro", "test_macro_mean", "test_macro_std",
                     "config"] + CLASSES)
        for name, r in results.items():
            vm = r["val_macro"] if r["val_macro"] is not None else ""
            w.writerow([name, vm, r["macro_mean"], r["macro_std"],
                        r["config_str"], *r["per_mean"]])
    print(f"\nSaved {out_path}")
    print(f"Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
