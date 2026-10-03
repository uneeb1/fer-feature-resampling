#!/usr/bin/env python3
"""Re-run final test eval only (no tuning) for already-selected configs on M2 features.
Reports accuracy + macro-F1 and saves to results_resampling_M2_with_accuracy.csv."""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import csv
import warnings
warnings.filterwarnings("ignore")

FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEEDS = [42, 123, 456]

Xtr = np.load(FEAT_DIR / "train_features.npy"); ytr = np.load(FEAT_DIR / "train_labels.npy")
Xva = np.load(FEAT_DIR / "val_features.npy");   yva = np.load(FEAT_DIR / "val_labels.npy")
Xte = np.load(FEAT_DIR / "test_features.npy");  yte = np.load(FEAT_DIR / "test_labels.npy")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
print(f"Test class counts: {dict(zip(CLASSES, np.bincount(yte)))}\n")


def fit_head(X, y, seed):
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, y)


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


def smote_resample(Xtr, ytr, seed, k_neighbors, target):
    counts = np.bincount(ytr)
    if target == "full":
        strategy = "auto"
    else:
        tgt = int(target)
        strategy = {c: max(counts[c], tgt) for c in range(len(CLASSES))}
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=strategy)
    return sm.fit_resample(Xtr, ytr)


def gaussian_resample(Xtr, ytr, seed, cov_method, alpha, target):
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr)
    tgt = counts.max() if target == "full" else int(target)
    Xout, yout = [Xtr], [ytr]
    for c in range(len(CLASSES)):
        Xc = Xtr[ytr == c]
        n_needed = tgt - len(Xc)
        if n_needed <= 0:
            continue
        mu = Xc.mean(0)
        if cov_method == "diagonal":
            cov = np.diag(Xc.var(axis=0) + 1e-6) * alpha
        else:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(Xc).covariance_ * alpha
        synth = rng.multivariate_normal(mu, cov, size=n_needed)
        synth = np.clip(synth, 0, None)
        Xout.append(synth)
        yout.append(np.full(n_needed, c))
    return np.vstack(Xout), np.concatenate(yout)


CONFIGS = {
    "Baseline":  {"fn": lambda s: (Xtr, ytr), "config_str": "none"},
    "Mixup*":    {"fn": lambda s: mixup_resample(Xtr, ytr, s, alpha=2.0, target="full"),
                  "config_str": "α=2.0, target=full"},
    "SMOTE*":    {"fn": lambda s: smote_resample(Xtr, ytr, s, k_neighbors=7, target=3000),
                  "config_str": "k=7, target=3000"},
    "Gaussian*": {"fn": lambda s: gaussian_resample(Xtr, ytr, s, "diagonal", 0.5, 3000),
                  "config_str": "cov=diagonal, α=0.5, target=3000"},
}

results = {}
for name, spec in CONFIGS.items():
    macros, accs, per_classes = [], [], []
    for s in SEEDS:
        Xr, yr = spec["fn"](s)
        clf = fit_head(Xr, yr, s)
        pred = clf.predict(Xte)
        macros.append(f1_score(yte, pred, average="macro"))
        accs.append(accuracy_score(yte, pred))
        per_classes.append(f1_score(yte, pred, average=None, labels=range(len(CLASSES))))
    results[name] = {
        "macro_mean": np.mean(macros), "macro_std": np.std(macros),
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "per_mean": np.mean(per_classes, 0),
        "config_str": spec["config_str"],
    }

bl_acc = results["Baseline"]["acc_mean"]
print(f"{'Method':<14} {'Macro-F1':>15} {'Accuracy':>15} {'Δ acc':>8}  Config")
print("-" * 80)
for name, r in results.items():
    delta = r["acc_mean"] - bl_acc
    print(f"{name:<14} {r['macro_mean']:.4f}±{r['macro_std']:.4f}  "
          f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}  {delta:+.4f}  {r['config_str']}")

out = Path("results_resampling_M2_with_accuracy.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "test_macro_mean", "test_macro_std",
                "test_acc_mean", "test_acc_std", "config"] + CLASSES)
    for name, r in results.items():
        w.writerow([name, r["macro_mean"], r["macro_std"],
                    r["acc_mean"], r["acc_std"], r["config_str"], *r["per_mean"]])
print(f"\nSaved {out}")
