#!/usr/bin/env python3
"""
Feature-Space Resampling Experiment v2 — SYMMETRIC protocol.

Every method (SMOTE, Gaussian, DC) is evaluated under the identical protocol:
  1. Val-tune over its FULL hyperparameter space (val macro-F1, mean over seeds)
  2. Report the val-selected config ONCE on test (mean over seeds)
  3. Bootstrap significance vs baseline (macro-F1 + disgust F1, 95% CI)

Each method is tuned over its full available hyperparameter space under the
identical val→test-once→bootstrap protocol. SMOTE has fewer knobs (2 axes)
because interpolation has no covariance or variance-scale parameter — the
protocol is identical; parameter count differs by method.

Methods:
  - Baseline: no resampling (reference)
  - SMOTE*: val-tuned (k_neighbors × target)
  - Gaussian*: val-tuned (cov_method × alpha × target)
  - DC*: val-tuned (k × alpha × target), Tukey λ=0.5 fixed

Runs locally on CPU in ~5 min on frozen 512-d M2 features.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import f1_score
import csv
import time
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
FEAT_DIR = Path("M2_seed42_backup (1)/features")
CLASSES  = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEEDS    = [42, 123, 456]
N_BOOTSTRAP = 1000

# --- SMOTE search grid (2 axes: k_neighbors × target) ---
SMOTE_K_GRID      = [3, 5, 7]
SMOTE_TARGET_GRID = [2000, 3000, "full"]

# --- Gaussian search grid (3 axes: cov × alpha × target) ---
GAUSS_COV_GRID    = ["ledoitwolf", "ridge", "diagonal"]
GAUSS_ALPHA_GRID  = [0.5, 0.75, 1.0, 1.5]
GAUSS_TARGET_GRID = [2000, 3000, "full"]

# --- DC search grid (3 axes: k × alpha × target) ---
# Tukey λ fixed at 0.5 (sqrt): standard for non-negative features, keeps DC
# at 3 tuned axes matching Gaussian's tuning budget.
DC_TUKEY_LAMBDA   = 0.5
DC_K_GRID         = [1, 2, 3]
DC_ALPHA_GRID     = [0.1, 0.21, 0.5]
DC_TARGET_GRID    = [2000, 3000, "full"]


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load():
    d = FEAT_DIR
    Xtr = np.load(d / "train_features.npy"); ytr = np.load(d / "train_labels.npy")
    Xva = np.load(d / "val_features.npy");   yva = np.load(d / "val_labels.npy")
    Xte = np.load(d / "test_features.npy");  yte = np.load(d / "test_labels.npy")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    print(f"train counts: {dict(zip(CLASSES, np.bincount(ytr)))}")
    print(f"feature min = {Xtr.min():.4f} (>=0 expected)\n")
    return Xtr, ytr, Xva, yva, Xte, yte


# ----------------------------------------------------------------------
# Covariance estimators (for Gaussian)
# ----------------------------------------------------------------------
def estimate_cov(Xc, method):
    d = Xc.shape[1]
    if len(Xc) < 2:
        return np.eye(d) * 1e-2
    if method == "ledoitwolf":
        return LedoitWolf().fit(Xc).covariance_
    if method == "diagonal":
        return np.diag(Xc.var(axis=0) + 1e-6)
    return np.cov(Xc, rowvar=False) + 1e-2 * np.eye(d)


# ----------------------------------------------------------------------
# SMOTE resampling
# ----------------------------------------------------------------------
def smote_resample(Xtr, ytr, seed, k_neighbors, target):
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(ytr)
    max_count = counts.max()
    if target == "full":
        strategy = "auto"
    else:
        tgt = int(target)
        strategy = {c: max(counts[c], tgt) for c in range(len(CLASSES))}
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors,
               sampling_strategy=strategy)
    return sm.fit_resample(Xtr, ytr)


# ----------------------------------------------------------------------
# Gaussian oversampling
# ----------------------------------------------------------------------
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
        mu  = Xc.mean(0)
        cov = estimate_cov(Xc, cov_method) * alpha
        synth = rng.multivariate_normal(mu, cov, size=n_needed)
        synth = np.clip(synth, 0, None)
        Xout.append(synth); yout.append(np.full(n_needed, c))
    return np.vstack(Xout), np.concatenate(yout)


# ----------------------------------------------------------------------
# Distribution Calibration (DC)
# ----------------------------------------------------------------------
def tukey(x, lam=DC_TUKEY_LAMBDA):
    return np.power(np.clip(x, 1e-6, None), lam) if lam != 0 else np.log(np.clip(x, 1e-6, None))


def dc_resample(Xtr, ytr, seed, k, alpha, target):
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr)
    tgt = counts.max() if target == "full" else int(target)
    Xt = tukey(Xtr)

    mu  = {c: Xt[ytr == c].mean(0) for c in range(len(CLASSES))}
    cov = {c: np.cov(Xt[ytr == c], rowvar=False) for c in range(len(CLASSES))}

    d = Xt.shape[1]
    # precompute Cholesky for each unique (class-set, alpha) to speed up sampling
    Xout, yout = [Xt], [ytr]
    for c in range(len(CLASSES)):
        Xc = Xt[ytr == c]
        n_needed = tgt - len(Xc)
        if n_needed <= 0:
            continue
        per_sample = max(1, n_needed // len(Xc) + 1)
        others = [cc for cc in range(len(CLASSES)) if cc != c]
        gen = []
        for x in Xc:
            dists = sorted(others, key=lambda cc: np.sum((mu[cc] - x) ** 2))[:k]
            cal_mu = (np.sum([mu[cc] for cc in dists], 0) + x) / (k + 1)
            cal_cov = np.mean([cov[cc] for cc in dists], 0) + alpha * np.eye(d)
            L = np.linalg.cholesky(cal_cov)
            z = rng.standard_normal((per_sample, d))
            samples = cal_mu + z @ L.T
            gen.append(samples)
        gen = np.vstack(gen)[:n_needed]
        Xout.append(gen)
        yout.append(np.full(len(gen), c))
    return np.vstack(Xout), np.concatenate(yout)


# ----------------------------------------------------------------------
# Head train/eval
# ----------------------------------------------------------------------
def fit_head(Xtr, ytr, seed):
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xtr, ytr)

def score_macro(clf, X, y):
    return f1_score(y, clf.predict(X), average="macro")

def scores_full(clf, X, y):
    pred = clf.predict(X)
    return (f1_score(y, pred, average="macro"),
            f1_score(y, pred, average=None, labels=range(len(CLASSES))))


# ----------------------------------------------------------------------
# Val-tuning helpers
# ----------------------------------------------------------------------
def tune_smote(Xtr, ytr, Xva, yva):
    print("--- Tuning SMOTE on VALIDATION (k_neighbors × target) ---")
    best = {"val_macro": -1}
    for k in SMOTE_K_GRID:
        for target in SMOTE_TARGET_GRID:
            macros = []
            for s in SEEDS:
                Xr, yr = smote_resample(Xtr, ytr, s, k, target)
                m = score_macro(fit_head(Xr, yr, s), Xva, yva)
                macros.append(m)
            vm = float(np.mean(macros))
            if vm > best["val_macro"]:
                best = {"val_macro": vm, "k_neighbors": k, "target": target}
    print(f"  -> Best: k={best['k_neighbors']}, target={best['target']}, "
          f"val macro-F1={best['val_macro']:.4f}\n")
    return best


def tune_gaussian(Xtr, ytr, Xva, yva):
    print("--- Tuning Gaussian on VALIDATION (cov × alpha × target) ---")
    best = {"val_macro": -1}
    for cov in GAUSS_COV_GRID:
        for alpha in GAUSS_ALPHA_GRID:
            for target in GAUSS_TARGET_GRID:
                macros = []
                for s in SEEDS:
                    Xr, yr = gaussian_resample(Xtr, ytr, s, cov, alpha, target)
                    m = score_macro(fit_head(Xr, yr, s), Xva, yva)
                    macros.append(m)
                vm = float(np.mean(macros))
                if vm > best["val_macro"]:
                    best = {"val_macro": vm, "cov": cov, "alpha": alpha, "target": target}
    print(f"  -> Best: cov={best['cov']}, alpha={best['alpha']}, "
          f"target={best['target']}, val macro-F1={best['val_macro']:.4f}\n")
    return best


def tune_dc(Xtr, ytr, Xva, yva):
    print("--- Tuning DC on VALIDATION (k × alpha × target), λ=0.5 fixed ---")
    Xva_tukey = tukey(Xva)
    best = {"val_macro": -1}
    total = len(DC_K_GRID) * len(DC_ALPHA_GRID) * len(DC_TARGET_GRID)
    i = 0
    for k in DC_K_GRID:
        for alpha in DC_ALPHA_GRID:
            for target in DC_TARGET_GRID:
                i += 1
                if i % 9 == 0 or i == total:
                    print(f"  [{i}/{total}] ...", flush=True)
                macros = []
                for s in SEEDS:
                    Xr, yr = dc_resample(Xtr, ytr, s, k, alpha, target)
                    m = score_macro(fit_head(Xr, yr, s), Xva_tukey, yva)
                    macros.append(m)
                vm = float(np.mean(macros))
                if vm > best["val_macro"]:
                    best = {"val_macro": vm, "k": k, "alpha": alpha, "target": target}
    print(f"  -> Best: k={best['k']}, alpha={best['alpha']}, "
          f"target={best['target']}, val macro-F1={best['val_macro']:.4f}\n")
    return best


# ----------------------------------------------------------------------
# Bootstrap significance (method vs baseline)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run():
    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte = load()

    # ---- 1. Val-tune all methods ----
    smote_cfg   = tune_smote(Xtr, ytr, Xva, yva)
    gauss_cfg   = tune_gaussian(Xtr, ytr, Xva, yva)
    dc_cfg      = tune_dc(Xtr, ytr, Xva, yva)

    # ---- 2. Evaluate on TEST (once per method, mean over seeds) ----
    Xte_tukey = tukey(Xte)

    methods = {
        "Baseline": {
            "fn": lambda s: (Xtr, ytr),
            "Xeval": Xte,
            "tuned": False,
            "val_macro": None,
            "config_str": "none",
        },
        "SMOTE*": {
            "fn": lambda s: smote_resample(Xtr, ytr, s,
                                           smote_cfg["k_neighbors"],
                                           smote_cfg["target"]),
            "Xeval": Xte,
            "tuned": True,
            "val_macro": smote_cfg["val_macro"],
            "config_str": f"k={smote_cfg['k_neighbors']}, target={smote_cfg['target']}",
        },
        "Gaussian*": {
            "fn": lambda s: gaussian_resample(Xtr, ytr, s,
                                              gauss_cfg["cov"],
                                              gauss_cfg["alpha"],
                                              gauss_cfg["target"]),
            "Xeval": Xte,
            "tuned": True,
            "val_macro": gauss_cfg["val_macro"],
            "config_str": f"cov={gauss_cfg['cov']}, α={gauss_cfg['alpha']}, target={gauss_cfg['target']}",
        },
        "DC*": {
            "fn": lambda s: dc_resample(Xtr, ytr, s,
                                        dc_cfg["k"],
                                        dc_cfg["alpha"],
                                        dc_cfg["target"]),
            "Xeval": Xte_tukey,
            "tuned": True,
            "val_macro": dc_cfg["val_macro"],
            "config_str": f"k={dc_cfg['k']}, α={dc_cfg['alpha']}, target={dc_cfg['target']}, λ=0.5 (fixed)",
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
            m, per = scores_full(clf, spec["Xeval"], yte)
            ms.append(m); ps.append(per)
            last_pred = clf.predict(spec["Xeval"])
        results[name] = {
            "macro_mean": np.mean(ms), "macro_std": np.std(ms),
            "per_mean": np.mean(ps, 0),
            "val_macro": spec["val_macro"],
            "config_str": spec["config_str"],
        }
        preds_by_method[name] = last_pred

    # ---- 3. Print val-selected configs ----
    print("=" * 84)
    print("VAL-SELECTED CONFIGS (tuned on validation macro-F1, mean over seeds)")
    print("=" * 84)
    print(f"{'Method':<14} {'Val F1':>8}  {'Test F1':>8}  Config")
    print("-" * 84)
    for name, r in results.items():
        vf = f"{r['val_macro']:.4f}" if r['val_macro'] is not None else "   N/A"
        print(f"{name:<14} {vf:>8}  {r['macro_mean']:8.4f}  {r['config_str']}")

    print()
    print("Every method is tuned over its full hyperparameter space under the")
    print("identical val→test-once→bootstrap protocol. SMOTE has fewer knobs (2 axes)")
    print("because interpolation has no covariance/variance-scale parameter.")
    print("Protocol is identical; parameter count differs by method.")

    # ---- 4. Print results table ----
    print()
    print("=" * 100)
    print("TEST RESULTS (mean over seeds) — * = val-tuned")
    print("=" * 100)
    hdr = f"{'Method':<14} {'MacroF1':<15} " + " ".join(f"{c[:4]:>6}" for c in CLASSES)
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<14} {r['macro_mean']:.4f}±{r['macro_std']:.4f}  "
              + " ".join(f"{v:6.3f}" for v in r['per_mean']))

    # ---- 5. Bootstrap significance for all methods vs baseline ----
    di = CLASSES.index("disgust")
    print()
    print("=" * 100)
    print("SIGNIFICANCE (bootstrap, seed-456 predictions, each method vs Baseline)")
    print("=" * 100)
    print(f"{'Method':<14} {'Metric':<12} {'Delta':>8} {'95% CI':>20} {'p(no-imp)':>10}  Verdict")
    print("-" * 84)
    base_pred = preds_by_method["Baseline"]
    for name in ["SMOTE*", "Gaussian*", "DC*"]:
        pred = preds_by_method[name]
        for metric, cls, lbl in [("macro", None, "macro-F1"), ("perclass", di, "disgust F1")]:
            d, lo, hi, p = bootstrap_delta(base_pred, pred, yte,
                                           metric=metric, cls=cls)
            sig = "SIGNIFICANT" if lo > 0 else "not significant"
            print(f"{name:<14} {lbl:<12} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {p:10.3f}  {sig}")

    # ---- 6. Save CSV ----
    outpath = Path("results_resampling_v2_symmetric.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "val_macro", "test_macro_mean", "test_macro_std",
                     "config"] + CLASSES)
        for name, r in results.items():
            vm = r["val_macro"] if r["val_macro"] is not None else ""
            w.writerow([name, vm, r["macro_mean"], r["macro_std"],
                        r["config_str"], *r["per_mean"]])
    print(f"\nSaved {outpath}")
    print(f"Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
