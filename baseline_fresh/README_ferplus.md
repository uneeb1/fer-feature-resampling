# FERPlus Replication of Geometry-Loss Experiment

## Overview

This replicates the FER2013 geometry-loss experiment on FERPlus labels.
FERPlus relabels the **same 35,887 images** via crowd-sourced votes
(Microsoft FERPlus, github.com/microsoft/FERPlus). Only the dataset
changes; the model, loss, recipe, split logic, leakage filter, feature
extraction, and resampling protocol are unchanged.

## Data Preparation

### fer2013new.csv

Download from: https://github.com/microsoft/FERPlus  
Attach as a Kaggle dataset alongside fer2013.csv.

### Vote-column mapping

fer2013new.csv vote columns (file order):
```
[neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF]
   0         1          2         3       4       5      6       7         8      9
```

COL_TO_FER7 mapping (contempt/unknown/NF → drop):
```
{0: neutral→6, 1: happiness→3, 2: surprise→5, 3: sadness→4,
 4: anger→0,   5: disgust→1,   6: fear→2}
```

### Majority-vote scheme (Microsoft FERPlus standard)

1. **Outlier removal**: zero any vote count < 1.0 + ε (single-vote outlier)
2. **Total check**: if sum of remaining votes ≤ 0 → drop row
3. **Majority check**: if max vote ≤ 0.5 × total → drop row (no majority)
4. **Map**: argmax (first-max tie-break) → COL_TO_FER7; None for indices 7/8/9

### Ground-truth counts (pre-leakage-filter)

| Class    | Train | Val  | Test |
|----------|-------|------|------|
| angry    | 2100  | 287  | 273  |
| disgust  | 119   | 24   | 18   |
| fear     | 532   | 62   | 83   |
| happy    | 7287  | 865  | 893  |
| sad      | 3014  | 351  | 384  |
| surprise | 3149  | 415  | 396  |
| neutral  | 8740  | 1182 | 1090 |

**KEPT**: train=24941, val=3186, test=3137 → **31264 total**  
**DROPPED**: 4623 of 35887  
- No majority: 4200  
- NF: 176  
- Contempt: 148  
- Unknown: 99  

## Run Commands

### On Kaggle (recommended)

Use `kaggle_run_ferplus.py` — it auto-detects both CSVs, clones the
`ferplus` branch, trains, extracts features, and runs Gaussian resampling.

### Manual (local)

```bash
cd baseline_fresh

# Baseline (CE only)
python train_geometry_loss.py --config config_m2.yaml \
  --csv /path/to/fer2013.csv --ferplus-csv /path/to/fer2013new.csv \
  --dataset ferplus --lam-center 0 --lam-sep 0 \
  --seeds 42 123 456 --extract-features

# Geometry (center + separation)
python train_geometry_loss.py --config config_m2.yaml \
  --csv /path/to/fer2013.csv --ferplus-csv /path/to/fer2013new.csv \
  --dataset ferplus --lam-center 0.001 --lam-sep 0.01 \
  --seeds 42 123 456 --extract-features

# Gaussian-only resampling
python resampling_experiment.py \
  --feat-dir ferplus_M2geo_lamc0_lams0/features \
  --methods gaussian \
  --out results_resampling_ferplus_baseline.csv

python resampling_experiment.py \
  --feat-dir ferplus_M2geo_lamc0.001_lams0.01/features \
  --methods gaussian \
  --out results_resampling_ferplus_geometry.csv
```

### Unit test

```bash
python test_ferplus_mapping.py
```

## Files Changed vs geometry-loss Branch

| File | Change |
|------|--------|
| `src/dataset.py` | Added `ferplus_majority_label`, `load_ferplus_csv`, `COL_TO_FER7`; factored `_apply_leakage_filter`; `load_fer2013_csv` unchanged in behavior |
| `train_geometry_loss.py` | Added `--dataset`, `--ferplus-csv` flags; `ferplus_` prefix on output dirs; `dataset` field in config/metrics JSON |
| `resampling_experiment.py` | Vendored from final-experiment branch + CLI (`--feat-dir`, `--out`, `--methods`) |
| `kaggle_run_ferplus.py` | New orchestrator mirroring `kaggle_run_geometry.py` |
| `test_ferplus_mapping.py` | Unit test for mapping counts |
| `README_ferplus.md` | This file |

## Expected Outputs

- `ferplus_M2geo_lamc0_lams0/` — baseline (CE only) on FERPlus
- `ferplus_M2geo_lamc0.001_lams0.01/` — geometry loss on FERPlus
- `results_resampling_ferplus_baseline.csv` — Gaussian resampling on baseline features
- `results_resampling_ferplus_geometry.csv` — Gaussian resampling on geometry features

## FER2013 vs FERPlus Headline Comparison

Assemble after running both experiments:

| Metric | FER2013 Baseline | FER2013 Geometry | FER2013 Geo+Gaussian | FERPlus Baseline | FERPlus Geometry | FERPlus Geo+Gaussian |
|--------|-----------------|-----------------|---------------------|-----------------|-----------------|---------------------|
| Macro-F1 | | | | | | |
| fear F1 | | | | | | |
| angry F1 | | | | | | |
| sad F1 | | | | | | |
| neutral F1 | | | | | | |
| disgust F1 | | | | | | |
| Silhouette (overall) | | | | | | |
| Silhouette (fear/angry) | | | | | | |
| Silhouette (sad/neutral) | | | | | | |
