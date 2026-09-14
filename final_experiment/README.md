# Finalized Thesis Experiment

**"Feature-Level Resampling for Handling Class Imbalance in Facial Expression Recognition"**

## Final Spine

- **Baseline**: ResNet-18 (IMAGENET1K_V1) fine-tuned on FER2013 with CE + label smoothing (M2 recipe). Backbone macro-F1 ≈ 0.678 (3-seed); seed-42 resampling reference 0.6874.
- **Proposed method**: Gaussian feature-space resampling on 512-d embeddings.
- **Comparison methods**: SMOTE and DC (Data Cooking).
- **Feature substrates** (increasing geometry quality):
  1. **M2 original** — standard M2 baseline features
  2. **CE-revised** — 60-epoch recipe, same architecture (λ_c=0, λ_s=0)
  3. **Center-only** — center loss compaction (λ_c=0.001, λ_s=0)
  4. **Geometry** — center loss + bounded inter-class separation (λ_c=0.001, λ_s=0.01)
- **Result**: 12 comparisons (4 substrates × 3 methods), **all non-significant**. The geometry substrate is the strongest standalone representation but resampling adds nothing on top.

## Pipeline

```
FER2013 image (48×48 grayscale)
  → RGB 224×224, augmentation (HFlip, crop, rotation, MixUp)
  → ResNet-18 backbone (IMAGENET1K_V1)
  → 512-d post-avgpool/post-ReLU embedding
  → [optional] geometry loss shaping (center + separation)
  → feature extraction (.npy)
  → [optional] resampling (Gaussian / SMOTE / DC)
  → logistic regression head
  → macro-F1 evaluation (3-seed, bootstrapped significance)
```

## Headline Results

### Resampling: Macro-F1 (mean ± std over 3 seeds)

| Substrate | Baseline | Gaussian* | SMOTE* | DC* |
|---|---|---|---|---|
| M2 original | **0.6874** | 0.6872 | 0.6915 ±.002 | 0.6870 ±.000 |
| CE revised | 0.6550 | 0.6568 ±.001 | 0.6558 ±.001 | 0.6354 ±.001 |
| Center-only | 0.6789 | 0.6799 ±.001 | 0.6765 ±.001 | 0.6697 ±.001 |
| Geometry | **0.6926** | 0.6901 | 0.6916 ±.001 | 0.6780 ±.001 |

\* Val-tuned under identical protocol. **None of the 12 deltas are statistically significant** (bootstrap, p < 0.05).

### Substrate Quality (seed 42, test set)

| Substrate | Silhouette | Sep/Compact | fear F1 | angry F1 | sad F1 | neutral F1 | disgust F1 |
|---|---|---|---|---|---|---|---|
| M2 original | — | — | .545 | .644 | .583 | .693 | .645 |
| CE revised | 0.102 | 0.039 | .505 | .610 | .542 | .655 | .629 |
| Center-only | 0.184 | 0.158 | .541 | .606 | .570 | .688 | .653 |
| Geometry | **0.209** | **0.199** | **.552** | **.639** | **.591** | **.696** | **.674** |

## Reproduction Commands

All commands run from `code/` with the venv activated. GPU required for training; resampling is CPU-only.

### 1. Train each substrate (GPU)

```bash
# M2 original baseline (100 epochs, 3 seeds)
python main.py --config config_m2.yaml --csv <path/to/fer2013.csv> --seeds 42 123 456

# CE-revised (60-epoch recipe)
python train_geometry_loss.py --config config_m2.yaml --csv <CSV> --lam-center 0 --lam-sep 0 --seeds 42 123 456 --extract-features

# Center-only
python train_geometry_loss.py --config config_m2.yaml --csv <CSV> --lam-center 0.001 --lam-sep 0 --seeds 42 123 456 --extract-features

# Geometry (winner)
python train_geometry_loss.py --config config_m2.yaml --csv <CSV> --lam-center 0.001 --lam-sep 0.01 --seeds 42 123 456 --extract-features
```

### 2. Run resampling on each substrate (CPU, ~100 min each)

```bash
# Edit FEAT_DIR in the sed command or use the runner:
bash run_resampling.sh ../features/m2_original
bash run_resampling.sh ../features/ce_revised
bash run_resampling.sh ../features/center_only
bash run_resampling.sh ../features/geometry
```

## Folder Map

```
final_experiment/
├── README.md                          # this file
├── code/
│   ├── train_geometry_loss.py         # unified trainer (CE / center / geometry via flags)
│   ├── main.py                        # M2 baseline trainer
│   ├── resampling_experiment.py       # v2-symmetric resampling (Gaussian/SMOTE/DC)
│   ├── run_resampling.sh              # runner script (patches FEAT_DIR + output path)
│   ├── config_m2.yaml                 # M2 hyperparameter config
│   └── src/                           # shared modules (dataset, model, train, features, transforms, plots)
├── features/                          # 512-d embeddings, 6 .npy per substrate
│   ├── m2_original/                   # {train,val,test}_{features,labels}.npy
│   ├── ce_revised/
│   ├── center_only/
│   └── geometry/                      # λ_c=0.001, λ_s=0.01 (winner)
└── results/
    ├── resampling_results.csv         # 4 substrates × 4 methods (baseline + 3 resampled)
    ├── per_substrate/
    │   ├── m2_original/metrics.json
    │   ├── ce_revised/metrics.json + geometry.json
    │   ├── center_only/metrics.json + geometry.json
    │   └── geometry/metrics.json + geometry.json + history.json
    └── figures/                        # geometry run PNGs (10 files + contact sheet)
```

## Caveats

1. **Recipe confound**: The geometry substrate uses a different training recipe (60 epochs, patience 10, min_delta 0.001) vs M2 original (100 epochs, patience 20, min_delta 0.0). The CE-revised substrate controls for this (same 60-epoch recipe, no geometry loss), but the geometry-vs-M2 comparison is confounded.
2. **Seed count**: Stage A (λ sweep) used seed 42 only. All resampling results and final comparisons use 3 seeds (42, 123, 456). Single-seed numbers overstate the effect.
3. **Disgust variance**: Disgust has only ~55 test samples (389 train). Per-class F1 for disgust has high variance across seeds and bootstrap samples. The ±0.08 CIs on disgust F1 deltas reflect this.
4. **Geometry metrics**: Silhouette and cosine similarity are computed on seed-42 features only (the extracted checkpoint). M2 original has no geometry.json because it was trained without center loss (no learned centers to report).
