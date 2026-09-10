# FER2013 Baseline Fresh

Imbalance-agnostic baseline for the thesis "Feature-Level Resampling for Handling
Class Imbalance in Facial Expression Recognition."

## Recipe

- ResNet-18 (IMAGENET1K_V1), fine-tune all layers
- Plain CE + label smoothing 0.1 (NO rebalancing)
- SGD lr=0.01, momentum=0.9, wd=5e-4, warmup 5ep + cosine 100ep
- Mild augmentation: HFlip + RandomCrop(pad=4) + Rotation(±15°)
- TTA: ten-crop + flip averaged logits
- 3 seeds (42, 123, 456), reported as mean ± std
- Primary metric: macro-F1 on natural (imbalanced) distribution

## Pinned Dependencies

```
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
Pillow==10.4.0
opencv-python-headless==4.10.0.84
PyYAML==6.0.1
tqdm==4.66.4
```

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 numpy==1.26.4 pandas==2.2.2 \
    scikit-learn==1.5.1 matplotlib==3.9.1 seaborn==0.13.2 Pillow==10.4.0 \
    opencv-python-headless==4.10.0.84 PyYAML==6.0.1 tqdm==4.66.4
```

## Run Smoke Test (CPU)

```bash
bash run_smoke.sh
```

## Run Full (GPU)

```bash
bash run_full.sh
```

## How to Run on Kaggle (5 Steps)

1. **Create a new Kaggle notebook.**
2. **Add FER2013 as a dataset input:** Search "FER2013" in Kaggle Datasets → Add.
   The CSV should appear at `/kaggle/input/fer2013/fer2013.csv`.
3. **Enable GPU:** Notebook Settings → Accelerator → GPU T4 x2.
4. **Upload source code:** Upload the contents of `baseline_fresh/` (main.py,
   config.yaml, src/ folder) — either as a Kaggle dataset named `baseline-fresh-src`,
   or by pasting into notebook cells. The notebook auto-copies from the dataset path.
5. **Run all cells** in `kaggle_run.ipynb` (or paste cells into the Kaggle notebook).
   Outputs land in `/kaggle/working/baseline_fresh/`. Download `baseline_fresh_results.tar.gz`.

## Outputs

```
baseline_fresh/
├── config.yaml              # full hyperparameter config
├── main.py                  # orchestrator CLI
├── src/                     # modules (dataset, model, train, features, plots, transforms)
├── checkpoints/             # best_seed{42,123,456}.pt
├── features/                # {train,val,test}_{features,labels}.npy (512-d, non-negative)
├── figures/                 # all PNGs (300 dpi)
├── metrics.json             # per-seed + aggregated results
├── results.md               # final table + per-class diagnostic
├── run_full.sh              # full run launcher
├── run_smoke.sh             # smoke test launcher
├── kaggle_run.ipynb         # Kaggle T4 notebook
└── README.md                # this file
```

## Feature Tap

512-d features are extracted from the penultimate layer (after global avgpool,
after ReLU, before fc head). These are **non-negative** by construction (ReLU
output) and ready for SMOTE-family resampling in the next phase.

---

## Phase 5: Geometry Loss (Center + Bounded Inter-Class Separation)

### Rationale

Phase 4 center loss compacted classes but did NOT separate them — entangled
classes (fear/angry, sad/neutral) still overlap in the 512-d space. Geometry
loss adds a **bounded, margin-hinged separation term** that pushes class centers
apart (cosine sim → 0) while center loss pulls samples inward. The resulting
features are better-separated for downstream feature-space resampling.

**No backbone/head/logit changes.** The loss acts ONLY on the 512-d embedding
and its class centers. Features are extracted identically and feed resampling.

### Revised recipe (overfitting-tuned)

| Param          | Value | Note                        |
|----------------|-------|-----------------------------|
| epochs         | 60    | cosine-annealed             |
| warmup         | 5     | unchanged                   |
| patience       | 10    | stricter early stop          |
| min_delta      | 0.001 | small improvement threshold  |
| weight_decay   | 0.002 | may escalate to 0.005        |
| lr             | 0.005 | unchanged                   |
| dropout        | 0.5   | unchanged                   |
| label_smooth   | 0.1   | unchanged                   |
| mixup_alpha    | 0.2   | unchanged                   |

### Stage A — Sweep (seed 42, on Kaggle GPU)

```bash
cd baseline_fresh

# CE baseline (revised recipe)
python train_geometry_loss.py --config config_m2.yaml --lam-center 0 --lam-sep 0 --extract-features

# Center-only (control)
python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0 --extract-features

# Geometry sweep
python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.001 --extract-features
python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.01  --extract-features
python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.1   --extract-features
```

Pick best `--lam-sep` by **val macro-F1**.

### Stage B — 3-seed confirmation

```bash
# Best geometry config (replace 0.01 with sweep winner)
python train_geometry_loss.py --config config_m2.yaml --lam-center 0.001 --lam-sep 0.01 --seeds 42 123 456 --extract-features

# Matched CE baseline
python train_geometry_loss.py --config config_m2.yaml --lam-center 0 --lam-sep 0 --seeds 42 123 456 --extract-features
```

### Step 2 — Resampling on geometry features

```bash
# (a) M2 features (locked reference)
bash run_resampling_on_center.sh "M2_seed42_backup (1)/features"

# (b) Revised CE baseline features
bash run_resampling_on_geometry.sh M2geo_lamc0_lams0/features

# (c) Geometry features
bash run_resampling_on_geometry.sh M2geo_lamc0.001_lams0.01/features
```

Output lands in `results_resampling_geometry.csv` — never overwrites
`results_resampling_v2_symmetric.csv`.

### Kaggle Commit Note

Upload `train_geometry_loss.py` alongside the existing `src/` directory.
Same dependencies, same notebook setup as Phase 4. GPU runtime required.

### Output structure

```
M2geo_lamc{x}_lams{y}/
├── checkpoints/best_seed{s}.pt
├── features/{train,val,test}_{features,labels}.npy
├── predictions/{test,val}_{preds,labels}.npy
├── geometry/geometry.json
├── graphs/                    # all PNGs + figures.png contact sheet
├── config.json                # resolved hyperparameters
├── metrics.json               # per-seed + aggregate + geometry
└── history.json               # full per-epoch training curves
```
