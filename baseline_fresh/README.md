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
