# Thesis Handover — Feature-Level Resampling for Handling Class Imbalance in FER

> Last updated: 2026-07-10
> Status: **Baseline locked. Project cleaned. Ready for systematic experiment phase.**

---

## Project Summary

Master's thesis: **"Feature-Level Resampling for Handling Class Imbalance in Facial Expression Recognition"**

Scope: Extract deep features from a frozen ResNet-18 backbone, apply SMOTE-family resampling in the 512-d feature space, retrain a linear classifier head, and evaluate with macro-F1 and class-wise analysis on FER2013.

---

## Current State

- Baseline is **locked**: ResNet-18 fine-tuned on FER2013, Val Macro-F1 = **0.6440** (exp_03, epoch 10)
- Baseline tuning history preserved (exp_01 through exp_03) — documents the regularization tradeoff for the thesis
- Embedding analysis complete: Silhouette scores (Train: 0.057, Val: 0.038) and t-SNE visualizations show severe manifold overlap among negative emotions
- All pilot SMOTE scripts and results have been removed — everything will be rerun systematically with seeds

---

## Dataset — FER2013

| Property | Value |
|---|---|
| Train samples | 28,709 |
| Val samples | 7,178 |
| Classes | angry, disgust, fear, happy, neutral, sad, surprise |
| Image size | 48x48 grayscale -> 224x224 RGB |
| Primary metric | **Macro-F1** (not accuracy) |

**Class distribution (train):**

| Class | Approx. samples | Note |
|---|---|---|
| happy | ~7,000 | majority |
| neutral | ~5,000 | |
| sad | ~4,700 | |
| fear | ~4,100 | |
| angry | ~3,900 | |
| surprise | ~3,100 | |
| disgust | ~450 | **severe minority** |

---

## Baseline Architecture & Config

```python
# Backbone: ResNet-18 (IMAGENET1K_V1)
# Head: Dropout(0.3) -> Linear(512, 7)
# Full fine-tuning with differential LR

BACKBONE_LR  = 3e-5
HEAD_LR      = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT      = 0.3
PATIENCE     = 5          # early stopping
NUM_EPOCHS   = 30
SCHEDULER    = StepLR(step_size=5, gamma=0.1)
OPTIMIZER    = Adam (differential LR)
LOSS         = CrossEntropyLoss
```

---

## Baseline Tuning History

| Exp | Backbone LR | WD | Dropout | Scheduler | Val Macro-F1 | Gap | Verdict |
|---|---|---|---|---|---|---|---|
| exp_01 | 3e-5 | 3e-4 | 0.3 | ReduceLROnPlateau | — | — | Config only (no full results) |
| exp_02 | 3e-5 | 3e-4 | 0.3 | ReduceLROnPlateau | 0.6730 | 0.261 | Best F1 but severe overfit |
| **exp_03** | 3e-5 | 3e-4 | 0.3 | StepLR step=5 | **0.6440** | **0.084** | **Official baseline** |

**exp_03 per-class results:**

```
              precision    recall  f1-score   support
       angry       0.59      0.62      0.60       958
     disgust       0.73      0.41      0.53       111
        fear       0.58      0.43      0.49      1024
       happy       0.86      0.89      0.88      1774
     neutral       0.61      0.70      0.65      1233
         sad       0.57      0.56      0.57      1247
    surprise       0.77      0.81      0.79       831

    macro avg       0.67      0.63      0.64      7178
```

Worst classes: **disgust** (F1: 0.53) and **fear** (F1: 0.49) — the imbalance targets.

---

## Embedding Analysis Summary

- **Silhouette scores:** Train: 0.057, Val: 0.038 (poor separation, < 0.25 threshold)
- **t-SNE findings:** happy and surprise form clean clusters; angry, fear, sad, neutral are severely entangled
- **Implication:** Feature-space interpolation in the overlap zone risks generating cross-class noise. Boundary-aware variants may partially mitigate this.
- **Plots saved:** `outputs/embedding_analysis/`

---

## Experiment Plan — Systematic Phase

### Resampling Variants (5)

All use offline feature extraction from the frozen exp_03 backbone, then retrain a fresh `Dropout(0.3) -> Linear(512, 7)` head.

| # | Method | Library | Purpose |
|---|---|---|---|
| 1 | Plain SMOTE | `imblearn.over_sampling.SMOTE` | Vanilla baseline for the SMOTE family |
| 2 | BorderlineSMOTE | `imblearn.over_sampling.BorderlineSMOTE` | Targets boundary samples only |
| 3 | SVMSMOTE | `imblearn.over_sampling.SVMSMOTE` | Uses SVM support vectors to guide synthesis |
| 4 | ADASYN | `imblearn.over_sampling.ADASYN` | Adaptive — generates more where harder to learn |
| 5 | SMOTETomek | `imblearn.combine.SMOTETomek` | SMOTE + Tomek link cleanup |

### Comparison Method (1)

| # | Method | Purpose |
|---|---|---|
| 6 | Class-weighted CrossEntropyLoss | Full fine-tune with same baseline config but inverse-sqrt class weights. Shows whether resampling outperforms simple loss reweighting. |

### Resampling Strategies (3)

Each SMOTE variant is tested under three balancing strategies:

| Strategy | Description |
|---|---|
| Full balance | Oversample all minority classes to match the majority class count |
| Partial balance | Oversample to the median class count |
| Minority-only | Only oversample disgust, fear, and surprise (the three worst classes) |

### Seeds

3 seeds per configuration: **42, 123, 999**. Report mean +/- std.

### Evaluation

- **Primary:** Val Macro-F1
- **Secondary:** Per-class precision, recall, F1
- **Diagnostic:** Confusion matrices, train/val gap
- **Benchmark:** All results compared against baseline 0.6440

### Total Experiment Count

- 5 SMOTE variants x 3 strategies x 3 seeds = **45 runs**
- 1 class-weighted loss x 3 seeds = **3 runs**
- **48 total runs**

---

## Environment

```bash
# Setup
cd experiment2
python3 -m venv venv
source venv/bin/activate
bash setup_linux.sh   # or: pip install -r requirements.txt

# Run baseline
./venv/bin/python3 -u train_baseline.py

# Summarize experiments
./venv/bin/python3 summarize_experiments.py
```

Device priority: CUDA > MPS > CPU

**Dependencies:** torch, torchvision, scikit-learn, matplotlib, seaborn, numpy, imbalanced-learn

---

## Project Files After Cleanup

```
experiment2/
├── train_baseline.py              — Locked baseline training script
├── check_embeddings.py            — Feature extraction + t-SNE + silhouette analysis
├── summarize_experiments.py       — Prints comparison table of all experiments
├── requirements.txt               — Python dependencies
├── setup_linux.sh                 — Linux environment setup
├── CLAUDE.md                      — Claude CLI context file
├── HANDOVER.md                    — This file
├── FINDINGS.md                    — Baseline findings and analysis
├── codex-for-claude (2).md        — Codex CLI reference
├── data/
│   ├── train/                     — 28,709 images (7 class folders)
│   └── test/                      — 7,178 images (7 class folders)
├── experiments/
│   ├── exp_01/                    — Baseline tuning run (config only)
│   ├── exp_02/                    — Baseline tuning (ReduceLROnPlateau, F1: 0.6730)
│   └── exp_03/                    — Official baseline (F1: 0.6440) — LOCKED
│       ├── best_model.pth         — Canonical checkpoint for feature extraction
│       ├── config.txt, results.txt
│       └── *.png                  — Loss, accuracy, F1 curves + confusion matrix
├── outputs/
│   └── embedding_analysis/        — t-SNE and SMOTE visualizations
│       ├── tsne_train_real.png
│       ├── tsne_real_vs_smote.png
│       └── tsne_val.png
└── venv/                          — Python virtual environment
```

---

## Rules

- Do NOT modify `train_baseline.py` or `exp_03/best_model.pth`
- Do NOT add data augmentation beyond what's in the baseline script
- Primary metric is **Macro-F1**, not accuracy
- One variable change at a time per experiment
- If an error occurs, show full traceback before attempting a fix
- If anything is unclear, ask before running
