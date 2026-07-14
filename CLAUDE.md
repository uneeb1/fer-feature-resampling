# FER2013 Thesis Experiment — Claude CLI Context

## Project
Master's thesis: "Feature-Level Resampling for Handling Class Imbalance in Facial Expression Recognition"

## Scope
Extract 512-d deep features from a frozen ResNet-18 backbone (exp_03), apply SMOTE-family resampling in feature space, retrain a linear classifier head, evaluate with macro-F1 and class-wise analysis on FER2013.

## Current State
- Baseline locked: Val Macro-F1 = 0.6440 (exp_03, epoch 10)
- Project cleaned and ready for systematic experiment phase
- See HANDOVER.md for full context, experiment plan, and file listing

## Dataset
- FER2013: 28,709 train / 7,178 val images, 7 classes
- Grayscale 48x48 -> RGB 224x224
- Class-imbalanced (disgust: ~450 samples vs happy: ~7,000)

## Baseline Config (exp_03 — LOCKED)
- Backbone: ResNet-18 (IMAGENET1K_V1)
- Head: Dropout(0.3) -> Linear(512, 7)
- Differential LR: Backbone 3e-5, Head 1e-4
- Weight Decay: 3e-4, Patience: 5, StepLR(step=5, gamma=0.1)
- Checkpoint: `experiments/exp_03/best_model.pth`

## Experiment Plan
- 5 SMOTE variants: Plain SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTETomek
- 1 comparison: Class-weighted CrossEntropyLoss (full fine-tune)
- 3 resampling strategies: full balance, partial balance, minority-only
- 3 seeds per config (42, 123, 999)
- Primary metric: Macro-F1

## How to Run
```bash
source venv/bin/activate
./venv/bin/python3 -u train_baseline.py        # baseline (already done)
./venv/bin/python3 summarize_experiments.py     # compare experiments
```

## Rules
- Do NOT modify train_baseline.py or exp_03/best_model.pth
- Do NOT add data augmentation beyond what's in the baseline
- Primary metric is Macro-F1, not accuracy
- Do NOT change hyperparameters without asking
- If an error occurs, show full traceback before fixing
- If unclear, ask before running
