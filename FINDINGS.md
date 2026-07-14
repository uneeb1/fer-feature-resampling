# FER2013 Baseline — Findings

## Research Context

**Thesis:** Feature-Level Resampling for Handling Class Imbalance in Facial Expression Recognition

**Scope:** Extract deep features from a frozen baseline backbone, apply SMOTE-family resampling in the 512-d feature space, retrain a linear classifier, and evaluate against the baseline with macro-F1 and class-wise metrics.

---

## Baseline Results (exp_03 — LOCKED)

```
Best epoch    : 10
Best Macro-F1 : 0.6440
Accuracy      : 68%

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

---

## Key Findings

### 1. Overfitting is the central challenge

ResNet-18 (11M params) on 28k images is high risk. Without regularization, train F1 reaches 0.88 while val F1 = 0.69 (gap: 0.187). The balanced config (exp_03) controls the gap to 0.084.

### 2. Regularization tradeoff

| Config | Val Macro-F1 | Gap | Verdict |
|---|---|---|---|
| No regularization | 0.6925 | 0.187 | Memorizing |
| Heavy regularization (WD=1e-3, DO=0.5) | 0.5415 | 0.003 | Underfitting |
| Balanced (exp_03) | 0.6440 | 0.084 | Generalizing |
| ReduceLROnPlateau (exp_02) | 0.6730 | 0.261 | Peak but overfitting |

### 3. Disgust and fear are the hardest classes

- **Disgust** (F1: 0.53): Only ~450 training samples — severe underrepresentation
- **Fear** (F1: 0.49): Visually similar to sad/angry — class separability problem
- **Happy** (F1: 0.88): Abundant samples + distinct facial features

### 4. Embedding space is poorly separated

- Silhouette scores: Train 0.057, Val 0.038 (well below the 0.25 "poor" threshold)
- t-SNE shows happy and surprise form clean clusters, but angry/fear/sad/neutral are severely entangled
- Implication: feature-space interpolation in overlap zones risks cross-class noise generation

### 5. Scheduler tradeoff

- **StepLR (step=5):** Aggressive LR drops cause early plateau, but controlled gap
- **ReduceLROnPlateau:** Higher peak F1 (0.6730) but allows gap to grow to 0.261
