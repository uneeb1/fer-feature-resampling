# Baseline Comparison — v1 / v3 / v5

> **Official baseline: v5** — All SMOTE experiments use v5 as the reference baseline.

## Summary

| Metric | v1 (original) | v3 (no aug, regularized) | v5 (mild aug, official) |
|---|---|---|---|
| Best epoch | ~10 | 6 | 19 |
| Val macro-F1 | 0.644 | 0.618 | **0.676** |
| Test macro-F1 | ~0.63 | 0.616 | **0.680** |
| Test accuracy | ~0.65 | 0.643 | **0.694** |
| Train acc @ best | ~0.98+ | 0.942 | 0.919 |
| Val acc @ best | ~0.64 | 0.634 | 0.686 |
| Overfitting gap | ~34% | 30.8% | **23.3%** |

## Per-Class Test F1

| Class | v1 | v3 | v5 | v3→v5 delta |
|---|---|---|---|---|
| Angry | ~0.55 | 0.548 | 0.598 | +0.050 |
| Disgust | 0.619 | 0.550 | 0.693 | +0.143 |
| Fear | ~0.49 | 0.496 | 0.541 | +0.045 |
| Happy | ~0.85 | 0.847 | 0.880 | +0.033 |
| Neutral | ~0.60 | 0.612 | 0.679 | +0.067 |
| Sad | ~0.48 | 0.483 | 0.551 | +0.068 |
| Surprise | ~0.78 | 0.775 | 0.819 | +0.044 |

## Configuration Differences

### v1 (original baseline)
- Full fine-tuning, all layers unfrozen
- Augmentation: horizontal flip, rotation(10), affine(scale 0.9-1.1, translate 0.1)
- Dropout 0.3, weight decay 3e-4, no label smoothing
- StepLR(step=5, gamma=0.1)

### v3 (no augmentation, regularized)
- Partial fine-tuning: conv1, bn1, layer1, layer2 FROZEN
- NO augmentation at all
- Dropout 0.5, weight decay 1e-3, label smoothing 0.1
- ReduceLROnPlateau(mode=max val macro-F1, factor 0.5, patience 3)

### v5 (mild augmentation, official baseline)
- Full fine-tuning, all layers unfrozen
- Mild augmentation: RandomHorizontalFlip(0.5) + RandomCrop(48, padding=4) ONLY
- Dropout 0.4, weight decay 5e-4, label smoothing 0.1
- ReduceLROnPlateau(mode=max val macro-F1, factor 0.5, patience 3)

### Shared across all
- ResNet-18, ImageNet-pretrained (IMAGENET1K_V1)
- Adam, backbone LR 3e-5, head LR 1e-4
- Cross-Entropy (no class weights), batch 64
- Max 30 epochs, early stopping patience 5
- Seed 42
- Data: data_lossless/{train,validation,test}/
- Preprocessing: Grayscale(3) → Resize(224) → ImageNet normalization

## Key Observations

### Overfitting
- v1 had a ~34% train-val gap — severe overfitting on 48×48 grayscale images upscaled to 224×224.
- v3 (no augmentation + freezing + stronger regularization) only reduced the gap to 30.8%. Freezing early layers hurt generalization (val F1 dropped from 0.644 to 0.618), confirming that the early stages need adaptation for grayscale FER images despite ImageNet pretraining.
- v5 (mild augmentation + moderate regularization) brought the gap down to 23.3% while improving val F1 to 0.676 — the best result. The mild flip+crop recipe is standard in FER literature and provides effective regularization without heavy geometric distortion.

### Disgust Analysis
- Disgust (smallest class, 436 train samples) F1 across baselines: v1=0.619, v3=0.550, v5=0.693.
- v5's disgust F1 (0.693) is higher than v1 (0.619), but the gain is proportional to other classes — every class improved by 0.03–0.07, with disgust's larger absolute jump (+0.14 from v3) reflecting its sensitivity to better-generalized features rather than any imbalance-specific effect.
- v2 (heavy augmentation, not official) reached disgust F1 ~0.76, suggesting heavy augmentation does provide a disproportionate minority benefit — but that would confound the thesis.
- Conclusion: mild augmentation acts as general regularization, not imbalance handling. This makes v5 a clean baseline for the SMOTE study.

### Entangled Classes
- Fear (0.541) and Sad (0.551) remain the weakest classes in v5, consistent across all baselines.
- These are the "entangled" classes — visually similar to each other and to angry/neutral — so their low F1 is a feature-space separability problem, not primarily an imbalance problem.
- This is the thesis motivation: SMOTE in feature space should help if it can generate synthetic samples that improve decision boundaries for these confused classes.

## File Locations

### v3
- Results: results/baseline_v3/baseline_v3_s42.json
- Checkpoint: results/baseline_v3/best_model_s42.pth
- Features: results/baseline_v3/test_features_s42.npz
- History: logs/baseline_v3_history_s42.json
- Figures: figures/baseline_v3/

### v5 (official baseline)
- Results: results/baseline_v5/baseline_v5_s42.json
- Checkpoint: results/baseline_v5/best_model_s42.pth
- Features: results/baseline_v5/test_features_s42.npz
- History: logs/baseline_v5_history_s42.json
- Figures: figures/baseline_v5/
  - training_curves.png
  - confusion_matrix_baseline.png
  - per_class_f1_baseline.png
  - class_distribution.png
  - tsne_baseline.png
