# FER2013 Baseline Fresh — Results

## Summary

*Placeholder — will be populated after the full Kaggle run (3 seeds, 224x224, 100 epochs).*

**Smoke test completed successfully** (1 seed, 112x112, 2 epochs, 500 samples per split, CPU).
All pipeline artifacts verified: checkpoints, features, figures, metrics.

## Configuration

| Setting | Value |
|---|---|
| Backbone | ResNet-18 (IMAGENET1K_V1), fine-tune all layers |
| Loss | CrossEntropy + label smoothing 0.1, NO rebalancing |
| Optimizer | SGD, lr=0.01, momentum=0.9, wd=5e-4 |
| Schedule | Linear warmup (5 ep) + cosine annealing (100 ep) |
| Augmentation | HFlip + RandomCrop(pad=4) + Rotation(±15°) |
| Preprocessing | Resize 224 → 3-ch → ImageNet normalize (CLAHE OFF) |
| TTA | Ten-crop + HFlip averaged logits |
| Seeds | 42, 123, 456 |
| Primary metric | Macro-F1 (imbalanced evaluation) |

## Results (to be filled after Kaggle run)

### Aggregate

| Metric | Mean ± Std |
|---|---|
| Test Macro-F1 | *pending* |
| Test Accuracy | *pending* |

### Per-Class F1 (Best Seed)

| Class | F1 | Train Count | Silhouette | Weakness Category |
|---|---|---|---|---|
| angry | *pending* | 3,995 | *pending* | Mild count-minority |
| disgust | *pending* | 436 | *pending* | Extreme count-minority (7x fewer than happy) |
| fear | *pending* | 4,097 | *pending* | Geometry/overlap (fear↔sad↔angry mutual confusion) |
| happy | *pending* | 7,215 | *pending* | Majority class (reference) |
| sad | *pending* | 4,830 | *pending* | Geometry/overlap (sad↔fear↔neutral mutual confusion) |
| surprise | *pending* | 3,171 | *pending* | Mild count-minority, but separable |
| neutral | *pending* | 4,965 | *pending* | Geometry/overlap (neutral↔sad confusion) |

### Per-Class Diagnostic Notes

- **angry** (3,995): Mild count-minority. Weakness likely count-driven; augmentation and resampling may help if separability is reasonable.
- **disgust** (436): Extreme count-minority — 7× fewer than the majority (happy). Prior work showed this class is separable (positive silhouette) but its tiny count makes it fragile. Weakness is count-driven.
- **fear** (4,097): Moderate count but historically the worst-performing class. Prior silhouette analysis showed negative values and heavy confusion with sad and angry. Weakness is separability-driven (geometric overlap), not count-driven.
- **happy** (7,215): Majority class, consistently highest F1. Reference point for evaluating resampling effects.
- **sad** (4,830): Adequate count but geometrically entangled with fear and neutral. Weakness is separability-driven.
- **surprise** (3,171): Mild count-minority but historically forms a clean cluster (positive silhouette). Expected to perform well despite lower count.
- **neutral** (4,965): Adequate count but overlaps with sad. Weakness is partly separability-driven.

## Figures

All saved in `figures/` at 300 dpi:
- `class_distribution.png` — per-class counts for train/val/test
- `training_curves.png` — train/val loss + val macro-F1 & accuracy
- `lr_schedule.png` — realized learning rate curve
- `confusion_matrix_raw.png` — test confusion matrix (counts)
- `confusion_matrix_norm.png` — test confusion matrix (row-normalized)
- `per_class_f1.png` — per-class F1 bar chart (test)
- `tsne_train_features.png` — t-SNE of 512-d train features
- `tsne_test_features.png` — t-SNE of 512-d test features
- `per_class_silhouette.png` — silhouette score per class (test)

## Feature Extraction

512-d features saved from the post-avgpool, post-ReLU tap (non-negative, verified).
Ready for SMOTE-family resampling in the next phase.
