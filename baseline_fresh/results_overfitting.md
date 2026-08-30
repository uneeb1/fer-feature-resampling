# Overfitting-Reduction Sweep — Results

*Populate after running sweeps on Kaggle T4.*

## Configs

| Config | Freeze | Dropout | WD | LS | MixUp | LR Groups | Notes |
|---|---|---|---|---|---|---|---|
| **G0** (baseline) | None | 0.4 | 1e-3 | 0.1 | Off | Uniform | Reference |
| **G1** (freeze+discLR) | conv1,bn1,L1,L2 | 0.4 | 1e-3 | 0.1 | Off | L3=lr/10, L4=lr/3, fc=lr | Partial freeze |
| **G3** (heavy reg) | None | 0.5 | 5e-3 | 0.2 | α=0.3 | Uniform | Over-regularized |
| **M1** (gentle mix) | None | 0.4 | 2e-3 | 0.1 | α=0.1 | Uniform | Lightest mixup dose |
| **M2** (moderate mix) | None | 0.5 | 2e-3 | 0.1 | α=0.2 | Uniform | Higher dropout + mixup |
| **M3** (moderate+LS) | None | 0.5 | 3e-3 | 0.15 | α=0.2 | Uniform | M2 + more LS and WD |

All share: ResNet-18 IMAGENET1K_V1, 224x224, plain CE (no class weights), mild aug
(flip + rot15 + crop, NO erasing), lr 0.005 SGD m=0.9, warmup 5 + cosine 100ep,
early stopping patience 20, ten-crop+flip TTA, imbalanced eval, leakage-filtered splits.

**Note:** Train F1 under MixUp (M1/M2/M3/G3) is measured on mixed inputs and is
artificially depressed — it is not directly comparable to non-MixUp train F1.
The meaningful generalization check is **val vs test** gap.

## Comparison Table (seed=42)

| Config | Train F1 | Val F1 | Test F1 | Gap | Best Ep | Stop Ep | Disgust | Fear | Sad | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 (baseline) | — | — | — | — | — | — | — | — | — | — |
| G1 (freeze+discLR) | — | — | — | — | — | — | — | — | — | — |
| G3 (heavy reg) | — | — | — | — | — | — | — | — | — | — |
| M1 (gentle mix) | — | — | — | — | — | — | — | — | — | — |
| M2 (moderate mix) | — | — | — | — | — | — | — | — | — | — |
| M3 (moderate+LS) | — | — | — | — | — | — | — | — | — | — |

**Target zone:** gap 10–15%, macro-F1 ≥ 0.66, disgust F1 ≥ 0.56.

## Per-Class Test F1 (seed=42)

| Class | G0 | G1 | G3 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|
| angry | — | — | — | — | — | — |
| disgust | — | — | — | — | — | — |
| fear | — | — | — | — | — | — |
| happy | — | — | — | — | — | — |
| sad | — | — | — | — | — | — |
| surprise | — | — | — | — | — | — |
| neutral | — | — | — | — | — | — |

## Interpretation

*To be written after results. No winner auto-selected — the table is for informed
manual selection.*
