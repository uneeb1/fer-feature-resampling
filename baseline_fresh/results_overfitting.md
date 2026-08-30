# Overfitting-Reduction Sweep — Results

*Populate after running `run_overfit_sweep.sh` on Kaggle T4.*

## Configs

| Config | Freeze | LR Groups | MixUp | WD | Other |
|---|---|---|---|---|---|
| **G0** (baseline) | None | Uniform 0.005 | Off | 1e-3 | Reference config |
| **G1** | conv1,bn1,layer1,layer2 | layer3=lr/10, layer4=lr/3, fc=lr | Off | 1e-3 | Discriminative LR |
| **G2** | Same as G1 | Same as G1 | α=0.2 | 3e-3 | Soft-label CE mixup |

All share: ResNet-18 IMAGENET1K_V1, 224x224, plain CE + LS 0.1, mild aug
(flip + rot15 + crop), warmup 5 + cosine 100ep, early stopping patience 20,
ten-crop+flip TTA, imbalanced eval, leakage-filtered splits.

## Comparison Table (seed=42)

| Config | Train F1 | Val F1 | Test F1 | Gap | Best Ep | Stop Ep | Disgust | Fear | Sad | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 (baseline) | *pending* | *pending* | *pending* | *pending* | — | — | *pending* | *pending* | *pending* | — |
| G1 (freeze+discLR) | *pending* | *pending* | *pending* | *pending* | — | — | *pending* | *pending* | *pending* | — |
| G2 (G1+mixup+wd) | *pending* | *pending* | *pending* | *pending* | — | — | *pending* | *pending* | *pending* | — |

## Per-Class Test F1 (seed=42)

| Class | G0 | G1 | G2 |
|---|---|---|---|
| angry | — | — | — |
| disgust | — | — | — |
| fear | — | — | — |
| happy | — | — | — |
| sad | — | — | — |
| surprise | — | — | — |
| neutral | — | — | — |

## Interpretation

*To be written after results are available. Key questions:*
- Does partial freezing (G1) reduce the train-val gap without sacrificing macro-F1?
- Does MixUp (G2) further regularize, and at what cost to disgust (extreme minority)?
- Which config best balances generalization gap vs absolute performance?

*No winner auto-selected — the table is for informed manual selection.*
