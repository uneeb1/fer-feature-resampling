# Overfitting-Reduction Sweep — Results

*Populate after running `run_overfit_sweep.sh` on Kaggle T4.*

## Configs

| Config | Freeze | LR Groups | Dropout | WD | LS | MixUp | Notes |
|---|---|---|---|---|---|---|---|
| **G0** (baseline) | None | Uniform 0.005 | 0.4 | 1e-3 | 0.1 | Off | Reference |
| **G1** | conv1,bn1,L1,L2 | L3=lr/10, L4=lr/3, fc=lr | 0.4 | 1e-3 | 0.1 | Off | Partial freeze + discLR |
| **G2** | Same as G1 | Same as G1 | 0.4 | 3e-3 | 0.1 | α=0.2 | G1 + MixUp + higher WD |
| **G3** | None | Uniform 0.005 | 0.5 | 5e-3 | 0.2 | α=0.3 | Full fine-tune + stacked reg |

All share: ResNet-18 IMAGENET1K_V1, 224x224, plain CE (no class weights), mild aug
(flip + rot15 + crop, NO erasing), warmup 5 + cosine 100ep, early stopping
patience 20, ten-crop+flip TTA, imbalanced eval, leakage-filtered splits.

## Comparison Table (seed=42)

| Config | Train F1 | Val F1 | Test F1 | Gap | Best Ep | Stop Ep | Disgust | Fear | Sad | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 (baseline) | — | — | — | — | — | — | — | — | — | — |
| G1 (freeze+discLR) | — | — | — | — | — | — | — | — | — | — |
| G2 (G1+mixup+wd) | — | — | — | — | — | — | — | — | — | — |
| G3 (full+stacked reg) | — | — | — | — | — | — | — | — | — | — |

## Per-Class Test F1 (seed=42)

| Class | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| angry | — | — | — | — |
| disgust | — | — | — | — |
| fear | — | — | — | — |
| happy | — | — | — | — |
| sad | — | — | — | — |
| surprise | — | — | — | — |
| neutral | — | — | — | — |

## Interpretation

*To be written after results are available. Key questions:*
- Does partial freezing (G1) reduce the train-val gap without sacrificing macro-F1?
- Does MixUp (G2) further regularize, and at what cost to disgust (extreme minority)?
- Does G3 (full fine-tune + stacked regularizers) achieve a better gap-vs-F1 tradeoff
  than G1/G2 by avoiding the freeze that starves minority classes?
- Which config best balances generalization gap vs absolute performance?

*No winner auto-selected — the table is for informed manual selection.*
