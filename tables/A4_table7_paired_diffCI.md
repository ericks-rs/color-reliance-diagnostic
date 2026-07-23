# A4 — Table 7: pairwise contrasts (difference with 95% CI)

Differences are reported with 95% confidence intervals from the per-seed paired differences (t, df = 4). P-values, effect sizes, and significance markers are deliberately omitted: with five seeds and a single checkpoint per arm they overstate the evidence.

## Flowers-102 — Clean accuracy

| contrast | difference | 95% CI | n |
|---|---|---|---|
| ResNet-50 − ConvNeXt-T | -0.0804 | [-0.0881, -0.0727] | 5 |
| ResNet-50 − ViT-S | -0.0272 | [-0.0317, -0.0226] | 5 |
| ResNet-50 − Swin-T | -0.0873 | [-0.0945, -0.0801] | 5 |
| ConvNeXt-T − ViT-S | +0.0533 | [+0.0440, +0.0626] | 5 |
| ConvNeXt-T − Swin-T | -0.0069 | [-0.0151, +0.0014] | 5 |
| ViT-S − Swin-T | -0.0601 | [-0.0707, -0.0496] | 5 |

## Flowers-102 — Color-Reliance (CR)

| contrast | difference | 95% CI | n |
|---|---|---|---|
| ResNet-50 − ConvNeXt-T | +0.0968 | [+0.0543, +0.1394] | 5 |
| ResNet-50 − ViT-S | -0.1849 | [-0.2187, -0.1512] | 5 |
| ResNet-50 − Swin-T | +0.0660 | [+0.0466, +0.0854] | 5 |
| ConvNeXt-T − ViT-S | -0.2817 | [-0.3000, -0.2635] | 5 |
| ConvNeXt-T − Swin-T | -0.0308 | [-0.0902, +0.0286] | 5 |
| ViT-S − Swin-T | +0.2509 | [+0.2038, +0.2981] | 5 |

## Flowers-102 — per-arm means

| backbone | clean acc | grayscale acc | CR |
|---|---|---|---|
| ResNet-50 | 0.8492 ± 0.0031 | 0.4987 ± 0.0173 | 0.3504 ± 0.0150 |
| ConvNeXt-T | 0.9296 ± 0.0054 | 0.6760 ± 0.0220 | 0.2536 ± 0.0215 |
| ViT-S | 0.8763 ± 0.0055 | 0.3410 ± 0.0113 | 0.5353 ± 0.0142 |
| Swin-T | 0.9365 ± 0.0032 | 0.6521 ± 0.0249 | 0.2844 ± 0.0278 |

## CUB-200 — Clean accuracy

| contrast | difference | 95% CI | n |
|---|---|---|---|
| ResNet-50 − ConvNeXt-T | -0.0502 | [-0.0549, -0.0454] | 5 |
| ResNet-50 − ViT-S | -0.0079 | [-0.0109, -0.0049] | 5 |
| ResNet-50 − Swin-T | -0.0595 | [-0.0643, -0.0548] | 5 |
| ConvNeXt-T − ViT-S | +0.0423 | [+0.0353, +0.0493] | 5 |
| ConvNeXt-T − Swin-T | -0.0093 | [-0.0162, -0.0025] | 5 |
| ViT-S − Swin-T | -0.0516 | [-0.0586, -0.0447] | 5 |

## CUB-200 — Color-Reliance (CR)

| contrast | difference | 95% CI | n |
|---|---|---|---|
| ResNet-50 − ConvNeXt-T | -0.0003 | [-0.0119, +0.0112] | 5 |
| ResNet-50 − ViT-S | -0.0811 | [-0.0979, -0.0644] | 5 |
| ResNet-50 − Swin-T | -0.0137 | [-0.0491, +0.0216] | 5 |
| ConvNeXt-T − ViT-S | -0.0808 | [-0.0914, -0.0702] | 5 |
| ConvNeXt-T − Swin-T | -0.0134 | [-0.0388, +0.0121] | 5 |
| ViT-S − Swin-T | +0.0674 | [+0.0359, +0.0989] | 5 |

## CUB-200 — per-arm means

| backbone | clean acc | grayscale acc | CR |
|---|---|---|---|
| ResNet-50 | 0.7873 ± 0.0029 | 0.3712 ± 0.0083 | 0.4160 ± 0.0107 |
| ConvNeXt-T | 0.8375 ± 0.0034 | 0.4211 ± 0.0037 | 0.4164 ± 0.0042 |
| ViT-S | 0.7951 ± 0.0052 | 0.2980 ± 0.0093 | 0.4971 ± 0.0109 |
| Swin-T | 0.8468 ± 0.0036 | 0.4171 ± 0.0205 | 0.4297 ± 0.0195 |


> Inference is conditional on a single checkpoint per arm; the variance reflects fine-tuning, not the pretraining population. Seeds are matched across arms: the data loader uses a seeded generator and worker initialisation, so the sample order is identical across architectures at the same seed.

