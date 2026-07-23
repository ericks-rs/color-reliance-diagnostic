# A7 — R2.3 three-cell matrix (train mode x test mode)

All entries are absolute top-1 accuracy. We deliberately do not report a recovery ratio as a headline metric: its denominator (the colour-reliance gap) differs across backbones, so the ratio is not comparable between models. The residual column is the absolute distance that grayscale training leaves unrecovered.

## Flowers-102

| backbone | RGB-train / RGB-test | RGB-train / gray-test | gray-train / gray-test | residual (RGB/RGB − gray/gray) |
|---|---|---|---|---|
| ResNet-50 | 0.8492 ± 0.0031 | 0.4987 ± 0.0173 | 0.8015 ± 0.0021 | 0.0477 |
| ConvNeXt-T | 0.9296 ± 0.0054 | 0.6760 ± 0.0220 | 0.9090 ± 0.0021 | 0.0206 |
| ViT-S | 0.8763 ± 0.0055 | 0.3410 ± 0.0113 | 0.8413 ± 0.0044 | 0.0350 |
| Swin-T | 0.9365 ± 0.0032 | 0.6521 ± 0.0249 | 0.9103 ± 0.0052 | 0.0262 |

## CUB-200

| backbone | RGB-train / RGB-test | RGB-train / gray-test | gray-train / gray-test | residual (RGB/RGB − gray/gray) |
|---|---|---|---|---|
| ResNet-50 | 0.7873 ± 0.0029 | 0.3712 ± 0.0083 | 0.6498 ± 0.0040 | 0.1374 |
| ConvNeXt-T | 0.8375 ± 0.0034 | 0.4211 ± 0.0037 | 0.7397 ± 0.0038 | 0.0978 |
| ViT-S | 0.7951 ± 0.0052 | 0.2980 ± 0.0093 | 0.6165 ± 0.0123 | 0.1786 |
| Swin-T | 0.8468 ± 0.0036 | 0.4171 ± 0.0205 | 0.7310 ± 0.0015 | 0.1158 |


> Inference is conditional on a single checkpoint per arm; the variance reflects fine-tuning, not the pretraining population. The contrasts in this table are within-architecture, with seeds matched across arms by a seeded data loader.

