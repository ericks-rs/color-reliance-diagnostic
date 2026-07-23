# T4 — Colour reliance, CUB-200

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

Colour reliance is the absolute drop, clean minus grayscale, computed per seed. We do not report it as a ratio: the denominator would differ across backbones and the values would stop being comparable.

| backbone | clean | grayscale | CR |
|:---|:---|:---|:---|
| ResNet-50 | 0.7873+-0.0029 | 0.3712+-0.0083 | 0.4160+-0.0107 |
| ConvNeXt-T | 0.8375+-0.0034 | 0.4211+-0.0037 | 0.4164+-0.0042 |
| ViT-S | 0.7951+-0.0052 | 0.2980+-0.0093 | 0.4971+-0.0109 |
| Swin-T | 0.8468+-0.0036 | 0.4171+-0.0205 | 0.4297+-0.0195 |

