# T4 — Colour reliance, Flowers-102

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

Colour reliance is the absolute drop, clean minus grayscale, computed per seed. We do not report it as a ratio: the denominator would differ across backbones and the values would stop being comparable.

| backbone | clean | grayscale | CR |
|:---|:---|:---|:---|
| ResNet-50 | 0.8492+-0.0031 | 0.4987+-0.0173 | 0.3504+-0.0150 |
| ConvNeXt-T | 0.9296+-0.0054 | 0.6760+-0.0220 | 0.2536+-0.0215 |
| ViT-S | 0.8763+-0.0055 | 0.3410+-0.0113 | 0.5353+-0.0142 |
| Swin-T | 0.9365+-0.0032 | 0.6521+-0.0249 | 0.2844+-0.0278 |

