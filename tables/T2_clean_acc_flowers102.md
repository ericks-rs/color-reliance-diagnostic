# T2 — Clean top-1 accuracy, Flowers-102

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

| backbone | n seeds | accuracy | macro-F1 | UAR |
|:---|---:|:---|:---|:---|
| ResNet-50 | 5 | 0.8492+-0.0031 | 0.8416+-0.0043 | 0.8692+-0.0029 |
| ConvNeXt-T | 5 | 0.9296+-0.0054 | 0.9283+-0.0049 | 0.9446+-0.0038 |
| ViT-S | 5 | 0.8763+-0.0055 | 0.8735+-0.0050 | 0.8965+-0.0040 |
| Swin-T | 5 | 0.9365+-0.0032 | 0.9348+-0.0015 | 0.9491+-0.0009 |

