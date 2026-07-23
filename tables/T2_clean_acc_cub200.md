# T2 — Clean top-1 accuracy, CUB-200

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

| backbone | n seeds | accuracy | macro-F1 | UAR |
|:---|---:|:---|:---|:---|
| ResNet-50 | 5 | 0.7873+-0.0029 | 0.7873+-0.0031 | 0.7891+-0.0028 |
| ConvNeXt-T | 5 | 0.8375+-0.0034 | 0.8368+-0.0037 | 0.8395+-0.0037 |
| ViT-S | 5 | 0.7951+-0.0052 | 0.7936+-0.0057 | 0.7967+-0.0054 |
| Swin-T | 5 | 0.8468+-0.0036 | 0.8458+-0.0037 | 0.8485+-0.0040 |

