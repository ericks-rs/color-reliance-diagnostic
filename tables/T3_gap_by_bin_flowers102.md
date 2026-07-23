# T3b — Accuracy gap to the convolutional baselines, by bin, Flowers-102

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

Gaps are computed per seed and then summarized, not as a difference of two means, so the reported spread reflects the pairs actually observed. The trend column is high minus low, computed per seed. It is descriptive and is not a test for trend.

| backbone vs baseline | low | mid | high | trend (high − low) |
|:---|:---|:---|:---|:---|
| ViT-S − ResNet-50 | 0.0388+-0.0084 | 0.0256+-0.0060 | 0.0171+-0.0088 | -0.0217+-0.0153 |
| ViT-S − ConvNeXt-T | -0.0443+-0.0093 | -0.0483+-0.0100 | -0.0672+-0.0082 | -0.0229+-0.0114 |
| Swin-T − ResNet-50 | 0.0936+-0.0060 | 0.0834+-0.0079 | 0.0850+-0.0098 | -0.0086+-0.0109 |
| Swin-T − ConvNeXt-T | 0.0104+-0.0078 | 0.0094+-0.0084 | 0.0007+-0.0077 | -0.0098+-0.0098 |

