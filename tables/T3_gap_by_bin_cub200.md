# T3b — Accuracy gap to the convolutional baselines, by bin, CUB-200

*Arms equalized to ImageNet-1k (R1.2): ResNet-50 `a1_in1k`, ConvNeXt-T `fb_in1k`, ViT-S `augreg_in1k`, Swin-T `ms_in1k`.*

Gaps are computed per seed and then summarized, not as a difference of two means, so the reported spread reflects the pairs actually observed. The trend column is high minus low, computed per seed. It is descriptive and is not a test for trend.

| backbone vs baseline | low | mid | high | trend (high − low) |
|:---|:---|:---|:---|:---|
| ViT-S − ResNet-50 | 0.0079+-0.0105 | 0.0113+-0.0049 | 0.0045+-0.0066 | -0.0034+-0.0165 |
| ViT-S − ConvNeXt-T | -0.0441+-0.0099 | -0.0411+-0.0082 | -0.0418+-0.0054 | 0.0024+-0.0122 |
| Swin-T − ResNet-50 | 0.0615+-0.0037 | 0.0583+-0.0065 | 0.0588+-0.0133 | -0.0027+-0.0112 |
| Swin-T − ConvNeXt-T | 0.0094+-0.0099 | 0.0060+-0.0081 | 0.0125+-0.0127 | 0.0031+-0.0163 |

