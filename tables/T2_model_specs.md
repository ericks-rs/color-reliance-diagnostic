# T2 — Backbone specifications

Parameter counts and FLOPs depend on the architecture and not on the loaded weights, so they are unchanged from the first submission. The checkpoint column is new. All four backbones are initialised from ImageNet-1k weights, which is what makes the comparison in Tables 3 to 8 a comparison of architectures rather than of pretraining budgets. The first submission described the set as fully ImageNet-1k while ViT-S in fact used `augreg_in21k_ft_in1k` and ConvNeXt-T used `in12k_ft_in1k`. Those two checkpoints are now reported only in the confound table, where they are compared against their equalised counterparts.

| backbone | design | params (M) | GFLOPs | checkpoint | pretraining data |
|:---|:---|---:|---:|:---|:---|
| ResNet-50 | classic convolutional | 23.72 | 8.26 | `a1_in1k` | ImageNet-1k |
| ConvNeXt-T | modern convolutional | 27.90 | 8.91 | `fb_in1k` | ImageNet-1k |
| ViT-S | global attention | 21.70 | 8.48 | `augreg_in1k` | ImageNet-1k |
| Swin-T | hierarchical attention | 27.60 | 8.74 | `ms_in1k` | ImageNet-1k |
