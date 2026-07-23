# A5 — Table 8: ResNet-50 under four fine-tuning configurations

ConvNeXt-T (ImageNet-1k) is shown as the upper reference. Differences and 95% CIs are against the shared configuration (S1) and are within-architecture (seeds matched).

**Reading notes.**

- **S11 changes two factors at once** (pretraining checkpoint *and* optimizer) relative to S1. It is reported as a configuration, not as a single-factor contrast, and it is not used to attribute any effect to the checkpoint alone.
- **S13 is the weakest configuration.** Holding the checkpoint fixed at `a1_in1k`, SGD trails AdamW by a wide margin. Read together with S11, which reaches the highest accuracy using the same SGD setting on the legacy checkpoint, this indicates a checkpoint-by-optimizer interaction: the `a1_in1k` weights respond poorly to plain SGD fine-tuning. It is not evidence that SGD is inherently unsuited to ResNet-50.

## Flowers-102

| id | configuration | clean acc | grayscale acc | CR | diff vs shared (95% CI) |
|---|---|---|---|---|---|
| S1 | ResNet-50 / shared (AdamW 1e-4) | 0.8492 ± 0.0031 | 0.4987 ± 0.0173 | 0.3504 ± 0.0150 | reference |
| S11 | ResNet-50 / legacy (tv weights, SGD 1e-2) | 0.9202 ± 0.0020 | 0.6698 ± 0.0106 | 0.2504 ± 0.0118 | +0.0710 [+0.0663, +0.0757] |
| S12 | ResNet-50 / proper (AdamW 1e-3) | 0.9076 ± 0.0062 | 0.5974 ± 0.0231 | 0.3102 ± 0.0211 | +0.0584 [+0.0477, +0.0691] |
| S13 | ResNet-50 / SGD (a1 weights, SGD 1e-2) | 0.7660 ± 0.0053 | 0.3793 ± 0.0194 | 0.3867 ± 0.0145 | -0.0832 [-0.0932, -0.0731] |
| S2 | ConvNeXt-T (IN1k) — reference | 0.9296 ± 0.0054 | 0.6760 ± 0.0220 | 0.2536 ± 0.0215 | not compared (cross-architecture) |

## CUB-200

| id | configuration | clean acc | grayscale acc | CR | diff vs shared (95% CI) |
|---|---|---|---|---|---|
| S1 | ResNet-50 / shared (AdamW 1e-4) | 0.7873 ± 0.0029 | 0.3712 ± 0.0083 | 0.4160 ± 0.0107 | reference |
| S11 | ResNet-50 / legacy (tv weights, SGD 1e-2) | 0.8125 ± 0.0038 | 0.3783 ± 0.0027 | 0.4343 ± 0.0040 | +0.0253 [+0.0200, +0.0305] |
| S12 | ResNet-50 / proper (AdamW 1e-3) | 0.8187 ± 0.0016 | 0.3916 ± 0.0122 | 0.4271 ± 0.0106 | +0.0315 [+0.0261, +0.0369] |
| S13 | ResNet-50 / SGD (a1 weights, SGD 1e-2) | 0.7525 ± 0.0030 | 0.3456 ± 0.0036 | 0.4069 ± 0.0050 | -0.0347 [-0.0373, -0.0322] |
| S2 | ConvNeXt-T (IN1k) — reference | 0.8375 ± 0.0034 | 0.4211 ± 0.0037 | 0.4164 ± 0.0042 | not compared (cross-architecture) |


> Inference is conditional on a single checkpoint per arm; the variance reflects fine-tuning, not the pretraining population. The contrasts in this table are within-architecture, with seeds matched across arms by a seeded data loader.

