# A8 — Architecture contrast for R1.3 (convolutional side)

**We do not report a legacy-to-modern ladder.** The legacy configuration (S11) differs from the shared configuration (S1) in both the pretraining checkpoint and the optimizer, so any step between them mixes two factors and cannot isolate the pretraining recipe. We therefore restrict R1.3 to the contrast that is interpretable: ResNet-50 against ConvNeXt-T with the optimizer, weight decay, schedule, augmentation, and budget held fixed.

The primary figure is the **conservative** one: ResNet-50 is given the learning rate that favours it (1e-3), while ConvNeXt-T stays at the shared 1e-4. Only the trailing model is tuned up, so the residual advantage is a lower bound on the architecture effect. The matched-learning-rate contrast (both at 1e-4) is reported as the upper bound; there ResNet-50 is under-tuned, so it overstates the same effect.

## Flowers-102

| contrast | ResNet-50 | ConvNeXt-T | difference | reading |
|---|---|---|---|---|
| **S12 → S2** (ResNet at 1e-3, ConvNeXt at 1e-4) | 0.9076 | 0.9296 | **+0.0220** | conservative lower bound |
| S1 → S2 (both at 1e-4, matched) | 0.8492 | 0.9296 | +0.0804 | upper bound (ResNet under-tuned) |

Architecture effect is bracketed between +0.0220 and +0.0804 on Flowers-102.

## CUB-200

| contrast | ResNet-50 | ConvNeXt-T | difference | reading |
|---|---|---|---|---|
| **S12 → S2** (ResNet at 1e-3, ConvNeXt at 1e-4) | 0.8187 | 0.8375 | **+0.0187** | conservative lower bound |
| S1 → S2 (both at 1e-4, matched) | 0.7873 | 0.8375 | +0.0502 | upper bound (ResNet under-tuned) |

Architecture effect is bracketed between +0.0187 and +0.0502 on CUB-200.

## What this does and does not license

It licenses: under a fixed fine-tuning protocol, a modern convolutional backbone retains an advantage over ResNet-50 that survives giving ResNet-50 a more favourable learning rate.

It does not license: attributing that advantage to the pretraining recipe alone. ConvNeXt-T bundles architectural and training-recipe modernisation, and the fourth cell of that factorisation (ConvNeXt under a classic recipe) is not obtainable without pretraining from scratch. We state this as a limitation rather than estimate it.


> Inference is conditional on a single checkpoint per arm; the variance reflects fine-tuning, not the pretraining population. The contrasts in this table are between architectures and are reported as point differences without intervals, since the bracket is bounded by two configurations rather than estimated from a single paired contrast.

