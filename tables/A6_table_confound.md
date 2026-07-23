# A6 — Pretraining-data confound: equalized vs non-equalized

Non-equalized checkpoints (extra pretraining data) are reported here only. They never enter the main tables. Differences are within-architecture (same backbone, matched seeds), so CIs apply.

## Flowers-102 — clean accuracy

| backbone | equalized (IN1k) | non-equalized | clean (eq) | clean (non-eq) | diff (95% CI) |
|---|---|---|---|---|---|
| ConvNeXt-T | fb_in1k (IN1k) | in12k_ft_in1k (IN12k) | 0.9296 ± 0.0054 | 0.9898 ± 0.0012 | +0.0602 [+0.0525, +0.0679] |
| ViT-S | augreg_in1k (IN1k) | augreg_in21k_ft_in1k (IN21k) | 0.8763 ± 0.0055 | 0.9686 ± 0.0066 | +0.0922 [+0.0837, +0.1008] |

## Flowers-102 — grayscale accuracy and Color-Reliance

| backbone | gray (eq) | gray (non-eq) | CR (eq) | CR (non-eq) | ΔCR (non-eq − eq), 95% CI |
|---|---|---|---|---|---|
| ConvNeXt-T | 0.6760 ± 0.0220 | 0.8274 ± 0.0410 | 0.2536 ± 0.0215 | 0.1624 ± 0.0414 | -0.0912 [-0.1433, -0.0391] |
| ViT-S | 0.3410 ± 0.0113 | 0.7141 ± 0.0493 | 0.5353 ± 0.0142 | 0.2545 ± 0.0466 | -0.2809 [-0.3291, -0.2326] |

**Reliance ordering.** Equalized: ViT-S 0.5353 vs ConvNeXt-T 0.2536 (ViT-S higher). Non-equalized: ViT-S 0.2545 vs ConvNeXt-T 0.1624 (ViT-S higher). The ordering holds under both pretraining regimes, even though ViT-S is the arm with the larger pretraining corpus.

## CUB-200 — clean accuracy

| backbone | equalized (IN1k) | non-equalized | clean (eq) | clean (non-eq) | diff (95% CI) |
|---|---|---|---|---|---|
| ConvNeXt-T | fb_in1k (IN1k) | in12k_ft_in1k (IN12k) | 0.8375 ± 0.0034 | 0.8866 ± 0.0027 | +0.0492 [+0.0417, +0.0566] |
| ViT-S | augreg_in1k (IN1k) | augreg_in21k_ft_in1k (IN21k) | 0.7951 ± 0.0052 | 0.8498 ± 0.0030 | +0.0547 [+0.0519, +0.0575] |

## CUB-200 — grayscale accuracy and Color-Reliance

| backbone | gray (eq) | gray (non-eq) | CR (eq) | CR (non-eq) | ΔCR (non-eq − eq), 95% CI |
|---|---|---|---|---|---|
| ConvNeXt-T | 0.4211 ± 0.0037 | 0.5270 ± 0.0084 | 0.4164 ± 0.0042 | 0.3596 ± 0.0092 | -0.0567 [-0.0705, -0.0429] |
| ViT-S | 0.2980 ± 0.0093 | 0.4272 ± 0.0094 | 0.4971 ± 0.0109 | 0.4226 ± 0.0111 | -0.0745 [-0.0886, -0.0605] |

**Reliance ordering.** Equalized: ViT-S 0.4971 vs ConvNeXt-T 0.4164 (ViT-S higher). Non-equalized: ViT-S 0.4226 vs ConvNeXt-T 0.3596 (ViT-S higher). The ordering holds under both pretraining regimes, even though ViT-S is the arm with the larger pretraining corpus.


> Inference is conditional on a single checkpoint per arm; the variance reflects fine-tuning, not the pretraining population. The contrasts in this table are within-architecture, with seeds matched across arms by a seeded data loader.

