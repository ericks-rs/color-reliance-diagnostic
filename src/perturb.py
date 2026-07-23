"""Perturbasi warna TEST-TIME (E2). Diterapkan pada batch RGB [0,1] (B,3,H,W)
SEBELUM normalisasi ImageNet. Tidak ada training ulang.

Kondisi:
  - grayscale penuh (luminance -> replikasi 3 channel)
  - hue rotation: factor {-0.5,-0.25,-0.083,+0.083,+0.25,+0.5} (~ +-30,+-90,+-180 deg)
  - channel shuffle: semua 5 permutasi RGB non-identitas (rata-rata)
  - color quantization: {2,4,8,16} level per channel
"""
from itertools import permutations

import torch
import torchvision.transforms.functional as TF

# 5 permutasi RGB non-identitas
CHANNEL_PERMS = [p for p in permutations([0, 1, 2]) if p != (0, 1, 2)]


def grayscale(x):
    """x: (B,3,H,W) in [0,1]. Luminance -> 3 channel."""
    g = TF.rgb_to_grayscale(x, num_output_channels=3)
    return g


def hue_rotate(x, factor):
    """factor in [-0.5, 0.5]."""
    return TF.adjust_hue(x, factor)


def channel_shuffle(x, perm):
    """perm: tuple permutasi index channel, mis. (1,0,2)."""
    return x[:, list(perm), :, :]


def quantize(x, levels):
    """Kuantisasi ke `levels` nilai per channel (uniform)."""
    levels = int(levels)
    q = torch.round(x * (levels - 1)) / (levels - 1)
    return q.clamp(0, 1)


def build_conditions(cfg):
    """Return list of (condition_name, kind, param, fn). fn: batch[0,1]->batch[0,1]."""
    pc = cfg["perturb"]
    conds = []
    conds.append(("grayscale", "grayscale", None, lambda x: grayscale(x)))
    for f in pc["hue_factors"]:
        conds.append((f"hue_{f:+.3f}", "hue", float(f),
                      (lambda f: (lambda x: hue_rotate(x, f)))(f)))
    for i, perm in enumerate(CHANNEL_PERMS):
        conds.append((f"shuffle_{''.join(map(str, perm))}", "shuffle", perm,
                      (lambda p: (lambda x: channel_shuffle(x, p)))(perm)))
    for lv in pc["quant_levels"]:
        conds.append((f"quant_{lv}", "quant", int(lv),
                      (lambda lv: (lambda x: quantize(x, lv)))(lv)))
    return conds
