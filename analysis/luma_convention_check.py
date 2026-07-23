# -*- coding: utf-8 -*-
"""Apakah Color-Reliance bergantung pada konvensi luminance?

Skor di paper memakai ITU-R 601 (0.2989, 0.5870, 0.1140), bobot default
torchvision dan Pillow. Skrip ini mengevaluasi ulang checkpoint yang sama di
bawah ITU-R BT.709 (0.2126, 0.7152, 0.0722) dan melaporkan selisihnya.

Tidak ada pelatihan ulang. Satu seed, empat backbone equalized, dua dataset.
Jalankan: python analysis/luma_convention_check.py [seed]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src import data as data_mod          # noqa: E402
from src import models as models_mod      # noqa: E402
from src.metrics import compute_metrics   # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)

W601 = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(DEV)
W709 = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1).to(DEV)

ARMS = [("resnet50", "resnet50", "ResNet-50"),
        ("convnext_tiny_in1k", "convnext_tiny_in1k", "ConvNeXt-T"),
        ("vit_small_in1k", "vit_small_in1k", "ViT-S"),
        ("swin_tiny", "swin_tiny", "Swin-T")]
DATASETS = [("flowers102", "Flowers-102", 102), ("cub200", "CUB-200", 200)]


def gray(x, w):
    """x: (B,3,H,W) di [0,1] -> luminance direplikasi ke tiga kanal."""
    return (x * w).sum(1, keepdim=True).expand(-1, 3, -1, -1)


@torch.no_grad()
def run(model, loader, fn):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEV, non_blocking=True)
        if fn is not None:
            x = fn(x)
        x = (x - MEAN) / STD
        with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
rows = []
print("seed %d, perangkat %s\n" % (SEED, DEV))
print("%-12s %-11s %8s %9s %9s %9s %9s %9s"
      % ("dataset", "backbone", "clean", "gray601", "gray709", "CR601", "CR709", "selisih"))
print("-" * 88)

for dsk, dsn, ncls in DATASETS:
    ds, _ = data_mod.get_eval_dataset_with_ids(dsk, cfg, normalize=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    for arm, mkey, name in ARMS:
        ck = os.path.join(ROOT, "checkpoints", "%s_%s_seed%d.pt" % (dsk, arm, SEED))
        assert os.path.exists(ck), ck
        model, _ = models_mod.load_checkpoint(ck, mkey, cfg, ncls, device=DEV)

        y, p = run(model, loader, None)
        clean = compute_metrics(y, p)["acc"]
        y, p = run(model, loader, lambda x: gray(x, W601))
        g601 = compute_metrics(y, p)["acc"]
        y, p = run(model, loader, lambda x: gray(x, W709))
        g709 = compute_metrics(y, p)["acc"]

        cr601, cr709 = clean - g601, clean - g709
        print("%-12s %-11s %8.4f %9.4f %9.4f %9.4f %9.4f %+9.4f"
              % (dsn, name, clean, g601, g709, cr601, cr709, cr709 - cr601))
        rows.append(dict(dataset=dsn, backbone=name, seed=SEED,
                         clean=round(clean, 4), gray_601=round(g601, 4), gray_709=round(g709, 4),
                         CR_601=round(cr601, 4), CR_709=round(cr709, 4),
                         delta_CR=round(cr709 - cr601, 4)))
        del model
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
out = os.path.join(ROOT, "tables", "luma_convention_check.csv")
df.to_csv(out, index=False)
print("-" * 88)
print("selisih CR terbesar (mutlak): %.4f" % df.delta_CR.abs().max())
print("urutan CR sama di kedua konvensi:")
for dsn in df.dataset.unique():
    d = df[df.dataset == dsn]
    a = list(d.sort_values("CR_601", ascending=False).backbone)
    b = list(d.sort_values("CR_709", ascending=False).backbone)
    print("   %-12s %s" % (dsn, "YA" if a == b else "TIDAK: %s vs %s" % (a, b)))
print("\nditulis:", out)
