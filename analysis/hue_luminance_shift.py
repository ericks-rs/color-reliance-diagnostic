# -*- coding: utf-8 -*-
"""How much does hue rotation shift luminance?

The paper (Section VI-E) states that hue rotation preserves saturation and value
and shifts mean luminance only slightly, while the grayscale conversion behind the
Color-Reliance score leaves luminance unchanged. This script measures both on the
full test set using the actual evaluation pipeline and the actual perturbation
functions, and writes tables/hue_luminance_shift.csv.

No training and no GPU. Run:
  python analysis/hue_luminance_shift.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data import get_eval_dataset_with_ids     # noqa: E402
from src.perturb import hue_rotate, grayscale       # noqa: E402

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
FACTORS = [float(f) for f in cfg["perturb"]["hue_factors"]]

# Rec.601 luma weights, matching torchvision's rgb_to_grayscale.
W = torch.tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1)


def luma(x):
    return (x * W).sum(1)


rows = []
for ds in ["flowers102", "cub200"]:
    dset, _ = get_eval_dataset_with_ids(ds, cfg, normalize=False)  # RGB in [0,1]
    loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=0)

    totals = {f: [0.0, 0.0] for f in FACTORS}          # sum(delta), sum(|delta|)
    totals["grayscale"] = [0.0, 0.0]
    base, seen = 0.0, 0
    for xb, _ in loader:
        y0 = luma(xb)
        base += y0.mean(dim=(1, 2)).sum().item()
        seen += xb.shape[0]
        for f in FACTORS:
            d = luma(hue_rotate(xb, f)) - y0
            totals[f][0] += d.mean(dim=(1, 2)).sum().item()
            totals[f][1] += d.abs().mean(dim=(1, 2)).sum().item()
        dg = luma(grayscale(xb)) - y0
        totals["grayscale"][0] += dg.mean(dim=(1, 2)).sum().item()
        totals["grayscale"][1] += dg.abs().mean(dim=(1, 2)).sum().item()

    base_mean = base / seen
    for key in FACTORS + ["grayscale"]:
        s, sa = totals[key]
        rows.append(dict(
            dataset=ds,
            condition=("hue_%+.3f" % key) if key != "grayscale" else "grayscale",
            n_test_images=seen,
            mean_luminance_clean=round(base_mean, 4),
            delta_mean_luminance=round(s / seen, 4),
            mean_abs_delta_luminance=round(sa / seen, 4),
        ))

out = ROOT / "tables" / "hue_luminance_shift.csv"
pd.DataFrame(rows).to_csv(out, index=False)
worst = max(abs(r["delta_mean_luminance"]) for r in rows if r["condition"] != "grayscale")
print(f"wrote {out}")
print(f"largest mean-luminance shift under hue rotation: {worst:.4f}")
print(f"grayscale mean-luminance shift: "
      f"{[r['delta_mean_luminance'] for r in rows if r['condition']=='grayscale']}")
