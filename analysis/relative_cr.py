# -*- coding: utf-8 -*-
"""Relative Color-Reliance, CR_rel = CR / acc_clean, per seed.

TIDAK dipakai di naskah. Ini amunisi cadangan untuk response letter kalau ada
reviewer yang meminta reliance ternormalisasi.

Kenapa tidak masuk paper: penyebutnya (clean accuracy) berbeda antar model, jadi
model dengan akurasi bersih rendah otomatis terlihat lebih bergantung warna. Pada
angka equalized yang berlaku sekarang, ResNet-50 pindah dari peringkat TERAKHIR di
CR absolut ke peringkat KEDUA di CR relatif pada CUB-200, murni karena akurasi
bersihnya paling rendah. Klaim yang bertahan di kedua ukuran hanya puncaknya,
yaitu ViT-S paling color-reliant pada kedua dataset.

Jalankan: python analysis/relative_cr.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

# empat arm equalized, semuanya ImageNet-1k, sama dengan tabel utama paper
ARMS = {
    "resnet50": "ResNet-50",
    "convnext_tiny_in1k": "ConvNeXt-T",
    "vit_small_in1k": "ViT-S",
    "swin_tiny": "Swin-T",
}
DATASETS = {"flowers102": "Flowers-102", "cub200": "CUB-200"}


def per_seed(ds):
    """Kembalikan DataFrame per (model, seed) dengan clean, gray, CR, CR_rel."""
    df = pd.read_csv(os.path.join(RESULTS, "e2_perturb_%s.csv" % ds))
    df = df[df.model.isin(ARMS) & df.condition.isin(["clean", "grayscale"])]
    wide = df.pivot_table(index=["model", "seed"], columns="condition", values="acc")
    wide = wide.reset_index()
    wide["CR"] = wide["clean"] - wide["grayscale"]
    wide["CR_rel"] = wide["CR"] / wide["clean"]
    wide["backbone"] = wide.model.map(ARMS)
    return wide


def ci95(v):
    """Selang t 95 persen untuk rerata, n kecil (5 seed)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2:
        return (np.nan, np.nan)
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return (v.mean() - h, v.mean() + h)


rows = []
for ds, label in DATASETS.items():
    w = per_seed(ds)
    print("=" * 74)
    print(label)
    print("  %-12s %8s %8s %9s %9s %-22s" % ("backbone", "clean", "CR", "CR_rel", "SD_rel", "CI95 CR_rel"))
    agg = []
    for arm, name in ARMS.items():
        g = w[w.model == arm]
        lo, hi = ci95(g.CR_rel)
        print("  %-12s %8.4f %8.4f %9.4f %9.4f [%.4f, %.4f]"
              % (name, g["clean"].mean(), g.CR.mean(), g.CR_rel.mean(), g.CR_rel.std(ddof=1), lo, hi))
        agg.append((name, g["clean"].mean(), g.CR.mean(), g.CR_rel.mean()))
        rows.append(dict(dataset=label, backbone=name, n_seed=len(g),
                         clean_mean=round(g["clean"].mean(), 4),
                         CR_mean=round(g.CR.mean(), 4),
                         CR_rel_mean=round(g.CR_rel.mean(), 4),
                         CR_rel_sd=round(g.CR_rel.std(ddof=1), 4),
                         CR_rel_ci_lo=round(lo, 4), CR_rel_ci_hi=round(hi, 4)))
    a = [n for n, c, cr, r in sorted(agg, key=lambda x: -x[2])]
    b = [n for n, c, cr, r in sorted(agg, key=lambda x: -x[3])]
    print("  urutan CR absolut : " + " > ".join(a))
    print("  urutan CR relatif : " + " > ".join(b))
    print("  urutan sama       : " + ("YA" if a == b else "TIDAK, penyebut menggeser peringkat"))
    print("  puncak bertahan   : " + ("YA" if a[0] == b[0] == "ViT-S" else "TIDAK"))
    print()

out = os.path.join(ROOT, "tables", "relative_cr.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("ditulis:", out)
