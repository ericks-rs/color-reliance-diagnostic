"""fig7 (distribusi Color-Reliance per-species) - R2.4. Ukuran cetak IEEE Access.

Untuk tiap (dataset, model), CR dihitung per KELAS (V7_perspecies_CR_{ds}.csv, kolom
CR_mean per y_true). Satu violin per model, dua panel (Flowers-102, CUB-200). Figur ini
menunjukkan sebaran per-kelas yang skor agregat (Table 6) sembunyikan: sebagian kelas
punya CR < 0 (sedikit lebih baik tanpa warna) dan sebagian mendekati 1.0 (kolaps tanpa
warna). Distribusi ViT-S bergeser paling tinggi, jadi ViT paling reliant bukan hanya di
rata-rata tetapi menyeluruh di distribusi kelas.

Print-spec sama dgn build_fig1-5: figsize = 6.989 in (= \\textwidth ieeeaccess.cls),
savefig TANPA bbox_inches="tight" (8 pt = 8 pt cetak), >= 8 pt, dpi 600, arm dari
revision/arms.py. Violin = deterministik (tak ada jitter acak), jadi md5 stabil.

python revision/build_fig7_perspecies.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

from analysis.arms import MAIN_ARMS as MODEL_ORDER, MLAB, MCOLOR, DSLAB, check

TEXTWIDTH_IN = 6.989
COLWIDTH_IN = 3.36   # \columnwidth ieeeaccess.cls (single column)


def _print_style():
    plt.rcParams.update({
        "savefig.dpi": 600, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })


def build(perclass, datasets, out_png):
    _print_style()
    # single-column: dua dataset ditumpuk ATAS-BAWAH (bukan kiri-kanan), tiap
    # panel dapat lebar kolom penuh.
    fig, axes = plt.subplots(len(datasets), 1, figsize=(COLWIDTH_IN, 3.8),
                             squeeze=False)
    # rentang-Y disamakan lintas panel: kedua dataset punya CR per-kelas di kisaran
    # yang sama (~-0.3 sampai 1.0), jadi bentuk distribusi bisa dibandingkan langsung.
    lo = float(perclass.CR_mean.min()) - 0.05
    hi = 1.02
    for c, ds in enumerate(datasets):
        ax = axes[c][0]
        sub = perclass[perclass.dataset == ds]
        data = [sub[sub.arm == m].CR_mean.dropna().values for m in MODEL_ORDER]
        pos = list(range(len(MODEL_ORDER)))
        parts = ax.violinplot(data, positions=pos, widths=0.32,
                              showmeans=False, showextrema=False, showmedians=True)
        for body, m in zip(parts["bodies"], MODEL_ORDER):
            body.set_facecolor(MCOLOR[m]); body.set_edgecolor("0.3")
            body.set_alpha(0.55); body.set_linewidth(0.6)
        parts["cmedians"].set_color("0.15"); parts["cmedians"].set_linewidth(1.1)
        ax.axhline(0.0, color="0.45", ls=":", lw=0.9, zorder=1)   # acuan CR = 0
        ax.set_xticks(pos)
        ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(lo, hi)
        ax.set_title(DSLAB[ds])
        ax.set_ylabel("per-class Color-Reliance")   # tiap panel (kini ditumpuk)
    fig.tight_layout(h_pad=3.0)            # jarak vertikal antar panel atas-bawah
    fig.savefig(out_png, dpi=600)          # TANPA bbox_inches="tight"
    plt.close(fig)


def main():
    chdir_to_root()
    cfg = load_config()
    tdir = Path(cfg["paths"]["tables"])
    fdir = Path(cfg["paths"]["figures"]) / "revised"
    fdir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in ["flowers102", "cub200"]
                if (tdir / f"V7_perspecies_CR_{d}.csv").exists()]
    if not datasets:
        raise SystemExit("V7_perspecies_CR_*.csv tidak ada")
    perclass = pd.concat([pd.read_csv(tdir / f"V7_perspecies_CR_{d}.csv")
                          for d in datasets], ignore_index=True)
    perclass = perclass[perclass.arm.isin(MODEL_ORDER)].copy()
    check(perclass.rename(columns={"arm": "model"}))   # jaring pengaman arm setara

    out = fdir / "fig7_perspecies.png"
    build(perclass, datasets, out)

    from PIL import Image
    im = Image.open(out)
    w_in = im.size[0] / im.info["dpi"][0]
    print(f"  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {w_in:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"     kelas per (dataset): {perclass.groupby('dataset').y_true.nunique().to_dict()}")


if __name__ == "__main__":
    main()
