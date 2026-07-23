"""fig4 (Color-Reliance) dibangun ULANG, arm setara + ukuran cetak.

Masalah versi lama (figures/pub/P2_color_reliance.png = paper/latex/fig4.png):
  - figsize 12.4 in dipasang \\textwidth = 6.989 in -> skala x0.56, anotasi 9 pt
    tercetak 5.1 pt.
  - memakai `convnext_tiny` (in12k) dan `vit_small` (in21k), melanggar R1.2.
  - berkasnya 26 Jun, mendahului rebuild 130-run.
  - ConvNeXt-T diwarnai hijau dan tiga model lain diabu-abukan. Docstring v1
    menyatakan tujuannya terang-terangan: supaya pembaca langsung membaca
    "ConvNeXt terpisah, tiga sisanya tak-terbedakan".

SOAL HIGHLIGHT. Pola itu tidak dipakai lagi, dan tidak sekadar dipindahkan ke
ViT-S. Tiga alasan.

  1. Klaim yang disangganya sudah tidak benar. Pada arm setara, CUB memberi
     ViT 0.4971, Swin 0.4297, ConvNeXt 0.4164, ResNet 0.4160. ConvNeXt praktis
     seri dengan ResNet, jadi "ConvNeXt terpisah dari tiga lainnya" gugur.
  2. Memindahkan highlight ke ViT berarti memakai alat persuasi yang sama untuk
     klaim yang baru. Reviewer 1 sudah menekan soal klaim yang melampaui bukti
     (R1.1). Menyorot satu batang adalah menyimpulkan di dalam figur, sebelum
     pembaca menilai angkanya.
  3. fig1, fig2, dan fig3 memakai palet per-model yang dikunci di
     revision/arms.py. Skema sorot-dan-redam hanya di fig4 membuat warna berarti
     satu hal di tiga figur dan hal lain di figur keempat.

Selisih ViT cukup besar untuk terlihat tanpa dibantu: di Flowers 0.5353 lawan
0.3504 milik peringkat kedua. Kalau di CUB keempatnya terlihat rapat, itu memang
keadaannya, dan figur tidak seharusnya menyembunyikan hal itu.

Yang DIPERTAHANKAN dari versi lama: rentang Y disamakan kedua panel, supaya
besaran reliance bisa dibandingkan lintas dataset secara visual.

python revision/build_fig4_colorreliance.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from analysis.arms import MAIN_ARMS, MLAB, MCOLOR, DSLAB, PRETRAIN_TAG, check

TEXTWIDTH_IN = 6.989


def _print_style():
    plt.rcParams.update({
        "savefig.dpi": 600, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
    })


def cr_per_seed(e2, arm):
    c = e2[(e2.model == arm) & (e2.condition == "clean")].set_index("seed").acc.sort_index()
    g = e2[(e2.model == arm) & (e2.condition == "grayscale")].set_index("seed").acc.sort_index()
    i = c.index.intersection(g.index)
    return (c.loc[i] - g.loc[i]).values


def build(data, datasets, out_png):
    _print_style()
    top = 0.0
    for ds in datasets:
        for v in data[ds].values():
            top = max(top, v.mean() + v.std(ddof=1))
    ylim = top + 0.075

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(TEXTWIDTH_IN, 2.85), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        x = np.arange(len(MAIN_ARMS))
        means = np.array([data[ds][a].mean() for a in MAIN_ARMS])
        sds = np.array([data[ds][a].std(ddof=1) for a in MAIN_ARMS])
        ax.bar(x, means, yerr=sds, capsize=3, width=0.62,
               color=[MCOLOR[a] for a in MAIN_ARMS],
               edgecolor="black", lw=0.5, zorder=2)
        for xi, a, mn, sd in zip(x, MAIN_ARMS, means, sds):
            v = data[ds][a]
            ax.plot(np.full(len(v), xi), v, linestyle="none", marker="o",
                    markersize=1.9, markerfacecolor="black",
                    markeredgecolor="none", alpha=0.45, zorder=4)
            ax.text(xi, mn + sd + 0.012, f"{mn:.3f}", ha="center", va="bottom",
                    fontsize=8, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([MLAB[a] for a in MAIN_ARMS], rotation=12)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(r"color reliance (acc$_\mathrm{clean}$ $-$ acc$_\mathrm{gray}$)")
        ax.set_title(DSLAB[ds])

    dots = Line2D([0], [0], color="black", ls="none", marker="o", markersize=1.9,
                  markeredgecolor="none", alpha=0.6)
    fig.legend([dots], ["seeds (n = 5)"], loc="lower center", ncol=1,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    fdir = Path(cfg["paths"]["figures"]) / "revised"
    fdir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in ["flowers102", "cub200"]
                if (rdir / f"e2_perturb_{d}.csv").exists()]
    data, rows = {}, []
    for ds in datasets:
        e2 = pd.read_csv(rdir / f"e2_perturb_{ds}.csv")
        e2 = e2[e2.model.isin(MAIN_ARMS)].copy()
        check(e2)
        data[ds] = {a: cr_per_seed(e2, a) for a in MAIN_ARMS}
        for a in MAIN_ARMS:
            v = data[ds][a]
            rows.append(dict(dataset=ds, arm=a, backbone=MLAB[a],
                             pretrain=PRETRAIN_TAG[a], n_seeds=len(v),
                             CR_mean=v.mean(), CR_sd=v.std(ddof=1),
                             CR_min=v.min(), CR_max=v.max()))

    out = fdir / "fig4_colorreliance.png"
    build(data, datasets, out)

    df = pd.DataFrame(rows)
    csv = Path(cfg["paths"]["tables"]) / "fig4_colorreliance_values.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  urutan CR per dataset (tertinggi lebih dulu):")
    for ds in datasets:
        d = df[df.dataset == ds].sort_values("CR_mean", ascending=False)
        order = "  >  ".join(f"{r.backbone} {r.CR_mean:.4f}" for _, r in d.iterrows())
        print(f"    {ds:11} {order}")
        top2 = d.head(2)
        gap = top2.CR_mean.iloc[0] - top2.CR_mean.iloc[1]
        pooled = np.hypot(top2.CR_sd.iloc[0], top2.CR_sd.iloc[1])
        print(f"    {'':11} jarak peringkat 1 ke 2 = {gap:+.4f} "
              f"({gap/pooled:.1f}x SD gabungan)")

    from PIL import Image
    im = Image.open(out)
    print(f"\n  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {im.size[0]/im.info['dpi'][0]:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
