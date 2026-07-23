"""fig2 (gap atensi minus konvolusi per bin) dibangun ULANG, arm setara + ukuran cetak.

Masalah versi lama (figures/pub/P3_gap_vs_bin.png = paper/latex/fig2.png):
  - figsize 12.4 in dipasang \\textwidth = 6.989 in -> skala x0.56, legenda 9 pt
    tercetak 5.1 pt.
  - memakai `convnext_tiny` (in12k) dan `vit_small` (in21k), melanggar R1.2.
  - berkasnya 26 Jun, mendahului rebuild 130-run.
  - gap dihitung dari selisih dua rata-rata, sehingga pita galatnya tidak
    mencerminkan pasangan seed yang benar-benar diamati.

Yang DIPERTAHANKAN dari versi lama (keputusan sengaja):
  Rentang Y disamakan kedua panel. Kalau tiap panel auto-scale, riak kecil di
  dataset yang rentangnya sempit akan tampak seperti modulasi. Skala seragam
  itulah yang membuat garis terbaca datar dan menyangga klaim null-modulation.

Yang berubah isinya, bukan hanya tampilannya. Pada arm setara, Swin-T UNGGUL
atas ConvNeXt-T di enam dari enam sel (CUB +0.0094/+0.0060/+0.0125, Flowers
+0.0104/+0.0094/+0.0007), sedangkan versi lama menempatkan semua garis lawan
ConvNeXt di bawah nol. Caption lama ("dashed < 0") tidak lagi menggambarkan
figur ini dan harus ditulis ulang.

Gap dihitung PER SEED lalu diringkas, memakai irisan indeks seed supaya tidak
mengandaikan kedua arm punya seed yang sama. Kelima seed juga digambar sebagai
titik, mengikuti standar fig1/fig3.

python revision/build_fig2_gapbybin.py
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
from analysis.arms import MAIN_ARMS, MLAB, MCOLOR, DSLAB, check

BINS = ["low", "mid", "high"]
CONVS = ["resnet50", "convnext_tiny_in1k"]
ATTNS = ["vit_small_in1k", "swin_tiny"]
TEXTWIDTH_IN = 6.989
# garis penuh = lawan ResNet-50, putus-putus = lawan ConvNeXt-T.
# warna mengikuti model ATENSI-nya, sesuai palet terkunci di revision/arms.py.
LS = {"resnet50": "-", "convnext_tiny_in1k": "--"}
MK = {"resnet50": "o", "convnext_tiny_in1k": "s"}


def _print_style():
    plt.rcParams.update({
        "savefig.dpi": 600, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
    })


def gaps(e1, arms_pair):
    """Gap per seed untuk tiap bin. Return {bin: array}."""
    at, cv = arms_pair
    out = {}
    for b in BINS:
        sa = e1[(e1.model == at) & (e1["bin"] == b)].set_index("seed").acc.sort_index()
        sb = e1[(e1.model == cv) & (e1["bin"] == b)].set_index("seed").acc.sort_index()
        common = sa.index.intersection(sb.index)
        out[b] = (sa.loc[common] - sb.loc[common]).values
    return out


def build(data, datasets, out_png):
    _print_style()
    # rentang Y global supaya kedua panel identik
    lo, hi = 1e9, -1e9
    for ds in datasets:
        for g in data[ds].values():
            for b in BINS:
                v = g[b]
                lo = min(lo, v.min()); hi = max(hi, v.max())
    pad = 0.012

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(TEXTWIDTH_IN, 3.05), squeeze=False)
    x = np.arange(len(BINS))
    pairs = [(at, cv) for at in ATTNS for cv in CONVS]
    # geser tiap seri sedikit di sumbu-x supaya galat dan titik seed tidak
    # saling menimpa di posisi bin yang sama
    dodge = np.linspace(-0.13, 0.13, len(pairs))

    for ax, ds in zip(axes[0], datasets):
        for (at, cv), dx in zip(pairs, dodge):
            g = data[ds][(at, cv)]
            m = np.array([g[b].mean() for b in BINS])
            s = np.array([g[b].std(ddof=1) for b in BINS])
            ax.errorbar(x + dx, m, yerr=s, ls=LS[cv], marker=MK[cv],
                        color=MCOLOR[at], capsize=2.5, elinewidth=0.9)
            for xi, b in zip(x, BINS):
                v = g[b]
                ax.plot(np.full(len(v), xi + dx), v, linestyle="none", marker="o",
                        markersize=1.8, markerfacecolor=MCOLOR[at],
                        markeredgecolor="none", alpha=0.5, zorder=5)
        ax.axhline(0, color="0.35", lw=0.9, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(BINS)
        ax.set_xlim(-0.42, len(BINS) - 0.58)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("colorfulness bin")
        # Sumbu tidak menyebut "attention". Kata itu memihak pembacaan bahwa
        # figur ini membandingkan atensi lawan konvolusi sebagai dua kelas,
        # padahal Swin-T dan ViT-S berperilaku berbeda terhadap ConvNeXt-T.
        # Pesan utama figur tetap null-modulation, yaitu garis yang datar
        # lintas bin. Pembingkaian selebihnya ditangani teks naskah.
        ax.set_ylabel("accuracy gap to convolutional baseline")
        ax.set_title(DSLAB[ds])

    handles = [Line2D([0], [0], color=MCOLOR[at], ls=LS[cv], marker=MK[cv],
                      markersize=3.2, lw=1.1) for at in ATTNS for cv in CONVS]
    labels = [f"{MLAB[at]} − {MLAB[cv]}" for at in ATTNS for cv in CONVS]
    dots = Line2D([0], [0], color="0.35", ls="none", marker="o", markersize=1.8,
                  markeredgecolor="none", alpha=0.75)
    fig.legend(handles + [dots], labels + ["seeds (n = 5)"],
               # lima entri berlabel panjang tidak muat satu baris pada 6.989 in
               # (kurung tutup "seeds (n = 5)" terpotong), jadi dipecah dua baris
               loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.012), columnspacing=1.1, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.135, 1, 1))
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    fdir = Path(cfg["paths"]["figures"]) / "revised"
    fdir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in ["flowers102", "cub200"]
                if (rdir / f"e1_clean_{d}.csv").exists()]
    data, rows = {}, []
    for ds in datasets:
        e1 = pd.read_csv(rdir / f"e1_clean_{ds}.csv")
        e1 = e1[e1.model.isin(MAIN_ARMS)].copy()
        check(e1)
        data[ds] = {}
        for at in ATTNS:
            for cv in CONVS:
                g = gaps(e1, (at, cv))
                data[ds][(at, cv)] = g
                for b in BINS:
                    rows.append(dict(dataset=ds, backbone=MLAB[at],
                                     baseline=MLAB[cv], bin=b,
                                     n_seeds=len(g[b]), gap_mean=g[b].mean(),
                                     gap_sd=g[b].std(ddof=1)))

    out = fdir / "fig2_gapbybin.png"
    build(data, datasets, out)

    df = pd.DataFrame(rows)
    csv = Path(cfg["paths"]["tables"]) / "fig2_gapbybin_values.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # berapa sel yang menyeberang nol, karena itu yang mengubah caption
    neg = df[df.gap_mean < 0]
    print(f"\n  sel bertanda negatif: {len(neg)}/{len(df)}")
    for k, g in df.groupby(["backbone", "baseline"]):
        sign = "semua pos" if (g.gap_mean > 0).all() else (
            "semua neg" if (g.gap_mean < 0).all() else "campuran")
        print(f"    {k[0]:11} vs {k[1]:11} : {sign}")

    from PIL import Image
    im = Image.open(out)
    print(f"\n  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {im.size[0]/im.info['dpi'][0]:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
