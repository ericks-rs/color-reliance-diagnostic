"""fig5 (ablasi resep ResNet-50) dibangun ULANG, arm setara + ukuran cetak.

Masalah versi lama (figures/ablation_recipe.png = paper/latex/fig5.png):
  - figsize 11 in dipasang \\includegraphics[width=\\columnwidth] = 3.358 in
    -> skala x0.31, anotasi 9 pt tercetak 2.9 pt. Ini figur terparah dari lima.
  - acuannya `convnext_tiny` (in12k), melanggar R1.2. Diganti `convnext_tiny_in1k`.
  - berkasnya 26 Jun, mendahului rebuild 130-run.
  - hanya menampilkan DUA konfigurasi ResNet, S1 (shared AdamW) dan S13 (SGD),
    yaitu dua yang terlemah. Table 8 memuat empat. Menyembunyikan S11 dan S12
    dari figur sementara keduanya ada di tabel adalah penyajian selektif, dan
    justru keduanya yang menjawab R1.5 ("SGD single untuned config").

Empat konfigurasi kini ditampilkan, sama persis dengan Table 8 (A5). Akibatnya
ResNet-50 terlihat nyaris menyamai ConvNeXt-T pada Flowers-102 (0.920 lawan
0.930). Itu memang keadaan datanya. Batas bawah efek arsitektur dilaporkan di A8
sebagai bracket konservatif (+0.0220 Flowers, +0.0187 CUB), bukan dibaca dari
tinggi batang di figur ini.

PERINGATAN BACA yang ikut digambar: S11 mengubah DUA faktor sekaligus terhadap
S1, yaitu checkpoint (tv_in1k) dan optimizer (SGD). Batangnya diberi tanda supaya
pembaca tidak menyusun tangga S13 -> S1 -> S12 -> S11 seolah satu sumbu tunggal.
A8 menolak pembacaan tangga itu secara eksplisit.

Warna mengikuti palet terkunci di revision/arms.py: keempat batang ResNet memakai
biru ResNet-50, acuan memakai hijau ConvNeXt-T. Warna tetap berarti ARSITEKTUR,
sama seperti di fig1 sampai fig4. Konfigurasi dibedakan lewat label, bukan warna.

python revision/build_fig5_ablation.py
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
from analysis.arms import MLAB, MCOLOR, DSLAB, check

TEXTWIDTH_IN = 6.989
# (id, arm, label dua baris, apakah mengubah dua faktor terhadap S1)
CFG = [
    ("S1", "resnet50", "AdamW\n0.0001", False),
    ("S12", "resnet50_adamw_lr1e3", "AdamW\n0.001", False),
    ("S13", "resnet50_sgd", "SGD\na1_in1k", False),
    ("S11", "resnet50_tv_sgd", "SGD\ntv_in1k", True),
]
REF = ("S2", "convnext_tiny_in1k", "ConvNeXt-T\nfb_in1k")


def _print_style():
    plt.rcParams.update({
        "savefig.dpi": 600, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
    })


def series(e2, arm):
    c = e2[(e2.model == arm) & (e2.condition == "clean")].set_index("seed").acc.sort_index()
    g = e2[(e2.model == arm) & (e2.condition == "grayscale")].set_index("seed").acc.sort_index()
    i = c.index.intersection(g.index)
    return c.loc[i].values, (c.loc[i] - g.loc[i]).values


def build(data, datasets, out_png):
    _print_style()
    order = [c[0] for c in CFG] + [REF[0]]
    labs = [c[2] for c in CFG] + [REF[2]]
    cols = [MCOLOR["resnet50"]] * len(CFG) + [MCOLOR["convnext_tiny_in1k"]]
    two_factor = {c[0]: c[3] for c in CFG}

    # rentang Y disamakan antar-dataset per metrik, supaya besaran bisa
    # dibandingkan lintas baris dan bukan hanya bentuknya
    lims = {}
    for k, idx in [("acc", 0), ("cr", 1)]:
        top = max(v[idx].mean() + v[idx].std(ddof=1)
                  for ds in datasets for v in data[ds].values())
        bot = min(v[idx].mean() - v[idx].std(ddof=1)
                  for ds in datasets for v in data[ds].values())
        lims[k] = (max(0.0, bot - 0.10) if k == "acc" else 0.0, top + 0.085)

    nrow = len(datasets)
    fig, axes = plt.subplots(nrow, 2, figsize=(TEXTWIDTH_IN, 2.55 * nrow + 0.5),
                             squeeze=False)
    x = np.arange(len(order))
    for r, ds in enumerate(datasets):
        for col, (key, idx, ylab) in enumerate(
                [("acc", 0, "top-1 accuracy"),
                 ("cr", 1, r"color reliance (acc$_\mathrm{clean}$ $-$ acc$_\mathrm{gray}$)")]):
            ax = axes[r][col]
            m = np.array([data[ds][k][idx].mean() for k in order])
            s = np.array([data[ds][k][idx].std(ddof=1) for k in order])
            bars = ax.bar(x, m, yerr=s, capsize=3, width=0.62, color=cols,
                          edgecolor="black", lw=0.5, zorder=2)
            # tanda dua-faktor: arsiran pada batang yang mengubah checkpoint DAN
            # optimizer sekaligus, supaya tidak dibaca sebagai satu langkah tangga
            for b, k in zip(bars, order):
                if two_factor.get(k):
                    b.set_hatch("///")
            for xi, k, mn, sd in zip(x, order, m, s):
                v = data[ds][k][idx]
                ax.plot(np.full(len(v), xi), v, linestyle="none", marker="o",
                        markersize=1.8, markerfacecolor="black",
                        markeredgecolor="none", alpha=0.45, zorder=4)
                ax.text(xi, mn + sd + 0.011, f"{mn:.3f}", ha="center",
                        va="bottom", fontsize=7.5, zorder=5)
            ax.set_xticks(x); ax.set_xticklabels(labs)
            ax.set_ylim(*lims[key])
            ax.set_ylabel(ylab)
            ax.set_title(f"{DSLAB[ds]}, {'clean accuracy' if col == 0 else 'color reliance'}")

    h = [Line2D([0], [0], color=MCOLOR["resnet50"], lw=5),
         Line2D([0], [0], color=MCOLOR["convnext_tiny_in1k"], lw=5),
         plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                       hatch="///", lw=0.5),
         Line2D([0], [0], color="black", ls="none", marker="o", markersize=1.8,
                markeredgecolor="none", alpha=0.6)]
    lb = ["ResNet-50 configurations", "ConvNeXt-T reference",
          "changes checkpoint and optimizer", "seeds (n = 5)"]
    fig.legend(h, lb, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.3, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.062, 1, 1))
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
    allarms = [(c[0], c[1]) for c in CFG] + [(REF[0], REF[1])]
    for ds in datasets:
        e2 = pd.read_csv(rdir / f"e2_perturb_{ds}.csv")
        e2 = e2[e2.model.isin([a for _, a in allarms])].copy()
        check(e2)
        data[ds] = {}
        for sid, arm in allarms:
            acc, cr = series(e2, arm)
            data[ds][sid] = (acc, cr)
            rows.append(dict(dataset=ds, id=sid, arm=arm, n_seeds=len(acc),
                             clean_mean=acc.mean(), clean_sd=acc.std(ddof=1),
                             CR_mean=cr.mean(), CR_sd=cr.std(ddof=1)))

    out = fdir / "fig5_ablation.png"
    build(data, datasets, out)

    df = pd.DataFrame(rows)
    csv = Path(cfg["paths"]["tables"]) / "fig5_ablation_values.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Seberapa jauh RESEP menggeser CR di dalam SATU arsitektur. Kalau rentang ini
    # sebanding dengan rentang antar-arsitektur, CR bukan sifat arsitektur murni
    # dan naskah harus mengatakannya.
    print("\n  rentang CR akibat RESEP, dalam ResNet-50 saja:")
    for ds in datasets:
        d = df[(df.dataset == ds) & (df.id != "S2")]
        span = d.CR_mean.max() - d.CR_mean.min()
        print(f"    {ds:11} {d.CR_mean.min():.4f} .. {d.CR_mean.max():.4f}  "
              f"rentang {span:.4f}  ({d.loc[d.CR_mean.idxmin(),'id']} .. "
              f"{d.loc[d.CR_mean.idxmax(),'id']})")

    from PIL import Image
    im = Image.open(out)
    print(f"\n  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {im.size[0]/im.info['dpi'][0]:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
