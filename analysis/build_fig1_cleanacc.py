"""fig1 (clean accuracy) dibangun ULANG pada ukuran cetak IEEE Access.

Masalah versi lama (figures/pub/P1_clean_acc.png = paper/latex/fig1.png):
  - figsize 12.4 in dipasang \\textwidth = 6.989 in -> skala x0.56, anotasi 9 pt
    tercetak 5.1 pt, di bawah batas keterbacaan IEEE (8 pt).
  - berkasnya bertanggal 26 Jun, mendahului rebuild 130-run.
  - hanya menampilkan mean + error bar; sebaran seed tidak terlihat.

Yang DIPERTAHANKAN dari versi lama (keputusan sengaja, jangan dibalik):
  Skala Y disamakan kedua panel. Kalau tiap panel auto-scale, selisih kecil
  antar model akan terlihat besar di panel yang rentangnya sempit.

Yang diperbaiki:
  1. figsize = ukuran terbit persis (6.989 in), savefig TANPA bbox_inches="tight".
  2. semua huruf >= 8 pt pada ukuran cetak. 600 dpi.
  3. kelima seed digambar sebagai titik di atas tiap batang, mengikuti standar
     yang ditetapkan fig3: di mana pun ada ringkasan sebaran, seed aslinya ikut
     tampil. Aturan seragam, bukan hanya di titik yang menarik.
  4. warna dan urutan model dikunci sama dengan fig3.

Generator lama TIDAK diubah. P1_clean_acc.png tetap ada sebagai pembanding v1.

python revision/build_fig1_cleanacc.py
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
from src import stats as st

from analysis.arms import MAIN_ARMS as MODEL_ORDER, MLAB, MCOLOR, DSLAB, check
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


def build(e1, datasets, out_png):
    _print_style()
    # pass 1: kumpulkan semua supaya rentang Y bisa disamakan kedua panel
    data, gymin = {}, 1.0
    for ds in datasets:
        ps = st.clean_acc_per_seed(e1, ds)
        per_seed = {m: ps[m].values for m in MODEL_ORDER if m in ps}
        means = [float(np.mean(per_seed[m])) for m in MODEL_ORDER]
        stds = [float(np.std(per_seed[m], ddof=1)) for m in MODEL_ORDER]
        data[ds] = (means, stds, per_seed)
        gymin = min(gymin, min(means))
    ylo = gymin - 0.08

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(TEXTWIDTH_IN, 2.85), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        means, stds, per_seed = data[ds]
        x = np.arange(len(MODEL_ORDER))
        ax.bar(x, means, yerr=stds, capsize=3, width=0.62,
               color=[MCOLOR[m] for m in MODEL_ORDER],
               edgecolor="black", lw=0.5, zorder=2)
        for xi, m, mn in zip(x, MODEL_ORDER, means):
            v = per_seed[m]
            ax.plot(np.full(len(v), xi), v, linestyle="none", marker="o",
                    markersize=1.9, markerfacecolor="black",
                    markeredgecolor="none", alpha=0.45, zorder=4)
            ax.text(xi, mn + stds[MODEL_ORDER.index(m)] + 0.012, f"{mn:.3f}",
                    ha="center", va="bottom", fontsize=8, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER], rotation=12)
        ax.set_ylim(ylo, 1.03)
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(f"{DSLAB[ds]}, clean")

    dots = Line2D([0], [0], color="black", ls="none", marker="o", markersize=1.9,
                  markeredgecolor="none", alpha=0.6)
    fig.legend([dots], ["seeds (n = 5)"], loc="lower center", ncol=1,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
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
                if (rdir / f"e1_clean_{d}.csv").exists()]
    e1 = pd.concat([pd.read_csv(rdir / f"e1_clean_{d}.csv") for d in datasets],
                   ignore_index=True)
    e1 = e1[e1.model.isin(MODEL_ORDER)].copy()
    check(e1)          # jaring pengaman R1.2: arm non-setara tidak boleh lolos

    out = fdir / "fig1_cleanacc.png"
    build(e1, datasets, out)

    # angka mentah ke stdout + csv, supaya bisa dicocokkan dengan tabel naskah
    rows = []
    for ds in datasets:
        ps = st.clean_acc_per_seed(e1, ds)
        for m in MODEL_ORDER:
            v = ps[m].values
            rows.append(dict(dataset=ds, model=MLAB[m], n_seed=len(v),
                             mean=v.mean(), sd=v.std(ddof=1),
                             vmin=v.min(), vmax=v.max()))
    df = pd.DataFrame(rows)
    csv = Path(cfg["paths"]["tables"]) / "fig1_cleanacc_values.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    from PIL import Image
    im = Image.open(out)
    print(f"\n  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {im.size[0]/im.info['dpi'][0]:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
