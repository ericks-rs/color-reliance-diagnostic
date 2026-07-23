"""fig3 (drop-curves) dibangun ULANG pada ukuran cetak IEEE Access.

Masalah versi lama (figures/pub/P4_drop_curves.png, dipakai sebagai paper/latex/fig3.png):
  - figsize 11.0 in, dipasang \\includegraphics[width=\\columnwidth] = 3.36 in
    -> seluruh teks menyusut x0.31, legenda 8 pt tercetak 2.5 pt.
  - berkasnya bertanggal 26 Jun, jadi datanya mendahului rebuild 130-run.

Perbaikan di sini:
  1. figsize = ukuran terbit persis (6.99 in = \\textwidth ieeeaccess.cls), dan
     savefig TANPA bbox_inches="tight" supaya PNG keluar tepat 6.99 in. Dengan
     begitu 8 pt di skrip = 8 pt di kertas, tidak ada penskalaan sama sekali.
     Konsekuensi: naskah harus memakai figure* (dua kolom), bukan columnwidth.
  2. semua ukuran huruf >= 8 pt (batas keterbacaan IEEE).
  3. satu legenda bersama di bawah, bukan legenda per panel. Empat panel x
     legenda 4 entri memakan ruang gambar dan memaksa font kecil.
  4. dpi 600 (line-art/kombinasi).
  5. dibaca dari results/e2_perturb_*.csv yang sekarang, jadi ikut rebuild.
  5b. arm diambil dari revision/arms.py, bukan disalin dari analyze_extra.py.
     Versi pertama berkas ini memakai convnext_tiny (in12k) dan vit_small
     (in21k) karena MODEL_ORDER disalin mentah dari generator v1. Itu melanggar
     keputusan R1.2 (semua arm utama disetarakan ke IN1k) dan membalik temuan:
     di arm non-setara ViT tampak peringkat 2-3 dalam color reliance, di arm
     setara ViT TERTINGGI di kedua dataset. Lihat revision/arms.py.
  6. kelima seed digambar sebagai titik di atas pita. Lihat _seed_dots(): pada
     ResNet-50/Flowers di kuantisasi 2 level sebarannya BIMODAL (dua seed ~0.18,
     tiga seed ~0.02), sehingga pita mean +- SD memberi kesan sebaran tunggal di
     0.087 yang tidak dialami satu seed pun. Diperiksa di
     revision/diag_resnet_flowers_quant.py: tiga seed itu kolaps degeneratif,
     membuang 45-56% gambar uji ke satu kelas (entropi prediksi 1.5-3.0 bit dari
     maksimum 6.67), sementara ConvNeXt sebagai kontrol memakai 102/102 kelas.

Generator lama TIDAK diubah. P4_drop_curves.png tetap ada sebagai pembanding v1.

python revision/build_fig3_dropcurves.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

from analysis.arms import MAIN_ARMS as MODEL_ORDER, MLAB, MCOLOR, DSLAB, check

# ieeeaccess.cls: \textwidth 177.53mm = 6.989in, \columnwidth 85.29mm = 3.358in
TEXTWIDTH_IN = 6.989


def _print_style():
    """Ukuran huruf = ukuran TERCETAK. Tidak ada satu pun di bawah 8 pt."""
    plt.rcParams.update({
        "savefig.dpi": 600, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
    })


def _seed_dots(ax, g, color):
    """Sebar kelima seed di tiap titik-x, di atas pita simpangan.

    Alasannya bukan hiasan. Pita mean +- SD mengandaikan satu populasi, dan pada
    ResNet-50/Flowers di kuantisasi 2 level pengandaian itu keliru: dua seed sehat
    di ~0.18 dan tiga seed kolaps di ~0.02, sehingga mean 0.087 menggambarkan
    keadaan yang tidak dialami satu seed pun. Titik per-seed membuat campuran dua
    rezim itu terlihat langsung.

    Aturannya diterapkan SERAGAM ke semua model, dataset, dan kondisi, bukan hanya
    ke titik yang bermasalah. Kalau hanya satu titik yang diberi perlakuan khusus,
    figurnya jadi terbaca seperti hasil pilih-pilih. Di titik yang seed-nya rapat
    kelima penanda saling menumpuk dan gambarnya tetap bersih dengan sendirinya.
    """
    ax.plot(g["param"].values, g["acc"].values, linestyle="none", marker="o",
            markersize=1.9, markerfacecolor=color, markeredgecolor="none",
            alpha=0.55, zorder=5)


def build(e2, datasets, out_png):
    _print_style()
    nrow = len(datasets)
    # tinggi per baris 2.55 in -> panel 3.4 x 2.55, rasio wajar untuk kurva garis.
    # +0.55 in di bawah untuk legenda bersama.
    fig, axes = plt.subplots(nrow, 2, figsize=(TEXTWIDTH_IN, 2.55 * nrow + 0.55),
                             squeeze=False)

    # Rentang Y disamakan ANTARA PANEL KIRI DAN KANAN pada baris yang sama, yaitu
    # per dataset. Dengan begitu besarnya kerusakan akibat hue rotation dan akibat
    # channel shuffle bisa dibandingkan langsung; sebelumnya tiap panel menskala
    # sendiri sehingga jatuhnya terlihat sebanding padahal tidak.
    #
    # Baris TIDAK disamakan satu sama lain. CUB-200 memang duduk lebih rendah dari
    # Flowers-102 di semua kondisi, dan memaksa satu skala global akan memipihkan
    # baris CUB tanpa menambah informasi. Yang dibandingkan lintas dataset adalah
    # bentuknya, bukan tinggi mutlaknya.
    ylim = {}
    for ds in datasets:
        sub = e2[e2.dataset == ds]
        lo, hi = 1.0, 0.0
        # hue: kurva dengan param numerik (mean +- SD antar seed)
        gh = sub[sub.kind == "hue"].copy()
        gh["p"] = gh["param"].astype(float)
        agg = gh.groupby(["model", "p"]).acc.agg(["mean", "std"])
        lo = min(lo, (agg["mean"] - agg["std"].fillna(0)).min())
        hi = max(hi, (agg["mean"] + agg["std"].fillna(0)).max())
        # channel shuffle: rerata per permutasi (di atas seed), lima nilai per model
        pm = sub[sub.kind == "shuffle"].groupby(["model", "param"]).acc.mean()
        lo, hi = min(lo, pm.min()), max(hi, pm.max())
        # garis acuan clean ikut diperhitungkan supaya tidak terpotong
        hi = max(hi, sub[sub.condition == "clean"].acc.max())
        pad = 0.04 * (hi - lo)
        ylim[ds] = (max(0.0, lo - pad), min(1.0, hi + pad))

    handles = None
    for r, ds in enumerate(datasets):
        sub = e2[e2.dataset == ds]
        clean_mean = {m: sub[(sub.model == m) & (sub.condition == "clean")].acc.mean()
                      for m in MODEL_ORDER}

        axh = axes[r][0]
        for m in MODEL_ORDER:
            g = sub[(sub.model == m) & (sub.kind == "hue")].copy()
            if not len(g):
                continue
            g["param"] = g["param"].astype(float)
            agg = g.groupby("param").acc.agg(["mean", "std"]).sort_index()
            axh.plot(agg.index, agg["mean"], marker="o", color=MCOLOR[m], label=MLAB[m])
            axh.fill_between(agg.index, agg["mean"] - agg["std"].fillna(0),
                             agg["mean"] + agg["std"].fillna(0),
                             color=MCOLOR[m], alpha=0.12)
            _seed_dots(axh, g, MCOLOR[m])
        hue_vals = sorted(sub[sub.kind == "hue"].param.astype(float).unique())
        axh.set_xticks(hue_vals)
        # nilai asli faktor rotasi: 0.083 ditulis apa adanya (bukan dibulatkan 0.08),
        # supaya cocok dengan teks III-D dan caption tak perlu catatan pembulatan.
        axh.set_xticklabels([f"{v:g}" for v in hue_vals])
        axh.set_xlabel("hue rotation factor")
        axh.set_ylabel("accuracy")
        axh.set_title(f"{DSLAB[ds]}, hue rotation")
        axh.set_ylim(*ylim[ds])
        if handles is None:
            handles, _lbl = axh.get_legend_handles_labels()

        # Panel kanan = channel shuffle. Shuffle bukan sumbu-kekuatan numerik
        # (lima permutasi kategoris), jadi bukan kurva melainkan batang: tinggi =
        # rerata akurasi di atas kelima permutasi dan kelima seed, whisker = SD antar
        # RERATA-permutasi (seberapa jauh permutasi berbeda saling menyimpang), dan
        # tik titik-titik = akurasi bersih tiap model sebagai acuan penurunan.
        axs = axes[r][1]
        xpos = list(range(len(MODEL_ORDER)))
        for i, m in enumerate(MODEL_ORDER):
            g = sub[(sub.model == m) & (sub.kind == "shuffle")]
            if not len(g):
                continue
            perm_means = g.groupby("param").acc.mean()      # 5 rerata-permutasi
            bar, spread = perm_means.mean(), perm_means.std()
            axs.bar(i, bar, width=0.62, color=MCOLOR[m], alpha=0.55,
                    edgecolor="none", zorder=2)
            axs.errorbar(i, bar, yerr=spread, fmt="none", ecolor="0.3",
                         elinewidth=0.9, capsize=2.5, zorder=4)
            axs.hlines(clean_mean[m], i - 0.31, i + 0.31, color=MCOLOR[m],
                       ls=":", lw=0.9, alpha=0.7, zorder=3)
        axs.set_xticks(xpos)
        axs.set_xticklabels([MLAB[m] for m in MODEL_ORDER])
        axs.set_ylabel("accuracy")
        axs.set_title(f"{DSLAB[ds]}, channel shuffle")
        axs.set_ylim(*ylim[ds])

    # Garis titik-titik di panel kuantisasi = akurasi bersih tiap model. Judul lama
    # menyebutnya "(dotted=clean)"; di sini keterangan itu dipindah ke legenda
    # bersama supaya judul pendek TAPI garisnya tidak jadi tanpa penjelasan.
    ref = Line2D([0], [0], color="0.35", ls=":", lw=0.9)
    dots = Line2D([0], [0], color="0.35", ls="none", marker="o", markersize=1.9,
                  markeredgecolor="none", alpha=0.75)
    fig.legend(handles + [ref, dots],
               [MLAB[m] for m in MODEL_ORDER]
               # label dipendekkan: versi panjang ("clean accuracy (reference)",
               # "individual seeds (n = 5)") menyisakan margin kiri hanya 19 px
               # dari 4193 (0.03 in), terlalu rapat ke tepi. Keterangan penuhnya
               # ada di caption.
               + ["clean accuracy", "seeds (n = 5)"],
               loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.1, handlelength=1.6)
    # rect menyisakan pita bawah untuk legenda; tight_layout MEMPERTAHANKAN figsize.
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    # TANPA bbox_inches="tight": itu akan memangkas kanvas dan mengubah lebar
    # keluaran, sehingga jaminan "8 pt = 8 pt" hilang.
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
    if not datasets:
        raise SystemExit("e2_perturb_*.csv tidak ada")
    e2 = pd.concat([pd.read_csv(rdir / f"e2_perturb_{d}.csv") for d in datasets],
                   ignore_index=True)
    e2 = e2[e2.model.isin(MODEL_ORDER)].copy()
    check(e2)          # jaring pengaman: gagalkan kalau arm non-setara lolos

    out = fdir / "fig3_dropcurves.png"
    build(e2, datasets, out)

    from PIL import Image
    im = Image.open(out)
    w_in = im.size[0] / im.info["dpi"][0]
    print(f"  -> {out}")
    print(f"     {im.size[0]}x{im.size[1]} px @ {round(im.info['dpi'][0])} dpi "
          f"= {w_in:.3f} in (target {TEXTWIDTH_IN:.3f})")
    print(f"     seed per titik: {e2.groupby('dataset').seed.nunique().to_dict()}")
    print("     pasang di naskah sebagai figure* (dua kolom), width=\\textwidth")


if __name__ == "__main__":
    main()
