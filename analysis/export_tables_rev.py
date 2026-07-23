"""Ekspor satu CSV per tabel naskah, ditandai _rev.

Berkas kerja di tables/ dinamai menurut tahap yang membuatnya (A4, A5, V5, T2,
dan seterusnya), bukan menurut nomor tabel di naskah, dan sebagian terpecah per
dataset. Nama itu berguna saat membangun, menyusahkan saat menulis: "T2" adalah
Table 3, dan "T4" adalah Table 6.

Berkas ini menghasilkan himpunan kanonik untuk fase tulis. Satu CSV per tabel
naskah, dinamai menurut nomor tabelnya, kedua dataset digabung dalam satu berkas
dengan kolom `dataset`, dan akhiran _rev supaya tidak mungkin tertukar dengan
keluaran v1 di tables/_superseded_v1_nonequalized/.

TIDAK ADA ANGKA YANG DIHITUNG ULANG di sini. Semuanya dibaca dari CSV yang sudah
dibangun, digabung, dan ditulis ulang. Menghitung ulang berarti membuka peluang
angka di sini menyimpang dari angka di .md yang dipakai untuk memeriksa.

python revision/export_tables_rev.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = ["flowers102", "cub200"]

# (nama keluaran, keterangan, daftar sumber). Sumber berupa satu berkas, atau
# pola per-dataset yang akan digabung dengan kolom `dataset`.
SPEC = [
    ("table1_colorfulness_rev.csv",
     "colourfulness statistics, whole image and foreground",
     ["T1_colorfulness.csv"]),
    ("table2_model_specs_rev.csv",
     "backbone specifications with checkpoint tags",
     ["T2_model_specs.csv"]),
    ("table3_clean_accuracy_rev.csv",
     "clean top-1 accuracy, macro-F1, UAR",
     ["T2_clean_acc_{ds}.csv"]),
    ("table4_perbin_whole_vs_foreground_rev.csv",
     "per-bin clean, grayscale, CR under both bin definitions",
     ["V4_perbin_wholeimg_{ds}.csv", "V5_perbin_foreground_{ds}.csv"]),
    ("table5_perbin_foreground_rev.csv",
     "per-bin foreground only, kept separate in case Table 5 stays standalone",
     ["V5_perbin_foreground_{ds}.csv"]),
    ("table6_colour_reliance_rev.csv",
     "clean, grayscale, and colour reliance",
     ["T4_color_reliance_{ds}.csv"]),
    ("table7_paired_diff_ci_rev.csv",
     "pairwise contrasts, difference with 95% CI",
     ["A4_table7_paired_diffCI.csv"]),
    ("table8_resnet_configurations_rev.csv",
     "ResNet-50 under four configurations with ConvNeXt-T reference",
     ["A5_table8_resnet4cfg.csv"]),
    ("table9_confound_pretraining_rev.csv",
     "R1.2 confound, equalized against non-equalized checkpoints",
     ["A6_table_confound.csv"]),
    ("table10_grayscale_training_matrix_rev.csv",
     "R2.3 three-cell matrix, train mode against test mode",
     ["A7_table_R23_matrix.csv"]),
    ("table11_gap_by_bin_rev.csv",
     "accuracy gap to convolutional baselines per bin, supports Fig. 2",
     ["T3_gap_by_bin_{ds}.csv"]),
    ("table12_architecture_contrast_rev.csv",
     "R1.3 architecture contrast, point differences without intervals",
     ["A8_arch_contrast_R13.csv"]),
]


def load(tdir, pattern):
    """Satu berkas, atau pola per-dataset yang digabung dengan kolom dataset."""
    if "{ds}" not in pattern:
        p = tdir / pattern
        return pd.read_csv(p) if p.exists() else None
    parts = []
    for ds in DS:
        p = tdir / pattern.format(ds=ds)
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "dataset" not in d.columns:
            d.insert(0, "dataset", ds)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    chdir_to_root()
    cfg = load_config()
    tdir = Path(cfg["paths"]["tables"])
    odir = tdir / "rev"
    odir.mkdir(parents=True, exist_ok=True)

    print(f"{'keluaran':46}{'baris':>7}{'kolom':>7}  sumber")
    print("-" * 110)
    missing = []
    for out, desc, sources in SPEC:
        frames = []
        used = []
        for s in sources:
            d = load(tdir, s)
            if d is None:
                continue
            if len(sources) > 1:
                # gabungan V4 + V5: tandai lingkup binnya
                d = d.copy()
                d.insert(1, "bin_scope",
                         "whole image" if "wholeimg" in s else "foreground")
            frames.append(d)
            used.append(s)
        if not frames:
            missing.append((out, sources))
            print(f"{out:46}{'-':>7}{'-':>7}  *** SUMBER TIDAK ADA: {sources}")
            continue
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df.to_csv(odir / out, index=False)
        print(f"{out:46}{len(df):>7}{df.shape[1]:>7}  {', '.join(used)}")

    # indeks kecil supaya fase tulis tahu berkas mana untuk tabel mana
    idx = pd.DataFrame([{"file": o, "contents": d, "sources": "; ".join(s)}
                        for o, d, s in SPEC])
    idx.to_csv(odir / "INDEX_rev.csv", index=False)
    print(f"\n  indeks -> {odir/'INDEX_rev.csv'}")

    if missing:
        print("\n  BELUM ADA SUMBERNYA:")
        for o, s in missing:
            print(f"    {o}  <- {s}")


if __name__ == "__main__":
    main()
