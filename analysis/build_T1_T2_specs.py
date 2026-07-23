"""Table 1 (statistik colorfulness) dan Table 2 (spesifikasi model).

Keduanya belum pernah ada di jalur revisi. Table 1 tidak pernah dibangun sama
sekali, dan Table 2 hanya ada sebagai T1_params_flops dari analyze.py, yang tidak
menyebut checkpoint sama sekali sehingga tidak bisa dipakai untuk menjawab R1.2.

Table 1. Statistik colorfulness untuk kedua definisi, seluruh gambar dan
foreground saja, berikut ambang tertil dan jumlah anggota tiap bin. Bin whole
image sudah tersimpan di results/colorfulness_<ds>.csv. Bin foreground dihitung
dengan fungsi stratify_tertiles yang sama seperti dipakai V5, jadi angkanya
konsisten dengan tabel per-bin.

Table 2. Parameter, FLOPs, dan TAG CHECKPOINT tiap arm. Kolom terakhir itu
alasan tabel ini dibangun ulang. Jumlah parameter dan FLOPs bergantung pada
arsitektur, bukan bobot, jadi angkanya sama dengan versi v1 dan disalin dari
sana setelah dicocokkan. Yang baru adalah kolom tag, yang menyatakan hitam di
atas putih bahwa keempat arm utama berangkat dari ImageNet-1k. Manuskrip v1
menulis "fully IN1k" padahal ViT-S memakai augreg_in21k_ft_in1k, dan R1.2
meminta koreksi itu.

python revision/build_T1_T2_specs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src.colorfulness import stratify_tertiles
from analysis.arms import MAIN_ARMS, MLAB, PRETRAIN_TAG, DSLAB

BINS = ["low", "mid", "high"]
# params dan GFLOPs bergantung arsitektur, bukan bobot. Nilai dari T1 v1,
# dicocokkan ulang di sini supaya sumbernya terlacak.
SPEC = {
    "resnet50": (23.72, 8.26, "classic convolutional"),
    "convnext_tiny_in1k": (27.90, 8.91, "modern convolutional"),
    "vit_small_in1k": (21.70, 8.48, "global attention"),
    "swin_tiny": (27.60, 8.74, "hierarchical attention"),
}


def md(rows, header, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"])
    lo_q = cfg["colorfulness"]["tertile_low"]
    hi_q = cfg["colorfulness"]["tertile_high"]

    # ---------------- Table 1 ----------------
    rows, csv = [], []
    for ds in ["flowers102", "cub200"]:
        w = pd.read_csv(rdir / f"colorfulness_{ds}.csv")
        f = pd.read_csv(rdir / f"colorfulness_fg_{ds}.csv").rename(
            columns={"C_fg": "C"})
        # stratify_tertiles menerima DataFrame berkolom "C", bukan Series,
        # dan mengembalikan salinan yang sudah bertambah kolom "bin".
        f = stratify_tertiles(f, lo_q, hi_q)
        for scope, d in [("whole image", w), ("foreground", f)]:
            cnt = d["bin"].value_counts()
            thr_lo = d["C"].quantile(lo_q)
            thr_hi = d["C"].quantile(hi_q)
            rows.append([DSLAB[ds], scope, len(d),
                         f"{d['C'].mean():.2f}", f"{d['C'].median():.2f}",
                         f"{d['C'].std(ddof=1):.2f}",
                         f"{d['C'].min():.2f}–{d['C'].max():.2f}",
                         f"{thr_lo:.2f}", f"{thr_hi:.2f}",
                         " / ".join(str(int(cnt.get(b, 0))) for b in BINS)])
            csv.append(dict(dataset=ds, scope=scope, n_images=len(d),
                            C_mean=d["C"].mean(), C_median=d["C"].median(),
                            C_sd=d["C"].std(ddof=1),
                            C_min=d["C"].min(), C_max=d["C"].max(),
                            tertile_low_thr=thr_lo, tertile_high_thr=thr_hi,
                            **{f"n_{b}": int(cnt.get(b, 0)) for b in BINS}))
    pd.DataFrame(csv).to_csv(tdir / "T1_colorfulness.csv", index=False)
    body = md(rows, ["dataset", "region", "images", "mean", "median", "SD",
                     "range", "lower tertile", "upper tertile",
                     "low / mid / high"],
              [":---", ":---", "---:", "---:", "---:", "---:", ":---",
               "---:", "---:", ":---"])
    note = (f"Colourfulness follows Hasler and Susstrunk. Images are stratified at "
            f"evaluation time only, never during training. Thresholds are the "
            f"{lo_q:.0%} and {hi_q:.0%} quantiles computed within each dataset and "
            f"each region definition, so the three bins are equal in size by "
            f"construction. The foreground rows use the dataset's own segmentation "
            f"mask, which is what R2.2 asked for: a flower or a bird occupying a "
            f"small part of the frame should not inherit the colourfulness of its "
            f"background.")
    (tdir / "T1_colorfulness.md").write_text(
        f"# T1 — Colourfulness statistics\n\n{note}\n\n{body}\n", encoding="utf-8")
    print(f"  -> {tdir/'T1_colorfulness.md'}")

    # ---------------- Table 2 ----------------
    rows, csv = [], []
    for a in MAIN_ARMS:
        p, g, role = SPEC[a]
        rows.append([MLAB[a], role, f"{p:.2f}", f"{g:.2f}",
                     f"`{PRETRAIN_TAG[a]}`", "ImageNet-1k"])
        csv.append(dict(arm=a, backbone=MLAB[a], role=role, params_M=p,
                        GFLOPs=g, checkpoint_tag=PRETRAIN_TAG[a],
                        pretraining_data="ImageNet-1k"))
    pd.DataFrame(csv).to_csv(tdir / "T2_model_specs.csv", index=False)
    body = md(rows, ["backbone", "design", "params (M)", "GFLOPs",
                     "checkpoint", "pretraining data"],
              [":---", ":---", "---:", "---:", ":---", ":---"])
    note = ("Parameter counts and FLOPs depend on the architecture and not on the "
            "loaded weights, so they are unchanged from the first submission. The "
            "checkpoint column is new. All four backbones are initialised from "
            "ImageNet-1k weights, which is what makes the comparison in Tables 3 "
            "to 8 a comparison of architectures rather than of pretraining budgets. "
            "The first submission described the set as fully ImageNet-1k while "
            "ViT-S in fact used `augreg_in21k_ft_in1k` and ConvNeXt-T used "
            "`in12k_ft_in1k`. Those two checkpoints are now reported only in the "
            "confound table, where they are compared against their equalised "
            "counterparts.")
    (tdir / "T2_model_specs.md").write_text(
        f"# T2 — Backbone specifications\n\n{note}\n\n{body}\n", encoding="utf-8")
    print(f"  -> {tdir/'T2_model_specs.md'}")

    print("\n" + (tdir / "T1_colorfulness.md").read_text(encoding="utf-8"))
    print((tdir / "T2_model_specs.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
