"""Export CSV per-V yang belum punya berkas sendiri (V1, V2, V3, V8, V9).

V4/V5 -> sudah ada (V4_perbin_wholeimg_*, V5_perbin_foreground_*)
V6    -> figur, tanpa CSV
V7    -> sudah ada (results/per_species_CR_*)

Output di tables/ (agregat mean+sd ddof=1) dan tables/perseed/ (nilai tiap seed):
  V1_clean_acc.csv           S1-S4 clean top-1
  V2_grayscale_CR.csv        S1-S4 clean, gray, CR
  V3_degradation_suite.csv   S1-S4 semua kondisi perturbasi
  V8_confound_eval.csv       S5-S6 clean, gray, CR  (non-equalized)
  V9_grayscale_eval.csv      S7-S10 gray-train di test gray

python revision/export_V_csv.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = ["flowers102", "cub200"]
MAIN = [("S1", "ResNet-50", "resnet50"), ("S2", "ConvNeXt-T", "convnext_tiny_in1k"),
        ("S3", "ViT-S", "vit_small_in1k"), ("S4", "Swin-T", "swin_tiny")]
CONF = [("S5", "ConvNeXt-T (in12k_ft_in1k)", "convnext_tiny"),
        ("S6", "ViT-S (augreg_in21k_ft_in1k)", "vit_small")]
GRAYA = [("S7", "ResNet-50", "resnet50_gray"), ("S8", "ConvNeXt-T", "convnext_tiny_in1k_gray"),
         ("S9", "ViT-S", "vit_small_in1k_gray"), ("S10", "Swin-T", "swin_tiny_gray")]


def S(d, a, c):
    return d[(d.model == a) & (d.condition == c)].set_index("seed")["acc"].sort_index()


def agg(rows):
    return pd.DataFrame(rows)


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)
    pdir = tdir / "perseed"; pdir.mkdir(exist_ok=True)
    E = {ds: pd.read_csv(rdir / f"e2_perturb_{ds}.csv") for ds in DS}

    v1, v2, v3, v8, v9 = [], [], [], [], []
    p1, p2, p8, p9 = [], [], [], []

    for ds in DS:
        d = E[ds]
        # V1 + V2 (S1-S4)
        for sid, name, arm in MAIN:
            c, g = S(d, arm, "clean"), S(d, arm, "grayscale")
            cr = c - g
            v1.append(dict(dataset=ds, id=sid, backbone=name, arm=arm,
                           clean_mean=c.mean(), clean_sd=c.std(ddof=1), n_seed=len(c)))
            v2.append(dict(dataset=ds, id=sid, backbone=name, arm=arm,
                           clean_mean=c.mean(), clean_sd=c.std(ddof=1),
                           gray_mean=g.mean(), gray_sd=g.std(ddof=1),
                           CR_mean=cr.mean(), CR_sd=cr.std(ddof=1), n_seed=len(cr)))
            for s in c.index:
                p1.append(dict(dataset=ds, id=sid, backbone=name, seed=int(s), clean=c[s]))
                p2.append(dict(dataset=ds, id=sid, backbone=name, seed=int(s),
                               clean=c[s], gray=g[s], CR=c[s] - g[s]))
        # V3 semua kondisi (S1-S4)
        for sid, name, arm in MAIN:
            sub = d[d.model == arm]
            for cond, grp in sub.groupby("condition"):
                v3.append(dict(dataset=ds, id=sid, backbone=name, condition=cond,
                               kind=grp["kind"].iloc[0] if "kind" in grp else "",
                               param=grp["param"].iloc[0] if "param" in grp else "",
                               acc_mean=grp["acc"].mean(), acc_sd=grp["acc"].std(ddof=1),
                               n_seed=grp["seed"].nunique()))
        # V8 (S5-S6)
        for sid, name, arm in CONF:
            c, g = S(d, arm, "clean"), S(d, arm, "grayscale")
            cr = c - g
            v8.append(dict(dataset=ds, id=sid, backbone=name, arm=arm,
                           clean_mean=c.mean(), clean_sd=c.std(ddof=1),
                           gray_mean=g.mean(), gray_sd=g.std(ddof=1),
                           CR_mean=cr.mean(), CR_sd=cr.std(ddof=1), n_seed=len(cr)))
            for s in c.index:
                p8.append(dict(dataset=ds, id=sid, backbone=name, seed=int(s),
                               clean=c[s], gray=g[s], CR=c[s] - g[s]))
        # V9 (S7-S10)
        for sid, name, arm in GRAYA:
            g = S(d, arm, "grayscale")
            v9.append(dict(dataset=ds, id=sid, backbone=name, arm=arm,
                           gray_train_mean=g.mean(), gray_train_sd=g.std(ddof=1),
                           n_seed=len(g)))
            for s in g.index:
                p9.append(dict(dataset=ds, id=sid, backbone=name, seed=int(s), gray_train=g[s]))

    for nm, rows in [("V1_clean_acc", v1), ("V2_grayscale_CR", v2),
                     ("V3_degradation_suite", v3), ("V8_confound_eval", v8),
                     ("V9_grayscale_eval", v9)]:
        p = tdir / f"{nm}.csv"; agg(rows).to_csv(p, index=False)
        print(f"  -> {p}  ({len(rows)} baris)")
    for nm, rows in [("V1_clean_acc", p1), ("V2_grayscale_CR", p2),
                     ("V8_confound_eval", p8), ("V9_grayscale_eval", p9)]:
        p = pdir / f"{nm}_perseed.csv"; agg(rows).to_csv(p, index=False)
        print(f"  -> {p}  ({len(rows)} baris)")


if __name__ == "__main__":
    main()
