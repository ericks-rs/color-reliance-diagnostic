"""Export tabel A4-A8 ke CSV tidy (pendamping versi .md).

Output di tables/:
  A4_table7_paired_diffCI.csv   dataset,metric,contrast,diff,ci_lo,ci_hi,n
  A4_perarm_means.csv           dataset,backbone,clean_*,gray_*,CR_*
  A5_table8_resnet4cfg.csv      dataset,id,config,checkpoint,optimizer,lr,wd,clean_*,gray_*,CR_*,diff_vs_S1,ci_*
  A6_table_confound.csv         dataset,backbone,regime,tag,clean_*,gray_*,CR_*
  A6_confound_diffs.csv         dataset,backbone,metric,diff,ci_lo,ci_hi
  A7_table_R23_matrix.csv       dataset,backbone,rgb_rgb_*,rgb_gray_*,gray_gray_*,residual
  A8_arch_contrast_R13.csv      dataset,contrast,resnet,convnext,diff,reading

python revision/export_A_csv.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = ["flowers102", "cub200"]
ARMS = [("ResNet-50", "resnet50"), ("ConvNeXt-T", "convnext_tiny_in1k"),
        ("ViT-S", "vit_small_in1k"), ("Swin-T", "swin_tiny")]
RES = [("S1", "resnet50", "shared", "a1_in1k", "adamw", 1e-4, 0.05),
       ("S11", "resnet50_tv_sgd", "legacy", "tv_in1k", "sgd", 1e-2, 1e-4),
       ("S12", "resnet50_adamw_lr1e3", "proper", "a1_in1k", "adamw", 1e-3, 0.05),
       ("S13", "resnet50_sgd", "sgd", "a1_in1k", "sgd", 1e-2, 1e-4)]
CONF = [("ConvNeXt-T", "convnext_tiny_in1k", "fb_in1k", "convnext_tiny", "in12k_ft_in1k"),
        ("ViT-S", "vit_small_in1k", "augreg_in1k", "vit_small", "augreg_in21k_ft_in1k")]
GRAY = {"ResNet-50": "resnet50_gray", "ConvNeXt-T": "convnext_tiny_in1k_gray",
        "ViT-S": "vit_small_in1k_gray", "Swin-T": "swin_tiny_gray"}


def S(d, a, c):
    return d[(d.model == a) & (d.condition == c)].set_index("seed")["acc"].sort_index()


def ci(a, b):
    i = a.index.intersection(b.index)
    v = (a.loc[i] - b.loc[i]).values
    m, sd, n = v.mean(), v.std(ddof=1), len(v)
    h = tdist.ppf(.975, n - 1) * sd / np.sqrt(n)
    return m, m - h, m + h, n


def ms(s):
    return s.mean(), s.std(ddof=1)


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"]); tdir = Path(cfg["paths"]["tables"])
    E = {ds: pd.read_csv(rdir / f"e2_perturb_{ds}.csv") for ds in DS}

    a4, a4m, a5, a6, a6d, a7, a8 = [], [], [], [], [], [], []
    for ds in DS:
        d = E[ds]
        clean = {n: S(d, a, "clean") for n, a in ARMS}
        gray = {n: S(d, a, "grayscale") for n, a in ARMS}
        cr = {n: clean[n] - gray[n] for n, _ in ARMS}

        # A4
        for metric, data in [("clean_acc", clean), ("CR", cr)]:
            for (n1, _), (n2, _) in combinations(ARMS, 2):
                m, lo, hi, n = ci(data[n1], data[n2])
                a4.append(dict(dataset=ds, metric=metric, contrast=f"{n1} - {n2}",
                               diff=m, ci_lo=lo, ci_hi=hi, n=n))
        for n, _ in ARMS:
            cm, cs = ms(clean[n]); gm, gs = ms(gray[n]); rm, rs = ms(cr[n])
            a4m.append(dict(dataset=ds, backbone=n, clean_mean=cm, clean_sd=cs,
                            gray_mean=gm, gray_sd=gs, CR_mean=rm, CR_sd=rs))

        # A5
        base = S(d, "resnet50", "clean")
        for sid, arm, name, ck, opt, lr, wd in RES:
            c, g = S(d, arm, "clean"), S(d, arm, "grayscale")
            r = c - g
            cm, cs = ms(c); gm, gs = ms(g); rm, rs = ms(r)
            if sid == "S1":
                dv, lo, hi = np.nan, np.nan, np.nan
            else:
                dv, lo, hi, _ = ci(c, base)
            a5.append(dict(dataset=ds, id=sid, config=name, checkpoint=ck, optimizer=opt,
                           lr=lr, wd=wd, clean_mean=cm, clean_sd=cs, gray_mean=gm,
                           gray_sd=gs, CR_mean=rm, CR_sd=rs,
                           diff_vs_S1=dv, ci_lo=lo, ci_hi=hi))
        c2, g2 = S(d, "convnext_tiny_in1k", "clean"), S(d, "convnext_tiny_in1k", "grayscale")
        cm, cs = ms(c2); gm, gs = ms(g2); rm, rs = ms(c2 - g2)
        a5.append(dict(dataset=ds, id="S2", config="ConvNeXt-T reference",
                       checkpoint="fb_in1k", optimizer="adamw", lr=1e-4, wd=0.05,
                       clean_mean=cm, clean_sd=cs, gray_mean=gm, gray_sd=gs,
                       CR_mean=rm, CR_sd=rs, diff_vs_S1=np.nan, ci_lo=np.nan, ci_hi=np.nan))

        # A6
        for name, eq, eqtag, ne, netag in CONF:
            for regime, arm, tag in [("equalized", eq, eqtag), ("non_equalized", ne, netag)]:
                c, g = S(d, arm, "clean"), S(d, arm, "grayscale")
                cm, cs = ms(c); gm, gs = ms(g); rm, rs = ms(c - g)
                a6.append(dict(dataset=ds, backbone=name, regime=regime, tag=tag,
                               clean_mean=cm, clean_sd=cs, gray_mean=gm, gray_sd=gs,
                               CR_mean=rm, CR_sd=rs))
            ce, cn = S(d, eq, "clean"), S(d, ne, "clean")
            ge, gn = S(d, eq, "grayscale"), S(d, ne, "grayscale")
            for metric, aa, bb in [("clean_acc", cn, ce), ("gray_acc", gn, ge),
                                   ("CR", cn - gn, ce - ge)]:
                m, lo, hi, _ = ci(aa, bb)
                a6d.append(dict(dataset=ds, backbone=name, metric=metric,
                                diff_noneq_minus_eq=m, ci_lo=lo, ci_hi=hi))

        # A7
        for name, arm in ARMS:
            rr, rg = clean[name], gray[name]
            gg = S(d, GRAY[name], "grayscale")
            a7.append(dict(dataset=ds, backbone=name,
                           rgb_rgb_mean=rr.mean(), rgb_rgb_sd=rr.std(ddof=1),
                           rgb_gray_mean=rg.mean(), rgb_gray_sd=rg.std(ddof=1),
                           gray_gray_mean=gg.mean(), gray_gray_sd=gg.std(ddof=1),
                           residual=rr.mean() - gg.mean()))

        # A8
        s1, s12, s2 = S(d, "resnet50", "clean"), S(d, "resnet50_adamw_lr1e3", "clean"), c2
        a8.append(dict(dataset=ds, contrast="S12 -> S2 (ResNet 1e-3 vs ConvNeXt 1e-4)",
                       resnet=s12.mean(), convnext=s2.mean(), diff=s2.mean() - s12.mean(),
                       reading="conservative lower bound"))
        a8.append(dict(dataset=ds, contrast="S1 -> S2 (both 1e-4, matched)",
                       resnet=s1.mean(), convnext=s2.mean(), diff=s2.mean() - s1.mean(),
                       reading="upper bound (ResNet under-tuned)"))

    for name, rows in [("A4_table7_paired_diffCI", a4), ("A4_perarm_means", a4m),
                       ("A5_table8_resnet4cfg", a5), ("A6_table_confound", a6),
                       ("A6_confound_diffs", a6d), ("A7_table_R23_matrix", a7),
                       ("A8_arch_contrast_R13", a8)]:
        p = tdir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"  -> {p}  ({len(rows)} baris)")


if __name__ == "__main__":
    main()
