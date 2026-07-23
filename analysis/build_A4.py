"""A4 — Table 7: kontras berpasangan antar backbone, diff + 95% CI.

Kontrak spec: LAPOR diff + CI. JANGAN masukkan p-value, d_z, atau bintang
signifikansi. std ddof=1. CI dari paired-diff per-seed (t, df=4).

Pemasangan lintas-arsitektur SAH di pipeline ini karena DataLoader memakai
generator ter-seed + worker_init_fn, sehingga urutan data identik lintas
arsitektur pada seed yang sama. Tetap dibatasi catatan kaki wajib.

python revision/build_A4.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = [("flowers102", "Flowers-102"), ("cub200", "CUB-200")]
ARMS = [("ResNet-50", "resnet50"), ("ConvNeXt-T", "convnext_tiny_in1k"),
        ("ViT-S", "vit_small_in1k"), ("Swin-T", "swin_tiny")]
FOOT = ("\n> Inference is conditional on a single checkpoint per arm; the variance "
        "reflects fine-tuning, not the pretraining population. Seeds are matched "
        "across arms: the data loader uses a seeded generator and worker "
        "initialisation, so the sample order is identical across architectures at "
        "the same seed.\n")


def series(df, arm, cond):
    return df[(df.model == arm) & (df.condition == cond)].set_index("seed")["acc"].sort_index()


def diff_ci(a, b):
    i = a.index.intersection(b.index)
    d = (a.loc[i] - b.loc[i]).values
    m, sd, n = d.mean(), d.std(ddof=1), len(d)
    half = tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, m - half, m + half, n


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)

    L = ["# A4 — Table 7: pairwise contrasts (difference with 95% CI)", "",
         "Differences are reported with 95% confidence intervals from the per-seed "
         "paired differences (t, df = 4). P-values, effect sizes, and significance "
         "markers are deliberately omitted: with five seeds and a single checkpoint "
         "per arm they overstate the evidence.", ""]

    for ds, lab in DS:
        e2 = pd.read_csv(rdir / f"e2_perturb_{ds}.csv")
        clean = {n: series(e2, a, "clean") for n, a in ARMS}
        gray = {n: series(e2, a, "grayscale") for n, a in ARMS}
        cr = {n: clean[n] - gray[n] for n, _ in ARMS}

        for metric, data in [("Clean accuracy", clean), ("Color-Reliance (CR)", cr)]:
            L += [f"## {lab} — {metric}", "",
                  "| contrast | difference | 95% CI | n |", "|---|---|---|---|"]
            for (n1, _), (n2, _) in combinations(ARMS, 2):
                m, lo, hi, n = diff_ci(data[n1], data[n2])
                L.append(f"| {n1} − {n2} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {n} |")
            L.append("")

        L += [f"## {lab} — per-arm means", "",
              "| backbone | clean acc | grayscale acc | CR |", "|---|---|---|---|"]
        for n, _ in ARMS:
            L.append(f"| {n} | {clean[n].mean():.4f} ± {clean[n].std(ddof=1):.4f} | "
                     f"{gray[n].mean():.4f} ± {gray[n].std(ddof=1):.4f} | "
                     f"{cr[n].mean():.4f} ± {cr[n].std(ddof=1):.4f} |")
        L.append("")

    L.append(FOOT)
    out = tdir / "A4_table7_paired_diffCI.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
