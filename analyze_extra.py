"""Analisis tambahan:
  T5 uji statistik berpasangan (paired t-test + Wilcoxon + Cohen's dz)
  Figur publikasi (P1-P4): gaya bersih, error bar std lintas seed, tanda signifikansi.

Usage: python analyze_extra.py --datasets flowers102 cub200
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root
from src import stats as st

MODEL_ORDER = ["resnet50", "convnext_tiny", "vit_small", "swin_tiny"]
MLAB = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T",
        "vit_small": "ViT-S", "swin_tiny": "Swin-T"}
MCOLOR = {"resnet50": "#4C72B0", "convnext_tiny": "#55A868",
          "vit_small": "#C44E52", "swin_tiny": "#8172B3"}
DSLAB = {"flowers102": "Flowers-102", "cub200": "CUB-200"}
BINS = ["low", "mid", "high"]
CONVS = ["resnet50", "convnext_tiny"]
ATTNS = ["vit_small", "swin_tiny"]
# perbandingan kunci RQ3: atensi vs tiap baseline konv + ResNet vs ConvNeXt
COMPARISONS = [("vit_small", "resnet50"), ("vit_small", "convnext_tiny"),
               ("swin_tiny", "resnet50"), ("swin_tiny", "convnext_tiny"),
               ("resnet50", "convnext_tiny")]


def _pub_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linestyle": "--", "legend.frameon": False, "legend.fontsize": 10,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })


def _stars(p):
    if p != p:
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p(p):
    if p != p:
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


# --------------------------------------------------------------------------
def stats_tables(e1, e2, datasets, tdir):
    all_rows = []
    for ds in datasets:
        acc_ps = st.clean_acc_per_seed(e1, ds)
        cr_ps = st.cr_per_seed(e2, ds) if len(e2) else {}
        all_rows.append(st.comparison_table(acc_ps, COMPARISONS, ds, "clean_acc"))
        if cr_ps:
            all_rows.append(st.comparison_table(cr_ps, COMPARISONS, ds, "CR"))
    df = pd.concat(all_rows, ignore_index=True)
    # format presentasi
    pres = df.copy()
    pres["comparison"] = pres["comparison"].map(
        lambda s: " - ".join(MLAB.get(x, x) for x in s.split(" - ")))
    pres["mean_diff"] = pres.apply(
        lambda r: f"{r['mean_diff']:+.4f}+-{r['sd_diff']:.4f}", axis=1)
    pres["p_ttest"] = df["p_ttest"].map(fmt_p)
    pres["p_wilcoxon"] = df["p_wilcoxon"].map(fmt_p)
    pres["cohen_dz"] = df["cohen_dz"].map(lambda v: f"{v:+.2f}" if v == v else "n/a")
    pres["sig"] = df["p_ttest"].map(_stars)
    pres = pres[["dataset", "metric", "comparison", "mean_diff", "t",
                 "p_ttest", "p_wilcoxon", "cohen_dz", "sig", "n"]]
    pres["t"] = pres["t"].map(lambda v: f"{v:+.2f}" if v == v else "n/a")
    tdir = Path(tdir); tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "T5_paired_stats.md").write_text(pres.to_markdown(index=False), encoding="utf-8")
    (tdir / "T5_paired_stats.tex").write_text(pres.to_latex(index=False), encoding="utf-8")
    print(f"  [table] T5_paired_stats -> {tdir}/T5_paired_stats.md/.tex")
    return df, pres


# --------------------------------------------------------------------------
def _mean_std(per_seed, model):
    s = per_seed.get(model)
    if s is None:
        return np.nan, np.nan
    return float(s.mean()), float(s.std(ddof=0))


def fig_clean_acc(e1, datasets, fdir, statdf):
    """P1: akurasi clean grouped bar, kedua dataset, error bar std seed."""
    _pub_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.6),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ps = st.clean_acc_per_seed(e1, ds)
        x = np.arange(len(MODEL_ORDER))
        means = [(_mean_std(ps, m)[0]) for m in MODEL_ORDER]
        stds = [(_mean_std(ps, m)[1]) for m in MODEL_ORDER]
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[MCOLOR[m] for m in MODEL_ORDER], edgecolor="black", lw=0.6)
        for b, mn in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mn + 0.005, f"{mn:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER], rotation=12)
        ax.set_ylim(min(means) - 0.08, 1.01)
        ax.set_ylabel("Top-1 accuracy")
        ax.set_title(f"{DSLAB[ds]} (clean)")
    fig.suptitle("Clean accuracy: ConvNeXt (modern-recipe CNN) leads on both datasets",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = Path(fdir) / "P1_clean_acc.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")


def fig_color_reliance(e2, datasets, fdir, statdf):
    """P2: color-reliance bar, kedua dataset, error bar + bracket signifikansi
    ResNet-vs-ConvNeXt & atensi-vs-ConvNeXt."""
    _pub_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ps = st.cr_per_seed(e2, ds)
        x = np.arange(len(MODEL_ORDER))
        means = [(_mean_std(ps, m)[0]) for m in MODEL_ORDER]
        stds = [(_mean_std(ps, m)[1]) for m in MODEL_ORDER]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=[MCOLOR[m] for m in MODEL_ORDER], edgecolor="black", lw=0.6)
        for xi, mn, sd in zip(x, means, stds):
            ax.text(xi, mn + sd + 0.006, f"{mn:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER], rotation=12)
        ax.set_ylabel("Color-Reliance  (acc$_{clean}$ - acc$_{gray}$)")
        ax.set_title(f"{DSLAB[ds]}")
        ax.set_ylim(0, max(means) + max(stds) + 0.07)
    fig.suptitle("Color-reliance: ConvNeXt is the most structural (lowest) on both datasets",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = Path(fdir) / "P2_color_reliance.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")


def fig_gap_vs_bin(e1, datasets, fdir):
    """P3: gap (atensi - konv) vs colorfulness bin, error bar std seed."""
    _pub_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.6),
                             squeeze=False)
    styles = {("vit_small", "resnet50"): ("-", "o", "#C44E52"),
              ("vit_small", "convnext_tiny"): ("--", "s", "#C44E52"),
              ("swin_tiny", "resnet50"): ("-", "o", "#8172B3"),
              ("swin_tiny", "convnext_tiny"): ("--", "s", "#8172B3")}
    for ax, ds in zip(axes[0], datasets):
        binps = {b: st.bin_acc_per_seed(e1, ds, b) for b in BINS}
        x = np.arange(len(BINS))
        for (a, c), (ls, mk, col) in styles.items():
            gm, gs = [], []
            for b in BINS:
                sa, sb = binps[b].get(a), binps[b].get(c)
                if sa is None or sb is None:
                    gm.append(np.nan); gs.append(0); continue
                common = sa.index.intersection(sb.index)
                d = (sa.loc[common] - sb.loc[common]).values
                gm.append(d.mean()); gs.append(d.std(ddof=0))
            ax.errorbar(x, gm, yerr=gs, ls=ls, marker=mk, color=col, capsize=3,
                        label=f"{MLAB[a]}-{MLAB[c]}")
        ax.axhline(0, color="gray", lw=1, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(BINS)
        ax.set_xlabel("colorfulness bin"); ax.set_ylabel("acc gap (attention - conv)")
        ax.set_title(f"{DSLAB[ds]}")
        ax.legend(fontsize=9)
    fig.suptitle("E1 gap vs colorfulness: attention beats ResNet but trails ConvNeXt (dashed<0)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = Path(fdir) / "P3_gap_vs_bin.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")


def fig_drop_curves(e2, datasets, fdir):
    """P4: drop-curve perturbasi (hue rotation & quantization) dgn error band."""
    _pub_style()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.4 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        sub = e2[e2.dataset == ds]
        clean_mean = {m: sub[(sub.model == m) & (sub.condition == "clean")].acc.mean()
                      for m in MODEL_ORDER}
        # hue
        axh = axes[r][0]
        for m in MODEL_ORDER:
            g = sub[(sub.model == m) & (sub.kind == "hue")]
            if not len(g):
                continue
            agg = g.groupby("param").acc.agg(["mean", "std"]).sort_index()
            axh.plot(agg.index, agg["mean"], marker="o", color=MCOLOR[m], label=MLAB[m])
            axh.fill_between(agg.index, agg["mean"] - agg["std"].fillna(0),
                             agg["mean"] + agg["std"].fillna(0), color=MCOLOR[m], alpha=0.15)
        axh.set_xlabel("hue rotation factor"); axh.set_ylabel("accuracy")
        axh.set_title(f"{DSLAB[ds]} - hue rotation"); axh.legend(fontsize=8, ncol=2)
        # quantization
        axq = axes[r][1]
        for m in MODEL_ORDER:
            g = sub[(sub.model == m) & (sub.kind == "quant")]
            if not len(g):
                continue
            agg = g.groupby("param").acc.agg(["mean", "std"]).sort_index()
            axq.plot(agg.index, agg["mean"], marker="s", color=MCOLOR[m], label=MLAB[m])
            axq.fill_between(agg.index, agg["mean"] - agg["std"].fillna(0),
                             agg["mean"] + agg["std"].fillna(0), color=MCOLOR[m], alpha=0.15)
            axq.axhline(clean_mean[m], color=MCOLOR[m], ls=":", lw=0.8, alpha=0.6)
        axq.set_xlabel("quantization levels / channel"); axq.set_ylabel("accuracy")
        axq.set_xscale("log", base=2)
        axq.set_title(f"{DSLAB[ds]} - quantization (dotted=clean)"); axq.legend(fontsize=8, ncol=2)
    fig.suptitle("E2 test-time color degradation drop-curves (band = std across seeds)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    p = Path(fdir) / "P4_drop_curves.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["flowers102", "cub200"])
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = cfg["paths"]["tables"]
    fdir = Path(cfg["paths"]["figures"]) / "pub"
    fdir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in args.datasets if (rdir / f"e1_clean_{d}.csv").exists()]
    e1 = pd.concat([pd.read_csv(rdir / f"e1_clean_{d}.csv") for d in datasets],
                   ignore_index=True)
    e2parts = [pd.read_csv(rdir / f"e2_perturb_{d}.csv") for d in datasets
               if (rdir / f"e2_perturb_{d}.csv").exists()]
    e2 = pd.concat(e2parts, ignore_index=True) if e2parts else pd.DataFrame()

    print("=== T5 paired stats ===")
    statdf, _ = stats_tables(e1, e2, datasets, tdir)
    print("\n=== publication figures ===")
    fig_clean_acc(e1, datasets, fdir, statdf)
    if len(e2):
        fig_color_reliance(e2, datasets, fdir, statdf)
        fig_drop_curves(e2, datasets, fdir)
    fig_gap_vs_bin(e1, datasets, fdir)

    # ringkasan signifikansi RQ3 ke stdout
    print("\n=== RINGKASAN SIGNIFIKANSI (RQ3, clean_acc) ===")
    s = statdf[statdf.metric == "clean_acc"]
    for _, r in s.iterrows():
        a, b = r["comparison"].split(" - ")
        print(f"  [{r['dataset']}] {MLAB.get(a,a)} vs {MLAB.get(b,b)}: "
              f"diff={r['mean_diff']:+.4f} p_t={fmt_p(r['p_ttest'])} "
              f"dz={r['cohen_dz']:+.2f} {_stars(r['p_ttest'])}")
    print("\n=== analyze_extra done ===")


if __name__ == "__main__":
    main()
