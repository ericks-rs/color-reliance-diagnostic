"""Ablation resep ResNet-50 (AdamW vs SGD), referensi atas ConvNeXt-T.

T6 + metrik turunan:
  1. dAcc_recipe = acc(SGD) - acc(AdamW), paired t-test n=5 + Cohen dz
  2. recipe gap recovered % = (acc_SGD - acc_AdamW)/(acc_ConvNeXt - acc_AdamW)*100
  3. dCR_recipe = CR(SGD) - CR(AdamW), paired test
  4. sisa gap ResNet-SGD -> ConvNeXt (acc + signifikansi)  [Skenario A vs B]
Figur: bar clean-acc & CR utk {ResNet-AdamW, ResNet-SGD, ConvNeXt} per dataset.

Usage: python analyze_ablation.py --datasets flowers102 cub200
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root
from src import stats as st

# arm yang dibandingkan
ADAMW = "resnet50"
SGD = "resnet50_sgd"
REF = "convnext_tiny"
ALAB = {ADAMW: "ResNet-50 (AdamW)", SGD: "ResNet-50 (SGD)", REF: "ConvNeXt-T"}
ACOLOR = {ADAMW: "#4C72B0", SGD: "#DD8452", REF: "#55A868"}
DSLAB = {"flowers102": "Flowers-102", "cub200": "CUB-200"}
ARM_ROWS = [REF, ADAMW, SGD]


def fmt_p(p):
    if p != p:
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def stars(p):
    if p != p:
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _msd(s):
    return f"{s.mean():.4f}+-{s.std(ddof=0):.4f}"


def build_t6(e1, e2, ds):
    """Return (df_t6, derived dict)."""
    acc_ps = st.clean_acc_per_seed(e1, ds)        # dict arm->Series(seed)
    cr_ps = st.cr_per_seed(e2, ds)
    # gray acc + macro_f1 per arm (mean+-std)
    e1o = e1[(e1.dataset == ds) & (e1.bin == "overall")]
    gray = e2[(e2.dataset == ds) & (e2.condition == "grayscale")]

    rows = []
    for arm in ARM_ROWS:
        if arm not in acc_ps:
            continue
        f1 = e1o[e1o.model == arm].set_index("seed")["macro_f1"].sort_index()
        gr = gray[gray.model == arm].set_index("seed")["acc"].sort_index()
        rows.append({
            "arm": ALAB[arm],
            "clean_acc": _msd(acc_ps[arm]),
            "gray_acc": _msd(gr) if len(gr) else "n/a",
            "CR": _msd(cr_ps[arm]) if arm in cr_ps else "n/a",
            "macro_f1": _msd(f1) if len(f1) else "n/a",
        })
    df_t6 = pd.DataFrame(rows)

    derived = {}
    if ADAMW in acc_ps and SGD in acc_ps:
        a, s = acc_ps[ADAMW], acc_ps[SGD]
        common = a.index.intersection(s.index)
        r = st.paired_test(s.loc[common].values, a.loc[common].values)
        derived["dAcc_recipe"] = r
        derived["acc_adamw"] = float(a.mean())
        derived["acc_sgd"] = float(s.mean())
        if REF in acc_ps:
            ref_mean = float(acc_ps[REF].mean())
            derived["acc_convnext"] = ref_mean
            denom = ref_mean - float(a.mean())
            derived["recovered_pct"] = (float(s.mean()) - float(a.mean())) / denom * 100 \
                if abs(denom) > 1e-9 else float("nan")
            # sisa gap SGD -> ConvNeXt (ConvNeXt - SGD)
            cref = acc_ps[REF]
            common2 = s.index.intersection(cref.index)
            rr = st.paired_test(cref.loc[common2].values, s.loc[common2].values)
            derived["residual_gap"] = rr
    if ADAMW in cr_ps and SGD in cr_ps:
        a, s = cr_ps[ADAMW], cr_ps[SGD]
        common = a.index.intersection(s.index)
        derived["dCR_recipe"] = st.paired_test(s.loc[common].values, a.loc[common].values)
    return df_t6, derived


def interpret(ds, d):
    print(f"\n=== ABLATION RESEP ({DSLAB.get(ds, ds)}) ===")
    if "dAcc_recipe" in d:
        r = d["dAcc_recipe"]
        print(f"  acc: AdamW={d['acc_adamw']:.4f}  SGD={d['acc_sgd']:.4f}  "
              f"ConvNeXt={d.get('acc_convnext', float('nan')):.4f}")
        print(f"  dAcc_recipe (SGD-AdamW) = {r['mean_diff']:+.4f} "
              f"(p={fmt_p(r['p_ttest'])} {stars(r['p_ttest'])}, dz={r['cohen_dz']:+.2f})")
    if "recovered_pct" in d:
        print(f"  recipe gap recovered = {d['recovered_pct']:.1f}% "
              f"dari jurang ResNet->ConvNeXt")
    if "residual_gap" in d:
        rr = d["residual_gap"]
        print(f"  sisa gap SGD->ConvNeXt = {rr['mean_diff']:+.4f} "
              f"(p={fmt_p(rr['p_ttest'])} {stars(rr['p_ttest'])})")
    if "dCR_recipe" in d:
        rc = d["dCR_recipe"]
        print(f"  dCR_recipe (SGD-AdamW) = {rc['mean_diff']:+.4f} "
              f"(p={fmt_p(rc['p_ttest'])} {stars(rc['p_ttest'])})  "
              f"[CR turun = lebih sedikit nyandar warna]")
    # interpretasi Skenario A vs B (ambang sederhana, netral)
    rec = d.get("recovered_pct", float("nan"))
    resid_p = d.get("residual_gap", {}).get("p_ttest", float("nan"))
    resid_d = d.get("residual_gap", {}).get("mean_diff", float("nan"))
    if rec == rec:
        big_recover = rec > 60
        residual_sig = (resid_p == resid_p) and resid_p < 0.05 and resid_d > 0.01
        if big_recover and not residual_sig:
            verdict = "condong SKENARIO A (recipe matters: SGD memulihkan sebagian besar gap, sisa ke ConvNeXt kecil/tak signifikan)"
        elif not big_recover and residual_sig:
            verdict = "condong SKENARIO B (optimizer bukan penyebab utama: sisa gap ke ConvNeXt tetap signifikan)"
        else:
            verdict = (f"campuran (recovered={rec:.1f}%, sisa gap "
                       f"{'signifikan' if residual_sig else 'tak signifikan'}) "
                       "-- laporkan apa adanya, jangan dipaksakan")
        print(f"  >> INTERPRETASI: {verdict}")


def fig_ablation(e1, e2, datasets, fdir, tag=""):
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
                         "legend.frameon": False})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.3 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        acc_ps = st.clean_acc_per_seed(e1, ds)
        cr_ps = st.cr_per_seed(e2, ds)
        arms = [a for a in ARM_ROWS if a in acc_ps]
        x = np.arange(len(arms))
        # clean acc
        ax = axes[r][0]
        means = [acc_ps[a].mean() for a in arms]
        stds = [acc_ps[a].std(ddof=0) for a in arms]
        ax.bar(x, means, yerr=stds, capsize=4, color=[ACOLOR[a] for a in arms],
               edgecolor="black", lw=0.6)
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([ALAB[a] for a in arms], rotation=10)
        ax.set_ylabel("Top-1 accuracy"); ax.set_title(f"{DSLAB[ds]} - clean accuracy")
        ax.set_ylim(min(means) - 0.08, 1.01)
        # CR
        ax = axes[r][1]
        cm = [cr_ps[a].mean() for a in arms]
        cs = [cr_ps[a].std(ddof=0) for a in arms]
        ax.bar(x, cm, yerr=cs, capsize=4, color=[ACOLOR[a] for a in arms],
               edgecolor="black", lw=0.6)
        for xi, m in zip(x, cm):
            ax.text(xi, m + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([ALAB[a] for a in arms], rotation=10)
        ax.set_ylabel("Color-Reliance"); ax.set_title(f"{DSLAB[ds]} - color reliance")
    fig.suptitle("Effect of training recipe on ResNet-50 (AdamW vs SGD, ConvNeXt ref)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    p = Path(fdir) / f"ablation_recipe{tag}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["flowers102", "cub200"])
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)

    # hanya dataset yg punya arm SGD
    datasets = []
    for d in args.datasets:
        e1p = rdir / f"e1_clean_{d}.csv"
        if e1p.exists() and (SGD in pd.read_csv(e1p)["model"].unique()):
            datasets.append(d)
    if not datasets:
        print("Belum ada arm resnet50_sgd di hasil. Jalankan run_all --recipe sgd dulu.")
        return

    e1 = pd.concat([pd.read_csv(rdir / f"e1_clean_{d}.csv") for d in datasets],
                   ignore_index=True)
    e2 = pd.concat([pd.read_csv(rdir / f"e2_perturb_{d}.csv") for d in datasets],
                   ignore_index=True)

    all_t6 = []
    for ds in datasets:
        df_t6, derived = build_t6(e1, e2, ds)
        df_t6.insert(0, "dataset", DSLAB.get(ds, ds))
        all_t6.append(df_t6)
        interpret(ds, derived)
    t6 = pd.concat(all_t6, ignore_index=True)
    (tdir / "T6_recipe_ablation.md").write_text(t6.to_markdown(index=False), encoding="utf-8")
    (tdir / "T6_recipe_ablation.tex").write_text(t6.to_latex(index=False), encoding="utf-8")
    print(f"\n  [table] T6_recipe_ablation -> {tdir}/T6_recipe_ablation.md/.tex")

    print("\n=== figures ===")
    fig_ablation(e1, e2, datasets, Path(cfg["paths"]["figures"]))
    pubdir = Path(cfg["paths"]["figures"]) / "pub"; pubdir.mkdir(parents=True, exist_ok=True)
    fig_ablation(e1, e2, datasets, pubdir, tag="_pub")
    print("\n=== analyze_ablation done ===")


if __name__ == "__main__":
    main()
