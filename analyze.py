"""Agregasi lintas seed -> tabel (markdown + latex) & figur (E1/E2/E3).

Tabel:
  T1 param/FLOPs per model
  T2 akurasi clean per dataset (mean+-std lintas seed)
  T3 E1 akurasi per colorfulness-bin + gap (atensi - konv)
  T4 E3 color-reliance (acc_clean - acc_grayscale) + kontrol ConvNeXt
Figur:
  F1 gap-vs-bin (E1)
  F2 drop-curve perturbasi (E2)
  F3 bar color-reliance per model (E3)
  F4 confusion matrices

Usage: python analyze.py --datasets flowers102 [cub200]
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
from src import models as models_mod

# urutan & label model
MODEL_ORDER = ["resnet50", "convnext_tiny", "vit_small", "swin_tiny"]
MODEL_LABEL = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T",
               "vit_small": "ViT-S", "swin_tiny": "Swin-T"}
CONV_BASELINES = ["resnet50", "convnext_tiny"]
ATTN_MODELS = ["vit_small", "swin_tiny"]
BIN_ORDER = ["low", "mid", "high"]


def _msd(series):
    """mean+-std string."""
    m = series.mean()
    s = series.std(ddof=0) if len(series) > 1 else 0.0
    return f"{m:.4f}+-{s:.4f}"


def save_table(df, name, tdir):
    tdir = Path(tdir); tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    (tdir / f"{name}.tex").write_text(df.to_latex(index=False), encoding="utf-8")
    print(f"  [table] {name} -> {tdir/name}.md/.tex")


# --------------------------------------------------------------------------
def table_params_flops(cfg, datasets, tdir):
    """T1: param & GFLOPs aktual per model (head dataset pertama)."""
    ds0 = datasets[0]
    nc = cfg["datasets"][ds0]["num_classes"]
    rows = []
    for mk in MODEL_ORDER:
        if mk not in cfg["models"]:
            continue
        model = models_mod.build_model(mk, cfg, nc, pretrained=False)
        total, _ = models_mod.count_params(model)
        gflops = models_mod.measure_flops(model, cfg["train"]["image_size"], "cpu")
        rows.append({"model": MODEL_LABEL[mk], "role": cfg["models"][mk]["role"],
                     "params_M": round(total / 1e6, 2),
                     "GFLOPs": round(gflops, 2) if gflops else None})
    df = pd.DataFrame(rows)
    save_table(df, f"T1_params_flops_{ds0}", tdir)
    return df


def table_clean_acc(e1, ds, tdir):
    """T2: akurasi clean (overall) mean+-std lintas seed."""
    sub = e1[(e1.dataset == ds) & (e1.bin == "overall")]
    rows = []
    for mk in MODEL_ORDER:
        g = sub[sub.model == mk]
        if len(g) == 0:
            continue
        rows.append({"model": MODEL_LABEL[mk],
                     "n_seeds": g.seed.nunique(),
                     "acc": _msd(g.acc), "macro_f1": _msd(g.macro_f1),
                     "uar": _msd(g.uar)})
    df = pd.DataFrame(rows)
    save_table(df, f"T2_clean_acc_{ds}", tdir)
    return df


def table_e1_bins(e1, ds, tdir):
    """T3: akurasi per bin (mean+-std) + gap atensi-konv per bin."""
    sub = e1[(e1.dataset == ds) & (e1.bin.isin(BIN_ORDER))]
    # acc per (model,bin) mean lintas seed
    piv = sub.groupby(["model", "bin"]).acc.mean().unstack("bin").reindex(
        index=[m for m in MODEL_ORDER if m in sub.model.unique()], columns=BIN_ORDER)
    rows = []
    for mk in piv.index:
        rows.append({"model": MODEL_LABEL[mk],
                     **{b: round(piv.loc[mk, b], 4) for b in BIN_ORDER}})
    df_acc = pd.DataFrame(rows)
    save_table(df_acc, f"T3_e1_acc_by_bin_{ds}", tdir)

    # gap = attn - conv per bin (per pasangan)
    gap_rows = []
    for attn in ATTN_MODELS:
        if attn not in piv.index:
            continue
        for conv in CONV_BASELINES:
            if conv not in piv.index:
                continue
            row = {"comparison": f"{MODEL_LABEL[attn]} - {MODEL_LABEL[conv]}"}
            for b in BIN_ORDER:
                row[b] = round(piv.loc[attn, b] - piv.loc[conv, b], 4)
            row["trend(high-low)"] = round(row["high"] - row["low"], 4)
            gap_rows.append(row)
    df_gap = pd.DataFrame(gap_rows)
    save_table(df_gap, f"T3_e1_gap_by_bin_{ds}", tdir)
    return df_acc, df_gap, piv


def table_color_reliance(e2, ds, tdir):
    """T4: CR = acc_clean - acc_grayscale (mean+-std) + kontrol ConvNeXt."""
    sub = e2[(e2.dataset == ds) & (e2.condition.isin(["clean", "grayscale"]))]
    piv = sub.pivot_table(index=["model", "seed"], columns="condition",
                          values="acc").reset_index()
    piv["CR"] = piv["clean"] - piv["grayscale"]
    rows = []
    for mk in MODEL_ORDER:
        g = piv[piv.model == mk]
        if len(g) == 0:
            continue
        rows.append({"model": MODEL_LABEL[mk],
                     "acc_clean": _msd(g["clean"]),
                     "acc_gray": _msd(g["grayscale"]),
                     "CR": _msd(g["CR"])})
    df = pd.DataFrame(rows)
    save_table(df, f"T4_color_reliance_{ds}", tdir)
    return df, piv


# --------------------------------------------------------------------------
def fig_gap_vs_bin(piv, ds, fdir):
    """F1: gap (attn - conv) vs bin."""
    fdir = Path(fdir); fdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(BIN_ORDER))
    for attn in ATTN_MODELS:
        if attn not in piv.index:
            continue
        for conv in CONV_BASELINES:
            if conv not in piv.index:
                continue
            gaps = [piv.loc[attn, b] - piv.loc[conv, b] for b in BIN_ORDER]
            plt.plot(x, gaps, marker="o",
                     label=f"{MODEL_LABEL[attn]}-{MODEL_LABEL[conv]}")
    plt.axhline(0, color="gray", ls=":", lw=1)
    plt.xticks(x, BIN_ORDER)
    plt.xlabel("colorfulness bin")
    plt.ylabel("acc gap (attention - conv)")
    plt.title(f"F1 E1 gap vs colorfulness bin ({ds})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = fdir / f"F1_gap_vs_bin_{ds}.png"
    plt.savefig(p, dpi=130); plt.close()
    print(f"  [fig] {p}")


def fig_drop_curve(e2, ds, fdir):
    """F2: drop-curve perturbasi (akurasi per kondisi, per model)."""
    fdir = Path(fdir)
    sub = e2[e2.dataset == ds]
    # quantization drop-curve (intensitas jelas: levels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # panel kiri: quantization
    ax = axes[0]
    for mk in MODEL_ORDER:
        g = sub[(sub.model == mk) & (sub.kind == "quant")]
        if len(g) == 0:
            continue
        gg = g.groupby("param").acc.mean().sort_index()
        ax.plot(gg.index, gg.values, marker="o", label=MODEL_LABEL[mk])
    ax.set_xlabel("quantization levels/channel"); ax.set_ylabel("accuracy")
    ax.set_title(f"quantization ({ds})"); ax.legend(fontsize=8)
    # panel kanan: bar grayscale + mean hue + mean shuffle vs clean
    ax = axes[1]
    kinds = ["clean", "grayscale", "hue", "shuffle"]
    width = 0.2
    xb = np.arange(len(kinds))
    for j, mk in enumerate([m for m in MODEL_ORDER if m in sub.model.unique()]):
        vals = []
        for k in kinds:
            if k == "clean":
                v = sub[(sub.model == mk) & (sub.condition == "clean")].acc.mean()
            else:
                v = sub[(sub.model == mk) & (sub.kind == k)].acc.mean()
            vals.append(v)
        ax.bar(xb + j * width, vals, width, label=MODEL_LABEL[mk])
    ax.set_xticks(xb + width * 1.5); ax.set_xticklabels(kinds)
    ax.set_ylabel("accuracy (mean over params)")
    ax.set_title(f"perturbation summary ({ds})"); ax.legend(fontsize=8)
    plt.tight_layout()
    p = fdir / f"F2_drop_curve_{ds}.png"
    plt.savefig(p, dpi=130); plt.close()
    print(f"  [fig] {p}")


def fig_color_reliance(cr_piv, ds, fdir):
    """F3: bar CR per model (mean+-std lintas seed)."""
    fdir = Path(fdir)
    plt.figure(figsize=(7, 4.5))
    means, stds, labels = [], [], []
    for mk in MODEL_ORDER:
        g = cr_piv[cr_piv.model == mk]
        if len(g) == 0:
            continue
        means.append(g["CR"].mean())
        stds.append(g["CR"].std(ddof=0) if len(g) > 1 else 0.0)
        labels.append(MODEL_LABEL[mk])
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"][:len(labels)]
    plt.bar(x, means, yerr=stds, capsize=4, color=colors)
    plt.xticks(x, labels)
    plt.ylabel("Color-Reliance (acc_clean - acc_gray)")
    plt.title(f"F3 color-reliance per model ({ds})")
    plt.tight_layout()
    p = fdir / f"F3_color_reliance_{ds}.png"
    plt.savefig(p, dpi=130); plt.close()
    print(f"  [fig] {p}")


def fig_confusion(cfg, ds, fdir):
    """F4: confusion matrix heatmap per model (seed terkecil)."""
    fdir = Path(fdir)
    cm_dir = Path(cfg["paths"]["results"]) / "confusion"
    present = [m for m in MODEL_ORDER
               if list(cm_dir.glob(f"{ds}_{m}_seed*.npy"))]
    if not present:
        return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, mk in zip(axes, present):
        f = sorted(cm_dir.glob(f"{ds}_{mk}_seed*.npy"))[0]
        cm = np.load(f).astype(float)
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(MODEL_LABEL[mk]); ax.set_xlabel("pred"); ax.set_ylabel("true")
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"F4 confusion (row-normalized) ({ds})")
    p = fdir / f"F4_confusion_{ds}.png"
    plt.savefig(p, dpi=120); plt.close()
    print(f"  [fig] {p}")


# --------------------------------------------------------------------------
def interpret(df_gap, cr_piv, ds):
    """Interpretasi otomatis sederhana RQ3 (efek mekanisme vs resep)."""
    print(f"\n=== INTERPRETASI ({ds}) ===")
    # apakah keunggulan atensi bertahan setelah ConvNeXt dimasukkan?
    cr_mean = cr_piv.groupby("model")["CR"].mean()
    if len(cr_mean):
        most = cr_mean.idxmax(); least = cr_mean.idxmin()
        print(f"  paling color-reliant: {MODEL_LABEL.get(most, most)} "
              f"(CR={cr_mean[most]:.4f})")
        print(f"  paling struktural   : {MODEL_LABEL.get(least, least)} "
              f"(CR={cr_mean[least]:.4f})")
    if df_gap is not None and len(df_gap):
        print("  gap atensi vs konv (kolom high):")
        for _, r in df_gap.iterrows():
            print(f"    {r['comparison']}: high={r['high']:+.4f} "
                  f"trend(high-low)={r['trend(high-low)']:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["flowers102"])
    args = ap.parse_args()

    chdir_to_root()
    cfg = load_config()
    tdir = cfg["paths"]["tables"]; fdir = cfg["paths"]["figures"]
    rdir = Path(cfg["paths"]["results"])

    print("=== T1 params/FLOPs ===")
    table_params_flops(cfg, args.datasets, tdir)

    for ds in args.datasets:
        print(f"\n######## dataset: {ds} ########")
        e1_p = rdir / f"e1_clean_{ds}.csv"
        e2_p = rdir / f"e2_perturb_{ds}.csv"
        if not e1_p.exists():
            print(f"  SKIP {ds}: {e1_p} belum ada"); continue
        e1 = pd.read_csv(e1_p)
        e2 = pd.read_csv(e2_p) if e2_p.exists() else pd.DataFrame()

        table_clean_acc(e1, ds, tdir)
        df_acc, df_gap, piv = table_e1_bins(e1, ds, tdir)
        fig_gap_vs_bin(piv, ds, fdir)

        cr_piv = None
        if len(e2):
            _, cr_piv = table_color_reliance(e2, ds, tdir)
            fig_drop_curve(e2, ds, fdir)
            fig_color_reliance(cr_piv, ds, fdir)
        fig_confusion(cfg, ds, fdir)

        interpret(df_gap, cr_piv if cr_piv is not None else pd.DataFrame(
            columns=["model", "CR"]), ds)

    print("\n=== analyze done ===")


if __name__ == "__main__":
    main()
