"""V4 + V5 per-bin LENGKAP: clean, grayscale, dan CR per bin colorfulness.

V4 = bin whole-image (P1).  V5 = bin foreground (P2).
Dihitung dari results/preds/ (per-image: y_true, pred_clean, pred_gray),
per-seed dulu lalu diagregat mean +- std (ddof=1) -> struktur reliance per bin
bisa dibandingkan whole-image vs foreground. Ini yang menjawab R2.2 + R1.7.

Output:
  tables/V4_perbin_wholeimg_<ds>.csv
  tables/V5_perbin_foreground_<ds>.csv
  tables/V4V5_perbin_compare.md   (side-by-side)

python revision/build_V4_V5.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src.colorfulness import stratify_tertiles

DS = [("flowers102", "Flowers-102"), ("cub200", "CUB-200")]
ARMS = [("ResNet-50", "resnet50"), ("ConvNeXt-T", "convnext_tiny_in1k"),
        ("ViT-S", "vit_small_in1k"), ("Swin-T", "swin_tiny")]
BINS = ["low", "mid", "high"]


def bin_maps(cfg, rdir, ds):
    """Return (whole_bins, fg_bins) sebagai Series image_id -> bin."""
    lo = cfg["colorfulness"]["tertile_low"]; hi = cfg["colorfulness"]["tertile_high"]
    whole = pd.read_csv(rdir / f"colorfulness_{ds}.csv").set_index("image_id")["bin"]
    fg = pd.read_csv(rdir / f"colorfulness_fg_{ds}.csv").rename(columns={"C_fg": "C"})
    fg = stratify_tertiles(fg, lo, hi).set_index("image_id")["bin"]
    return whole, fg


def per_seed_bins(rdir, ds, arm, binmap):
    """Return dict bin -> (clean list, gray list) per seed."""
    out = {b: {"clean": [], "gray": []} for b in BINS}
    for f in sorted((rdir / "preds").glob(f"{ds}_{arm}_seed*.csv")):
        d = pd.read_csv(f)
        d["bin"] = d.image_id.map(binmap)
        d["okc"] = (d.y_true == d.pred_clean).astype(float)
        d["okg"] = (d.y_true == d.pred_gray).astype(float)
        for b in BINS:
            s = d[d["bin"] == b]
            if len(s):
                out[b]["clean"].append(s.okc.mean())
                out[b]["gray"].append(s.okg.mean())
    return out


def rows_for(rdir, ds, binmap):
    rows = []
    for name, arm in ARMS:
        pb = per_seed_bins(rdir, ds, arm, binmap)
        for b in BINS:
            c = np.array(pb[b]["clean"]); g = np.array(pb[b]["gray"])
            if not len(c):
                continue
            cr = c - g
            rows.append(dict(dataset=ds, backbone=name, bin=b, n_seed=len(c),
                             clean=c.mean(), clean_sd=c.std(ddof=1),
                             gray=g.mean(), gray_sd=g.std(ddof=1),
                             CR=cr.mean(), CR_sd=cr.std(ddof=1)))
    return pd.DataFrame(rows)


def fmt(r, k):
    return f"{r[k]:.4f} ± {r[k+'_sd']:.4f}"


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)

    L = ["# V4 (whole-image bins) vs V5 (foreground bins) — per-bin clean / grayscale / CR", "",
         "Bins are colourfulness tertiles: V4 from the whole image (P1), V5 from the "
         "foreground region only (P2, dataset masks). Values are per-seed means "
         "aggregated over five seeds (± SD, ddof=1). CR = clean − grayscale within "
         "the same bin.", ""]

    for ds, lab in DS:
        whole, fg = bin_maps(cfg, rdir, ds)
        v4 = rows_for(rdir, ds, whole)
        v5 = rows_for(rdir, ds, fg)
        v4.to_csv(tdir / f"V4_perbin_wholeimg_{ds}.csv", index=False)
        v5.to_csv(tdir / f"V5_perbin_foreground_{ds}.csv", index=False)

        L += [f"## {lab}", "",
              "| backbone | bin | clean (whole) | gray (whole) | **CR whole** | clean (fg) | gray (fg) | **CR fg** | ΔCR (fg−whole) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, _ in ARMS:
            for b in BINS:
                a = v4[(v4.backbone == name) & (v4.bin == b)]
                c = v5[(v5.backbone == name) & (v5.bin == b)]
                if not len(a) or not len(c):
                    continue
                a = a.iloc[0]; c = c.iloc[0]
                L.append(f"| {name} | {b} | {fmt(a,'clean')} | {fmt(a,'gray')} | "
                         f"**{fmt(a,'CR')}** | {fmt(c,'clean')} | {fmt(c,'gray')} | "
                         f"**{fmt(c,'CR')}** | {c['CR']-a['CR']:+.4f} |")
        # trend high-low
        L += ["", f"### {lab} — CR trend (high bin − low bin)", "",
              "| backbone | CR trend whole | CR trend foreground |", "|---|---|---|"]
        for name, _ in ARMS:
            def tr(df):
                h = df[(df.backbone == name) & (df.bin == "high")]
                l = df[(df.backbone == name) & (df.bin == "low")]
                return (h.iloc[0]["CR"] - l.iloc[0]["CR"]) if len(h) and len(l) else np.nan
            L.append(f"| {name} | {tr(v4):+.4f} | {tr(v5):+.4f} |")
        L.append("")

    out = tdir / "V4V5_perbin_compare.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {out}")
    print(f"  -> tables/V4_perbin_wholeimg_*.csv, tables/V5_perbin_foreground_*.csv")


if __name__ == "__main__":
    main()
