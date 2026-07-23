"""V5 (per-bin foreground) + V7 (per-species CR) dari dump prediksi per-image
(results/preds/, dihasilkan eval_predictions.py). Murni pandas, non-GPU.

  python revision/eval_from_preds.py --task v7      # per-species CR -> R2.4
  python revision/eval_from_preds.py --task v5      # per-bin foreground -> R2.2
  python revision/eval_from_preds.py --task both

V7: group preds by y_true (=label kelas) -> acc(clean), acc(gray), CR per kelas per arm.
    + top/bottom species + korelasi colorfulness-kelas vs CR.
V5: join image_id -> bin foreground (re-stratifikasi tertile colorfulness_fg) -> acc per bin.
    dibanding per-bin whole-image (V4) sebagai cek robustness R2.2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src.colorfulness import stratify_tertiles

S1_S4 = ["resnet50", "convnext_tiny_in1k", "vit_small_in1k", "swin_tiny"]
LABEL = {"resnet50": "ResNet-50", "convnext_tiny_in1k": "ConvNeXt-T",
         "vit_small_in1k": "ViT-S", "swin_tiny": "Swin-T"}
DS = ["cub200", "flowers102"]


def load_preds(rdir, ds, arm):
    fs = sorted((rdir / "preds").glob(f"{ds}_{arm}_seed*.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def v7(cfg, rdir, tdir):
    print("=== V7 per-species CR ===")
    for ds in DS:
        cf = pd.read_csv(rdir / f"colorfulness_{ds}.csv").set_index("image_id")["C"]
        allc = []
        for arm in S1_S4:
            df = load_preds(rdir, ds, arm)
            if df is None:
                print(f"  {ds}/{arm}: preds belum ada (jalankan eval_predictions dulu)"); continue
            df["ok_c"] = (df.y_true == df.pred_clean).astype(float)
            df["ok_g"] = (df.y_true == df.pred_gray).astype(float)
            g = df.groupby("y_true").agg(clean=("ok_c", "mean"),
                                         gray=("ok_g", "mean"), n=("ok_c", "size"))
            g["CR"] = g["clean"] - g["gray"]
            g["arm"] = arm
            allc.append(g.reset_index())
            # korelasi colorfulness kelas vs CR
            cls_cf = df.assign(C=df.image_id.map(cf)).groupby("y_true")["C"].mean()
            r = np.corrcoef(cls_cf.reindex(g.index).values, g["CR"].values)[0, 1]
            top = g["CR"].sort_values(ascending=False).head(3).round(3).to_dict()
            print(f"  {ds}/{LABEL[arm]}: mean CR={g.CR.mean():.3f} "
                  f"corr(colorfulness_kelas,CR)={r:.3f} top-CR classes={top}")
        if allc:
            out = pd.concat(allc, ignore_index=True)
            p = rdir / f"per_species_CR_{ds}.csv"
            out.to_csv(p, index=False)
            print(f"  -> {p}")


def v5(cfg, rdir, tdir):
    print("=== V5 per-bin foreground (vs whole-image) ===")
    lo = cfg["colorfulness"]["tertile_low"]; hi = cfg["colorfulness"]["tertile_high"]
    for ds in DS:
        fgp = rdir / f"colorfulness_fg_{ds}.csv"
        if not fgp.exists():
            print(f"  {ds}: colorfulness_fg belum ada (P2)"); continue
        fg = pd.read_csv(fgp).rename(columns={"C_fg": "C"})
        fg = stratify_tertiles(fg, lo, hi)[["image_id", "bin"]].rename(
            columns={"bin": "bin_fg"}).set_index("image_id")
        rows = []
        for arm in S1_S4:
            df = load_preds(rdir, ds, arm)
            if df is None:
                print(f"  {ds}/{arm}: preds belum ada"); continue
            df["bin_fg"] = df.image_id.map(fg["bin_fg"])
            df["ok_c"] = (df.y_true == df.pred_clean).astype(float)
            for b in ["low", "mid", "high"]:
                sub = df[df.bin_fg == b]
                if len(sub):
                    rows.append({"dataset": ds, "arm": arm, "bin": b,
                                 "n": len(sub) // 5, "clean_acc_fg": sub.ok_c.mean()})
        if rows:
            out = pd.DataFrame(rows)
            p = tdir / f"V5_perbin_foreground_{ds}.csv"
            out.to_csv(p, index=False)
            print(f"  -> {p}")
            for arm in S1_S4:
                a = out[out.arm == arm]
                if len(a):
                    accs = {r["bin"]: f"{r['clean_acc_fg']:.3f}" for _, r in a.iterrows()}
                    print(f"    {LABEL.get(arm,arm)}: {accs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["v5", "v7", "both"], default="both")
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)
    if args.task in ("v7", "both"):
        v7(cfg, rdir, tdir)
    if args.task in ("v5", "both"):
        v5(cfg, rdir, tdir)


if __name__ == "__main__":
    main()
