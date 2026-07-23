"""V7 per-species CR — dihitung PER-SEED lalu diagregat (mean ± SD, ddof=1).

Versi lama (results/per_species_CR_*.csv) mem-pool 5 seed jadi satu, sehingga tidak
punya estimasi ketidakpastian dan menyimpang dari konvensi tabel lain. Di sini tiap
(kelas, seed) dihitung dulu, baru diringkas.

Output tables/:
  V7_perspecies_CR_<ds>.csv        per kelas: clean/gray/CR mean+sd lintas 5 seed
  V7_perspecies_summary.csv        distribusi CR per (dataset, arm)
  perseed/V7_perspecies_perseed_<ds>.csv   nilai tiap (kelas, seed)

python revision/build_V7.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = ["flowers102", "cub200"]
ARMS = [("S1", "ResNet-50", "resnet50"), ("S2", "ConvNeXt-T", "convnext_tiny_in1k"),
        ("S3", "ViT-S", "vit_small_in1k"), ("S4", "Swin-T", "swin_tiny")]


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)
    pdir = tdir / "perseed"; pdir.mkdir(exist_ok=True)

    summary = []
    for ds in DS:
        per_seed, per_class = [], []
        for sid, name, arm in ARMS:
            files = sorted((rdir / "preds").glob(f"{ds}_{arm}_seed*.csv"))
            if not files:
                print(f"  SKIP {ds}/{arm}: preds tidak ada"); continue
            for f in files:
                seed = int(f.stem.split("seed")[-1])
                d = pd.read_csv(f)
                d["okc"] = (d.y_true == d.pred_clean).astype(float)
                d["okg"] = (d.y_true == d.pred_gray).astype(float)
                g = d.groupby("y_true").agg(clean=("okc", "mean"), gray=("okg", "mean"),
                                            n_img=("okc", "size")).reset_index()
                g["CR"] = g["clean"] - g["gray"]
                g.insert(0, "seed", seed); g.insert(0, "arm", arm)
                g.insert(0, "backbone", name); g.insert(0, "id", sid)
                g.insert(0, "dataset", ds)
                per_seed.append(g)
            ps = pd.concat(per_seed[-len(files):], ignore_index=True)
            agg = ps.groupby("y_true").agg(
                n_img=("n_img", "first"), n_seed=("seed", "nunique"),
                clean_mean=("clean", "mean"), clean_sd=("clean", lambda s: s.std(ddof=1)),
                gray_mean=("gray", "mean"), gray_sd=("gray", lambda s: s.std(ddof=1)),
                CR_mean=("CR", "mean"), CR_sd=("CR", lambda s: s.std(ddof=1))).reset_index()
            agg.insert(0, "arm", arm); agg.insert(0, "backbone", name)
            agg.insert(0, "id", sid); agg.insert(0, "dataset", ds)
            per_class.append(agg)

            c = agg["CR_mean"]
            summary.append(dict(dataset=ds, id=sid, backbone=name, n_class=len(c),
                                CR_mean=c.mean(), CR_median=c.median(),
                                CR_sd=c.std(ddof=1), CR_min=c.min(), CR_p25=c.quantile(.25),
                                CR_p75=c.quantile(.75), CR_max=c.max(),
                                n_class_CR_le0=int((c <= 0).sum()),
                                top1_class=int(c.idxmax()), top1_CR=c.max(),
                                bottom1_class=int(c.idxmin()), bottom1_CR=c.min()))

        pd.concat(per_class, ignore_index=True).to_csv(
            tdir / f"V7_perspecies_CR_{ds}.csv", index=False)
        pd.concat(per_seed, ignore_index=True).to_csv(
            pdir / f"V7_perspecies_perseed_{ds}.csv", index=False)
        print(f"  -> tables/V7_perspecies_CR_{ds}.csv")
        print(f"  -> tables/perseed/V7_perspecies_perseed_{ds}.csv")

    pd.DataFrame(summary).to_csv(tdir / "V7_perspecies_summary.csv", index=False)
    print("  -> tables/V7_perspecies_summary.csv")


if __name__ == "__main__":
    main()
