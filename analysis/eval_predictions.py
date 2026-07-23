"""Fondasi V5/V7 (dan V4): dump prediksi PER-IMAGE tiap arm S1-S4 di bawah
clean + grayscale (grayscale-fn identik dgn V2/eval_perturb, biar CR konsisten).

Output: results/preds/<ds>_<arm>_seed<s>.csv  kolom [image_id,y_true,pred_clean,pred_gray]

Dari sini (pandas, non-GPU):
  V7 per-species CR = per-kelas acc(clean) - acc(gray)
  V5 per-bin foreground = join image_id -> bin foreground (colorfulness_fg) -> acc per bin
  V4 per-bin whole = join -> colorfulness (whole) -> acc per bin (cek vs e1_clean)

WAJIB pakai checkpoint FINAL (setelah consist). Jalan dari color_complexity/.
  python revision/eval_predictions.py --arms resnet50 convnext_tiny_in1k vit_small_in1k swin_tiny
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src import data as data_mod
from src import models as models_mod
from src import perturb as perturb_mod

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
S1_S4 = ["resnet50", "convnext_tiny_in1k", "vit_small_in1k", "swin_tiny"]


@torch.no_grad()
def _preds(model, loader, device, fn, mean, std):
    ps = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x = (fn(x) - mean) / std
        with torch.amp.autocast("cuda"):
            ps.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(ps)


def run(cfg, ds, arm, seed, gray_fn, device):
    num_classes = cfg["datasets"][ds]["num_classes"]
    ckpt = Path(cfg["paths"]["checkpoints"]) / f"{ds}_{arm}_seed{seed}.pt"
    if not ckpt.exists():
        print(f"  SKIP {ds}/{arm}/seed{seed}: checkpoint belum ada"); return None
    model, _ = models_mod.load_checkpoint(ckpt, arm, cfg, num_classes, device)
    dset, ids = data_mod.get_eval_dataset_with_ids(ds, cfg, normalize=False)
    loader = DataLoader(dset, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    ys = np.concatenate([y.numpy() for _, y in DataLoader(
        dset, batch_size=cfg["eval"]["batch_size"], shuffle=False)])
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    pc = _preds(model, loader, device, lambda x: x, mean, std)
    pg = _preds(model, loader, device, gray_fn, mean, std)
    return pd.DataFrame({"image_id": ids, "y_true": ys,
                         "pred_clean": pc, "pred_gray": pg})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=S1_S4)
    ap.add_argument("--datasets", nargs="+", default=["cub200", "flowers102"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(cfg["paths"]["results"]) / "preds"
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        gray_fn = {c[0]: c[3] for c in perturb_mod.build_conditions(cfg)}["grayscale"]
        for arm in args.arms:
            for s in args.seeds:
                df = run(cfg, ds, arm, s, gray_fn, device)
                if df is not None:
                    p = outdir / f"{ds}_{arm}_seed{s}.csv"
                    df.to_csv(p, index=False)
                    acc_c = (df.y_true == df.pred_clean).mean()
                    acc_g = (df.y_true == df.pred_gray).mean()
                    print(f"  {p.name}: clean={acc_c:.4f} gray={acc_g:.4f} CR={acc_c-acc_g:.4f}")
    print("done -> results/preds/")


if __name__ == "__main__":
    main()
