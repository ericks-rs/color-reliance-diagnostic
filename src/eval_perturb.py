"""E2: eval checkpoint clean di bawah perturbasi warna test-time.
Tidak ada training ulang. Output: akurasi per kondisi perturbasi.

Baris: dataset, model, seed, condition, kind, param, n, acc, macro_f1, uar.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as data_mod
from . import models as models_mod
from . import perturb as perturb_mod
from .metrics import compute_metrics

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _predict_perturbed(model, loader, device, fn, mean, std):
    ys, ps = [], []
    for x, y in loader:                      # x in [0,1]
        x = x.to(device, non_blocking=True)
        x = fn(x)                            # perturbasi pada [0,1]
        x = (x - mean) / std                 # normalisasi ImageNet
        with torch.amp.autocast("cuda"):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def eval_perturb(cfg, dataset_name, arm, seed, model_key=None):
    """arm: nama output/checkpoint. model_key: builder timm (default=arm)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["datasets"][dataset_name]["num_classes"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / \
        f"{dataset_name}_{arm}_seed{seed}.pt"
    if model_key is None:
        model_key = arm
    model, _ = models_mod.load_checkpoint(ckpt_path, model_key, cfg,
                                          num_classes, device)

    ds, _ = data_mod.get_eval_dataset_with_ids(dataset_name, cfg, normalize=False)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    base = {"dataset": dataset_name, "model": arm, "seed": seed}
    rows = []

    # baseline clean (identity) sebagai referensi drop-curve
    y_true, y_pred = _predict_perturbed(model, loader, device,
                                        lambda x: x, mean, std)
    m = compute_metrics(y_true, y_pred, num_classes)
    rows.append({**base, "condition": "clean", "kind": "clean", "param": 0,
                 "n": m["n"], "acc": m["acc"], "macro_f1": m["macro_f1"],
                 "uar": m["uar"]})

    for cond_name, kind, param, fn in perturb_mod.build_conditions(cfg):
        yt, yp = _predict_perturbed(model, loader, device, fn, mean, std)
        mm = compute_metrics(yt, yp, num_classes)
        rows.append({**base, "condition": cond_name, "kind": kind,
                     "param": param if param is not None else "",
                     "n": mm["n"], "acc": mm["acc"], "macro_f1": mm["macro_f1"],
                     "uar": mm["uar"]})

    return pd.DataFrame(rows)
