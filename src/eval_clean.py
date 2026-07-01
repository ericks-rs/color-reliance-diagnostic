"""E1: eval clean test set + akurasi per colorfulness-bin (low/mid/high).

Output baris: dataset, model, seed, bin, n, acc, macro_f1, uar.
'bin' juga punya nilai 'overall' (semua sampel).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as data_mod
from . import models as models_mod
from .metrics import compute_metrics, confusion


@torch.no_grad()
def _predict(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def eval_clean(cfg, dataset_name, arm, seed, model_key=None, save_confusion=True):
    """arm: nama output/checkpoint (mis. 'resnet50' atau 'resnet50_sgd').
    model_key: builder timm (default=arm; isi terpisah utk arm ablation)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["datasets"][dataset_name]["num_classes"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / \
        f"{dataset_name}_{arm}_seed{seed}.pt"
    if model_key is None:
        model_key = arm
    model, _ = models_mod.load_checkpoint(ckpt_path, model_key, cfg,
                                          num_classes, device)

    ds, ids = data_mod.get_eval_dataset_with_ids(dataset_name, cfg, normalize=True)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    y_true, y_pred = _predict(model, loader, device)

    # join colorfulness bin
    cf_path = Path(cfg["paths"]["results"]) / f"colorfulness_{dataset_name}.csv"
    cf = pd.read_csv(cf_path).set_index("image_id")
    bins = np.array([cf.loc[i, "bin"] if i in cf.index else "NA" for i in ids])

    rows = []
    base = {"dataset": dataset_name, "model": arm, "seed": seed}
    # overall
    m = compute_metrics(y_true, y_pred, num_classes)
    rows.append({**base, "bin": "overall", "n": m["n"], "acc": m["acc"],
                 "macro_f1": m["macro_f1"], "uar": m["uar"]})
    # per bin
    for b in ["low", "mid", "high"]:
        mask = bins == b
        if mask.sum() == 0:
            continue
        mb = compute_metrics(y_true[mask], y_pred[mask], num_classes)
        rows.append({**base, "bin": b, "n": mb["n"], "acc": mb["acc"],
                     "macro_f1": mb["macro_f1"], "uar": mb["uar"]})

    if save_confusion:
        cm = confusion(y_true, y_pred, num_classes)
        cm_dir = Path(cfg["paths"]["results"]) / "confusion"
        cm_dir.mkdir(parents=True, exist_ok=True)
        np.save(cm_dir / f"{dataset_name}_{arm}_seed{seed}.npy", cm)

    return pd.DataFrame(rows)
