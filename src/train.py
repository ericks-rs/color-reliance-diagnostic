"""Latih 1 (dataset, model, seed) -> simpan checkpoint val-acc terbaik.

RESEP TERKUNCI (identik semua model & dataset, lihat config.yaml train:):
AdamW lr=1e-4 wd=0.05, cosine + warmup 5ep, total 60ep, label_smoothing=0.1,
AMP, fine-tune full network, effective batch 64.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import data as data_mod
from . import models as models_mod
from .metrics import compute_metrics
from .utils import set_seed


def _cosine_warmup_lambda(warmup, total):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return compute_metrics(y_true, y_pred)


def _build_optimizer(model, recipe_cfg, lr):
    """Bangun optimizer dari paket recipe. Hanya optimizer+lr+wd yang beda antar arm."""
    opt = recipe_cfg["optimizer"].lower()
    wd = recipe_cfg["weight_decay"]
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=recipe_cfg.get("momentum", 0.9),
                               nesterov=recipe_cfg.get("nesterov", True),
                               weight_decay=wd)
    else:
        raise ValueError(f"unknown optimizer {opt}")


def train_one_run(cfg, dataset_name, model_key, seed, epochs=None, logger=None,
                  recipe="adamw", arm=None):
    """recipe: 'adamw' (default arm) | 'sgd' (ablation). arm: nama output/checkpoint
    (default = model_key utk adamw, atau '<model_key>_<recipe>' utk lainnya)."""
    t0 = time.time()
    tcfg = cfg["train"]
    # recipe pack: ambil dari cfg['recipes'][recipe], fallback ke train: block (adamw)
    rcfg = cfg.get("recipes", {}).get(recipe)
    if rcfg is None:
        rcfg = {"optimizer": tcfg["optimizer"], "lr": tcfg["lr"],
                "weight_decay": tcfg["weight_decay"]}
    if arm is None:
        arm = model_key if recipe == "adamw" else f"{model_key}_{recipe}"
    num_classes = cfg["datasets"][dataset_name]["num_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = epochs or tcfg["epochs"]
    warmup = tcfg["warmup_epochs"]
    accum = tcfg.get("grad_accum_steps", 1)
    bs = tcfg["batch_size"]
    nw = tcfg["num_workers"]
    base_lr = rcfg["lr"]
    lr_fallback = rcfg.get("lr_fallback", None)

    def log(msg):
        (logger.info if logger else print)(msg)

    set_seed(seed, deterministic=True)

    # data
    train_ds = data_mod.get_dataset(dataset_name, "train", cfg, train_aug=True)
    val_ds = data_mod.get_dataset(dataset_name, "val", cfg, train_aug=False)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=True, drop_last=True,
                          persistent_workers=(nw > 0))
    val_ld = DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=True,
                        persistent_workers=(nw > 0))

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{dataset_name}_{arm}_seed{seed}.pt"
    DIV_CHECK_STEPS = 30   # window deteksi divergensi di epoch 0

    # daftar lr untuk dicoba: base_lr, lalu lr_fallback (kalau divergen di awal)
    lr_attempts = [base_lr] + ([lr_fallback] if lr_fallback else [])

    result = None
    for attempt, cur_lr in enumerate(lr_attempts):
        set_seed(seed, deterministic=True)  # restart bersih
        model = models_mod.build_model(model_key, cfg, num_classes, pretrained=True)
        model = model.to(device)
        optimizer = _build_optimizer(model, rcfg, cur_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _cosine_warmup_lambda(warmup, epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=tcfg["amp"])

        best_acc, best_metrics, history = -1.0, None, []
        diverged = False
        log(f"[train] {dataset_name}/{arm}/seed{seed} recipe={recipe} "
            f"opt={rcfg['optimizer']} lr={cur_lr} wd={rcfg['weight_decay']} "
            f"epochs={epochs} device={device} n_train={len(train_ds)} "
            f"n_val={len(val_ds)} bs={bs} accum={accum}"
            + (f"  [RETRY lr_fallback]" if attempt > 0 else ""))

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running, nb = 0.0, 0
            pbar = tqdm(train_ld, desc=f"ep{epoch+1}/{epochs}", leave=False)
            for i, (x, y) in enumerate(pbar):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=tcfg["amp"]):
                    out = model(x)
                    loss = criterion(out, y) / accum
                # pengaman numerik: deteksi divergensi di awal epoch 0
                if epoch == 0 and i < DIV_CHECK_STEPS and not torch.isfinite(loss):
                    log(f"  !! loss non-finite di ep1 step{i} (lr={cur_lr}) -> divergen")
                    diverged = True
                    break
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += loss.item() * accum
                nb += 1
                pbar.set_postfix(loss=f"{running/nb:.3f}")
            if diverged:
                break
            scheduler.step()

            train_loss = running / max(1, nb)
            if epoch == 0 and not math.isfinite(train_loss):
                log(f"  !! train_loss NaN di ep1 (lr={cur_lr}) -> divergen")
                diverged = True
                break
            val_m = evaluate(model, val_ld, device)
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  ep{epoch+1}: train_loss={train_loss:.4f} "
                f"val_acc={val_m['acc']:.4f} val_f1={val_m['macro_f1']:.4f} "
                f"val_uar={val_m['uar']:.4f} lr={lr_now:.2e}")
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "lr": lr_now, **{f"val_{k}": v for k, v in val_m.items()}})

            if val_m["acc"] > best_acc:
                best_acc, best_metrics = val_m["acc"], val_m
                torch.save({"model_key": model_key, "arm": arm, "recipe": recipe,
                            "lr_used": cur_lr, "dataset": dataset_name,
                            "seed": seed, "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "val_metrics": val_m,
                            "num_classes": num_classes}, ckpt_path)

        if not diverged:
            result = {"best_acc": best_acc, "best_metrics": best_metrics,
                      "ckpt_path": str(ckpt_path), "history": history,
                      "arm": arm, "recipe": recipe, "lr_used": cur_lr}
            break
        else:
            log(f"  divergen pada lr={cur_lr}; "
                + ("coba lr_fallback..." if attempt < len(lr_attempts) - 1
                   else "TIDAK ada fallback lagi."))

    if result is None:
        raise RuntimeError(f"{dataset_name}/{arm}/seed{seed} divergen di semua lr {lr_attempts}")

    dt = time.time() - t0
    result["time_min"] = dt / 60
    log(f"[done] {arm} best_val_acc={result['best_acc']:.4f} "
        f"lr_used={result['lr_used']} ckpt={ckpt_path} time={dt/60:.1f}min")
    return result
