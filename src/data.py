"""Dataset + transforms.

PENTING (rigor rule #3): train transform TANPA color augmentation.
Hanya RandomResizedCrop + HorizontalFlip. Dilarang ColorJitter / RandAugment /
hue / saturation / grayscale / AutoAugment.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import Flowers102

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size=224, rrc_scale=(0.6, 1.0), grayscale=False):
    """NO COLOR AUG. Hanya crop geometris + flip.
    grayscale=True (arm R2.3): buang warna saat TRAINING (luminance -> 3 channel).
    Ini SATU-SATUNYA tempat grayscale boleh masuk training (ablation eksplisit)."""
    ops = [T.RandomResizedCrop(image_size, scale=tuple(rrc_scale)),
           T.RandomHorizontalFlip(0.5)]
    if grayscale:
        ops.append(T.Grayscale(num_output_channels=3))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def build_eval_transform(resize=256, center_crop=224, normalize=True,
                         grayscale=False):
    """normalize=False -> tensor [0,1] tanpa Normalize (buat perturbasi warna E2,
    perturbasi diterapkan pada RGB [0,1] sebelum normalisasi).
    grayscale=True: dipakai HANYA utk val-loader arm grayscale-training (biar
    pemilihan checkpoint konsisten dgn distribusi training grayscale)."""
    ops = [T.Resize(resize), T.CenterCrop(center_crop)]
    if grayscale:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.ToTensor())
    if normalize:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


# --------------------------------------------------------------------------
# CUB-200-2011
# --------------------------------------------------------------------------
class Cub2011(Dataset):
    """Caltech-UCSD Birds 200-2011. Baca images.txt / train_test_split.txt /
    image_class_labels.txt dari folder CUB_200_2011/.

    split in {train, val, test}. CUB tidak punya val resmi. Kalau val_per_class>0,
    val di-carve dari porsi TRAIN (is_training_img==1) secara per-kelas dgn
    split_seed TETAP (independen dari seed model), sisanya jadi train. Test
    (is_training_img==0) TIDAK PERNAH disentuh utk pemilihan checkpoint.
    Kalau val_per_class==0 -> perilaku lama (val == test) utk reproduksi baseline.
    """

    def __init__(self, root, split="train", transform=None, train=None,
                 val_per_class=0, split_seed=12345):
        self.root = Path(root)
        self.transform = transform
        # kompat lama: train=True/False -> split train/test
        if train is not None:
            split = "train" if train else "test"
        assert split in ("train", "val", "test"), split
        self.split = split
        self.val_per_class = int(val_per_class)
        self.split_seed = int(split_seed)
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(self.root / "images.txt", sep=" ",
                             names=["img_id", "filepath"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ",
                             names=["img_id", "target"])
        split = pd.read_csv(self.root / "train_test_split.txt", sep=" ",
                            names=["img_id", "is_training_img"])
        data = images.merge(labels, on="img_id").merge(split, on="img_id")

        if self.split == "test":
            self.data = data[data.is_training_img == 0].reset_index(drop=True)
            return

        trainval = data[data.is_training_img == 1].reset_index(drop=True)
        if self.val_per_class <= 0:
            # baseline lama: 'val' dipetakan ke test, 'train' = seluruh trainval
            if self.split == "val":
                self.data = data[data.is_training_img == 0].reset_index(drop=True)
            else:
                self.data = trainval
            return

        # carve val per-kelas dgn RNG tetap (deterministik lintas run/model-seed)
        rng = np.random.RandomState(self.split_seed)
        val_ids = []
        for _, grp in trainval.groupby("target"):
            idx = grp.index.to_numpy().copy()
            rng.shuffle(idx)
            k = min(self.val_per_class, max(0, len(idx) - 1))  # sisakan >=1 utk train
            val_ids.extend(idx[:k].tolist())
        val_mask = trainval.index.isin(val_ids)
        if self.split == "val":
            self.data = trainval[val_mask].reset_index(drop=True)
        else:
            self.data = trainval[~val_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def _path(self, idx):
        return self.root / "images" / self.data.iloc[idx].filepath

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(self._path(idx)).convert("RGB")
        target = int(row.target) - 1  # 1-indexed -> 0-indexed
        if self.transform:
            img = self.transform(img)
        return img, target


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def get_dataset(name, split, cfg, train_aug, grayscale=False):
    """Return torch Dataset. split in {train, val, test}.
    train_aug=True -> train transform (no color aug); else eval transform.
    grayscale=True (arm R2.3): input di-grayscale (train + val loader).
    """
    tcfg = cfg["train"]
    ecfg = cfg["eval"]
    transform = (build_train_transform(tcfg["image_size"], tcfg["rrc_scale"],
                                       grayscale=grayscale)
                 if train_aug else
                 build_eval_transform(ecfg["resize"], ecfg["center_crop"],
                                      grayscale=grayscale))

    if name == "flowers102":
        root = cfg["datasets"]["flowers102"]["root"]
        return Flowers102(root=root, split=split, transform=transform,
                          download=True)
    elif name == "cub200":
        root = cfg["datasets"]["cub200"]["root"]
        vcfg = _cub_val_cfg(cfg)
        return Cub2011(root=root, split=split, transform=transform,
                       val_per_class=vcfg["per_class"],
                       split_seed=vcfg["split_seed"])
    else:
        raise ValueError(f"unknown dataset {name}")


def _cub_val_cfg(cfg):
    """Baca konfigurasi val-split CUB. Default: nonaktif (val==test, baseline)."""
    v = (cfg.get("datasets", {}).get("cub200", {}).get("val_split")
         or cfg.get("cub_val") or {})
    enabled = bool(v.get("enabled", False))
    return {"per_class": int(v.get("per_class", 5)) if enabled else 0,
            "split_seed": int(v.get("split_seed", 12345))}


def get_eval_dataset_with_ids(name, cfg, normalize=True):
    """Return (dataset, ids) untuk SPLIT TEST. ids urut sama dgn dataset
    (DataLoader shuffle=False jaga urutan), dipakai utk join colorfulness bin.
    normalize=False -> tensor [0,1] (buat E2 perturbasi)."""
    ecfg = cfg["eval"]
    transform = build_eval_transform(ecfg["resize"], ecfg["center_crop"], normalize)
    if name == "flowers102":
        root = cfg["datasets"]["flowers102"]["root"]
        ds = Flowers102(root=root, split="test", transform=transform, download=True)
        ids = [Path(p).name for p in ds._image_files]
        return ds, ids
    elif name == "cub200":
        root = cfg["datasets"]["cub200"]["root"]
        ds = Cub2011(root=root, train=False, transform=transform)
        ids = [ds.data.iloc[i].filepath for i in range(len(ds))]
        return ds, ids
    else:
        raise ValueError(f"unknown dataset {name}")


def list_test_images(name, cfg):
    """Return list of (image_id, abs_path) untuk SPLIT TEST.
    Dipakai colorfulness.py — baca raw RGB tanpa transform."""
    if name == "flowers102":
        root = cfg["datasets"]["flowers102"]["root"]
        ds = Flowers102(root=root, split="test", download=True)
        out = []
        for p in ds._image_files:
            p = Path(p)
            out.append((p.name, str(p)))  # image_id = filename
        return out
    elif name == "cub200":
        root = cfg["datasets"]["cub200"]["root"]
        ds = Cub2011(root=root, train=False)
        out = []
        for i in range(len(ds)):
            row = ds.data.iloc[i]
            out.append((row.filepath, str(ds._path(i))))
        return out
    else:
        raise ValueError(f"unknown dataset {name}")
