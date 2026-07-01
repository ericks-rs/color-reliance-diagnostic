"""Dataset + transforms.

PENTING (rigor rule #3): train transform TANPA color augmentation.
Hanya RandomResizedCrop + HorizontalFlip. Dilarang ColorJitter / RandAugment /
hue / saturation / grayscale / AutoAugment.
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import Flowers102

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size=224, rrc_scale=(0.6, 1.0)):
    """NO COLOR AUG. Hanya crop geometris + flip."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=tuple(rrc_scale)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(resize=256, center_crop=224, normalize=True):
    """normalize=False -> tensor [0,1] tanpa Normalize (buat perturbasi warna E2,
    perturbasi diterapkan pada RGB [0,1] sebelum normalisasi)."""
    ops = [T.Resize(resize), T.CenterCrop(center_crop), T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


# --------------------------------------------------------------------------
# CUB-200-2011
# --------------------------------------------------------------------------
class Cub2011(Dataset):
    """Caltech-UCSD Birds 200-2011. Baca images.txt / train_test_split.txt /
    image_class_labels.txt dari folder CUB_200_2011/."""

    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.train = train
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(self.root / "images.txt", sep=" ",
                             names=["img_id", "filepath"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ",
                             names=["img_id", "target"])
        split = pd.read_csv(self.root / "train_test_split.txt", sep=" ",
                            names=["img_id", "is_training_img"])
        data = images.merge(labels, on="img_id").merge(split, on="img_id")
        if self.train:
            self.data = data[data.is_training_img == 1].reset_index(drop=True)
        else:
            self.data = data[data.is_training_img == 0].reset_index(drop=True)

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
def get_dataset(name, split, cfg, train_aug):
    """Return torch Dataset. split in {train, val, test}.
    train_aug=True -> train transform (no color aug); else eval transform.
    """
    tcfg = cfg["train"]
    ecfg = cfg["eval"]
    transform = (build_train_transform(tcfg["image_size"], tcfg["rrc_scale"])
                 if train_aug else
                 build_eval_transform(ecfg["resize"], ecfg["center_crop"]))

    if name == "flowers102":
        root = cfg["datasets"]["flowers102"]["root"]
        return Flowers102(root=root, split=split, transform=transform,
                          download=True)
    elif name == "cub200":
        root = cfg["datasets"]["cub200"]["root"]
        # CUB cuma punya train/test. val = test (dipakai utk pilih best ckpt).
        is_train = (split == "train")
        return Cub2011(root=root, train=is_train, transform=transform)
    else:
        raise ValueError(f"unknown dataset {name}")


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
