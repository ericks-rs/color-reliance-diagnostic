"""Hasler & Susstrunk (2003) colorfulness metric + tertile stratification.

C = sqrt(std(rg)^2 + std(yb)^2) + 0.3*sqrt(mean(rg)^2 + mean(yb)^2)
  rg = R - G
  yb = 0.5*(R + G) - B

Dihitung pada citra RGB MENTAH (0-255, sebelum normalisasi), pada resolusi asli.

RIGOR (rule #2): metrik & stratifikasi ini HANYA untuk grouping evaluasi (E1).
TIDAK PERNAH menyentuh training, sampling, atau loss.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def hasler_susstrunk(img_uint8: np.ndarray) -> float:
    """img_uint8: HxWx3 RGB array, 0-255."""
    img = img_uint8.astype(np.float64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def colorfulness_of_path(path: str) -> float:
    img = Image.open(path).convert("RGB")
    return hasler_susstrunk(np.asarray(img))


def compute_colorfulness_df(image_list) -> pd.DataFrame:
    """image_list: list of (image_id, path). Return df[image_id, C]."""
    rows = []
    for image_id, path in tqdm(image_list, desc="colorfulness"):
        rows.append({"image_id": image_id, "C": colorfulness_of_path(path)})
    return pd.DataFrame(rows)


def stratify_tertiles(df: pd.DataFrame, low_q=0.333, high_q=0.667) -> pd.DataFrame:
    """Tambah kolom 'bin' (low/mid/high) + simpan ambang ke attrs."""
    c = df["C"].values
    t_low = float(np.quantile(c, low_q))
    t_high = float(np.quantile(c, high_q))

    df = df.copy()
    df["bin"] = df["C"].apply(lambda x: "low" if x <= t_low
                              else ("mid" if x <= t_high else "high"))
    df.attrs["t_low"] = t_low
    df.attrs["t_high"] = t_high
    return df


def build_colorfulness_csv(name, cfg, out_path=None, low_q=None, high_q=None):
    """Pipeline lengkap untuk satu dataset (test split)."""
    from . import data as data_mod  # local import to avoid cycles
    if low_q is None:
        low_q = cfg["colorfulness"]["tertile_low"]
    if high_q is None:
        high_q = cfg["colorfulness"]["tertile_high"]

    image_list = data_mod.list_test_images(name, cfg)
    df = compute_colorfulness_df(image_list)
    df = stratify_tertiles(df, low_q, high_q)

    if out_path is None:
        out_path = Path(cfg["paths"]["results"]) / f"colorfulness_{name}.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df, out_path
