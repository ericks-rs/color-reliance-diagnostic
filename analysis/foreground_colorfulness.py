"""R2.2: colorfulness FOREGROUND (objek), bukan seluruh citra (yg bisa didominasi latar).

CUB   -> crop ke bounding box (bounding_boxes.txt).
Flowers -> mask background pakai segmim_XXXXX.jpg (Oxford VGG). Foreground = piksel
           non-background. Warna background di-deteksi dari modus sudut (robust).

Output per dataset:
  results/colorfulness_fg_{ds}.csv  [image_id, C_fg]
Lalu banding vs whole-image (results/colorfulness_{ds}.csv):
  - Pearson & Spearman r (C_fg vs C_whole)
  - re-stratifikasi tertile pada C_fg -> bin_fg; hitung % gambar yang TETAP di bin sama
Kalau r tinggi & bin agreement tinggi -> stratifikasi BUKAN artefak latar (jawab R2.2
tanpa perlu re-eval). Kalau rendah -> perlu re-eval per-bin foreground.

Usage: python revision/foreground_colorfulness.py --datasets cub200 flowers102
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src import data as data_mod
from src.colorfulness import hasler_susstrunk, stratify_tertiles


def hs_masked(img_uint8, mask=None):
    """Hasler-Susstrunk di piksel mask==True saja (mask None -> semua piksel)."""
    img = img_uint8.astype(np.float64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    if mask is not None:
        rg, yb = rg[mask], yb[mask]
    if rg.size < 10:               # foreground terlalu kecil -> fallback seluruh citra
        return hasler_susstrunk(img_uint8)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


# ---------- CUB: bbox ----------
def cub_fg(cfg):
    root = Path(cfg["datasets"]["cub200"]["root"])
    images = pd.read_csv(root / "images.txt", sep=" ", names=["iid", "fp"])
    bbox = pd.read_csv(root / "bounding_boxes.txt", sep=" ",
                       names=["iid", "x", "y", "w", "h"])
    fp2box = images.merge(bbox, on="iid").set_index("fp")[["x", "y", "w", "h"]]

    rows = []
    for image_id, path in tqdm(data_mod.list_test_images("cub200", cfg), desc="cub fg"):
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]
        b = fp2box.loc[image_id]
        x, y = int(round(b.x)), int(round(b.y))
        x2, y2 = int(round(b.x + b.w)), int(round(b.y + b.h))
        x, y = max(0, x), max(0, y)
        x2, y2 = min(W, x2), min(H, y2)
        crop = img[y:y2, x:x2]
        C = hasler_susstrunk(crop) if crop.size >= 30 else hasler_susstrunk(img)
        rows.append({"image_id": image_id, "C_fg": C})
    return pd.DataFrame(rows)


# ---------- Flowers: segmim mask ----------
def _bg_mask(segmim):
    """Background = warna dominan di 4 sudut (patch kecil). Return mask FOREGROUND."""
    h, w = segmim.shape[:2]
    k = max(2, min(h, w) // 20)
    corners = np.concatenate([
        segmim[:k, :k].reshape(-1, 3), segmim[:k, -k:].reshape(-1, 3),
        segmim[-k:, :k].reshape(-1, 3), segmim[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(segmim.astype(np.float64) - bg, axis=-1)
    fg = dist > 30.0                      # ambang jarak warna dari background
    return fg


def flowers_fg(cfg):
    root = Path(cfg["datasets"]["flowers102"]["root"])
    segdir = root / "flowers-102" / "segmim"
    if not segdir.exists():
        segdir = root / "segmim"       # fallback kalau root sudah menunjuk flowers-102/
    if not segdir.exists():
        print(f"  [SKIP flowers] segmim/ belum ada di {segdir} (download dulu)")
        return None
    rows = []
    for image_id, path in tqdm(data_mod.list_test_images("flowers102", cfg), desc="flowers fg"):
        num = image_id.replace("image_", "").replace(".jpg", "")
        segp = segdir / f"segmim_{num}.jpg"
        img = np.asarray(Image.open(path).convert("RGB"))
        if not segp.exists():
            rows.append({"image_id": image_id, "C_fg": hasler_susstrunk(img)})
            continue
        seg = np.asarray(Image.open(segp).convert("RGB"))
        if seg.shape[:2] != img.shape[:2]:
            seg = np.asarray(Image.open(segp).convert("RGB").resize(
                (img.shape[1], img.shape[0])))
        fg = _bg_mask(seg)
        rows.append({"image_id": image_id, "C_fg": hs_masked(img, fg)})
    return pd.DataFrame(rows)


def compare(ds, df_fg, cfg):
    rdir = Path(cfg["paths"]["results"])
    df_fg.to_csv(rdir / f"colorfulness_fg_{ds}.csv", index=False)
    whole = pd.read_csv(rdir / f"colorfulness_{ds}.csv")   # image_id, C, bin
    m = whole.merge(df_fg, on="image_id", how="inner")
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(m["C"], m["C_fg"])[0]
    sr = spearmanr(m["C"], m["C_fg"])[0]
    # re-stratifikasi fg
    lo = cfg["colorfulness"]["tertile_low"]
    hi = cfg["colorfulness"]["tertile_high"]
    fgb = stratify_tertiles(m.rename(columns={"C_fg": "C", "C": "C_whole"}), lo, hi)
    m["bin_fg"] = fgb["bin"].values
    agree = float((m["bin"] == m["bin_fg"]).mean())
    print(f"\n=== {ds} foreground vs whole-image colorfulness (n={len(m)}) ===")
    print(f"  Pearson r = {pr:.4f} | Spearman r = {sr:.4f}")
    print(f"  bin agreement (whole vs fg tertiles) = {agree*100:.1f}%")
    ct = pd.crosstab(m["bin"], m["bin_fg"])
    print("  crosstab whole(row) x fg(col):")
    print(ct.to_string())
    return {"ds": ds, "pearson": pr, "spearman": sr, "bin_agree": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub200", "flowers102"])
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    summ = []
    for ds in args.datasets:
        df_fg = cub_fg(cfg) if ds == "cub200" else flowers_fg(cfg)
        if df_fg is None:
            continue
        summ.append(compare(ds, df_fg, cfg))
    print("\n=== RINGKASAN R2.2 ===")
    for s in summ:
        verdict = ("ROBUST (bukan artefak latar)" if s["spearman"] > 0.8 and s["bin_agree"] > 0.7
                   else "PERLU re-eval per-bin foreground")
        print(f"  {s['ds']}: Spearman={s['spearman']:.3f} agree={s['bin_agree']*100:.1f}% -> {verdict}")


if __name__ == "__main__":
    main()
