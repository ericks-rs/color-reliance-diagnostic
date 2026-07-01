"""M1: Flowers-102 data + colorfulness pipeline + visualisasi distribusi & contoh."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config
from src import colorfulness as cf
from src import data as data_mod

cfg = load_config()
name = "flowers102"

print("=== M1: Flowers-102 ===")
# trigger download + sanity check split sizes
for split in ["train", "val", "test"]:
    ds = data_mod.get_dataset(name, split, cfg, train_aug=(split == "train"))
    print(f"  split {split}: {len(ds)} images")

# sanity: 1 batch shape
ds_test = data_mod.get_dataset(name, "test", cfg, train_aug=False)
img0, lbl0 = ds_test[0]
print(f"  eval tensor shape: {tuple(img0.shape)}, label example: {lbl0}")

# colorfulness on test set
print("\n[colorfulness] computing on TEST split...")
df, out_path = cf.build_colorfulness_csv(name, cfg)
print(f"  saved -> {out_path}  ({len(df)} rows)")

c = df["C"].values
print("\n[distribusi C]")
print(f"  n={len(c)}  min={c.min():.2f}  max={c.max():.2f}  "
      f"mean={c.mean():.2f}  median={np.median(c):.2f}  std={c.std():.2f}")
print(f"  tertile thresholds: t_low(33.3%)={df.attrs['t_low']:.3f}  "
      f"t_high(66.7%)={df.attrs['t_high']:.3f}")
print("\n[bin counts]")
print(df["bin"].value_counts().reindex(["low", "mid", "high"]).to_string())

# --- Figure: histogram distribusi C + ambang ---
fig_dir = Path(cfg["paths"]["figures"]); fig_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 4))
plt.hist(c, bins=50, color="#4C72B0", alpha=0.85)
plt.axvline(df.attrs["t_low"], color="red", ls="--", label=f"t_low={df.attrs['t_low']:.1f}")
plt.axvline(df.attrs["t_high"], color="green", ls="--", label=f"t_high={df.attrs['t_high']:.1f}")
plt.xlabel("Colorfulness C (Hasler-Susstrunk)")
plt.ylabel("count")
plt.title(f"Flowers-102 test colorfulness distribution (n={len(c)})")
plt.legend()
plt.tight_layout()
hist_path = fig_dir / "m1_colorfulness_hist_flowers102.png"
plt.savefig(hist_path, dpi=130)
plt.close()
print(f"\n[figure] histogram -> {hist_path}")

# --- Figure: contoh gambar low vs high colorfulness ---
id2path = {iid: p for iid, p in data_mod.list_test_images(name, cfg)}
low_ex = df.nsmallest(4, "C")
high_ex = df.nlargest(4, "C")

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for j, (_, r) in enumerate(low_ex.iterrows()):
    ax = axes[0, j]
    ax.imshow(Image.open(id2path[r.image_id]).convert("RGB"))
    ax.set_title(f"LOW C={r.C:.1f}", fontsize=10)
    ax.axis("off")
for j, (_, r) in enumerate(high_ex.iterrows()):
    ax = axes[1, j]
    ax.imshow(Image.open(id2path[r.image_id]).convert("RGB"))
    ax.set_title(f"HIGH C={r.C:.1f}", fontsize=10)
    ax.axis("off")
fig.suptitle("Flowers-102: lowest (top) vs highest (bottom) colorfulness", fontsize=12)
plt.tight_layout()
ex_path = fig_dir / "m1_examples_lowhigh_flowers102.png"
plt.savefig(ex_path, dpi=120)
plt.close()
print(f"[figure] examples -> {ex_path}")
print("\n=== M1 done ===")
