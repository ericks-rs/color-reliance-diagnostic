"""M4 prep CUB-200 (CPU only): verifikasi dataset load + build colorfulness CSV.
Aman dijalankan paralel dgn training GPU (tidak pakai GPU)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root
from src import data as data_mod
from src import colorfulness as cf

chdir_to_root()
cfg = load_config()
name = "cub200"

print("=== CUB-200 data sanity ===")
for split in ["train", "test"]:
    ds = data_mod.get_dataset(name, split, cfg, train_aug=(split == "train"))
    print(f"  split {split}: {len(ds)} images")
img0, lbl0 = data_mod.get_dataset(name, "test", cfg, train_aug=False)[0]
print(f"  eval tensor shape: {tuple(img0.shape)}, label range example: {lbl0}")

# label sanity: harus 0..199
ds_test = data_mod.get_dataset(name, "test", cfg, train_aug=False)
labels = [int(ds_test.data.iloc[i].target) - 1 for i in range(len(ds_test))]
print(f"  label min={min(labels)} max={max(labels)} n_classes={len(set(labels))}")

print("\n=== colorfulness CUB (test) ===")
df, out_path = cf.build_colorfulness_csv(name, cfg)
c = df["C"].values
print(f"  saved -> {out_path} ({len(df)} rows)")
print(f"  C: min={c.min():.2f} max={c.max():.2f} mean={c.mean():.2f} "
      f"median={np.median(c):.2f} std={c.std():.2f}")
print(f"  tertile: t_low={df.attrs['t_low']:.3f} t_high={df.attrs['t_high']:.3f}")
print("  bin counts:")
print(df["bin"].value_counts().reindex(["low", "mid", "high"]).to_string())
print("\n=== CUB prep done ===")
