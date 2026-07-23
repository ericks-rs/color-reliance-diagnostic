"""Arsip per-run permanen: tiap arm simpan snapshot KODE + CONFIG + LOG + HASIL
sendiri, di folder unik (timestamp+code-hash) -> TIDAK ketimpa run baru.

Struktur (nested, rapi buat GitHub):
  runs_archive/
    _code/<hash8>/            snapshot kode+config (1 per versi kode; dedup by hash)
      run_all.py, config.yaml, src/*.py, MANIFEST.txt
    <dataset>/<arm>/<ts>_<hash8>/   satu folder per invokasi run
      code_hash.txt           -> nunjuk snapshot kode mana
      train_summary_<ds>.csv   baris arm ini (subset)
      e1_clean_<ds>.csv        baris arm ini
      e2_perturb_<ds>.csv      baris arm ini
      train.log                log run (kalau ada)

Dipakai run_all.py. Semua dibungkus try/except di pemanggil -> arsip GAGAL != run gagal.
"""
import hashlib
import shutil
from pathlib import Path

import pandas as pd

CODE_FILES = [
    "run_all.py", "config.yaml",
    "src/train.py", "src/data.py", "src/models.py",
    "src/eval_clean.py", "src/eval_perturb.py", "src/colorfulness.py",
    "src/metrics.py", "src/stats.py", "src/utils.py",
]


def _hash_code(root: Path):
    h = hashlib.md5()
    parts = []
    for f in CODE_FILES:
        p = root / f
        if p.exists():
            b = p.read_bytes()
            h.update(f.encode()); h.update(b)
            parts.append((hashlib.md5(b).hexdigest(), f))
    return h.hexdigest()[:8], parts


def snapshot_code(root=".") -> str:
    """Simpan snapshot kode sekali per versi (dedup by hash). Return hash8."""
    root = Path(root)
    code_hash, parts = _hash_code(root)
    dest = root / "runs_archive" / "_code" / code_hash
    if not dest.exists():
        for _, f in parts:
            out = dest / f
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / f, out)
        (dest / "MANIFEST.txt").write_text(
            "\n".join(f"{md5}  {f}" for md5, f in parts), encoding="utf-8")
    return code_hash


def archive_arm(ds, arm, code_hash, ts, results_dir="results",
                log_file=None, root="."):
    """Simpan baris hasil arm + ref kode + log ke folder unik. Return path."""
    root = Path(root)
    rdir = root / results_dir
    dest = root / "runs_archive" / ds / arm / f"{ts}_{code_hash}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "code_hash.txt").write_text(code_hash, encoding="utf-8")
    for name in (f"train_summary_{ds}.csv", f"e1_clean_{ds}.csv",
                 f"e2_perturb_{ds}.csv"):
        p = rdir / name
        if p.exists():
            df = pd.read_csv(p)
            sub = df[df["model"] == arm]
            if len(sub):
                sub.to_csv(dest / name, index=False)
    if log_file and Path(log_file).exists():
        shutil.copy2(log_file, dest / "train.log")
    return dest
