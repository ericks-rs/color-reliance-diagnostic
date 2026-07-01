"""Utility: seeding, logging, reproducibility (env capture)."""
import os
import sys
import random
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """Set semua RNG supaya reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_logger(name: str, log_file: str = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_env(out_dir: str, config_path: str = None):
    """Simpan pip freeze + copy config buat reproducibility tiap run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (out / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as e:
        (out / "pip_freeze.txt").write_text(f"pip freeze failed: {e}", encoding="utf-8")
    info = [
        f"python: {sys.version}",
        f"executable: {sys.executable}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        info.append(f"device: {torch.cuda.get_device_name(0)}")
        info.append(f"capability: {torch.cuda.get_device_capability(0)}")
    (out / "env_info.txt").write_text("\n".join(info), encoding="utf-8")
    if config_path and Path(config_path).exists():
        (out / "config_used.yaml").write_text(
            Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
        )


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # color_complexity/


def chdir_to_root():
    """Pastikan CWD = color_complexity/ supaya semua path relatif di config
    (data, results, figures, ...) konsisten dari mana pun script dipanggil."""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def load_config(path: str = None):
    import yaml
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
