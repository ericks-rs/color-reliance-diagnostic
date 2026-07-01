"""M2 confirm: resnet50 seed0, 6 epoch (lewat warmup) -> pastikan val_acc NAIK
beneran, bukan cuma pipeline jalan. Insurance sebelum M3 full run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root, get_logger
from src.train import train_one_run


def main():
    chdir_to_root()
    cfg = load_config()
    logger = get_logger("m2c", log_file="logs/m2_confirm.log")
    res = train_one_run(cfg, "flowers102", "resnet50", seed=0, epochs=6,
                        logger=logger)
    accs = [h["val_acc"] for h in res["history"]]
    logger.info("=== CONFIRM ===")
    logger.info("val_acc per epoch: " + ", ".join(f"{a:.3f}" for a in accs))
    logger.info(f"best_val_acc={res['best_acc']:.4f}  "
                f"(naik signifikan: {res['best_acc'] > 0.5})")


if __name__ == "__main__":
    main()
