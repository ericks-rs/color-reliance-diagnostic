"""M2 smoke-test: resnet50, seed 0, 2 epoch di Flowers-102.
Verifikasi loss turun, AMP+GPU jalan, checkpoint tersimpan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root, get_logger, save_env
from src.train import train_one_run
import torch


def main():
    chdir_to_root()
    cfg = load_config()
    logger = get_logger("m2", log_file="logs/m2_smoke.log")
    logger.info(f"cuda={torch.cuda.is_available()} "
                f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    save_env("logs/m2_env")

    res = train_one_run(cfg, "flowers102", "resnet50", seed=0, epochs=2,
                        logger=logger)

    hist = res["history"]
    logger.info("=== SMOKE SUMMARY ===")
    logger.info(f"epoch1 loss={hist[0]['train_loss']:.4f} -> "
                f"epoch2 loss={hist[1]['train_loss']:.4f}  "
                f"(turun: {hist[1]['train_loss'] < hist[0]['train_loss']})")
    logger.info(f"best_val_acc={res['best_acc']:.4f}")
    ck = Path(res["ckpt_path"])
    logger.info(f"checkpoint exists={ck.exists()} size={ck.stat().st_size/1e6:.1f}MB")
    logger.info(f"time={res['time_min']:.1f} min")


if __name__ == "__main__":
    main()
