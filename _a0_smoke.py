"""A0 smoke: ResNet-SGD recipe, seed0, 2 epoch, Flowers.
Cek: loss turun (no NaN), GPU+AMP, checkpoint nama baru tersimpan, arm lama utuh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root, get_logger
from src.train import train_one_run


def main():
    chdir_to_root()
    cfg = load_config()
    logger = get_logger("a0", log_file="logs/a0_smoke.log")

    old_ckpt = Path("checkpoints/flowers102_resnet50_seed0.pt")
    old_exists_before = old_ckpt.exists()
    old_mtime_before = old_ckpt.stat().st_mtime if old_exists_before else None

    res = train_one_run(cfg, "flowers102", "resnet50", seed=0, epochs=2,
                        logger=logger, recipe="sgd")

    h = res["history"]
    new_ckpt = Path(res["ckpt_path"])
    logger.info("=== A0 SMOKE SUMMARY ===")
    logger.info(f"arm={res['arm']} recipe={res['recipe']} lr_used={res['lr_used']}")
    logger.info(f"ep1 loss={h[0]['train_loss']:.4f} -> ep2 loss={h[1]['train_loss']:.4f} "
                f"(turun: {h[1]['train_loss'] < h[0]['train_loss']}, "
                f"finite: {all(__import__('math').isfinite(x['train_loss']) for x in h)})")
    logger.info(f"new ckpt={new_ckpt.name} exists={new_ckpt.exists()} "
                f"size={new_ckpt.stat().st_size/1e6:.1f}MB")
    # cek arm lama (adamw) TIDAK tersentuh
    old_mtime_after = old_ckpt.stat().st_mtime if old_ckpt.exists() else None
    untouched = (old_exists_before == old_ckpt.exists()) and \
                (old_mtime_before == old_mtime_after)
    logger.info(f"arm lama (resnet50 adamw) UTUH (tak tertimpa): {untouched}")
    logger.info(f"time={res['time_min']:.1f}min")


if __name__ == "__main__":
    main()
