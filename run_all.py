"""Orkestrasi: loop (model x seed) -> train + E1(clean) + E2(perturb).
Hasil mentah diakumulasi ke results/*.csv (idempotent per (model,seed)).

Usage:
  python run_all.py --dataset flowers102 --models resnet50 convnext_tiny vit_small swin_tiny --seeds 0
  python run_all.py --dataset flowers102 --seeds 0 1 2          # semua model default
  python run_all.py --dataset flowers102 --models resnet50 --seeds 0 --epochs 2   # smoke
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config, chdir_to_root, get_logger, save_env
from src.train import train_one_run
from src.eval_clean import eval_clean
from src.eval_perturb import eval_perturb
from src import colorfulness as cf_mod


def _upsert(csv_path, new_df, keys):
    csv_path = Path(csv_path)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        merged = old.merge(new_df[keys].drop_duplicates(), on=keys,
                           how="left", indicator=True)
        old = old[merged["_merge"] == "left_only"].drop(columns=[])
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(csv_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flowers102")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke)")
    ap.add_argument("--recipe", default="adamw", help="adamw (default arm) | sgd (ablation)")
    args = ap.parse_args()

    chdir_to_root()
    cfg = load_config()
    ds = args.dataset
    models = args.models or list(cfg["models"].keys())
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]

    recipe = args.recipe
    res_dir = Path(cfg["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)
    log_suffix = ds if recipe == "adamw" else f"{ds}_{recipe}"
    logger = get_logger("run_all", log_file=f"logs/run_all_{log_suffix}.log")
    save_env(f"logs/run_all_{log_suffix}_env", config_path=str(Path("config.yaml")))
    logger.info(f"dataset={ds} models={models} seeds={seeds} epochs={args.epochs} recipe={recipe}")

    # pastikan colorfulness csv ada (buat E1 binning)
    cf_path = res_dir / f"colorfulness_{ds}.csv"
    if not cf_path.exists():
        logger.info(f"colorfulness csv belum ada -> compute {ds}")
        cf_mod.build_colorfulness_csv(ds, cfg)

    train_csv = res_dir / f"train_summary_{ds}.csv"
    e1_csv = res_dir / f"e1_clean_{ds}.csv"
    e2_csv = res_dir / f"e2_perturb_{ds}.csv"

    for model_key in models:
        arm = model_key if recipe == "adamw" else f"{model_key}_{recipe}"
        for seed in seeds:
            logger.info(f"==== {ds} / {arm} / seed{seed} (recipe={recipe}) ====")
            res = train_one_run(cfg, ds, model_key, seed, epochs=args.epochs,
                                logger=logger, recipe=recipe, arm=arm)
            _upsert(train_csv, pd.DataFrame([{
                "dataset": ds, "model": arm, "seed": seed,
                "best_val_acc": res["best_acc"], "time_min": res["time_min"],
                "lr_used": res["lr_used"], "recipe": recipe}]),
                keys=["dataset", "model", "seed"])

            logger.info("  [E1] eval clean + per colorfulness bin")
            df1 = eval_clean(cfg, ds, arm, seed, model_key=model_key)
            _upsert(e1_csv, df1, keys=["dataset", "model", "seed", "bin"])

            logger.info("  [E2] eval perturbasi warna")
            df2 = eval_perturb(cfg, ds, arm, seed, model_key=model_key)
            _upsert(e2_csv, df2, keys=["dataset", "model", "seed", "condition"])
            logger.info(f"  done {arm} seed{seed}")

    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
