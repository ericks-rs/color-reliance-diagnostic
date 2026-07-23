"""Run the full study: every arm x seed on both datasets.

This is a thin, deterministic loop over ``scripts/run_all.py``. Each arm on each
dataset is one ``run_all`` call that trains, then runs the clean (E1) and
color-degradation (E2) evaluations, and upserts the rows into ``results/*.csv``.
Reruns are idempotent per (model, seed), so a partial run can be resumed by
simply running again.

The study is 13 arms x 2 datasets x 5 seeds = 130 runs. On the reference GPU
(RTX 5080) a Flowers-102 run is about 3 minutes and a CUB-200 run about 12.

Arm groups (pass --arms to select):
  main      4 equalized backbones, ImageNet-1k, AdamW           (the main tables)
  gray      4 grayscale-trained counterparts                    (R2.3, Table 10)
  confound  2 non-equalized checkpoints                         (R1.2, Table 3 only)
  ablation  3 ResNet-50 recipe variants                         (R1.3/R1.4/R1.5, Table 9)

Usage:
  python scripts/run_study.py                          # everything
  python scripts/run_study.py --datasets flowers102    # one dataset
  python scripts/run_study.py --arms main gray         # selected groups
  python scripts/run_study.py --seeds 0                # smoke, one seed
"""
import argparse
import subprocess
import sys
from pathlib import Path

RUN_ALL = Path(__file__).resolve().parent / "run_all.py"

# (model_key, recipe, grayscale) -> reproduces the arm id run_all.py writes.
ARMS = {
    "main": [
        ("resnet50", "adamw", False),
        ("convnext_tiny_in1k", "adamw", False),
        ("vit_small_in1k", "adamw", False),
        ("swin_tiny", "adamw", False),
    ],
    "gray": [
        ("resnet50", "adamw", True),
        ("convnext_tiny_in1k", "adamw", True),
        ("vit_small_in1k", "adamw", True),
        ("swin_tiny", "adamw", True),
    ],
    "confound": [
        ("convnext_tiny", "adamw", False),
        ("vit_small", "adamw", False),
    ],
    "ablation": [
        ("resnet50", "adamw_lr1e3", False),
        ("resnet50", "sgd", False),
        ("resnet50_tv", "sgd", False),
    ],
}
DATASETS = ["flowers102", "cub200"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None, help="override, for a smoke run")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args()

    jobs = [(ds, m, r, g)
            for ds in args.datasets
            for grp in args.arms
            for (m, r, g) in ARMS[grp]]

    print(f"{len(jobs)} runs: {len(args.datasets)} datasets x "
          f"{sum(len(ARMS[g]) for g in args.arms)} arms, seeds {args.seeds}\n")

    for i, (ds, model, recipe, gray) in enumerate(jobs, 1):
        cmd = [sys.executable, str(RUN_ALL), "--dataset", ds,
               "--models", model, "--recipe", recipe,
               "--seeds", *map(str, args.seeds)]
        if gray:
            cmd.append("--grayscale")
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        arm = model + ("_" + recipe if recipe != "adamw" else "") + ("_gray" if gray else "")
        print(f"[{i}/{len(jobs)}] {ds} / {arm}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"run_all failed (exit {rc}) on {ds}/{arm}. Fix and rerun; "
                     f"completed arms are already saved in results/.")

    print("\nAll runs complete. Build the tables and figures with the scripts in analysis/.")


if __name__ == "__main__":
    main()
