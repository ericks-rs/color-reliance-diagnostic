<img src="assets/banner.png" alt="Color-Reliance Diagnostic" width="100%">

# A Colorfulness-Stratified Diagnostic of Color Reliance in Convolutional and Attention Models for Fine-Grained Classification

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3725483-00629B)](https://doi.org/10.1109/ACCESS.2026.3725483)
[![Published in IEEE Access](https://img.shields.io/badge/Published%20in-IEEE%20Access-00629B)](https://doi.org/10.1109/ACCESS.2026.3725483)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](requirements.txt)

**Authors**<br>
![Ericks Rachmat Swedia](https://img.shields.io/badge/Ericks%20Rachmat%20Swedia-5C2D91)<br>
![Astie Darmayantie](https://img.shields.io/badge/Astie%20Darmayantie-5C2D91)<br>
![Mochammad Akbar Marwan](https://img.shields.io/badge/Mochammad%20Akbar%20Marwan-5C2D91)<br>
![Aries Muslim](https://img.shields.io/badge/Aries%20Muslim-5C2D91)

Reproducibility code for the paper, accepted for publication in IEEE Access.

This repository holds the code, configuration, random seeds, per-run result CSVs, logs, tables, and figures that reproduce every number and plot in the paper. Trained weights and the datasets are not tracked: the weights regenerate from the fixed seeds, and the two datasets are public and download on first run. See [Reproducing the results](#reproducing-the-results).

## What the study does

One fine-tuning recipe is held fixed across four parameter- and FLOPs-matched backbones (ResNet-50, ConvNeXt-T, ViT-S, Swin-T), all initialized from ImageNet-1k and trained with no color augmentation. The test set is stratified by the Hasler-Susstrunk colorfulness metric only at evaluation. A **Color-Reliance score**, the top-1 accuracy a model loses when the image is reduced to grayscale, measures how much each backbone leans on color.

**Findings (five seeds, two datasets).** On equalized ImageNet-1k pretraining, Swin-T is the most accurate (0.9365 on Flowers-102, 0.8468 on CUB-200), sitting level with ConvNeXt-T on Flowers-102 and above it on CUB-200. Color reliance orders the backbones differently: the purest attention model, ViT-S, is the most color-reliant on both datasets (CR 0.5353 and 0.4971), while the windowed Swin-T cannot be separated from the convolutional ConvNeXt-T. Accuracy reveals none of this. Reliance is not fixed by architecture either: changing the training configuration of a single backbone moves it across a third to a half of the range that separates the four.

## Repository layout

```
color_complexity/
├── config.yaml            # the locked recipe, the ablation recipe pack, and paths
├── requirements.txt
├── src/                   # core modules
│   ├── colorfulness.py    # Hasler-Susstrunk score + tertile binning
│   ├── data.py            # loaders, transforms (no color aug), grayscale eval, CUB val split
│   ├── models.py          # timm backbone construction
│   ├── train.py           # one training run (locked recipe, AMP, cosine, warmup)
│   ├── eval_clean.py      # E1: clean + per-colorfulness-bin accuracy
│   ├── eval_perturb.py    # E2: test-time color-degradation suite
│   ├── perturb.py         # grayscale, hue rotation, channel shuffle, quantization
│   ├── metrics.py         # top-1, macro-F1, UAR
│   └── utils.py           # config loading, logging, environment capture
├── scripts/               # entry points (run from the repository root)
│   ├── check_env.py       # GPU / CUDA sanity check
│   ├── run_all.py         # one (model x seed): train + E1 + E2, upsert into results/
│   └── run_study.py       # the whole study: every arm x seed, a thin loop over run_all
├── analysis/              # build the tables and figures from results/ (no GPU)
│   ├── arms.py            # single source of truth for arm ids, labels, colors
│   ├── build_*.py         # tables and publication figures
│   ├── export_*.py        # tidy CSV exports of the tables
│   ├── eval_*.py          # per-image prediction dumps and Grad-CAM (need checkpoints)
│   ├── foreground_colorfulness.py   # object-only colorfulness (CUB box, Flowers mask)
│   ├── relative_cr.py             # relative Color-Reliance (robustness check)
│   ├── luma_convention_check.py   # BT.709 vs Rec.601 luminance check
│   └── hue_luminance_shift.py     # luminance shift under hue rotation
├── results/               # raw per-run CSVs, the single source of truth for every table
├── tables/                # exported tables (.md, .csv)
├── figures/               # publication figures and their value CSVs
├── logs/                  # per-run config_used.yaml, env_info.txt, pip_freeze.txt, *.log
├── checkpoints/           # (gitignored) regenerable weights
├── runs/                  # (gitignored) per-run working dirs
└── data/                  # (gitignored) datasets, downloaded on first run
```

## Setup

Built on Windows 11 with a single NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell, sm_120), Python 3.11, PyTorch 2.11.0 + CUDA 12.8.

```bash
conda create -n colorreliance python=3.11
conda activate colorreliance

# GPU build first (Blackwell needs the cu128 wheels; a different GPU may need another CUDA build):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py     # confirms CUDA is visible
```

## Datasets

Both are public. Place them under `data/` (paths are configurable in `config.yaml`):

- **Oxford Flowers-102** (102 classes): https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ → `data/flowers-102/`
- **CUB-200-2011** (200 bird classes): https://www.vision.caltech.edu/datasets/cub_200_2011/ → `data/CUB_200_2011/`

CUB-200 ships no validation split. A class-balanced set of 1,000 images (five per class) is carved from the training partition with a fixed split seed of 12345, leaving 4,994 for training; the test set is untouched until the single final evaluation. This is handled in `src/data.py` and configured under `datasets.cub200.val_split` in `config.yaml`.

## Reproducing the results

The recipe is locked in `config.yaml` and identical across architectures. The study uses five seeds (0 to 4).

```bash
# 1. Train the whole study: 13 arms x 2 datasets x 5 seeds = 130 runs.
#    Idempotent per (model, seed); a partial run resumes by rerunning.
python scripts/run_study.py

#    or a subset:
python scripts/run_study.py --arms main --datasets flowers102     # the 4 main arms, one dataset
python scripts/run_study.py --seeds 0 --epochs 2                  # quick smoke run

# 2. Build every table and figure from results/ (no GPU needed).
for f in analysis/build_*.py; do python "$f"; done

# 3. Optional foreground colorfulness and the robustness checks:
python analysis/foreground_colorfulness.py
python analysis/relative_cr.py
python analysis/luma_convention_check.py
python analysis/hue_luminance_shift.py
```

`run_all.py` is idempotent per `(model, seed)`: rerunning upserts rows into the result CSVs rather than duplicating them. Every table and figure is derived from `results/`, so once training is done the tables and figures rebuild without a GPU.

## What each arm is

`run_study.py` covers 13 arms:

| Group | Arms | Purpose |
|---|---|---|
| main | ResNet-50, ConvNeXt-T, ViT-S, Swin-T (all ImageNet-1k) | the main comparison |
| gray | the four main arms trained on grayscale input | color-free training |
| confound | ConvNeXt-T and ViT-S on their extra-data checkpoints | reported only in the pretraining-confound table |
| ablation | ResNet-50 under three recipe variants (AdamW 1e-3, SGD, legacy checkpoint + SGD) | recipe sensitivity |

## Outputs map (content → script → file)

The paper renumbers tables and figures; the mapping by content is:

| Content | Built by | File(s) |
|---|---|---|
| Colorfulness stats and thresholds | `build_T1_T2_specs.py` | `tables/T1_colorfulness_*` |
| Backbone specs and checkpoints | `build_T1_T2_specs.py` | `tables/T2_model_specs_*` |
| Clean accuracy, macro-F1, UAR | `build_T2T3T4_equalized.py` | `tables/T2_clean_acc_*` |
| Accuracy and gap per colorfulness bin | `build_T2T3T4_equalized.py` | `tables/T3_acc_by_bin_*`, `tables/T3_gap_by_bin_*` |
| Color-Reliance score | `build_T2T3T4_equalized.py` | `tables/T4_color_reliance_*` |
| Pairwise contrasts with 95% CI | `build_A4.py` | `tables/A4_table7_paired_diffCI_*` |
| ResNet-50 four-configuration ablation | `build_A5_A8.py` | `tables/A5_table8_resnet4cfg_*` |
| Pretraining-data confound (equalized vs not) | `build_A5_A8.py` | `tables/A6_table_confound_*` |
| Per-bin foreground stratification | `build_V4_V5.py` | `tables/V4_perbin_wholeimg_*`, `tables/V5_perbin_foreground_*` |
| Grayscale-training recovery | `export_V_csv.py` | `tables/V9_grayscale_eval_*` |
| Per-species Color-Reliance | `build_V7.py` | `tables/V7_perspecies_*` |
| Clean-accuracy figure | `build_fig1_cleanacc.py` | `figures/` |
| Gap-per-bin figure | `build_fig2_gapbybin.py` | `figures/` |
| Degradation drop-curves figure | `build_fig3_dropcurves.py` | `figures/` |
| Color-Reliance figure | `build_fig4_colorreliance.py` | `figures/` |
| Recipe-ablation figure | `build_fig5_ablation.py` | `figures/` |
| Per-species distribution figure | `build_fig7_perspecies.py` | `figures/` |
| Grad-CAM comparison figure | `analysis/eval_gradcam.py` | `figures/` |

## Checkpoints

The trained weights are not tracked; they regenerate from the fixed seeds by running the pipeline. Each run's exact `config_used.yaml`, `env_info.txt`, and `pip_freeze.txt` are committed under `logs/`, so the environment behind every number is recorded.

## Citation

```bibtex
@article{swedia2026colorreliance,
  author  = {Swedia, Ericks Rachmat and Darmayantie, Astie and Marwan, Mochammad Akbar and Muslim, Aries},
  title   = {A Colorfulness-Stratified Diagnostic of Color Reliance in Convolutional and Attention Models for Fine-Grained Classification},
  journal = {IEEE Access},
  volume  = {14},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3725483}
}
```

## License

MIT. See [LICENSE](LICENSE).
