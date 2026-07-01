# Color-Reliance Diagnostic: Convolution vs Attention in Fine-Grained Recognition

Reproducibility code for the paper **"A Colorfulness-Stratified Diagnostic of Color Reliance in Convolutional and Attention Models for Fine-Grained Classification."**

This repository contains the code, configurations, random seeds, raw per-run results, and analysis scripts that reproduce every table and figure in the paper. The trained checkpoints (about 4.7 GB) and the public datasets (about 3 GB) are not tracked here. See [Datasets](#datasets) for download links and [Checkpoints](#checkpoints) for availability.

## What the study does

We hold one training recipe fixed across four parameter- and FLOPs-matched backbones (ResNet-50, ConvNeXt-T, ViT-S, Swin-T), train without any color augmentation, and stratify the test set by the Hasler-Susstrunk colorfulness metric only at evaluation time. A named **Color-Reliance score** (the accuracy a model loses when color is removed by grayscaling) measures how much each architecture leans on color, and a ConvNeXt-T arm separates the attention mechanism from the modern training recipe. A ResNet-50 SGD arm rules out the wrong-optimizer objection.

**Headline findings (5 seeds, two datasets).** The modern convolutional network ConvNeXt-T is the most accurate (0.9902 on Flowers-102, 0.8900 on CUB-200) and the least color-reliant (CR 0.16 and 0.36). The attention models beat the classic ResNet-50 but trail ConvNeXt-T under the same recipe. Color complexity does not modulate the convolution-versus-attention gap. Retraining ResNet-50 with classic SGD lowers accuracy rather than closing the gap. The apparent advantage of attention is, in these experiments, an advantage of the training recipe.

## Repository layout

```
color_complexity/
├── config.yaml              # the locked training recipe + ablation recipe pack
├── run_all.py               # orchestrator: (model x seed) -> train + E1(clean) + E2(perturb)
├── analyze.py               # draft tables (T1-T5) and quick-look figures (F1-F4)
├── analyze_extra.py         # publication figures P1-P4 (300 DPI) -> figures/pub/
├── analyze_ablation.py      # recipe-ablation table (T6) and ablation figure
├── analyze_extra.py / ...   # see "Outputs map" below
├── requirements.txt
├── src/
│   ├── colorfulness.py      # Hasler-Susstrunk score + tertile binning
│   ├── data.py              # dataset loaders, transforms (NO color aug), grayscale eval
│   ├── models.py            # timm backbone construction
│   ├── train.py             # one training run (locked recipe, AMP, cosine, warmup)
│   ├── eval_clean.py        # E1: clean + per-colorfulness-bin accuracy
│   ├── eval_perturb.py      # E2: test-time color-degradation suite
│   ├── perturb.py           # grayscale, hue rotation, channel shuffle, quantization
│   ├── metrics.py           # top-1, macro-F1, UAR
│   ├── stats.py             # paired t-test, Wilcoxon, Cohen's d_z
│   └── utils.py             # config loading, logging, env capture
├── results/                 # raw per-run CSVs (the source of every table) + confusion/
├── tables/                  # exported tables (.md and .tex), T1-T6
├── figures/                 # quick-look figures + figures/pub/ (300 DPI, used in the paper)
├── logs/                    # per-run config_used.yaml, env_info.txt, pip_freeze.txt, *.log
├── checkpoints/             # (gitignored) trained weights
└── data/                    # (gitignored) datasets
```

## Setup

Built and validated on Windows 11 with a single NVIDIA RTX 5080 (Blackwell, sm_120), Python 3.11.15, PyTorch 2.11.0 + CUDA 12.8.

```bash
conda create -n colorreliance python=3.11
conda activate colorreliance

# GPU build first (RTX 5080 / Blackwell needs the cu128 wheels):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A different GPU generation may use a different CUDA build of PyTorch. Everything else in `requirements.txt` is generation-independent.

## Datasets

Both datasets are public. Download and place them under `data/` as below.

- **Oxford Flowers-102** (102 classes): https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ -> `data/flowers-102/`
- **CUB-200-2011** (200 bird classes): https://www.vision.caltech.edu/datasets/cub_200_2011/ -> `data/CUB_200_2011/`

CUB ships no official validation split, so the test set is used for best-checkpoint selection (documented in the paper). Paths are configurable in `config.yaml` under `datasets:`.

## Reproducing the results

The recipe is locked in `config.yaml` and is identical across architectures. The paper uses **five seeds (0 to 4)**. The committed `config.yaml` default lists `seeds: [0, 1, 2]` for a quick partial run, so pass `--seeds 0 1 2 3 4` to match the paper.

```bash
# 1. Main arm (AdamW), both datasets, all four models, five seeds.
#    Produces results/{train_summary,e1_clean,e2_perturb}_<ds>.csv and confusion matrices.
python run_all.py --dataset flowers102 --seeds 0 1 2 3 4
python run_all.py --dataset cub200     --seeds 0 1 2 3 4

# 2. Recipe-ablation arm (classic SGD on ResNet-50 only), five seeds.
python run_all.py --dataset flowers102 --models resnet50 --recipe sgd --seeds 0 1 2 3 4
python run_all.py --dataset cub200     --models resnet50 --recipe sgd --seeds 0 1 2 3 4

# 3. Build tables + quick-look figures (T1-T5, F1-F4).
python analyze.py

# 4. Build the 300 DPI publication figures (P1-P4 -> figures/pub/).
python analyze_extra.py

# 5. Build the recipe-ablation table T6 and its figure.
python analyze_ablation.py
```

`run_all.py` is idempotent per `(model, seed)`: rerunning upserts rows into the result CSVs rather than duplicating them. Training takes roughly 3 minutes per run on Flowers-102 and 12 minutes per run on CUB-200 on the reference GPU. The full study is 50 models (2 datasets x 4 models x 5 seeds = 40 main, plus 10 ResNet-SGD ablation).

## Outputs map (paper table/figure -> file)

The paper renumbers some tables and figures relative to the filenames here. The mapping is:

| Paper | Produced by | File(s) |
|---|---|---|
| Table I (colorfulness stats) | `analyze.py` | `results/colorfulness_<ds>.csv`, `tables/T1_*` |
| Table II (params / FLOPs) | `analyze.py` | `tables/T1_params_flops_*` |
| Table III (clean accuracy) | `analyze.py` | `tables/T2_clean_acc_*` |
| Table IV (acc per bin) | `analyze.py` | `tables/T3_e1_acc_by_bin_*` |
| Table V (gap per bin) | `analyze.py` | `tables/T3_e1_gap_by_bin_*` |
| Table VI (Color-Reliance) | `analyze.py` | `tables/T4_color_reliance_*` |
| Table VII (paired stats) | `analyze.py` | `tables/T5_paired_stats` |
| Table VIII (recipe ablation) | `analyze_ablation.py` | `tables/T6_recipe_ablation` |
| Fig. 1 (clean accuracy) | `analyze_extra.py` | `figures/pub/P1_clean_acc.png` |
| Fig. 2 (gap vs bin) | `analyze_extra.py` | `figures/pub/P3_gap_vs_bin.png` |
| Fig. 3 (drop curves) | `analyze_extra.py` | `figures/pub/P4_drop_curves.png` |
| Fig. 4 (color reliance) | `analyze_extra.py` | `figures/pub/P2_color_reliance.png` |
| Fig. 5 (recipe ablation) | `analyze_ablation.py` | `figures/pub/ablation_recipe_pub.png` |

The raw CSVs in `results/` are the single source of truth. Every table and figure is derived from them, so the figures and tables can be regenerated without retraining as long as `results/` is present (it is committed here).

## Checkpoints

The 50 trained checkpoints and full per-run logs are available from the corresponding author on reasonable request. The committed `logs/` already contain each run's exact `config_used.yaml`, `env_info.txt`, and `pip_freeze.txt`.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added once the paper has a DOI. For now:

```
Ericks Rachmat Swedia, Astie Darmayantie, Mochammad Akbar Marwan, and Aries Muslim,
"A Colorfulness-Stratified Diagnostic of Color Reliance in Convolutional and
Attention Models for Fine-Grained Classification," submitted to IEEE Access, 2026.
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
