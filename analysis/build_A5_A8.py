"""Bangun tabel A5-A8 (STEP 4 build) dari hasil final.

A5 Table 8   : ResNet-50 di 4 konfigurasi (S1 shared, S11 tv, S12 proper, S13 sgd)
               + ConvNeXt-in1k (S2) sebagai referensi atas.
A6 Confound  : non-eq (S5 in12k, S6 in21k) vs equalized (S2 fb, S3 augreg) side-by-side.
A7 R2.3      : matriks 3-sel RGB/RGB, RGB/gray, gray/gray. ABSOLUT (bukan recov%).
A8 Ladder    : dua bacaan R1.3 -> A: S11->S1->S2 ; B: S11->S12->S2.

Kontrak statistik: diff + 95% CI dari paired-diff per-seed (t, df=4). TANPA p-value
dan d_z. std ddof=1.

Soal CI lintas arsitektur: kebijakan tunggalnya ada di revision/STATS_POLICY.md dan
di bagian statistik Methods. Ringkasnya, interval dihitung dari selisih per-seed
dengan SATU checkpoint pretrained per backbone, jadi ia menggambarkan seberapa
konsisten sebuah selisih terulang di bawah protokol ini, bukan bagaimana arsitektur
akan dibandingkan lintas populasi run pretraining. Catatan kaki tiap tabel di sini
menerangkan kontras jenis apa yang tabel itu lakukan, bukan menetapkan aturan global.

python revision/build_A5_A8.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root

DS = [("flowers102", "Flowers-102"), ("cub200", "CUB-200")]
# Catatan kaki A5/A6/A7/A8 menerangkan APA YANG TABEL INI LAKUKAN, bukan aturan
# global. Versi sebelumnya berbunyi "Confidence intervals are reported only for
# within-architecture contrasts. Cross-architecture rows are descriptive." Kalimat
# itu benar untuk keempat tabel ini (A5 membandingkan konfigurasi ResNet terhadap
# S1, A6 membandingkan checkpoint setara lawan non-setara pada backbone yang sama,
# A7 residual per backbone, A8 tidak melaporkan CI sama sekali), TETAPI ia
# dirumuskan sebagai kebijakan global sehingga bertabrakan dengan A4, yang memang
# melaporkan CI antar-arsitektur dan punya pembenarannya sendiri.
#
# Perbaikannya bukan mencabut pembatasan, melainkan menurunkannya dari klaim
# global menjadi keterangan lokal. Tidak ada yang dilonggarkan, dan tiap kalimat
# menggambarkan isi tabelnya sendiri. Pernyataan yang mengikat semua tabel ada
# sekali di bagian statistik Methods, lihat revision/STATS_POLICY.md.
FOOT = ("\n> Inference is conditional on a single checkpoint per arm; the variance "
        "reflects fine-tuning, not the pretraining population. The contrasts in this "
        "table are within-architecture, with seeds matched across arms by a seeded "
        "data loader.\n")

# A8 adalah pengecualian dan HARUS memakai catatan kakinya sendiri. Isinya justru
# kontras antar-arsitektur (ResNet-50 lawan ConvNeXt-T), tetapi dilaporkan sebagai
# selisih titik tanpa interval. Memakai FOOT di atas akan menuliskan pernyataan
# yang tidak benar untuk tabel ini.
FOOT_A8 = ("\n> Inference is conditional on a single checkpoint per arm; the variance "
           "reflects fine-tuning, not the pretraining population. The contrasts in "
           "this table are between architectures and are reported as point "
           "differences without intervals, since the bracket is bounded by two "
           "configurations rather than estimated from a single paired contrast.\n")


def acc(df, arm, cond):
    return df[(df.model == arm) & (df.condition == cond)].set_index("seed")["acc"].sort_index()


def msd(s):
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def paired(a, b):
    """diff = a - b per seed -> (mean, lo, hi) 95% CI."""
    i = a.index.intersection(b.index)
    d = (a.loc[i] - b.loc[i]).values
    m, sd, n = d.mean(), d.std(ddof=1), len(d)
    half = tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, m - half, m + half


def w(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {path}")


def main():
    chdir_to_root()
    cfg = load_config()
    tdir = Path(cfg["paths"]["tables"]); tdir.mkdir(parents=True, exist_ok=True)
    rdir = Path(cfg["paths"]["results"])
    E = {ds: pd.read_csv(rdir / f"e2_perturb_{ds}.csv") for ds, _ in DS}

    # ---------- A5: Table 8 ----------
    CFG = [("S1", "resnet50", "ResNet-50 / shared (AdamW 1e-4)"),
           ("S11", "resnet50_tv_sgd", "ResNet-50 / legacy (tv weights, SGD 1e-2)"),
           ("S12", "resnet50_adamw_lr1e3", "ResNet-50 / proper (AdamW 1e-3)"),
           ("S13", "resnet50_sgd", "ResNet-50 / SGD (a1 weights, SGD 1e-2)")]
    L = ["# A5 — Table 8: ResNet-50 under four fine-tuning configurations", "",
         "ConvNeXt-T (ImageNet-1k) is shown as the upper reference. Differences and 95% "
         "CIs are against the shared configuration (S1) and are within-architecture "
         "(seeds matched).", "",
         "**Reading notes.**", "",
         "- **S11 changes two factors at once** (pretraining checkpoint *and* optimizer) "
         "relative to S1. It is reported as a configuration, not as a single-factor "
         "contrast, and it is not used to attribute any effect to the checkpoint alone.",
         "- **S13 is the weakest configuration.** Holding the checkpoint fixed at `a1_in1k`, "
         "SGD trails AdamW by a wide margin. Read together with S11, which reaches the "
         "highest accuracy using the same SGD setting on the legacy checkpoint, this "
         "indicates a checkpoint-by-optimizer interaction: the `a1_in1k` weights respond "
         "poorly to plain SGD fine-tuning. It is not evidence that SGD is inherently "
         "unsuited to ResNet-50.", ""]
    for ds, lab in DS:
        d = E[ds]
        L += [f"## {lab}", "",
              "| id | configuration | clean acc | grayscale acc | CR | diff vs shared (95% CI) |",
              "|---|---|---|---|---|---|"]
        base = acc(d, "resnet50", "clean")
        for sid, arm, name in CFG:
            c, g = acc(d, arm, "clean"), acc(d, arm, "grayscale")
            i = c.index.intersection(g.index)
            cr = c.loc[i] - g.loc[i]
            if sid == "S1":
                dd = "reference"
            else:
                m, lo, hi = paired(c, base)
                dd = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            L.append(f"| {sid} | {name} | {msd(c)} | {msd(g)} | {msd(cr)} | {dd} |")
        c2, g2 = acc(d, "convnext_tiny_in1k", "clean"), acc(d, "convnext_tiny_in1k", "grayscale")
        i2 = c2.index.intersection(g2.index)
        L.append(f"| S2 | ConvNeXt-T (IN1k) — reference | {msd(c2)} | {msd(g2)} | "
                 f"{msd(c2.loc[i2]-g2.loc[i2])} | not compared (cross-architecture) |")
        L.append("")
    L.append(FOOT)
    w(tdir / "A5_table8_resnet4cfg.md", L)

    # ---------- A6: confound ----------
    PAIR = [("ConvNeXt-T", "convnext_tiny_in1k", "fb_in1k (IN1k)", "convnext_tiny", "in12k_ft_in1k (IN12k)"),
            ("ViT-S", "vit_small_in1k", "augreg_in1k (IN1k)", "vit_small", "augreg_in21k_ft_in1k (IN21k)")]
    L = ["# A6 — Pretraining-data confound: equalized vs non-equalized", "",
         "Non-equalized checkpoints (extra pretraining data) are reported here only. "
         "They never enter the main tables. Differences are within-architecture "
         "(same backbone, matched seeds), so CIs apply.", ""]
    for ds, lab in DS:
        d = E[ds]
        L += [f"## {lab} — clean accuracy", "",
              "| backbone | equalized (IN1k) | non-equalized | clean (eq) | clean (non-eq) | diff (95% CI) |",
              "|---|---|---|---|---|---|"]
        for name, eq, eqtag, ne, netag in PAIR:
            ce, cn = acc(d, eq, "clean"), acc(d, ne, "clean")
            m, lo, hi = paired(cn, ce)
            L.append(f"| {name} | {eqtag} | {netag} | {msd(ce)} | {msd(cn)} | "
                     f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        L += ["", f"## {lab} — grayscale accuracy and Color-Reliance", "",
              "| backbone | gray (eq) | gray (non-eq) | CR (eq) | CR (non-eq) | ΔCR (non-eq − eq), 95% CI |",
              "|---|---|---|---|---|---|"]
        crs = {}
        for name, eq, eqtag, ne, netag in PAIR:
            ce, cn = acc(d, eq, "clean"), acc(d, ne, "clean")
            ge, gn = acc(d, eq, "grayscale"), acc(d, ne, "grayscale")
            cre, crn = ce - ge, cn - gn
            crs[name] = (cre.mean(), crn.mean())
            m, lo, hi = paired(crn, cre)
            L.append(f"| {name} | {msd(ge)} | {msd(gn)} | {msd(cre)} | {msd(crn)} | "
                     f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        cvx, vit = crs["ConvNeXt-T"], crs["ViT-S"]
        L += ["", f"**Reliance ordering.** Equalized: ViT-S {vit[0]:.4f} vs ConvNeXt-T "
              f"{cvx[0]:.4f} (ViT-S {'higher' if vit[0] > cvx[0] else 'lower'}). "
              f"Non-equalized: ViT-S {vit[1]:.4f} vs ConvNeXt-T {cvx[1]:.4f} "
              f"(ViT-S {'higher' if vit[1] > cvx[1] else 'lower'}). The ordering "
              f"{'holds' if (vit[0] > cvx[0]) == (vit[1] > cvx[1]) else 'reverses'} "
              "under both pretraining regimes, even though ViT-S is the arm with the "
              "larger pretraining corpus.", ""]
    L.append(FOOT)
    w(tdir / "A6_table_confound.md", L)

    # ---------- A7: R2.3 matrix ----------
    TRI = [("ResNet-50", "resnet50", "resnet50_gray"),
           ("ConvNeXt-T", "convnext_tiny_in1k", "convnext_tiny_in1k_gray"),
           ("ViT-S", "vit_small_in1k", "vit_small_in1k_gray"),
           ("Swin-T", "swin_tiny", "swin_tiny_gray")]
    L = ["# A7 — R2.3 three-cell matrix (train mode x test mode)", "",
         "All entries are absolute top-1 accuracy. We deliberately do not report a "
         "recovery ratio as a headline metric: its denominator (the colour-reliance "
         "gap) differs across backbones, so the ratio is not comparable between models. "
         "The residual column is the absolute distance that grayscale training leaves "
         "unrecovered.", ""]
    for ds, lab in DS:
        d = E[ds]
        L += [f"## {lab}", "",
              "| backbone | RGB-train / RGB-test | RGB-train / gray-test | gray-train / gray-test | residual (RGB/RGB − gray/gray) |",
              "|---|---|---|---|---|"]
        for name, rgb, gray in TRI:
            rr, rg = acc(d, rgb, "clean"), acc(d, rgb, "grayscale")
            gg = acc(d, gray, "grayscale")
            L.append(f"| {name} | {msd(rr)} | {msd(rg)} | {msd(gg)} | "
                     f"{rr.mean()-gg.mean():.4f} |")
        L.append("")
    L.append(FOOT)
    w(tdir / "A7_table_R23_matrix.md", L)

    # ---------- A8: architecture contrast for R1.3 ----------
    L = ["# A8 — Architecture contrast for R1.3 (convolutional side)", "",
         "**We do not report a legacy-to-modern ladder.** The legacy configuration (S11) "
         "differs from the shared configuration (S1) in both the pretraining checkpoint "
         "and the optimizer, so any step between them mixes two factors and cannot "
         "isolate the pretraining recipe. We therefore restrict R1.3 to the contrast "
         "that is interpretable: ResNet-50 against ConvNeXt-T with the optimizer, weight "
         "decay, schedule, augmentation, and budget held fixed.", "",
         "The primary figure is the **conservative** one: ResNet-50 is given the learning "
         "rate that favours it (1e-3), while ConvNeXt-T stays at the shared 1e-4. Only the "
         "trailing model is tuned up, so the residual advantage is a lower bound on the "
         "architecture effect. The matched-learning-rate contrast (both at 1e-4) is "
         "reported as the upper bound; there ResNet-50 is under-tuned, so it overstates "
         "the same effect.", ""]
    for ds, lab in DS:
        d = E[ds]
        a = {k: acc(d, v, "clean") for k, v in
             dict(S1="resnet50", S12="resnet50_adamw_lr1e3",
                  S2="convnext_tiny_in1k").items()}
        L += [f"## {lab}", "",
              "| contrast | ResNet-50 | ConvNeXt-T | difference | reading |",
              "|---|---|---|---|---|",
              f"| **S12 → S2** (ResNet at 1e-3, ConvNeXt at 1e-4) | {a['S12'].mean():.4f} | "
              f"{a['S2'].mean():.4f} | **{a['S2'].mean()-a['S12'].mean():+.4f}** | "
              "conservative lower bound |",
              f"| S1 → S2 (both at 1e-4, matched) | {a['S1'].mean():.4f} | "
              f"{a['S2'].mean():.4f} | {a['S2'].mean()-a['S1'].mean():+.4f} | "
              "upper bound (ResNet under-tuned) |", "",
              f"Architecture effect is bracketed between "
              f"{a['S2'].mean()-a['S12'].mean():+.4f} and "
              f"{a['S2'].mean()-a['S1'].mean():+.4f} on {lab}.", ""]
    L += ["## What this does and does not license", "",
          "It licenses: under a fixed fine-tuning protocol, a modern convolutional "
          "backbone retains an advantage over ResNet-50 that survives giving ResNet-50 a "
          "more favourable learning rate.", "",
          "It does not license: attributing that advantage to the pretraining recipe "
          "alone. ConvNeXt-T bundles architectural and training-recipe modernisation, and "
          "the fourth cell of that factorisation (ConvNeXt under a classic recipe) is not "
          "obtainable without pretraining from scratch. We state this as a limitation "
          "rather than estimate it.", ""]
    L.append(FOOT_A8)
    w(tdir / "A8_arch_contrast_R13.md", L)


if __name__ == "__main__":
    main()
