"""T2, T3, T4 dibangun ulang pada arm SETARA, plus gap-per-bin yang belum ada.

Keempat tabel ini sebelumnya lahir dari analyze.py, yang memakai `convnext_tiny`
(in12k_ft_in1k) dan `vit_small` (augreg_in21k_ft_in1k). Reviewer 1 menyoroti
ketimpangan pretraining itu di R1.2, dan revisi menyetarakan semua arm utama ke
IN1k. Versi lama dipindah ke tables/_superseded_v1_nonequalized/.

Yang dihasilkan, untuk tiap dataset:
  T2_clean_acc_<ds>.md        akurasi bersih, macro-F1, UAR (mean +- SD, 5 seed)
  T3_acc_by_bin_<ds>.md       akurasi per bin colorfulness
  T3_gap_by_bin_<ds>.md       gap atensi minus konvolusi per bin, plus tren
  T4_color_reliance_<ds>.md   clean, grayscale, dan CR
Plus .csv pendamping tiap tabel, dan T3_gap_by_bin per-seed untuk pemeriksaan.

Catatan soal gap-per-bin. Tabel ini satu-satunya yang belum punya pengganti sama
sekali, dan ia menyangga klaim null-modulation. Gap dihitung PER SEED lalu
diringkas, bukan dari selisih dua rata-rata, supaya SD-nya mencerminkan
keragaman pasangan yang benar-benar diamati. Kedua cara memberi mean yang sama,
tetapi hanya cara per-seed yang memberi ukuran ketidakpastian yang sah.

Tren ditulis sebagai high minus low. Ia deskriptif, bukan uji tren, jadi jangan
dibaca sebagai bukti ada atau tidaknya modulasi tanpa menyertakan sebarannya.

python revision/build_T2T3T4_equalized.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from analysis.arms import MAIN_ARMS, MLAB, PRETRAIN_TAG, DSLAB, check

BINS = ["low", "mid", "high"]
CONVS = ["resnet50", "convnext_tiny_in1k"]
ATTNS = ["vit_small_in1k", "swin_tiny"]
HEAD = ("*Arms equalized to ImageNet-1k (R1.2): "
        + ", ".join(f"{MLAB[a]} `{PRETRAIN_TAG[a]}`" for a in MAIN_ARMS) + ".*")


def ms(v):
    return f"{np.mean(v):.4f}+-{np.std(v, ddof=1):.4f}"


def md_table(rows, header, align=None):
    align = align or (["---"] * len(header))
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write(path, title, note, body, extra=""):
    path.write_text(f"# {title}\n\n{note}\n\n{body}\n{extra}\n", encoding="utf-8")
    print(f"  -> {path}")


def main():
    chdir_to_root()
    cfg = load_config()
    rdir = Path(cfg["paths"]["results"])
    tdir = Path(cfg["paths"]["tables"])
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "perseed").mkdir(exist_ok=True)

    for ds in ["flowers102", "cub200"]:
        e1p, e2p = rdir / f"e1_clean_{ds}.csv", rdir / f"e2_perturb_{ds}.csv"
        if not e1p.exists():
            continue
        e1 = pd.read_csv(e1p)
        e1 = e1[e1.model.isin(MAIN_ARMS)].copy()
        check(e1)
        e2 = pd.read_csv(e2p) if e2p.exists() else pd.DataFrame()
        if len(e2):
            e2 = e2[e2.model.isin(MAIN_ARMS)].copy()
            check(e2)
        ov = e1[e1["bin"] == "overall"]
        lab = DSLAB[ds]

        # ---- T2 akurasi bersih ------------------------------------------------
        rows, csv = [], []
        for a in MAIN_ARMS:
            g = ov[ov.model == a]
            rows.append([MLAB[a], g.seed.nunique(), ms(g.acc.values),
                         ms(g.macro_f1.values), ms(g.uar.values)])
            csv.append(dict(dataset=ds, arm=a, backbone=MLAB[a],
                            pretrain=PRETRAIN_TAG[a], n_seeds=g.seed.nunique(),
                            acc_mean=g.acc.mean(), acc_sd=g.acc.std(ddof=1),
                            macro_f1_mean=g.macro_f1.mean(),
                            macro_f1_sd=g.macro_f1.std(ddof=1),
                            uar_mean=g.uar.mean(), uar_sd=g.uar.std(ddof=1)))
        pd.DataFrame(csv).to_csv(tdir / f"T2_clean_acc_{ds}.csv", index=False)
        write(tdir / f"T2_clean_acc_{ds}.md",
              f"T2 — Clean top-1 accuracy, {lab}", HEAD,
              md_table(rows, ["backbone", "n seeds", "accuracy", "macro-F1", "UAR"],
                       [":---", "---:", ":---", ":---", ":---"]))

        # ---- T3 akurasi per bin ----------------------------------------------
        per = {a: {b: e1[(e1.model == a) & (e1["bin"] == b)]
                   .set_index("seed").acc.sort_index() for b in BINS}
               for a in MAIN_ARMS}
        rows, csv = [], []
        for a in MAIN_ARMS:
            rows.append([MLAB[a]] + [ms(per[a][b].values) for b in BINS])
            for b in BINS:
                v = per[a][b].values
                csv.append(dict(dataset=ds, arm=a, backbone=MLAB[a], bin=b,
                                n_seeds=len(v), acc_mean=v.mean(),
                                acc_sd=v.std(ddof=1)))
        pd.DataFrame(csv).to_csv(tdir / f"T3_acc_by_bin_{ds}.csv", index=False)
        write(tdir / f"T3_acc_by_bin_{ds}.md",
              f"T3a — Accuracy by colorfulness bin, {lab}", HEAD,
              md_table(rows, ["backbone"] + BINS,
                       [":---"] + [":---"] * len(BINS)))

        # ---- T3 gap per bin (yang sebelumnya hilang) --------------------------
        # Gap dihitung per seed lalu diringkas. Seed dipasangkan lewat irisan
        # indeks, jadi tidak ada asumsi bahwa kedua arm punya seed yang sama.
        rows, csv, ps_rows = [], [], []
        for at in ATTNS:
            for cv in CONVS:
                cells, tr = [], {}
                for b in BINS:
                    sa, sb = per[at][b], per[cv][b]
                    common = sa.index.intersection(sb.index)
                    d = (sa.loc[common] - sb.loc[common]).values
                    tr[b] = d
                    cells.append(ms(d))
                    # Kolom dinamai "backbone" dan "baseline", bukan "attention"
                    # dan "convolution". Nama lama mengelompokkan ViT-S dengan
                    # Swin-T sebagai satu kelas, padahal isi tabel ini justru
                    # memisahkan keduanya: Swin-T lawan ConvNeXt-T berkisar
                    # +0.001 sampai +0.013, sedangkan ViT-S lawan ConvNeXt-T
                    # berkisar -0.041 sampai -0.067. Label kolom sebaiknya tidak
                    # menyatukan apa yang datanya pisahkan. Sejalan dengan sumbu-Y
                    # fig2 yang juga tidak lagi menyebut "attention".
                    csv.append(dict(dataset=ds, backbone=MLAB[at],
                                    baseline=MLAB[cv], bin=b,
                                    n_seeds=len(d), gap_mean=d.mean(),
                                    gap_sd=d.std(ddof=1)))
                    for s, val in zip(common, d):
                        ps_rows.append(dict(dataset=ds, backbone=MLAB[at],
                                            baseline=MLAB[cv], bin=b,
                                            seed=int(s), gap=float(val)))
                # tren per seed = (high - low) tiap seed, baru diringkas
                t = tr["high"] - tr["low"]
                rows.append([f"{MLAB[at]} − {MLAB[cv]}"] + cells + [ms(t)])
                csv[-1]["trend_mean"] = t.mean()
                csv[-1]["trend_sd"] = t.std(ddof=1)
        pd.DataFrame(csv).to_csv(tdir / f"T3_gap_by_bin_{ds}.csv", index=False)
        pd.DataFrame(ps_rows).to_csv(
            tdir / "perseed" / f"T3_gap_by_bin_perseed_{ds}.csv", index=False)
        note = (HEAD + "\n\nGaps are computed per seed and then summarized, not as "
                "a difference of two means, so the reported spread reflects the "
                "pairs actually observed. The trend column is high minus low, "
                "computed per seed. It is descriptive and is not a test for trend.")
        write(tdir / f"T3_gap_by_bin_{ds}.md",
              f"T3b — Accuracy gap to the convolutional baselines, by bin, {lab}",
              note, md_table(rows, ["backbone vs baseline"] + BINS + ["trend (high − low)"],
                             [":---"] + [":---"] * (len(BINS) + 1)))

        # ---- T4 color reliance ------------------------------------------------
        if len(e2):
            rows, csv = [], []
            for a in MAIN_ARMS:
                c = e2[(e2.model == a) & (e2.condition == "clean")].set_index("seed").acc.sort_index()
                g = e2[(e2.model == a) & (e2.condition == "grayscale")].set_index("seed").acc.sort_index()
                common = c.index.intersection(g.index)
                cr = (c.loc[common] - g.loc[common]).values
                rows.append([MLAB[a], ms(c.loc[common].values),
                             ms(g.loc[common].values), ms(cr)])
                csv.append(dict(dataset=ds, arm=a, backbone=MLAB[a],
                                pretrain=PRETRAIN_TAG[a], n_seeds=len(cr),
                                clean_mean=c.loc[common].mean(),
                                clean_sd=c.loc[common].std(ddof=1),
                                gray_mean=g.loc[common].mean(),
                                gray_sd=g.loc[common].std(ddof=1),
                                CR_mean=cr.mean(), CR_sd=cr.std(ddof=1)))
            pd.DataFrame(csv).to_csv(tdir / f"T4_color_reliance_{ds}.csv", index=False)
            note4 = (HEAD + "\n\nColour reliance is the absolute drop, clean minus "
                     "grayscale, computed per seed. We do not report it as a ratio: "
                     "the denominator would differ across backbones and the values "
                     "would stop being comparable.")
            write(tdir / f"T4_color_reliance_{ds}.md",
                  f"T4 — Colour reliance, {lab}", note4,
                  md_table(rows, ["backbone", "clean", "grayscale", "CR"],
                           [":---", ":---", ":---", ":---"]))


if __name__ == "__main__":
    main()
