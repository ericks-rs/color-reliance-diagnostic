"""Uji statistik berpasangan antar model (paired by seed).

n=5 seed: paired t-test (primary) + Wilcoxon signed-rank (robustness, non-param)
+ Cohen's dz (effect size paired). Catatan: Wilcoxon n=5 -> p minimum two-sided
0.0625, jadi t-test lebih bertenaga untuk n kecil; kami laporkan keduanya.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps


def paired_test(a, b):
    """a,b: array nilai metrik per seed (paired, urutan seed sama).
    Return dict: mean_diff (a-b), sd_diff, t, p_t, p_wilcoxon, dz, n."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1)) if n > 1 else 0.0
    # paired t-test
    if n > 1 and sd_diff > 0:
        t, p_t = sps.ttest_rel(a, b)
    else:
        t, p_t = (np.nan, np.nan)
    # Wilcoxon (butuh variasi & n>=1 nonzero diff)
    try:
        if np.any(d != 0):
            _, p_w = sps.wilcoxon(a, b)
        else:
            p_w = np.nan
    except Exception:
        p_w = np.nan
    dz = mean_diff / sd_diff if sd_diff > 0 else np.nan
    return {"mean_diff": mean_diff, "sd_diff": sd_diff, "t": float(t) if t==t else np.nan,
            "p_ttest": float(p_t) if p_t == p_t else np.nan,
            "p_wilcoxon": float(p_w) if p_w == p_w else np.nan,
            "cohen_dz": float(dz) if dz == dz else np.nan, "n": n}


def _pivot_metric_per_seed(df, value_col, model_col="model", seed_col="seed"):
    """Return dict[model] -> pd.Series(index=seed, value)."""
    out = {}
    for m, g in df.groupby(model_col):
        out[m] = g.set_index(seed_col)[value_col].sort_index()
    return out


def clean_acc_per_seed(e1, dataset):
    sub = e1[(e1.dataset == dataset) & (e1.bin == "overall")]
    return _pivot_metric_per_seed(sub, "acc")


def cr_per_seed(e2, dataset):
    sub = e2[(e2.dataset == dataset) & (e2.condition.isin(["clean", "grayscale"]))]
    piv = sub.pivot_table(index=["model", "seed"], columns="condition",
                          values="acc").reset_index()
    piv["CR"] = piv["clean"] - piv["grayscale"]
    return _pivot_metric_per_seed(piv, "CR")


def bin_acc_per_seed(e1, dataset, bin_name):
    sub = e1[(e1.dataset == dataset) & (e1.bin == bin_name)]
    return _pivot_metric_per_seed(sub, "acc")


def comparison_table(per_seed, comparisons, dataset, metric_name):
    """per_seed: dict[model]->Series. comparisons: list of (a_model,b_model).
    Hanya pakai seed yang sama-sama ada (intersection)."""
    rows = []
    for a, b in comparisons:
        if a not in per_seed or b not in per_seed:
            continue
        sa, sb = per_seed[a], per_seed[b]
        common = sa.index.intersection(sb.index)
        r = paired_test(sa.loc[common].values, sb.loc[common].values)
        rows.append({"dataset": dataset, "metric": metric_name,
                     "comparison": f"{a} - {b}", **r})
    return pd.DataFrame(rows)
