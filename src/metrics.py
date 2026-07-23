"""Metrik evaluasi: top-1 acc, macro-F1, UAR (balanced accuracy), confusion."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes)) if num_classes else None
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def metrics_by_group(y_true, y_pred, groups, num_classes=None):
    """Hitung metrik per grup (mis. colorfulness bin). groups: array label grup.
    Return dict[group_value] -> metric dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[g] = compute_metrics(y_true[mask], y_pred[mask], num_classes)
    return out
