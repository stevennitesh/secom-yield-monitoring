from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryCounts:
    tn: int
    fp: int
    fn: int
    tp: int


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryCounts:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return BinaryCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def true_pos_rate(counts: BinaryCounts) -> float:
    denom = counts.tp + counts.fn
    if denom == 0:
        return 0.0
    return counts.tp / denom


def true_neg_rate(counts: BinaryCounts) -> float:
    denom = counts.tn + counts.fp
    if denom == 0:
        return 0.0
    return counts.tn / denom


def ber_from_counts(counts: BinaryCounts) -> float:
    tpr = true_pos_rate(counts)
    tnr = true_neg_rate(counts)
    return 1.0 - 0.5 * (tpr + tnr)


def predict_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=float))
    return np.concatenate((np.array([-np.inf]), uniq, np.array([np.inf])))


def find_ber_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_threshold = None
    best_ber = np.inf
    best_tpr = -np.inf

    for threshold in candidate_thresholds(scores):
        y_pred = predict_from_threshold(scores, float(threshold))
        counts = confusion_counts(y_true, y_pred)
        ber = ber_from_counts(counts)
        tpr = true_pos_rate(counts)

        is_better = False
        if ber < best_ber:
            is_better = True
        elif np.isclose(ber, best_ber):
            if tpr > best_tpr:
                is_better = True
            elif np.isclose(tpr, best_tpr):
                if best_threshold is None or float(threshold) < float(best_threshold):
                    is_better = True
        if is_better:
            best_threshold = float(threshold)
            best_ber = float(ber)
            best_tpr = float(tpr)

    assert best_threshold is not None
    y_best = predict_from_threshold(scores, best_threshold)
    counts_best = confusion_counts(y_true, y_best)
    return best_threshold, {
        "BER": ber_from_counts(counts_best),
        "True+": true_pos_rate(counts_best),
        "True-": true_neg_rate(counts_best),
    }


def extract_tpr_at_tnr(
    y_true: np.ndarray, scores: np.ndarray, target_tnr: float = 0.90
) -> tuple[float, float, float]:
    best_threshold = None
    best_tpr = -np.inf
    best_tnr = 0.0
    for threshold in candidate_thresholds(scores):
        y_pred = predict_from_threshold(scores, float(threshold))
        counts = confusion_counts(y_true, y_pred)
        tnr = true_neg_rate(counts)
        tpr = true_pos_rate(counts)
        if tnr >= target_tnr:
            if tpr > best_tpr:
                best_tpr = tpr
                best_tnr = tnr
                best_threshold = float(threshold)
            elif np.isclose(tpr, best_tpr):
                # Highest-TPR threshold with TNR>=target, tie -> lowest threshold.
                if best_threshold is None or float(threshold) < float(best_threshold):
                    best_threshold = float(threshold)
                    best_tnr = tnr
    if best_threshold is None:
        # If no threshold reaches target, use highest-threshold fallback.
        fallback = float(np.max(candidate_thresholds(scores)))
        y_pred = predict_from_threshold(scores, fallback)
        c = confusion_counts(y_true, y_pred)
        return fallback, true_neg_rate(c), true_pos_rate(c)
    return best_threshold, float(best_tnr), float(best_tpr)


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = predict_from_threshold(scores, threshold)
    counts = confusion_counts(y_true, y_pred)

    metrics = {
        "BER": ber_from_counts(counts),
        "True+": true_pos_rate(counts),
        "True-": true_neg_rate(counts),
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F2": fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0),
        "lockbox_n": float(len(y_true)),
        "lockbox_fails": float(np.sum(y_true == 1)),
        "FP": float(counts.fp),
        "FN": float(counts.fn),
    }

    # AUC metrics need both classes to be present.
    if np.unique(y_true).size == 2:
        metrics["ROC_AUC"] = roc_auc_score(y_true=y_true, y_score=scores)
        metrics["PR_AUC"] = average_precision_score(y_true=y_true, y_score=scores)
    return metrics


def bootstrap_ci_for_mean(
    values: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.95
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    idx = np.arange(vals.size)
    for i in range(n_boot):
        draw = rng.choice(idx, size=vals.size, replace=True)
        means[i] = float(np.mean(vals[draw]))
    lower_q = (1 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return (float(np.quantile(means, lower_q)), float(np.quantile(means, upper_q)))


def paired_bootstrap_delta_ci(
    left: np.ndarray,
    right: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.95,
) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired arrays must have same shape")
    deltas = left - right
    return bootstrap_ci_for_mean(deltas, n_boot=n_boot, seed=seed, alpha=alpha)


def expected_cost_per_wafer(fp: float, fn: float, n: float, cost_ratio: float) -> float:
    if n <= 0:
        return np.nan
    c_fp = 1.0
    c_fn = float(cost_ratio)
    return float((c_fp * fp + c_fn * fn) / n)

