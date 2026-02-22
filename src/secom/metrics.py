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


def _is_better_threshold(
    ber: float,
    tpr: float,
    threshold: float,
    best_ber: float,
    best_tpr: float,
    best_threshold: float | None,
) -> bool:
    if ber < best_ber:
        return True
    if np.isclose(ber, best_ber):
        if tpr > best_tpr:
            return True
        if np.isclose(tpr, best_tpr):
            if best_threshold is None or float(threshold) < float(best_threshold):
                return True
    return False


def _find_ber_optimal_threshold_bruteforce(
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
        if _is_better_threshold(
            ber=float(ber),
            tpr=float(tpr),
            threshold=float(threshold),
            best_ber=float(best_ber),
            best_tpr=float(best_tpr),
            best_threshold=best_threshold,
        ):
            best_threshold = float(threshold)
            best_ber = float(ber)
            best_tpr = float(tpr)

    if best_threshold is None:
        raise RuntimeError("Failed to find BER-optimal threshold")
    y_best = predict_from_threshold(scores, best_threshold)
    counts_best = confusion_counts(y_true, y_best)
    return best_threshold, {
        "BER": ber_from_counts(counts_best),
        "True+": true_pos_rate(counts_best),
        "True-": true_neg_rate(counts_best),
    }


def find_ber_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    y_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    if y_arr.size != scores_arr.size:
        raise ValueError("y_true and scores must have identical length")
    if y_arr.size == 0:
        raise ValueError("Cannot optimize threshold on empty arrays")
    if not np.all(np.isfinite(scores_arr)):
        return _find_ber_optimal_threshold_bruteforce(y_arr, scores_arr)

    order = np.argsort(scores_arr, kind="mergesort")[::-1]
    sorted_scores = scores_arr[order]
    sorted_y = y_arr[order]

    n_pos_total = int(np.sum(sorted_y == 1))
    n_neg_total = int(sorted_y.size - n_pos_total)
    tp = 0
    fp = 0
    fn = n_pos_total
    tn = n_neg_total

    best_threshold: float | None = None
    best_ber = np.inf
    best_tpr = -np.inf
    best_counts: BinaryCounts | None = None

    # threshold = +inf (predict all negatives)
    tpr = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    tnr = 0.0 if (tn + fp) == 0 else float(tn / (tn + fp))
    ber = float(1.0 - 0.5 * (tpr + tnr))
    if _is_better_threshold(ber, tpr, float(np.inf), best_ber, best_tpr, best_threshold):
        best_threshold = float(np.inf)
        best_ber = ber
        best_tpr = tpr
        best_counts = BinaryCounts(tn=tn, fp=fp, fn=fn, tp=tp)

    i = 0
    n = int(sorted_scores.size)
    while i < n:
        score_value = float(sorted_scores[i])
        j = i
        group_pos = 0
        while j < n and float(sorted_scores[j]) == score_value:
            group_pos += int(sorted_y[j] == 1)
            j += 1
        group_size = j - i
        group_neg = group_size - group_pos

        tp += group_pos
        fp += group_neg
        fn -= group_pos
        tn -= group_neg

        tpr = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
        tnr = 0.0 if (tn + fp) == 0 else float(tn / (tn + fp))
        ber = float(1.0 - 0.5 * (tpr + tnr))
        if _is_better_threshold(ber, tpr, score_value, best_ber, best_tpr, best_threshold):
            best_threshold = score_value
            best_ber = ber
            best_tpr = tpr
            best_counts = BinaryCounts(tn=tn, fp=fp, fn=fn, tp=tp)
        i = j

    # threshold = -inf (predict all positives), ties break to smallest threshold
    tpr = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    tnr = 0.0 if (tn + fp) == 0 else float(tn / (tn + fp))
    ber = float(1.0 - 0.5 * (tpr + tnr))
    if _is_better_threshold(ber, tpr, float(-np.inf), best_ber, best_tpr, best_threshold):
        best_threshold = float(-np.inf)
        best_counts = BinaryCounts(tn=tn, fp=fp, fn=fn, tp=tp)

    if best_threshold is None or best_counts is None:
        raise RuntimeError("Failed to find BER-optimal threshold")
    return best_threshold, {
        "BER": ber_from_counts(best_counts),
        "True+": true_pos_rate(best_counts),
        "True-": true_neg_rate(best_counts),
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
