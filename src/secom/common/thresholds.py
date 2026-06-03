"""Threshold helpers shared by temporal study workflows."""

from __future__ import annotations

import numpy as np

from secom.metrics import candidate_thresholds, confusion_counts, predict_from_threshold, true_pos_rate

MAX_WEEKLY_FLAG_FRACTION = 0.10


def weekly_flag_fraction(scores: np.ndarray, threshold: float, week_labels: np.ndarray) -> float:
    """Return the mean weekly fraction of wafers flagged at ``threshold``."""
    predictions = predict_from_threshold(scores, threshold)
    weeks = np.asarray(week_labels, dtype=int)

    if weeks.size == 0:
        return 0.0
    _unique_weeks, week_codes = np.unique(weeks, return_inverse=True)
    week_counts = np.bincount(week_codes)
    flagged_counts = np.bincount(week_codes, weights=predictions.astype(float), minlength=week_counts.size)
    return float(np.mean(flagged_counts / week_counts))


def _has_better_operating_tpr(tpr: float, threshold: float, best_tpr: float, best_threshold: float | None) -> bool:
    """Prefer higher TPR, then the lowest threshold for exact ties."""
    if tpr > best_tpr:
        return True
    if np.isclose(tpr, best_tpr):
        return best_threshold is None or threshold < best_threshold
    return False


def operational_threshold(scores: np.ndarray, y_true: np.ndarray, week_labels: np.ndarray) -> float:
    """Choose the lowest-threshold max-TPR operating point under the weekly flag cap."""
    scores_arr = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y_true, dtype=int)
    weeks = np.asarray(week_labels, dtype=int)
    if y_arr.size != scores_arr.size or weeks.size != scores_arr.size:
        raise ValueError("scores, y_true, and week_labels must have identical length")
    if not np.all(np.isfinite(scores_arr)):
        return _operational_threshold_bruteforce(scores_arr=scores_arr, y_arr=y_arr, weeks=weeks)

    _unique_weeks, week_codes = np.unique(weeks, return_inverse=True)
    week_counts = np.bincount(week_codes)
    flagged_counts = np.zeros(week_counts.size, dtype=float)
    n_pos_total = int(np.sum(y_arr == 1))
    tp = 0
    fn = n_pos_total

    best_threshold: float | None = None
    best_tpr = -np.inf
    order = np.argsort(scores_arr, kind="mergesort")[::-1]
    sorted_scores = scores_arr[order]
    sorted_y = y_arr[order]
    sorted_week_codes = week_codes[order]

    def consider(threshold: float) -> None:
        nonlocal best_threshold, best_tpr
        flag_fraction = float(np.mean(flagged_counts / week_counts)) if week_counts.size else 0.0
        if flag_fraction > MAX_WEEKLY_FLAG_FRACTION:
            return
        tpr = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
        if _has_better_operating_tpr(tpr, threshold, best_tpr, best_threshold):
            best_tpr = tpr
            best_threshold = threshold

    consider(float(np.inf))

    i = 0
    n = int(sorted_scores.size)
    while i < n:
        score_value = float(sorted_scores[i])
        j = i
        group_pos = 0
        while j < n and float(sorted_scores[j]) == score_value:
            group_pos += int(sorted_y[j] == 1)
            j += 1
        group_week_codes = sorted_week_codes[i:j]
        flagged_counts += np.bincount(group_week_codes, minlength=week_counts.size)
        tp += group_pos
        fn -= group_pos
        consider(score_value)
        i = j

    consider(float(-np.inf))

    if best_threshold is None:
        # No candidate can satisfy the operations cap, so downstream scoring sees no positives.
        return float(np.inf)
    return best_threshold


def _operational_threshold_bruteforce(*, scores_arr: np.ndarray, y_arr: np.ndarray, weeks: np.ndarray) -> float:
    """Slow operational-threshold path for non-finite score sentinels."""
    best_threshold: float | None = None
    best_tpr = -np.inf
    for candidate in candidate_thresholds(scores_arr):
        threshold = float(candidate)
        flag_fraction = weekly_flag_fraction(scores=scores_arr, threshold=threshold, week_labels=weeks)
        if flag_fraction > MAX_WEEKLY_FLAG_FRACTION:
            continue

        counts = confusion_counts(y_arr, predict_from_threshold(scores_arr, threshold))
        tpr = true_pos_rate(counts)
        if _has_better_operating_tpr(tpr, threshold, best_tpr, best_threshold):
            best_tpr = tpr
            best_threshold = threshold

    if best_threshold is None:
        # No candidate can satisfy the operations cap, so downstream scoring sees no positives.
        return float(np.inf)
    return best_threshold
