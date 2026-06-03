"""Threshold helpers shared by temporal study workflows."""

from __future__ import annotations

import numpy as np

from secom.metrics import candidate_thresholds, confusion_counts, predict_from_threshold, true_pos_rate

MAX_WEEKLY_FLAG_FRACTION = 0.10


def weekly_flag_fraction(scores: np.ndarray, threshold: float, week_labels: np.ndarray) -> float:
    """Return the mean weekly fraction of wafers flagged at ``threshold``."""
    predictions = predict_from_threshold(scores, threshold)
    weeks = np.asarray(week_labels, dtype=int)

    unique_weeks = np.unique(weeks)
    if unique_weeks.size == 0:
        return 0.0
    fractions = [float(np.mean(predictions[weeks == week])) for week in unique_weeks]
    return float(np.mean(fractions))


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
