"""Threshold helpers shared by temporal study workflows."""

from __future__ import annotations

import numpy as np

from secom.metrics import candidate_thresholds, confusion_counts, true_pos_rate

MAX_WEEKLY_FLAG_FRACTION = 0.10


def weekly_flag_fraction(scores: np.ndarray, threshold: float, week_labels: np.ndarray) -> float:
    """Return the mean weekly fraction of wafers flagged at ``threshold``."""
    predictions = (scores >= threshold).astype(int)
    weeks = np.asarray(week_labels, dtype=int)

    fractions: list[float] = []
    for week in sorted(np.unique(weeks).tolist()):
        week_indices = np.where(weeks == week)[0]
        if week_indices.size == 0:
            continue
        fractions.append(float(np.mean(predictions[week_indices])))

    if not fractions:
        return 0.0
    return float(np.mean(fractions))


def operational_threshold(scores: np.ndarray, y_true: np.ndarray, week_labels: np.ndarray) -> float:
    """Choose the lowest-threshold max-TPR operating point under the weekly flag cap."""
    best_threshold: float | None = None
    best_tpr = -np.inf
    for candidate in candidate_thresholds(scores):
        threshold = float(candidate)
        flag_fraction = weekly_flag_fraction(scores=scores, threshold=threshold, week_labels=week_labels)
        if flag_fraction > MAX_WEEKLY_FLAG_FRACTION:
            continue

        counts = confusion_counts(y_true, (scores >= threshold).astype(int))
        tpr = true_pos_rate(counts)
        if tpr > best_tpr:
            best_tpr = tpr
            best_threshold = threshold
        elif np.isclose(tpr, best_tpr):
            if best_threshold is None or threshold < best_threshold:
                best_threshold = threshold

    if best_threshold is None:
        # No candidate can satisfy the operations cap, so downstream scoring sees no positives.
        return float(np.inf)
    return best_threshold
