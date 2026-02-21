from __future__ import annotations

import numpy as np

from secom.metrics import candidate_thresholds, confusion_counts, true_pos_rate


def weekly_flag_fraction(scores: np.ndarray, threshold: float, week_labels: np.ndarray) -> float:
    preds = (scores >= threshold).astype(int)
    weeks = np.asarray(week_labels, dtype=int)
    fractions: list[float] = []
    for w in sorted(np.unique(weeks).tolist()):
        idx = np.where(weeks == w)[0]
        if idx.size == 0:
            continue
        fractions.append(float(np.mean(preds[idx])))
    if not fractions:
        return 0.0
    return float(np.mean(fractions))


def operational_threshold(scores: np.ndarray, y_true: np.ndarray, week_labels: np.ndarray) -> float:
    best_threshold = None
    best_tpr = -np.inf
    for t in candidate_thresholds(scores):
        frac = weekly_flag_fraction(scores=scores, threshold=float(t), week_labels=week_labels)
        if frac <= 0.10:
            counts = confusion_counts(y_true, (scores >= t).astype(int))
            tpr = true_pos_rate(counts)
            if tpr > best_tpr:
                best_tpr = tpr
                best_threshold = float(t)
            elif np.isclose(tpr, best_tpr):
                if best_threshold is None or float(t) < best_threshold:
                    best_threshold = float(t)
    if best_threshold is None:
        return float(np.inf)
    return best_threshold
