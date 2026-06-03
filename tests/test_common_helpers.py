from __future__ import annotations

import numpy as np
import pytest

from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold, weekly_flag_fraction
from secom.metrics import candidate_thresholds, confusion_counts, predict_from_threshold, true_pos_rate
from tests.assertions import threshold_equal


def _bruteforce_operational_threshold(scores: np.ndarray, y_true: np.ndarray, week_labels: np.ndarray) -> float:
    best_threshold: float | None = None
    best_tpr = -np.inf
    for candidate in candidate_thresholds(scores):
        threshold = float(candidate)
        if weekly_flag_fraction(scores=scores, threshold=threshold, week_labels=week_labels) > 0.10:
            continue

        counts = confusion_counts(y_true, predict_from_threshold(scores, threshold))
        tpr = true_pos_rate(counts)
        if tpr > best_tpr or (np.isclose(tpr, best_tpr) and (best_threshold is None or threshold < best_threshold)):
            best_tpr = tpr
            best_threshold = threshold
    return float(np.inf) if best_threshold is None else best_threshold


def test_psi_for_feature_is_zero_for_matching_distributions() -> None:
    values = np.asarray([0.1, 0.2, 0.3, np.nan, 0.4, 0.5], dtype=float)

    assert psi_for_feature(values, values.copy()) == 0.0


def test_psi_for_feature_counts_missing_values_as_explicit_shift() -> None:
    dev_values = np.asarray([0.0, 0.1, 0.2, 0.3, np.nan], dtype=float)
    lock_values = np.asarray([0.0, np.nan, np.nan, np.nan, np.nan], dtype=float)

    assert psi_for_feature(dev_values, lock_values) > 0.0


def test_weekly_flag_fraction_averages_each_week_equally() -> None:
    scores = np.asarray([0.9, 0.1, 0.8, 0.7, 0.2], dtype=float)
    week_labels = np.asarray([2, 2, 5, 5, 5], dtype=int)

    assert np.isclose(weekly_flag_fraction(scores, threshold=0.75, week_labels=week_labels), 5.0 / 12.0)


def test_operational_threshold_chooses_lowest_threshold_among_max_tpr_under_weekly_cap() -> None:
    scores = np.asarray([0.90, 0.80, 0.70, *([0.10] * 7)] * 2, dtype=float)
    y_true = np.asarray([1, 1, 1, *([0] * 7)] * 2, dtype=int)
    week_labels = np.asarray([0] * 10 + [1] * 10, dtype=int)

    assert operational_threshold(scores, y_true, week_labels) == 0.90


def test_operational_threshold_returns_infinity_when_cap_cannot_be_satisfied() -> None:
    scores = np.asarray([0.5, 0.5, 0.5], dtype=float)
    y_true = np.asarray([1, 0, 1], dtype=int)
    week_labels = np.asarray([0, 0, 0], dtype=int)

    assert np.isinf(operational_threshold(scores, y_true, week_labels))


def test_operational_threshold_rejects_mismatched_input_lengths() -> None:
    scores = np.asarray([0.5, 0.4, 0.3], dtype=float)
    y_true = np.asarray([1, 0], dtype=int)
    week_labels = np.asarray([0, 0, 0], dtype=int)

    with pytest.raises(ValueError, match="identical length"):
        operational_threshold(scores, y_true, week_labels)


def test_operational_threshold_matches_bruteforce_random() -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        scores = rng.normal(loc=0.0, scale=1.0, size=120).astype(float)
        y_true = rng.integers(0, 2, size=scores.size, dtype=int)
        week_labels = rng.integers(1, 7, size=scores.size, dtype=int)

        assert threshold_equal(
            operational_threshold(scores, y_true, week_labels),
            _bruteforce_operational_threshold(scores, y_true, week_labels),
        )


def test_operational_threshold_nonfinite_scores_fallback_matches_bruteforce() -> None:
    scores = np.asarray([0.8, np.nan, 0.7, np.inf, -np.inf, 0.2, 0.1], dtype=float)
    y_true = np.asarray([1, 0, 1, 0, 0, 1, 0], dtype=int)
    week_labels = np.asarray([1, 1, 2, 2, 3, 3, 3], dtype=int)

    assert threshold_equal(
        operational_threshold(scores, y_true, week_labels),
        _bruteforce_operational_threshold(scores, y_true, week_labels),
    )
