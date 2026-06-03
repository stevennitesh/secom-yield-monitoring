from __future__ import annotations

import numpy as np

from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold, weekly_flag_fraction


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
