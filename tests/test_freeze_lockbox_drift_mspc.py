from __future__ import annotations

import numpy as np

from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold
from secom.workflows.freeze_lockbox import _manager_weekly_metrics


def test_operational_threshold_enforces_cap() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.05, 0.3, 0.4])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    weeks = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    t = operational_threshold(scores=scores, y_true=y, week_labels=weeks)
    preds = (scores >= t).astype(int)
    week_means = [preds[weeks == w].mean() for w in sorted(np.unique(weeks))]
    assert np.mean(week_means) <= 0.10 + 1e-9


def test_psi_feature_handles_missing_and_out_of_range() -> None:
    dev = np.array([1.0, 1.2, 1.1, np.nan, 0.9, 1.05])
    lock = np.array([3.0, 3.1, np.nan, 2.9, 3.2, 3.3])
    psi = psi_for_feature(dev, lock)
    assert np.isfinite(psi)
    assert psi >= 0.0


def test_manager_weekly_metrics_aggregates_weekly_counts() -> None:
    y = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.2, 0.8, 0.7, 0.4, 0.3], dtype=float)
    weeks = np.array([1, 1, 2, 2, 3, 3], dtype=int)

    metrics = _manager_weekly_metrics(
        y_true=y,
        scores=scores,
        threshold=0.5,
        week_labels=weeks,
    )

    assert metrics["dev_sample_count"] == 6.0
    assert metrics["dev_week_count"] == 3.0
    assert metrics["weekly_rate"] == 2.0
    assert np.isclose(metrics["predicted_flag_fraction"], 0.5)
    assert np.isclose(metrics["mean_weekly_flagged_wafers"], 1.0)
    assert np.isclose(metrics["mean_weekly_fail_captures"], 2.0 / 3.0)
    assert np.isclose(metrics["mean_weekly_fail_misses"], 1.0 / 3.0)
