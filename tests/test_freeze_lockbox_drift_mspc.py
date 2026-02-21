from __future__ import annotations

import numpy as np

from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold


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
