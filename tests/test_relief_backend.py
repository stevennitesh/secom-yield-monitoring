"""Numerical regression tests against the pinned upstream ReliefF calculation."""

from __future__ import annotations

import numpy as np
import pytest
from skrebate import ReliefF

from secom.feature_select._relief_backend import VectorizedReliefF
from secom.feature_select.relief import relief_rank_features


@pytest.mark.parametrize("kind", ["continuous", "categorical", "mixed"])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("neighbors", [1, 5, 20])
def test_vectorized_scores_and_neighbors_exactly_match_reference(kind, layout, neighbors):
    """Cover ramp semantics, memory-layout rounding, ties and undersized failure classes."""
    rng = np.random.default_rng(23)
    continuous = rng.normal(size=(48, 16))
    categorical = rng.integers(0, 3, size=(48, 5)).astype(float)
    x = {
        "continuous": continuous,
        "categorical": categorical,
        "mixed": np.column_stack((continuous, categorical, np.ones(48))),
    }[kind]
    x = np.array(x, order=layout)
    y = np.zeros(48, dtype=int)
    y[[0, 5, 12]] = 1
    slow = ReliefF(n_neighbors=neighbors, n_jobs=1).fit(x, y)
    fast = VectorizedReliefF(n_neighbors=neighbors, n_jobs=1).fit(x, y)
    np.testing.assert_array_equal(fast.feature_importances_, slow.feature_importances_)
    np.testing.assert_array_equal(fast.top_features_, slow.top_features_)


def test_vectorized_backend_rejects_unverified_dependency(monkeypatch):
    """A dependency upgrade requires repeating equivalence validation first."""
    monkeypatch.setattr("secom.feature_select._relief_backend.version", lambda _name: "future-version")
    with pytest.raises(RuntimeError, match="verified skrebate"):
        VectorizedReliefF().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))


def test_vectorized_backend_rejects_inputs_outside_study_contract():
    with pytest.raises(ValueError, match="finite, imputed"):
        VectorizedReliefF().fit(np.array([[np.nan], [1.0]]), np.array([0, 1]))


def test_backend_choice_has_separate_score_cache(monkeypatch):
    """Reference runs must execute independently of cached accelerated scores."""
    from secom.feature_select import relief

    relief._SCORE_CACHE.clear()
    x = np.random.default_rng(3).normal(size=(30, 4))
    y = np.tile([0, 1], 15)
    monkeypatch.setenv("SECOM_RELIEF_N_JOBS", "1")
    monkeypatch.setenv("SECOM_RELIEF_BACKEND", "vectorized")
    fast_order, fast_scores = relief_rank_features(x, y, 3)
    monkeypatch.setenv("SECOM_RELIEF_BACKEND", "reference")
    slow_order, slow_scores = relief_rank_features(x, y, 3)
    assert len(relief._SCORE_CACHE) == 2
    np.testing.assert_array_equal(fast_scores, slow_scores)
    np.testing.assert_array_equal(fast_order, slow_order)
