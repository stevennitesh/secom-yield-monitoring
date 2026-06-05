"""Tests for univariate, Gram-Schmidt, and ReliefF selector ranking."""

from __future__ import annotations

import sys

import numpy as np

from secom.config import EPS_SELECTOR, SelectorName
from secom.feature_select.gram_schmidt import gram_schmidt_rank_features
from secom.feature_select.relief import relief_rank_features
from secom.feature_select import univariate


def test_score_s2n_matches_hand_calculated_class_separation() -> None:
    """S2N should equal absolute mean gap over summed class standard deviations."""
    x = np.asarray(
        [
            [0.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [5.0, 7.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    scores = univariate.score_s2n(x, y)

    expected_feature_0 = 3.0 / (np.sqrt(2.0) + np.sqrt(2.0) + EPS_SELECTOR)
    assert np.isclose(scores[0], expected_feature_0)
    assert np.isneginf(scores[1])


def test_score_welch_t_matches_hand_calculated_class_separation() -> None:
    """Welch-t should use class variances scaled by class sample counts."""
    x = np.asarray(
        [
            [0.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [5.0, 7.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    scores = univariate.score_welch_t(x, y)

    expected_feature_0 = 3.0 / np.sqrt((2.0 / 2.0) + (2.0 / 2.0) + EPS_SELECTOR)
    assert np.isclose(scores[0], expected_feature_0)
    assert np.isneginf(scores[1])


def test_score_pooled_ttest_matches_hand_calculated_class_separation() -> None:
    """Pooled Ttest should use the common two-sample pooled variance estimate."""
    x = np.asarray(
        [
            [0.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [5.0, 7.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    scores = univariate.score_pooled_ttest(x, y)

    pooled_var = ((2 - 1) * 2.0 + (2 - 1) * 2.0) / (2 + 2 - 2)
    expected_feature_0 = 3.0 / np.sqrt(pooled_var * (1.0 / 2.0 + 1.0 / 2.0) + EPS_SELECTOR)
    assert np.isclose(scores[0], expected_feature_0)
    assert np.isneginf(scores[1])


def test_pooled_ttest_is_available_through_rank_features() -> None:
    """The UCI-style Ttest selector should be a first-class univariate selector."""
    x = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    order, scores = univariate.rank_features(SelectorName.TTEST, x, y)

    assert order.tolist() == [0, 1, 2]
    assert scores[0] > scores[1]
    assert np.isneginf(scores[2])


def test_score_f_test_skips_constant_columns(monkeypatch) -> None:
    """F-test scoring should omit constant columns from sklearn scoring."""
    x = np.array(
        [
            [1.0, 0.0, 5.0, 2.0],
            [1.0, 1.0, 5.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
            [1.0, 1.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)

    called = {}

    def fake_f_classif(x_sub: np.ndarray, y_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Record the scored shape and return fixed F statistics."""
        called["shape"] = x_sub.shape
        assert np.array_equal(y_sub, y)
        return np.array([7.0, 3.0], dtype=float), np.array([0.1, 0.2], dtype=float)

    monkeypatch.setattr(univariate, "f_classif", fake_f_classif)

    scores = univariate.score_f_test(x, y)

    assert called["shape"] == (4, 2)
    assert np.isneginf(scores[[0, 2]]).all()
    assert np.allclose(scores[[1, 3]], [7.0, 3.0])


def test_f_test_and_pearson_rank_binary_labels_monotonically() -> None:
    """Binary-label F-test and absolute Pearson should agree on feature order."""
    x = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 4.0],
            [1.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    f_order, f_scores = univariate.rank_features(SelectorName.F_TEST, x, y)
    pearson_order, pearson_scores = univariate.rank_features(SelectorName.PEARSON, x, y)

    assert f_order.tolist() == pearson_order.tolist() == [0, 1, 2]
    assert f_scores[0] > f_scores[1]
    assert pearson_scores[0] > pearson_scores[1]
    assert np.isneginf(f_scores[2])
    assert np.isneginf(pearson_scores[2])


def test_score_pearson_marks_constant_columns_as_bottom_rank() -> None:
    """Pearson scoring should rank constant columns below usable features."""
    x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    scores = univariate.score_pearson(x, y)

    assert np.isneginf(scores[0])
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], 1.0)


def test_rank_features_uses_lower_feature_index_for_score_ties(monkeypatch) -> None:
    """Feature ranking ties should preserve lower-index deterministic order."""
    monkeypatch.setitem(univariate._SCORERS, SelectorName.S2N, lambda x, y: np.asarray([1.0, -np.inf, 2.0, 2.0]))

    order, scores = univariate.rank_features(SelectorName.S2N, np.zeros((3, 4)), np.asarray([0, 1, 0]))

    assert order.tolist() == [2, 3, 0, 1]
    assert np.isneginf(scores[1])


def test_gram_schmidt_rank_features_skips_constant_columns() -> None:
    """Gram-Schmidt ranking should not select constant columns."""
    x = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    order, scores = gram_schmidt_rank_features(x, y, k=2)

    assert 0 not in order.tolist()
    assert np.isneginf(scores[0])


def test_gram_schmidt_rank_features_skips_redundant_correlated_feature() -> None:
    """Gram-Schmidt should prefer a nonredundant residual signal over a duplicate."""
    x = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)

    order, scores = gram_schmidt_rank_features(x, y, k=2)

    assert order.tolist() == [0, 2]
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_relief_rank_features_fallback_matches_known_neighborhood_geometry(monkeypatch) -> None:
    """Fallback ReliefF should reward nearest misses and penalize nearest hits."""
    monkeypatch.setitem(sys.modules, "skrebate", None)
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    order, scores = relief_rank_features(x, y, n_neighbors=1)

    assert order.tolist() == [0, 1]
    assert np.allclose(scores, [2.0, -1.0])


def test_relief_rank_features_fallback_uses_deterministic_order(monkeypatch) -> None:
    """Fallback ReliefF ranking should sanitize invalid scores deterministically."""
    import secom.feature_select.relief as relief

    def fake_fallback(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Return fixed fallback ReliefF scores including an invalid value."""
        return np.asarray([0.5, np.nan, 1.0, 1.0], dtype=float)

    monkeypatch.setitem(sys.modules, "skrebate", None)
    monkeypatch.setattr(relief, "_fallback_relief_scores", fake_fallback)

    order, scores = relief_rank_features(
        np.zeros((4, 4), dtype=float),
        np.asarray([0, 1, 0, 1], dtype=int),
        n_neighbors=1,
    )

    assert order.tolist() == [2, 3, 0, 1]
    assert np.isneginf(scores[1])
