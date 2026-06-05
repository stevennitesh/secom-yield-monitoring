"""Tests for feature-selection dispatch, preprocessing, and tuning."""

from __future__ import annotations

import numpy as np
import pytest

from secom.config import ScalerName, SelectorName
from secom.selection import engine
from secom.selection.engine import fit_selector_pipeline, select_features
from secom.selection.tuning import select_best_inner_config


def test_select_features_applies_top_k_after_univariate_dispatch(monkeypatch) -> None:
    """Univariate dispatch should rank first and then apply top-k truncation."""
    scores = np.asarray([0.2, 0.1, 0.3], dtype=float)

    def fake_rank_features(method: str, x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic ranking while asserting dispatch metadata."""
        assert method == SelectorName.S2N
        return np.asarray([2, 0, 1], dtype=int), scores

    monkeypatch.setattr(engine, "rank_features", fake_rank_features)

    selected, returned_scores = select_features(
        method=SelectorName.S2N,
        x_train=np.zeros((4, 3), dtype=float),
        y_train=np.asarray([0, 1, 0, 1], dtype=int),
        k=2,
    )

    assert selected.tolist() == [2, 0]
    assert returned_scores is scores


def test_select_features_requires_relief_neighbors() -> None:
    """ReliefF selection should reject calls without a neighbor count."""
    with pytest.raises(ValueError, match="ReliefF requires n_neighbors"):
        select_features(
            method=SelectorName.RELIEFF,
            x_train=np.zeros((4, 3), dtype=float),
            y_train=np.asarray([0, 1, 0, 1], dtype=int),
            k=2,
        )


def test_fit_selector_pipeline_returns_selected_views_and_metadata() -> None:
    """Selector preprocessing should return train/eval views and feature metadata."""
    x_train = np.asarray(
        [
            [0.0, np.nan, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 3.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    x_eval = np.asarray([[4.0, np.nan, 1.0], [5.0, 6.0, 0.0]], dtype=float)
    y_train = np.asarray([0, 0, 1, 1], dtype=int)

    x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler = fit_selector_pipeline(
        x_train_raw=x_train,
        y_train=y_train,
        x_eval_raw=x_eval,
        method=SelectorName.S2N,
        k=2,
        scaler_name=ScalerName.STANDARD,
        add_indicator=True,
        n_neighbors=None,
    )

    assert x_train_sel.shape == (4, 2)
    assert x_eval_sel.shape == (2, 2)
    assert selected_local.shape == (2,)
    assert len(feature_meta) == imputer.transform(x_train).shape[1]
    assert scaler is not None


def test_fit_selector_pipeline_scores_imputed_and_scaled_training_data(monkeypatch) -> None:
    """Selector scoring should receive transformed training data, not raw values."""
    x_train = np.asarray(
        [
            [1.0, np.nan],
            [3.0, 1.0],
            [5.0, 3.0],
            [7.0, 5.0],
        ],
        dtype=float,
    )
    x_eval = np.asarray([[9.0, np.nan]], dtype=float)
    y_train = np.asarray([0, 0, 1, 1], dtype=int)
    captured: dict[str, np.ndarray] = {}

    def fake_rank_features(method: str, x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Capture selector input after preprocessing."""
        captured["x_train"] = x_train
        assert method == SelectorName.S2N
        assert np.array_equal(y_train, np.asarray([0, 0, 1, 1], dtype=int))
        return np.asarray([0, 1], dtype=int), np.asarray([1.0, 0.5], dtype=float)

    monkeypatch.setattr(engine, "rank_features", fake_rank_features)

    fit_selector_pipeline(
        x_train_raw=x_train,
        y_train=y_train,
        x_eval_raw=x_eval,
        method=SelectorName.S2N,
        k=1,
        scaler_name=ScalerName.STANDARD,
        add_indicator=False,
        n_neighbors=None,
    )

    assert not np.isnan(captured["x_train"]).any()
    assert np.allclose(np.mean(captured["x_train"], axis=0), [0.0, 0.0])
    assert np.allclose(np.std(captured["x_train"], axis=0, ddof=0), [1.0, 1.0])


def test_fit_selector_pipeline_rejects_mismatched_feature_counts() -> None:
    """Train and evaluation matrices must share the same raw feature width."""
    with pytest.raises(ValueError, match="same feature count"):
        fit_selector_pipeline(
            x_train_raw=np.zeros((4, 3), dtype=float),
            y_train=np.asarray([0, 1, 0, 1], dtype=int),
            x_eval_raw=np.zeros((2, 2), dtype=float),
            method=SelectorName.S2N,
            k=2,
            scaler_name=ScalerName.STANDARD,
            add_indicator=False,
            n_neighbors=None,
        )


def test_fit_selector_pipeline_rejects_mismatched_label_length() -> None:
    """Training labels must align one-to-one with training rows."""
    with pytest.raises(ValueError, match="y_train length"):
        fit_selector_pipeline(
            x_train_raw=np.zeros((4, 3), dtype=float),
            y_train=np.asarray([0, 1, 0], dtype=int),
            x_eval_raw=np.zeros((2, 3), dtype=float),
            method=SelectorName.S2N,
            k=2,
            scaler_name=ScalerName.STANDARD,
            add_indicator=False,
            n_neighbors=None,
        )


def test_fit_selector_pipeline_metadata_ignores_eval_only_missingness() -> None:
    """Missing-indicator metadata should be learned from training data only."""
    x_train = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )
    x_eval = np.asarray([[4.0, np.nan]], dtype=float)
    y_train = np.asarray([0, 0, 1, 1], dtype=int)

    _x_train_sel, _x_eval_sel, feature_meta, _selected_local, imputer, _scaler = fit_selector_pipeline(
        x_train_raw=x_train,
        y_train=y_train,
        x_eval_raw=x_eval,
        method=SelectorName.S2N,
        k=2,
        scaler_name=ScalerName.STANDARD,
        add_indicator=True,
        n_neighbors=None,
    )

    assert len(feature_meta) == 2
    assert imputer.transform(x_eval).shape[1] == 2


def test_select_best_inner_config_prefers_ber_within_near_best_auc_band() -> None:
    """Inner tuning should prefer BER inside the near-best AUC tolerance."""
    selected = select_best_inner_config(
        [
            {"mean_inner_ROC_AUC": 0.900, "mean_inner_BER": 0.30, "k": 10, "C": 1.0, "scaler": ScalerName.STANDARD},
            {"mean_inner_ROC_AUC": 0.891, "mean_inner_BER": 0.20, "k": 20, "C": 1.0, "scaler": ScalerName.STANDARD},
            {"mean_inner_ROC_AUC": 0.880, "mean_inner_BER": 0.10, "k": 5, "C": 1.0, "scaler": ScalerName.STANDARD},
        ]
    )

    assert selected["k"] == 20


def test_select_best_inner_config_uses_deterministic_simplicity_tie_breaks() -> None:
    """Equivalent inner configs should resolve through deterministic simplicity keys."""
    rows = [
        {
            "mean_inner_ROC_AUC": 0.80,
            "mean_inner_BER": 0.20,
            "k": 10,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": None,
        },
        {
            "mean_inner_ROC_AUC": 0.80,
            "mean_inner_BER": 0.20,
            "k": 10,
            "C": 1.0,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": 5,
        },
        {
            "mean_inner_ROC_AUC": 0.80,
            "mean_inner_BER": 0.20,
            "k": 20,
            "C": 0.1,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": None,
        },
    ]

    selected = select_best_inner_config(rows)

    assert selected["scaler"] == ScalerName.STANDARD
    assert selected["n_neighbors"] == 5
