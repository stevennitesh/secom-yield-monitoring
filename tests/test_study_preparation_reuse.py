"""Exact scientific and layout controls for local feature-budget reuse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from secom.config import ScalerName, SelectorName
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold
from secom.workflows import benchmark_common, benchmark_tuned, temporal_robustness


SELECTOR_CONFIGS = [
    (selector, neighbors)
    for selector in SelectorName.ACTIVE
    for neighbors in ([5, 10, 20] if selector == SelectorName.RELIEFF else [None])
]


def _missing_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(72)
    x = rng.normal(size=(110, 55))
    y = np.tile([0, 1], 55)
    x[:, :3] += y[:, None]
    x[::4, 3:8] = np.nan
    x[:, -2] = 1
    x[:, -1] = np.nan
    return x[:90], y[:90], x[90:], y[90:]


def _assert_same_matrix(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)
    assert actual.strides == expected.strides
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous


@pytest.mark.parametrize("selector,neighbors", SELECTOR_CONFIGS)
@pytest.mark.parametrize("add_indicator", [False, True])
@pytest.mark.parametrize("scaler", ScalerName.ALL)
def test_all_selector_budget_prefixes_match_independent_fits_and_model_scores(
    selector: str, neighbors: int | None, add_indicator: bool, scaler: str
) -> None:
    x_train, y_train, x_eval, y_eval = _missing_dataset()
    kwargs = dict(
        x_train_raw=x_train,
        y_train=y_train,
        x_eval_raw=x_eval,
        y_eval=y_eval,
        method=selector,
        scaler_name=scaler,
        add_indicator=add_indicator,
        n_neighbors=neighbors,
    )
    full = temporal_robustness._prepare_selector_eval_view(**kwargs, k=40)
    for k in [10, 20, 40]:
        prefix = temporal_robustness._selector_budget_views([full], k)[0]
        independent = temporal_robustness._prepare_selector_eval_view(**kwargs, k=k)
        for field in ["x_train_sel", "x_eval_sel", "selected_local"]:
            _assert_same_matrix(prefix[field], independent[field])
        assert prefix["feature_meta"] == independent["feature_meta"]
        assert prefix["imputer"] is full["imputer"]
        assert prefix["scaler"] is full["scaler"]
        assert prefix["y_train"] is full["y_train"]
        np.testing.assert_array_equal(prefix["imputer"].statistics_, independent["imputer"].statistics_)
        for classifier, config in [("krr", {"alpha": 1.0, "gamma": None}), ("logreg", {"C": 1.0})]:
            scores = []
            for view in [prefix, independent]:
                benchmark_common.reset_model_score_cache()
                scores.append(
                    benchmark_common.fit_classifier_scores(
                        classifier=classifier,
                        x_train_sel=view["x_train_sel"],
                        y_train=view["y_train"],
                        x_eval_sel=view["x_eval_sel"],
                        classifier_config=config,
                    )
                )
            for actual, expected in zip(scores[0], scores[1], strict=True):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("selector,neighbors", SELECTOR_CONFIGS)
@pytest.mark.parametrize("add_indicator", [False, True])
def test_tuned_budget_cache_fits_each_inner_training_split_once(
    monkeypatch, selector: str, neighbors: int | None, add_indicator: bool
) -> None:
    x, y, _, _ = _missing_dataset()
    original = benchmark_tuned._prepare_inner_selector_views
    calls = []

    def spy(**kwargs):
        calls.append(kwargs["selector_config"])
        return original(**kwargs)

    monkeypatch.setattr(benchmark_tuned, "_prepare_inner_selector_views", spy)
    cache = {}
    for k in [10, 20, 40]:
        kwargs = dict(
            x_outer_train_raw=x,
            y_outer_train=y,
            selector=selector,
            add_indicator=add_indicator,
            selector_config={"k": k, "n_neighbors": neighbors},
        )
        cached = benchmark_tuned._cached_inner_selector_views(cache, **kwargs)
        independent = original(**kwargs)
        for actual, expected in zip(cached, independent, strict=True):
            _assert_same_matrix(actual.x_train_sel, expected.x_train_sel)
            _assert_same_matrix(actual.x_eval_sel, expected.x_eval_sel)
            np.testing.assert_array_equal(actual.y_train, expected.y_train)
            np.testing.assert_array_equal(actual.y_eval, expected.y_eval)
        assert cached is benchmark_tuned._cached_inner_selector_views(cache, **kwargs)
    assert calls == [{"k": 40, "n_neighbors": neighbors}]
    assert np.shares_memory(cache[(10, neighbors)][0].x_train_sel, cache[(40, neighbors)][0].x_train_sel)
    # A new outer-fold owner prepares its own training context.
    benchmark_tuned._cached_inner_selector_views({}, **{**kwargs, "x_outer_train_raw": x + 10})
    assert len(calls) == 2


@pytest.mark.parametrize("eval_labels", [[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
@pytest.mark.parametrize("train_scores", [[0.2, 0.8, 0.4, 0.7], [0.5, 0.5, 0.5, 0.5]])
@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")
def test_inner_only_metrics_match_complete_metrics_and_single_class_fallback(
    monkeypatch, eval_labels: list[int], train_scores: list[float]
) -> None:
    y_train = np.array([0, 1, 0, 1])
    y_eval = np.array(eval_labels)
    train = np.array(train_scores)
    evaluated = np.array([0.3, 0.3, 0.9, 0.9])
    threshold, _ = find_ber_optimal_threshold(y_train, train)
    full = binary_metrics_at_threshold(y_eval, evaluated, threshold)
    expected_auc = float(full["ROC_AUC"]) if np.isfinite(full["ROC_AUC"]) else 0.5

    def unwanted_full_metrics(*args, **kwargs):
        raise AssertionError("Inner search should not compute unused extended diagnostics")

    monkeypatch.setattr(benchmark_tuned, "fit_classifier_scores", lambda **kwargs: (train, evaluated))
    monkeypatch.setattr(benchmark_tuned, "binary_metrics_at_threshold", unwanted_full_metrics)
    prepared = benchmark_tuned._InnerSelectorView(np.ones((4, 2)), y_train, np.ones((4, 2)), y_eval)
    summary = benchmark_tuned._inner_cv_summary_for_config(
        classifier="logreg", classifier_config={"C": 1}, prepared_inner_views=[prepared]
    )
    inner_threshold, _ = find_ber_optimal_threshold(y_eval, evaluated)
    expected_inner = binary_metrics_at_threshold(y_eval, evaluated, inner_threshold)
    assert summary["inner_selected_widths"] == "2"
    assert np.isnan(summary["gamma"])
    assert {key: summary[key] for key in ("mean_inner_ROC_AUC", "mean_inner_BER", "threshold_inner_oof")} == {
        "mean_inner_ROC_AUC": expected_auc,
        "mean_inner_BER": expected_inner["BER"],
        "threshold_inner_oof": inner_threshold,
    }
    monkeypatch.setattr(
        temporal_robustness,
        "_fit_temporal_logreg_view_scores",
        lambda **kwargs: (threshold, object(), train, evaluated),
    )
    monkeypatch.setattr(temporal_robustness, "binary_metrics_at_threshold", unwanted_full_metrics)
    assert temporal_robustness._score_prepared_inner_cv([{"y_eval": y_eval}], 1) == (expected_auc, full["BER"])


def test_temporal_selection_reuse_stays_within_fold_seed_and_scaler(monkeypatch) -> None:
    x, y, _, _ = _missing_dataset()
    original = temporal_robustness._prepare_inner_cv_views
    calls = []

    def spy(**kwargs):
        calls.append((kwargs["x_outer_train_raw"].copy(), kwargs["seed"], kwargs["scaler_name"], kwargs["k"]))
        return original(**kwargs)

    monkeypatch.setattr(temporal_robustness, "_prepare_inner_cv_views", spy)
    dates = [pd.Timestamp("2008-01-01"), pd.Timestamp("2008-02-01")]
    folds = [
        SimpleNamespace(
            train_index=np.arange(start, start + 60),
            test_index=np.arange(start + 60, start + 70),
            outer_fold=i,
            train_start_ts=dates[0],
            train_end_ts=dates[1],
            test_start_ts=dates[0],
            test_end_ts=dates[1],
        )
        for i, start in enumerate([0, 20], start=1)
    ]
    inner, outer = temporal_robustness._run_stage_b_model_selection(
        bundle=SimpleNamespace(
            fold_plan=SimpleNamespace(folds=folds),
            dev_with_weeks=pd.DataFrame(
                {
                    "raw_row_id": np.arange(len(y)),
                    "y_bin": y,
                    "timestamp": pd.date_range("2008-01-01", periods=len(y), freq="h"),
                }
            ),
        ),
        selectors_run=[SelectorName.F_TEST],
        x_dev=x,
        y_dev=y,
    )
    assert len(calls) == 2 * 2
    assert all(k == 40 for _, _, _, k in calls)
    assert {seed for _, seed, _, _ in calls} == {42}
    assert {scaler for _, _, scaler, _ in calls} == set(ScalerName.ALL)
    for trained, _, _, _ in calls[:2]:
        np.testing.assert_array_equal(trained, x[folds[0].train_index[:48]])
    for trained, _, _, _ in calls[2:]:
        np.testing.assert_array_equal(trained, x[folds[1].train_index[:48]])
    assert len(inner) == 2 * 3 * 4 * 2
    assert len(outer) == 2


def test_temporal_freeze_prepares_each_scaler_once_for_all_budgets(monkeypatch) -> None:
    x, y, _, _ = _missing_dataset()
    original = temporal_robustness._prepare_phase2_inner_views
    calls = []

    def spy(**kwargs):
        calls.append((kwargs["k"], kwargs["scaler_name"]))
        return original(**kwargs)

    monkeypatch.setattr(temporal_robustness, "_prepare_phase2_inner_views", spy)
    frame, _ = temporal_robustness._phase2_freeze_for_role(
        role="primary", selector=SelectorName.F_TEST, x_dev=x, y_dev=y
    )
    assert calls == [(40, ScalerName.STANDARD), (40, ScalerName.ROBUST)]
    assert len(frame) == 3 * 3 * 4 * 2
    assert frame.groupby(["k", "C", "scaler"]).size().eq(3).all()
