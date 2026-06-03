from __future__ import annotations

import numpy as np
import pytest

from secom.metrics import (
    candidate_thresholds,
    confusion_counts,
    extract_tpr_at_tnr,
    find_ber_optimal_threshold,
    roc_auc_or_default,
    safe_std,
)
from tests.assertions import threshold_equal


def _bruteforce_reference(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold: float | None = None
    best_ber = np.inf
    best_tpr = -np.inf
    best_metrics: dict[str, float] | None = None
    for threshold in candidate_thresholds(scores):
        y_pred = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
        counts = confusion_counts(np.asarray(y_true, dtype=int), y_pred)
        tpr_denom = counts.tp + counts.fn
        tnr_denom = counts.tn + counts.fp
        tpr = 0.0 if tpr_denom == 0 else float(counts.tp / tpr_denom)
        tnr = 0.0 if tnr_denom == 0 else float(counts.tn / tnr_denom)
        ber = float(1.0 - 0.5 * (tpr + tnr))
        if ber < best_ber:
            best_threshold = float(threshold)
            best_ber = ber
            best_tpr = tpr
            best_metrics = {"BER": ber, "True+": tpr, "True-": tnr}
        elif np.isclose(ber, best_ber):
            if tpr > best_tpr:
                best_threshold = float(threshold)
                best_tpr = tpr
                best_metrics = {"BER": ber, "True+": tpr, "True-": tnr}
            elif np.isclose(tpr, best_tpr):
                if best_threshold is None or float(threshold) < float(best_threshold):
                    best_threshold = float(threshold)
                    best_metrics = {"BER": ber, "True+": tpr, "True-": tnr}
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Bruteforce threshold search failed")
    return best_threshold, best_metrics


def _assert_fast_threshold_search_matches_bruteforce(y_true: np.ndarray, scores: np.ndarray) -> None:
    threshold_fast, metrics_fast = find_ber_optimal_threshold(y_true, scores)
    threshold_brute, metrics_brute = _bruteforce_reference(y_true, scores)

    assert threshold_equal(float(threshold_fast), float(threshold_brute))
    assert np.isclose(float(metrics_fast["BER"]), float(metrics_brute["BER"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True+"]), float(metrics_brute["True+"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True-"]), float(metrics_brute["True-"]), atol=1e-12)


def _bruteforce_tpr_at_tnr(y_true: np.ndarray, scores: np.ndarray, target_tnr: float) -> tuple[float, float, float]:
    best_threshold: float | None = None
    best_tpr = -np.inf
    best_tnr = 0.0
    for threshold in candidate_thresholds(scores):
        y_pred = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
        counts = confusion_counts(np.asarray(y_true, dtype=int), y_pred)
        tpr_denom = counts.tp + counts.fn
        tnr_denom = counts.tn + counts.fp
        tpr = 0.0 if tpr_denom == 0 else float(counts.tp / tpr_denom)
        tnr = 0.0 if tnr_denom == 0 else float(counts.tn / tnr_denom)
        if tnr >= target_tnr:
            if tpr > best_tpr:
                best_tpr = tpr
                best_tnr = tnr
                best_threshold = float(threshold)
            elif np.isclose(tpr, best_tpr):
                if best_threshold is None or float(threshold) < float(best_threshold):
                    best_threshold = float(threshold)
                    best_tnr = tnr
    if best_threshold is None:
        fallback = float(np.max(candidate_thresholds(scores)))
        y_pred = (np.asarray(scores, dtype=float) >= fallback).astype(int)
        counts = confusion_counts(np.asarray(y_true, dtype=int), y_pred)
        tpr_denom = counts.tp + counts.fn
        tnr_denom = counts.tn + counts.fp
        return (
            fallback,
            0.0 if tnr_denom == 0 else float(counts.tn / tnr_denom),
            0.0 if tpr_denom == 0 else float(counts.tp / tpr_denom),
        )
    return best_threshold, float(best_tnr), float(best_tpr)


def _assert_fast_tpr_at_tnr_matches_bruteforce(y_true: np.ndarray, scores: np.ndarray, target_tnr: float) -> None:
    threshold_fast, tnr_fast, tpr_fast = extract_tpr_at_tnr(y_true, scores, target_tnr=target_tnr)
    threshold_brute, tnr_brute, tpr_brute = _bruteforce_tpr_at_tnr(y_true, scores, target_tnr=target_tnr)

    assert threshold_equal(float(threshold_fast), float(threshold_brute))
    assert np.isclose(float(tnr_fast), float(tnr_brute), atol=1e-12)
    assert np.isclose(float(tpr_fast), float(tpr_brute), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_find_ber_optimal_threshold_matches_bruteforce_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 180
    y_true = rng.integers(0, 2, size=n, dtype=int)
    if np.unique(y_true).size < 2:
        y_true[0] = 1 - int(y_true[0])
    scores = rng.normal(loc=0.0, scale=1.0, size=n).astype(float)

    _assert_fast_threshold_search_matches_bruteforce(y_true, scores)


def test_find_ber_optimal_threshold_matches_bruteforce_with_duplicate_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    scores = np.asarray([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.2, 0.2], dtype=float)

    _assert_fast_threshold_search_matches_bruteforce(y_true, scores)


def test_find_ber_optimal_threshold_nonfinite_scores_fallback_equivalence() -> None:
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    scores = np.asarray([0.2, np.nan, -0.1, np.inf, -np.inf, 0.2, 0.8, -0.5], dtype=float)

    _assert_fast_threshold_search_matches_bruteforce(y_true, scores)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("target_tnr", [0.0, 0.5, 0.9, 1.0])
def test_extract_tpr_at_tnr_matches_bruteforce_random(seed: int, target_tnr: float) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=180, dtype=int)
    if np.unique(y_true).size < 2:
        y_true[0] = 1 - int(y_true[0])
    scores = rng.normal(loc=0.0, scale=1.0, size=y_true.size).astype(float)

    _assert_fast_tpr_at_tnr_matches_bruteforce(y_true, scores, target_tnr=target_tnr)


def test_extract_tpr_at_tnr_matches_bruteforce_with_duplicate_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    scores = np.asarray([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.2, 0.2], dtype=float)

    _assert_fast_tpr_at_tnr_matches_bruteforce(y_true, scores, target_tnr=0.5)


def test_extract_tpr_at_tnr_nonfinite_scores_fallback_equivalence() -> None:
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    scores = np.asarray([0.2, np.nan, -0.1, np.inf, -np.inf, 0.2, 0.8, -0.5], dtype=float)

    _assert_fast_tpr_at_tnr_matches_bruteforce(y_true, scores, target_tnr=0.75)


def test_extract_tpr_at_tnr_rejects_mismatched_input_lengths() -> None:
    y_true = np.asarray([0, 1, 0], dtype=int)
    scores = np.asarray([0.1, 0.9], dtype=float)

    with pytest.raises(ValueError, match="identical length"):
        extract_tpr_at_tnr(y_true, scores)


def test_safe_std_accepts_arrays_and_iterators() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=float)

    assert np.isclose(safe_std(values), 1.0)
    assert np.isclose(safe_std(iter(values.tolist())), 1.0)


def test_roc_auc_or_default_handles_single_class_eval() -> None:
    assert np.isclose(
        roc_auc_or_default(
            np.asarray([0, 1], dtype=int),
            np.asarray([0.1, 0.9], dtype=float),
        ),
        1.0,
    )
    assert np.isclose(
        roc_auc_or_default(
            np.asarray([1, 1], dtype=int),
            np.asarray([0.8, 0.9], dtype=float),
            default=0.25,
        ),
        0.25,
    )
