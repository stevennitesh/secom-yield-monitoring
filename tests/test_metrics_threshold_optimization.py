from __future__ import annotations

import numpy as np
import pytest

from secom.metrics import candidate_thresholds, confusion_counts, find_ber_optimal_threshold


def _threshold_equal(a: float, b: float) -> bool:
    if np.isinf(a) and np.isinf(b):
        return bool(np.sign(a) == np.sign(b))
    return bool(np.isclose(float(a), float(b), atol=1e-12))


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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_find_ber_optimal_threshold_matches_bruteforce_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 180
    y_true = rng.integers(0, 2, size=n, dtype=int)
    if np.unique(y_true).size < 2:
        y_true[0] = 1 - int(y_true[0])
    scores = rng.normal(loc=0.0, scale=1.0, size=n).astype(float)
    threshold_fast, metrics_fast = find_ber_optimal_threshold(y_true, scores)
    threshold_brute, metrics_brute = _bruteforce_reference(y_true, scores)
    assert _threshold_equal(float(threshold_fast), float(threshold_brute))
    assert np.isclose(float(metrics_fast["BER"]), float(metrics_brute["BER"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True+"]), float(metrics_brute["True+"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True-"]), float(metrics_brute["True-"]), atol=1e-12)


def test_find_ber_optimal_threshold_matches_bruteforce_with_duplicate_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    scores = np.asarray([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.2, 0.2], dtype=float)
    threshold_fast, metrics_fast = find_ber_optimal_threshold(y_true, scores)
    threshold_brute, metrics_brute = _bruteforce_reference(y_true, scores)
    assert _threshold_equal(float(threshold_fast), float(threshold_brute))
    assert np.isclose(float(metrics_fast["BER"]), float(metrics_brute["BER"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True+"]), float(metrics_brute["True+"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True-"]), float(metrics_brute["True-"]), atol=1e-12)


def test_find_ber_optimal_threshold_nonfinite_scores_fallback_equivalence() -> None:
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    scores = np.asarray([0.2, np.nan, -0.1, np.inf, -np.inf, 0.2, 0.8, -0.5], dtype=float)
    threshold_fast, metrics_fast = find_ber_optimal_threshold(y_true, scores)
    threshold_brute, metrics_brute = _bruteforce_reference(y_true, scores)
    assert _threshold_equal(float(threshold_fast), float(threshold_brute))
    assert np.isclose(float(metrics_fast["BER"]), float(metrics_brute["BER"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True+"]), float(metrics_brute["True+"]), atol=1e-12)
    assert np.isclose(float(metrics_fast["True-"]), float(metrics_brute["True-"]), atol=1e-12)
