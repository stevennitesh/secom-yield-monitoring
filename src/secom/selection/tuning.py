"""Inner-CV tuning helpers for benchmark and temporal selection workflows."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from secom.config import (
    BENCHMARK_INNER_SPLITS,
    BENCHMARK_KRR_ALPHA_GRID,
    BENCHMARK_KRR_GAMMA_GRID,
    BENCHMARK_LOGREG_C_GRID,
    ScalerName,
    SEED_BENCHMARK,
)
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold
from secom.models import (
    fit_benchmark_krr_model,
    make_benchmark_krr_model,
    make_benchmark_logreg_model,
)

_FLOAT_TOLERANCE = 1e-12
_NEAR_BEST_AUC_BAND = 0.01
_ScoreFn = Callable[[Any, np.ndarray], np.ndarray]


def gamma_sort_key(gamma: float | None) -> float:
    """Sort ``None`` before numeric RBF gamma values."""
    return -1.0 if gamma is None else float(gamma)


def _inner_split_count(y_train: np.ndarray) -> int:
    """Return the feasible stratified inner-CV split count for binary labels."""
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    return min(int(BENCHMARK_INNER_SPLITS), min_class)


def _inner_cv(n_splits: int) -> StratifiedKFold:
    """Create the deterministic inner-CV splitter used by benchmark tuning."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_BENCHMARK,
    )


def _iter_inner_train_val_folds(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
):
    inner_cv = _inner_cv(n_splits)
    for inner_train_idx, inner_val_idx in inner_cv.split(x_train_sel, y_train):
        yield (
            x_train_sel[inner_train_idx],
            y_train[inner_train_idx],
            x_train_sel[inner_val_idx],
            y_train[inner_val_idx],
        )


def _ber_threshold_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the BER-optimal threshold for already-computed scores."""
    threshold, _ = find_ber_optimal_threshold(y_true, scores)
    return float(threshold)


def _ber_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Evaluate BER for scores at a frozen threshold."""
    metrics = binary_metrics_at_threshold(y_true, scores, threshold=threshold)
    return float(metrics["BER"])


def _krr_scores(clf: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict(x), dtype=float)


def _logreg_scores(clf: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict_proba(x)[:, 1], dtype=float)


def _train_threshold(clf: Any, score_fn: _ScoreFn, x_train: np.ndarray, y_train: np.ndarray) -> float:
    train_scores = score_fn(clf, x_train)
    return _ber_threshold_from_scores(y_train, train_scores)


def _validation_ber_with_train_threshold(
    *,
    clf: Any,
    score_fn: _ScoreFn,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    threshold = _train_threshold(clf, score_fn, x_train, y_train)
    val_scores = score_fn(clf, x_val)
    return _ber_at_threshold(y_val, val_scores, threshold=threshold)


def select_krr_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float | None, Any, float, float]:
    """Select KRR alpha/gamma by inner BER and return the fitted final model."""
    y_train = np.asarray(y_train, dtype=int)
    n_splits = _inner_split_count(y_train)
    sorted_alphas = sorted(float(a) for a in BENCHMARK_KRR_ALPHA_GRID)
    sorted_gammas = sorted(
        (None if g is None else float(g) for g in BENCHMARK_KRR_GAMMA_GRID),
        key=gamma_sort_key,
    )

    if n_splits < 2:
        fallback_alpha = float(sorted_alphas[0])
        fallback_gamma = sorted_gammas[0]
        fallback_clf = fit_benchmark_krr_model(
            x_train_sel,
            y_train,
            alpha=fallback_alpha,
            gamma=fallback_gamma,
        )
        fallback_threshold = _train_threshold(fallback_clf, _krr_scores, x_train_sel, y_train)
        return fallback_alpha, fallback_gamma, fallback_clf, float(fallback_threshold), np.inf

    best_alpha: float | None = None
    best_gamma: float | None = None
    best_inner_ber = np.inf

    for alpha, gamma in product(sorted_alphas, sorted_gammas):
        fold_bers: list[float] = []
        for x_inner_train, y_inner_train, x_inner_val, y_inner_val in _iter_inner_train_val_folds(
            x_train_sel, y_train, n_splits
        ):
            clf_inner = fit_benchmark_krr_model(
                x_inner_train,
                y_inner_train,
                alpha=alpha,
                gamma=gamma,
            )
            fold_bers.append(
                _validation_ber_with_train_threshold(
                    clf=clf_inner,
                    score_fn=_krr_scores,
                    x_train=x_inner_train,
                    y_train=y_inner_train,
                    x_val=x_inner_val,
                    y_val=y_inner_val,
                )
            )

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - _FLOAT_TOLERANCE:
            best_inner_ber = mean_inner_ber
            best_alpha = alpha
            best_gamma = gamma
        elif np.isclose(mean_inner_ber, best_inner_ber):
            if best_alpha is None or alpha < best_alpha:
                best_alpha = alpha
                best_gamma = gamma
            elif best_alpha is not None and np.isclose(alpha, best_alpha):
                if gamma_sort_key(gamma) < gamma_sort_key(best_gamma):
                    best_gamma = gamma

    if best_alpha is None:
        raise RuntimeError("krr: failed to choose (alpha, gamma) from inner CV")

    final_clf = fit_benchmark_krr_model(
        x_train_sel,
        y_train,
        alpha=float(best_alpha),
        gamma=best_gamma,
    )
    final_threshold = _train_threshold(final_clf, _krr_scores, x_train_sel, y_train)
    return float(best_alpha), best_gamma, final_clf, float(final_threshold), float(best_inner_ber)


def select_logreg_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, Any, float, float]:
    """Select logistic-regression C by inner BER and return the fitted final model."""
    y_train = np.asarray(y_train, dtype=int)
    n_splits = _inner_split_count(y_train)
    sorted_c_values = sorted(float(c) for c in BENCHMARK_LOGREG_C_GRID)

    if n_splits < 2:
        fallback_c = float(sorted_c_values[0])
        fallback_clf = make_benchmark_logreg_model(c_value=fallback_c)
        fallback_clf.fit(x_train_sel, y_train)
        fallback_threshold = _train_threshold(fallback_clf, _logreg_scores, x_train_sel, y_train)
        return fallback_c, fallback_clf, float(fallback_threshold), np.inf

    best_c: float | None = None
    best_inner_ber = np.inf

    for c_value in sorted_c_values:
        fold_bers: list[float] = []
        for x_inner_train, y_inner_train, x_inner_val, y_inner_val in _iter_inner_train_val_folds(
            x_train_sel, y_train, n_splits
        ):
            clf_inner = make_benchmark_logreg_model(c_value=c_value)
            clf_inner.fit(x_inner_train, y_inner_train)
            fold_bers.append(
                _validation_ber_with_train_threshold(
                    clf=clf_inner,
                    score_fn=_logreg_scores,
                    x_train=x_inner_train,
                    y_train=y_inner_train,
                    x_val=x_inner_val,
                    y_val=y_inner_val,
                )
            )

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - _FLOAT_TOLERANCE:
            best_inner_ber = mean_inner_ber
            best_c = c_value
        elif np.isclose(mean_inner_ber, best_inner_ber):
            if best_c is None or c_value < best_c:
                best_c = c_value

    if best_c is None:
        raise RuntimeError("logreg: failed to choose C from inner CV")

    final_clf = make_benchmark_logreg_model(c_value=float(best_c))
    final_clf.fit(x_train_sel, y_train)
    final_threshold = _train_threshold(final_clf, _logreg_scores, x_train_sel, y_train)
    return float(best_c), final_clf, float(final_threshold), float(best_inner_ber)


def inner_cv_ber_krr_strict(x_train_sel: np.ndarray, y_train: np.ndarray) -> float:
    """Return strict KRR inner-CV BER for the optional benchmark replication path."""
    y_train = np.asarray(y_train, dtype=int)
    n_splits = _inner_split_count(y_train)
    if n_splits < 2:
        clf = make_benchmark_krr_model(alpha=1.0, gamma=None)
        y_train_krr = 2 * y_train - 1
        clf.fit(x_train_sel, y_train_krr)
        threshold = _train_threshold(clf, _krr_scores, x_train_sel, y_train)
        return _ber_at_threshold(y_train, _krr_scores(clf, x_train_sel), threshold=threshold)

    fold_bers: list[float] = []
    for x_inner_train, y_inner_train, x_inner_val, y_inner_val in _iter_inner_train_val_folds(
        x_train_sel, y_train, n_splits
    ):
        clf = make_benchmark_krr_model(alpha=1.0, gamma=None)
        y_inner_train_krr = 2 * y_inner_train - 1
        clf.fit(x_inner_train, y_inner_train_krr)
        fold_bers.append(
            _validation_ber_with_train_threshold(
                clf=clf,
                score_fn=_krr_scores,
                x_train=x_inner_train,
                y_train=y_inner_train,
                x_val=x_inner_val,
                y_val=y_inner_val,
            )
        )
    return float(np.mean(fold_bers))


def _inner_config_simplicity_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    nn = row.get("n_neighbors")
    nn_key = math.inf if nn is None else nn
    scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
    return (row["k"], row["C"], scaler_pref, nn_key)


def select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the temporal config by near-best AUC, BER, then deterministic simplicity."""
    if not config_rows:
        raise ValueError("No configs to select")
    best_auc = max(row["mean_inner_ROC_AUC"] for row in config_rows)
    near_best_auc = [
        row for row in config_rows if row["mean_inner_ROC_AUC"] >= best_auc - _NEAR_BEST_AUC_BAND - _FLOAT_TOLERANCE
    ]
    min_ber = min(row["mean_inner_BER"] for row in near_best_auc)
    tied_on_ber = [row for row in near_best_auc if np.isclose(row["mean_inner_BER"], min_ber)]

    return sorted(tied_on_ber, key=_inner_config_simplicity_key)[0]
