from __future__ import annotations

import math
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


def gamma_sort_key(gamma: float | None) -> float:
    return -1.0 if gamma is None else float(gamma)


def select_krr_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float | None, Any, float, float]:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(BENCHMARK_INNER_SPLITS), min_class)
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
        fallback_train_scores = np.asarray(fallback_clf.predict(x_train_sel), dtype=float)
        fallback_threshold, _ = find_ber_optimal_threshold(y_train, fallback_train_scores)
        return fallback_alpha, fallback_gamma, fallback_clf, float(fallback_threshold), np.inf

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_BENCHMARK,
    )
    best_alpha: float | None = None
    best_gamma: float | None = None
    best_inner_ber = np.inf

    for alpha, gamma in product(sorted_alphas, sorted_gammas):
        fold_bers: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(x_train_sel, y_train):
            x_inner_train = x_train_sel[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            x_inner_val = x_train_sel[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            clf_inner = fit_benchmark_krr_model(
                x_inner_train,
                y_inner_train,
                alpha=alpha,
                gamma=gamma,
            )
            inner_train_scores = np.asarray(clf_inner.predict(x_inner_train), dtype=float)
            inner_threshold, _ = find_ber_optimal_threshold(y_inner_train, inner_train_scores)
            inner_val_scores = np.asarray(clf_inner.predict(x_inner_val), dtype=float)
            inner_metrics = binary_metrics_at_threshold(
                y_inner_val,
                inner_val_scores,
                threshold=float(inner_threshold),
            )
            fold_bers.append(float(inner_metrics["BER"]))

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - 1e-12:
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
    final_train_scores = np.asarray(final_clf.predict(x_train_sel), dtype=float)
    final_threshold, _ = find_ber_optimal_threshold(y_train, final_train_scores)
    return float(best_alpha), best_gamma, final_clf, float(final_threshold), float(best_inner_ber)


def select_logreg_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, Any, float, float]:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(BENCHMARK_INNER_SPLITS), min_class)
    sorted_c_values = sorted(float(c) for c in BENCHMARK_LOGREG_C_GRID)

    if n_splits < 2:
        fallback_c = float(sorted_c_values[0])
        fallback_clf = make_benchmark_logreg_model(c_value=fallback_c)
        fallback_clf.fit(x_train_sel, y_train)
        fallback_train_scores = np.asarray(fallback_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
        fallback_threshold, _ = find_ber_optimal_threshold(y_train, fallback_train_scores)
        return fallback_c, fallback_clf, float(fallback_threshold), np.inf

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_BENCHMARK,
    )
    best_c: float | None = None
    best_inner_ber = np.inf

    for c_value in sorted_c_values:
        fold_bers: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(x_train_sel, y_train):
            x_inner_train = x_train_sel[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            x_inner_val = x_train_sel[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            clf_inner = make_benchmark_logreg_model(c_value=c_value)
            clf_inner.fit(x_inner_train, y_inner_train)
            inner_train_scores = np.asarray(clf_inner.predict_proba(x_inner_train)[:, 1], dtype=float)
            inner_threshold, _ = find_ber_optimal_threshold(y_inner_train, inner_train_scores)
            inner_val_scores = np.asarray(clf_inner.predict_proba(x_inner_val)[:, 1], dtype=float)
            inner_metrics = binary_metrics_at_threshold(
                y_inner_val,
                inner_val_scores,
                threshold=float(inner_threshold),
            )
            fold_bers.append(float(inner_metrics["BER"]))

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - 1e-12:
            best_inner_ber = mean_inner_ber
            best_c = c_value
        elif np.isclose(mean_inner_ber, best_inner_ber):
            if best_c is None or c_value < best_c:
                best_c = c_value

    if best_c is None:
        raise RuntimeError("logreg: failed to choose C from inner CV")

    final_clf = make_benchmark_logreg_model(c_value=float(best_c))
    final_clf.fit(x_train_sel, y_train)
    final_train_scores = np.asarray(final_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
    final_threshold, _ = find_ber_optimal_threshold(y_train, final_train_scores)
    return float(best_c), final_clf, float(final_threshold), float(best_inner_ber)


def inner_cv_ber_krr_strict(x_train_sel: np.ndarray, y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(BENCHMARK_INNER_SPLITS), min_class)
    if n_splits < 2:
        clf = make_benchmark_krr_model(alpha=1.0, gamma=None)
        y_train_krr = 2 * y_train - 1
        clf.fit(x_train_sel, y_train_krr)
        train_scores = np.asarray(clf.predict(x_train_sel), dtype=float)
        threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
        m = binary_metrics_at_threshold(y_train, train_scores, threshold=float(threshold))
        return float(m["BER"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_BENCHMARK)
    fold_bers: list[float] = []
    for inner_train_idx, inner_val_idx in skf.split(x_train_sel, y_train):
        x_inner_train = x_train_sel[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        x_inner_val = x_train_sel[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]
        clf = make_benchmark_krr_model(alpha=1.0, gamma=None)
        y_inner_train_krr = 2 * y_inner_train - 1
        clf.fit(x_inner_train, y_inner_train_krr)
        inner_train_scores = np.asarray(clf.predict(x_inner_train), dtype=float)
        threshold, _ = find_ber_optimal_threshold(y_inner_train, inner_train_scores)
        inner_val_scores = np.asarray(clf.predict(x_inner_val), dtype=float)
        m = binary_metrics_at_threshold(
            y_inner_val,
            inner_val_scores,
            threshold=float(threshold),
        )
        fold_bers.append(float(m["BER"]))
    return float(np.mean(fold_bers))


def select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not config_rows:
        raise ValueError("No configs to select")
    best_auc = max(r["mean_inner_ROC_AUC"] for r in config_rows)
    near = [r for r in config_rows if r["mean_inner_ROC_AUC"] >= best_auc - 0.01 - 1e-12]
    min_ber = min(r["mean_inner_BER"] for r in near)
    tied = [r for r in near if np.isclose(r["mean_inner_BER"], min_ber)]

    def key(row: dict[str, Any]) -> tuple[float, float, int, float]:
        nn = row.get("n_neighbors")
        nn_key = math.inf if nn is None else nn
        scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
        return (row["k"], row["C"], scaler_pref, nn_key)

    return sorted(tied, key=key)[0]
