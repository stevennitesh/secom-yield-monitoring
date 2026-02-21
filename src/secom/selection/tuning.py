from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from secom.config import (
    LANE_A_KRR_BALANCED_ALPHA_GRID,
    LANE_A_KRR_BALANCED_GAMMA_GRID,
    LANE_A_KRR_BALANCED_INNER_SPLITS,
    LANE_A_LOGREG_C_GRID,
    LANE_A_L1_SELECTOR_C_GRID,
    LANE_A_MRMR_LAMBDA_GRID,
    LANE_A_MUTUAL_INFO_N_NEIGHBORS_GRID,
    LaneAClassifier,
    ScalerName,
    SEED_LANE_A,
    SelectorName,
)
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold
from secom.models import (
    fit_lane_a_balanced_classifier,
    make_lane_a_classifier,
    make_lane_a_logreg_tuned_classifier,
)
from secom.selection.engine import select_features


def gamma_sort_key(gamma: float | None) -> float:
    return -1.0 if gamma is None else float(gamma)


def select_krr_balanced_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float | None, Any, float, float]:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(LANE_A_KRR_BALANCED_INNER_SPLITS), min_class)
    sorted_alphas = sorted(float(a) for a in LANE_A_KRR_BALANCED_ALPHA_GRID)
    sorted_gammas = sorted(
        (None if g is None else float(g) for g in LANE_A_KRR_BALANCED_GAMMA_GRID),
        key=gamma_sort_key,
    )

    if n_splits < 2:
        fallback_alpha = float(sorted_alphas[0])
        fallback_gamma = sorted_gammas[0]
        fallback_clf = fit_lane_a_balanced_classifier(
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
        random_state=SEED_LANE_A,
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

            clf_inner = fit_lane_a_balanced_classifier(
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
        raise RuntimeError("krr_balanced: failed to choose (alpha, gamma) from inner CV")

    final_clf = fit_lane_a_balanced_classifier(
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
    n_splits = min(int(LANE_A_KRR_BALANCED_INNER_SPLITS), min_class)
    sorted_c_values = sorted(float(c) for c in LANE_A_LOGREG_C_GRID)

    if n_splits < 2:
        fallback_c = float(sorted_c_values[0])
        fallback_clf = make_lane_a_logreg_tuned_classifier(c_value=fallback_c)
        fallback_clf.fit(x_train_sel, y_train)
        fallback_train_scores = np.asarray(fallback_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
        fallback_threshold, _ = find_ber_optimal_threshold(y_train, fallback_train_scores)
        return fallback_c, fallback_clf, float(fallback_threshold), np.inf

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_LANE_A,
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

            clf_inner = make_lane_a_logreg_tuned_classifier(c_value=c_value)
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

    final_clf = make_lane_a_logreg_tuned_classifier(c_value=float(best_c))
    final_clf.fit(x_train_sel, y_train)
    final_train_scores = np.asarray(final_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
    final_threshold, _ = find_ber_optimal_threshold(y_train, final_train_scores)
    return float(best_c), final_clf, float(final_threshold), float(best_inner_ber)


def inner_cv_ber_krr_strict(x_train_sel: np.ndarray, y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(LANE_A_KRR_BALANCED_INNER_SPLITS), min_class)
    if n_splits < 2:
        clf = make_lane_a_classifier(alpha=1.0, gamma=None)
        y_train_krr = 2 * y_train - 1
        clf.fit(x_train_sel, y_train_krr)
        train_scores = np.asarray(clf.predict(x_train_sel), dtype=float)
        threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
        m = binary_metrics_at_threshold(y_train, train_scores, threshold=float(threshold))
        return float(m["BER"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_LANE_A)
    fold_bers: list[float] = []
    for inner_train_idx, inner_val_idx in skf.split(x_train_sel, y_train):
        x_inner_train = x_train_sel[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        x_inner_val = x_train_sel[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]
        clf = make_lane_a_classifier(alpha=1.0, gamma=None)
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


def tune_classifier_for_selected_features(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
    classifier: str,
) -> tuple[float, dict[str, Any]]:
    if classifier == LaneAClassifier.KRR_BALANCED:
        alpha, gamma, clf, threshold, inner_ber = select_krr_balanced_config_with_inner_cv(
            x_train_sel=x_train_sel,
            y_train=y_train,
        )
        return float(inner_ber), {
            "chosen_alpha": float(alpha),
            "chosen_gamma": gamma,
            "chosen_C": None,
            "model": clf,
            "threshold": float(threshold),
        }
    if classifier == LaneAClassifier.LOGREG:
        c_value, clf, threshold, inner_ber = select_logreg_config_with_inner_cv(
            x_train_sel=x_train_sel,
            y_train=y_train,
        )
        return float(inner_ber), {
            "chosen_alpha": None,
            "chosen_gamma": None,
            "chosen_C": float(c_value),
            "model": clf,
            "threshold": float(threshold),
        }
    if classifier == LaneAClassifier.KRR_STRICT:
        inner_ber = inner_cv_ber_krr_strict(x_train_sel=x_train_sel, y_train=y_train)
        return float(inner_ber), {
            "chosen_alpha": None,
            "chosen_gamma": None,
            "chosen_C": None,
            "model": None,
            "threshold": None,
        }
    raise ValueError(f"Unknown Lane A classifier mode: {classifier}")


def select_selector_param_for_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classifier: str,
    selector_method: str,
    param_values: list[float | int],
    param_name: str,
    k: int = 40,
) -> tuple[float | int, np.ndarray, dict[str, Any]]:
    best_value: float | int | None = None
    best_selected_local: np.ndarray | None = None
    best_criterion = np.inf
    best_payload: dict[str, Any] = {}

    for value in param_values:
        selector_kwargs: dict[str, Any] = {
            "method": selector_method,
            "x_train": x_train,
            "y_train": y_train,
            "k": k,
        }
        if param_name == "mrmr_lambda":
            selector_kwargs["mrmr_lambda"] = float(value)
        elif param_name == "mutual_info_n_neighbors":
            selector_kwargs["mutual_info_n_neighbors"] = int(value)
        elif param_name == "l1_selector_c":
            selector_kwargs["l1_selector_c"] = float(value)
        else:
            raise ValueError(f"Unsupported selector param_name={param_name}")

        selected_local, _ = select_features(**selector_kwargs)
        x_train_sel = x_train[:, selected_local]
        criterion, payload = tune_classifier_for_selected_features(
            x_train_sel=x_train_sel,
            y_train=y_train,
            classifier=classifier,
        )

        if criterion < best_criterion - 1e-12:
            best_criterion = criterion
            best_value = value
            best_selected_local = selected_local
            best_payload = payload
        elif np.isclose(criterion, best_criterion):
            if best_value is None or value < best_value:
                best_value = value
                best_selected_local = selected_local
                best_payload = payload

    if best_value is None or best_selected_local is None:
        raise RuntimeError(
            f"{selector_method} selector tuning failed for param {param_name}"
        )
    return best_value, best_selected_local, best_payload


def select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not config_rows:
        raise ValueError("No configs to select")
    best_auc = max(r["mean_inner_ROC_AUC"] for r in config_rows)
    near = [
        r
        for r in config_rows
        if r["mean_inner_ROC_AUC"] >= best_auc - 0.01 - 1e-12
    ]
    min_ber = min(r["mean_inner_BER"] for r in near)
    tied = [r for r in near if np.isclose(r["mean_inner_BER"], min_ber)]

    def key(row: dict[str, Any]) -> tuple[float, float, int, float]:
        nn = row.get("n_neighbors")
        nn_key = math.inf if nn is None else nn
        scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
        return (row["k"], row["C"], scaler_pref, nn_key)

    return sorted(tied, key=key)[0]


def lane_a_param_grid_for_selector(selector: str, param_name: str) -> list[float | int]:
    if selector == SelectorName.MRMR and param_name == "mrmr_lambda":
        return sorted(float(v) for v in LANE_A_MRMR_LAMBDA_GRID)
    if selector == SelectorName.MUTUAL_INFO and param_name == "mutual_info_n_neighbors":
        return sorted(int(v) for v in LANE_A_MUTUAL_INFO_N_NEIGHBORS_GRID)
    if selector == SelectorName.L1_LOGREG and param_name == "l1_selector_c":
        return sorted(float(v) for v in LANE_A_L1_SELECTOR_C_GRID)
    raise ValueError(f"Unsupported selector/param pair: {selector}/{param_name}")
