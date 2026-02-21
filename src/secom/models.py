from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression


def make_lane_a_classifier(alpha: float = 1.0, gamma: float | None = None) -> KernelRidge:
    return KernelRidge(kernel="rbf", alpha=float(alpha), gamma=gamma)


def make_lane_a_notebook_classifier() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )


def make_lane_a_logreg_tuned_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )


def make_lane_b_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )


def fit_lane_b_classifier(
    x_train: np.ndarray, y_train_bin: np.ndarray, c_value: float
) -> LogisticRegression:
    clf = make_lane_b_classifier(c_value)
    clf.fit(x_train, np.asarray(y_train_bin, dtype=int))
    return clf


def fit_lane_a_and_score(
    x_train: np.ndarray,
    y_train_bin: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    y_krr = 2 * np.asarray(y_train_bin, dtype=int) - 1
    clf = make_lane_a_classifier(alpha=1.0)
    clf.fit(x_train, y_krr)
    return np.asarray(clf.predict(x_eval), dtype=float)


def _balanced_sample_weight(y_train_bin: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train_bin, dtype=int)
    n = int(y.size)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def fit_lane_a_balanced_classifier(
    x_train: np.ndarray,
    y_train_bin: np.ndarray,
    alpha: float = 1.0,
    gamma: float | None = None,
) -> KernelRidge:
    y_krr = 2 * np.asarray(y_train_bin, dtype=int) - 1
    sample_weight = _balanced_sample_weight(y_train_bin)
    clf = make_lane_a_classifier(alpha=alpha, gamma=gamma)
    clf.fit(x_train, y_krr, sample_weight=sample_weight)
    return clf


def fit_lane_a_notebook_and_score(
    x_train: np.ndarray,
    y_train_bin: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    clf = make_lane_a_notebook_classifier()
    clf.fit(x_train, np.asarray(y_train_bin, dtype=int))
    return np.asarray(clf.predict_proba(x_eval)[:, 1], dtype=float)


def fit_lane_b_and_score_proba(
    x_train: np.ndarray, y_train_bin: np.ndarray, x_eval: np.ndarray, c_value: float
) -> np.ndarray:
    clf = fit_lane_b_classifier(x_train, y_train_bin, c_value)
    return clf.predict_proba(x_eval)[:, 1]
