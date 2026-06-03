"""Model factories and fitting helpers for benchmark and temporal workflows."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression


def make_benchmark_krr_model(alpha: float = 1.0, gamma: float | None = None) -> KernelRidge:
    """Create the benchmark RBF Kernel Ridge classifier surrogate."""
    return KernelRidge(kernel="rbf", alpha=float(alpha), gamma=gamma)


def _make_balanced_logreg_model(c_value: float) -> LogisticRegression:
    """Create the shared balanced logistic-regression classifier."""
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )


def make_benchmark_logreg_model(c_value: float) -> LogisticRegression:
    """Create the benchmark balanced logistic-regression classifier."""
    return _make_balanced_logreg_model(c_value)


def make_temporal_logreg_model(c_value: float) -> LogisticRegression:
    """Create the temporal balanced logistic-regression classifier."""
    return _make_balanced_logreg_model(c_value)


def fit_temporal_logreg_model(x_train: np.ndarray, y_train_bin: np.ndarray, c_value: float) -> LogisticRegression:
    """Fit temporal logistic regression while suppressing convergence warnings."""
    clf = make_temporal_logreg_model(c_value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x_train, np.asarray(y_train_bin, dtype=int))
    return clf


def _balanced_sample_weight(y_train_bin: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for binary KRR targets."""
    y = np.asarray(y_train_bin, dtype=int)
    n = int(y.size)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def fit_benchmark_krr_model(
    x_train: np.ndarray,
    y_train_bin: np.ndarray,
    alpha: float = 1.0,
    gamma: float | None = None,
) -> KernelRidge:
    """Fit benchmark KRR on -1/+1 labels with balanced sample weights."""
    y_krr = 2 * np.asarray(y_train_bin, dtype=int) - 1
    sample_weight = _balanced_sample_weight(y_train_bin)
    clf = make_benchmark_krr_model(alpha=alpha, gamma=gamma)
    clf.fit(x_train, y_krr, sample_weight=sample_weight)
    return clf
