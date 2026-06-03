"""Univariate feature ranking methods for the SECOM study selectors."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.feature_selection import f_classif

from secom.config import EPS_SELECTOR, SelectorName


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Coerce unusable scores to the shared bottom-rank sentinel."""
    sanitized = np.asarray(scores, dtype=float).copy()
    sanitized[~np.isfinite(sanitized)] = -np.inf
    return sanitized


def _zero_variance_mask(x: np.ndarray) -> np.ndarray:
    """Identify columns that cannot support a univariate ranking signal."""
    std = np.std(np.asarray(x, dtype=float), axis=0, ddof=0)
    return std <= 0


def _rank_desc_with_index_tiebreak(scores: np.ndarray) -> np.ndarray:
    """Rank higher scores first, with deterministic lower-index tie breaks."""
    feature_indices = np.arange(scores.shape[0], dtype=int)
    return np.lexsort((feature_indices, -scores))


def score_s2n(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    """Score features by signal-to-noise separation between fail and pass wafers."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    fail = y == 1
    pass_ = y == 0

    mu_fail = np.nanmean(x[fail], axis=0)
    mu_pass = np.nanmean(x[pass_], axis=0)
    sd_fail = np.nanstd(x[fail], axis=0, ddof=1)
    sd_pass = np.nanstd(x[pass_], axis=0, ddof=1)
    score = np.abs(mu_fail - mu_pass) / (sd_fail + sd_pass + eps)
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


def score_welch_t(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    """Score features by absolute Welch t-style class separation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    fail = y == 1
    pass_ = y == 0
    n_fail = int(np.sum(fail))
    n_pass = int(np.sum(pass_))

    mu_fail = np.nanmean(x[fail], axis=0)
    mu_pass = np.nanmean(x[pass_], axis=0)
    sd_fail = np.nanstd(x[fail], axis=0, ddof=1)
    sd_pass = np.nanstd(x[pass_], axis=0, ddof=1)
    denom = np.sqrt((sd_fail**2) / max(n_fail, 1) + (sd_pass**2) / max(n_pass, 1) + eps)
    score = np.abs(mu_fail - mu_pass) / denom
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


def score_f_test(x: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Score non-constant features with scikit-learn's ANOVA F statistic."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    zero_var = _zero_variance_mask(x)
    score = np.full(x.shape[1], -np.inf, dtype=float)
    if not np.any(~zero_var):
        return score

    # f_classif warns on constant columns, so prefilter them and restore positions.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Features .* are constant\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"invalid value encountered in divide",
            category=RuntimeWarning,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            non_constant_scores, _ = f_classif(x[:, ~zero_var], y)
    score[~zero_var] = _sanitize_scores(non_constant_scores)
    return score


def score_pearson(x: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Score features by absolute Pearson correlation with the binary label."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=float)
    yc = y - np.mean(y)
    denom_y = np.linalg.norm(yc)
    score = np.full(x.shape[1], -np.inf, dtype=float)
    for j in range(x.shape[1]):
        xj = x[:, j] - np.mean(x[:, j])
        denom = np.linalg.norm(xj) * denom_y
        if denom > 0:
            score[j] = abs(float(np.dot(xj, yc) / denom))
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    SelectorName.S2N: score_s2n,
    SelectorName.WELCH_T: score_welch_t,
    SelectorName.F_TEST: score_f_test,
    SelectorName.PEARSON: score_pearson,
}


def rank_features(
    method: str,
    x: np.ndarray,
    y_bin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic feature order and raw scores for a univariate method."""
    scorer = _SCORERS.get(method)
    if scorer is None:
        raise ValueError(f"Unsupported univariate selector method={method}")

    scores = scorer(x, y_bin)
    order = _rank_desc_with_index_tiebreak(scores)
    return order, scores
