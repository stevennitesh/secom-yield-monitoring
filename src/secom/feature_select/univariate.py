"""Univariate feature ranking methods for the SECOM study selectors."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import f_classif

from secom.config import EPS_SELECTOR, SelectorName
from secom.feature_select._ranking import rank_desc_with_index_tiebreak, sanitize_scores


@dataclass(frozen=True)
class _ClassStats:
    """Class-conditional moments used by univariate separation scores."""

    mu_fail: np.ndarray
    mu_pass: np.ndarray
    sd_fail: np.ndarray
    sd_pass: np.ndarray
    n_fail: int
    n_pass: int


def _zero_variance_mask(x: np.ndarray) -> np.ndarray:
    """Identify columns that cannot support a univariate ranking signal."""
    std = np.std(np.asarray(x, dtype=float), axis=0, ddof=0)
    return std <= 0


def _sanitize_univariate_scores(scores: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize invalid scores and force constant columns to the bottom rank."""
    sanitized = sanitize_scores(scores)
    sanitized[_zero_variance_mask(x)] = -np.inf
    return sanitized


def _class_stats(x: np.ndarray, y_bin: np.ndarray) -> _ClassStats:
    """Return fail/pass column means, standard deviations, and class counts."""
    x_arr = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    fail = y == 1
    pass_ = y == 0

    return _ClassStats(
        mu_fail=np.nanmean(x_arr[fail], axis=0),
        mu_pass=np.nanmean(x_arr[pass_], axis=0),
        sd_fail=np.nanstd(x_arr[fail], axis=0, ddof=1),
        sd_pass=np.nanstd(x_arr[pass_], axis=0, ddof=1),
        n_fail=int(np.sum(fail)),
        n_pass=int(np.sum(pass_)),
    )


def score_s2n(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    """Score features by signal-to-noise separation between fail and pass wafers."""
    x = np.asarray(x, dtype=float)
    stats = _class_stats(x, y_bin)
    score = np.abs(stats.mu_fail - stats.mu_pass) / (stats.sd_fail + stats.sd_pass + eps)
    return _sanitize_univariate_scores(score, x)


def score_welch_t(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    """Score features by absolute Welch t-style class separation."""
    x = np.asarray(x, dtype=float)
    stats = _class_stats(x, y_bin)
    denom = np.sqrt((stats.sd_fail**2) / max(stats.n_fail, 1) + (stats.sd_pass**2) / max(stats.n_pass, 1) + eps)
    score = np.abs(stats.mu_fail - stats.mu_pass) / denom
    return _sanitize_univariate_scores(score, x)


def score_pooled_ttest(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    """Score features by absolute two-sample t statistic with pooled class variance."""
    x = np.asarray(x, dtype=float)
    stats = _class_stats(x, y_bin)
    fail_df = max(stats.n_fail - 1, 0)
    pass_df = max(stats.n_pass - 1, 0)
    pooled_df = max(fail_df + pass_df, 1)
    fail_var = np.nan_to_num(stats.sd_fail**2, nan=0.0, posinf=0.0, neginf=0.0)
    pass_var = np.nan_to_num(stats.sd_pass**2, nan=0.0, posinf=0.0, neginf=0.0)
    pooled_var = (fail_df * fail_var + pass_df * pass_var) / pooled_df
    denom = np.sqrt(pooled_var * (1.0 / max(stats.n_fail, 1) + 1.0 / max(stats.n_pass, 1)) + eps)
    score = np.abs(stats.mu_fail - stats.mu_pass) / denom
    return _sanitize_univariate_scores(score, x)


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
    score[~zero_var] = sanitize_scores(non_constant_scores)
    return score


def score_pearson(x: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Score features by absolute Pearson correlation with the binary label."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=float)
    yc = y - np.mean(y)
    denom_y = np.linalg.norm(yc)

    x_centered = x - np.mean(x, axis=0)
    denom = np.linalg.norm(x_centered, axis=0) * denom_y
    score = np.full(x.shape[1], -np.inf, dtype=float)
    valid = denom > 0
    score[valid] = np.abs(np.dot(yc, x_centered[:, valid]) / denom[valid])
    return _sanitize_univariate_scores(score, x)


_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    SelectorName.S2N: score_s2n,
    SelectorName.TTEST: score_pooled_ttest,
    SelectorName.WELCH_T: score_welch_t,
    SelectorName.F_TEST: score_f_test,
    SelectorName.PEARSON: score_pearson,
}
UNIVARIATE_SELECTORS = frozenset(_SCORERS)


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
    order = rank_desc_with_index_tiebreak(scores)
    return order, scores
