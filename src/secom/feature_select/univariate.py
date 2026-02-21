from __future__ import annotations

import numpy as np
from sklearn.feature_selection import f_classif

from secom.config import EPS_SELECTOR, SelectorName


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=float).copy()
    out[~np.isfinite(out)] = -np.inf
    return out


def _zero_variance_mask(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=0)
    return std <= 0


def _rank_desc_with_index_tiebreak(scores: np.ndarray) -> np.ndarray:
    idx = np.arange(scores.shape[0], dtype=int)
    # Desc score, asc index.
    return np.lexsort((idx, -scores))


def score_s2n(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    fail = y == 1
    pas = y == 0

    mu_fail = np.nanmean(x[fail], axis=0)
    mu_pass = np.nanmean(x[pas], axis=0)
    sd_fail = np.nanstd(x[fail], axis=0, ddof=1)
    sd_pass = np.nanstd(x[pas], axis=0, ddof=1)
    score = np.abs(mu_fail - mu_pass) / (sd_fail + sd_pass + eps)
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


def score_welch_t(x: np.ndarray, y_bin: np.ndarray, eps: float = EPS_SELECTOR) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    fail = y == 1
    pas = y == 0
    n_fail = int(np.sum(fail))
    n_pass = int(np.sum(pas))

    mu_fail = np.nanmean(x[fail], axis=0)
    mu_pass = np.nanmean(x[pas], axis=0)
    sd_fail = np.nanstd(x[fail], axis=0, ddof=1)
    sd_pass = np.nanstd(x[pas], axis=0, ddof=1)
    denom = np.sqrt((sd_fail**2) / max(n_fail, 1) + (sd_pass**2) / max(n_pass, 1) + eps)
    score = np.abs(mu_fail - mu_pass) / denom
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


def score_f_test(x: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    score, _ = f_classif(np.asarray(x, dtype=float), np.asarray(y_bin, dtype=int))
    score = _sanitize_scores(score)
    score[_zero_variance_mask(np.asarray(x, dtype=float))] = -np.inf
    return score


def score_pearson(x: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=float)
    yc = y - np.mean(y)
    denom_y = np.linalg.norm(yc)
    score = np.full(x.shape[1], -np.inf, dtype=float)
    for j in range(x.shape[1]):
        xj = x[:, j] - np.mean(x[:, j])
        denom = np.linalg.norm(xj) * denom_y
        if denom <= 0:
            score[j] = -np.inf
        else:
            score[j] = abs(float(np.dot(xj, yc) / denom))
    score = _sanitize_scores(score)
    score[_zero_variance_mask(x)] = -np.inf
    return score


def rank_features(method: str, x: np.ndarray, y_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if method == SelectorName.S2N:
        scores = score_s2n(x, y_bin)
    elif method == SelectorName.WELCH_T:
        scores = score_welch_t(x, y_bin)
    elif method == SelectorName.F_TEST:
        scores = score_f_test(x, y_bin)
    elif method == SelectorName.PEARSON:
        scores = score_pearson(x, y_bin)
    else:
        raise ValueError(f"Unsupported univariate selector method={method}")
    order = _rank_desc_with_index_tiebreak(scores)
    return order, scores

