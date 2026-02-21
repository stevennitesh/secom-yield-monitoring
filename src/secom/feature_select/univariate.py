from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

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


def score_mutual_info(
    x: np.ndarray,
    y_bin: np.ndarray,
    n_neighbors: int = 3,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    if np.unique(y).size < 2:
        return np.full(x.shape[1], -np.inf, dtype=float)
    k = max(1, min(int(n_neighbors), max(1, x.shape[0] - 1)))
    scores = mutual_info_classif(x, y, random_state=42, n_neighbors=k)
    scores = _sanitize_scores(scores)
    scores[_zero_variance_mask(x)] = -np.inf
    return scores


def score_l1_logreg(
    x: np.ndarray,
    y_bin: np.ndarray,
    c_value: float = 1.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    if np.unique(y).size < 2:
        return np.full(x.shape[1], -np.inf, dtype=float)
    clf = LogisticRegression(
        penalty="l1",
        C=float(c_value),
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=42,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(x, y)
        scores = np.abs(np.asarray(clf.coef_[0], dtype=float))
    except Exception:
        scores = np.full(x.shape[1], -np.inf, dtype=float)
    scores = _sanitize_scores(scores)
    scores[_zero_variance_mask(x)] = -np.inf
    return scores


def rank_mrmr_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    lambda_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    relevance = score_mutual_info(x, y_bin)
    zero_var = _zero_variance_mask(x)

    corr = np.corrcoef(x, rowvar=False)
    corr = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr, 0.0)

    remaining = list(range(p))
    selected: list[int] = []
    mrmr_scores = np.full(p, -np.inf, dtype=float)

    while remaining:
        best_feat: int | None = None
        best_score = -np.inf
        for feat in remaining:
            if zero_var[feat]:
                score = -np.inf
            elif not selected:
                score = float(relevance[feat])
            else:
                red = float(np.mean(corr[feat, np.asarray(selected, dtype=int)]))
                score = float(relevance[feat] - float(lambda_weight) * red)
            if not np.isfinite(score):
                score = -np.inf
            if best_feat is None or score > best_score + 1e-12:
                best_feat = feat
                best_score = score
            elif np.isclose(score, best_score) and feat < best_feat:
                best_feat = feat

        if best_feat is None:
            raise RuntimeError("mRMR selection failed to choose a feature")
        selected.append(best_feat)
        mrmr_scores[best_feat] = best_score
        remaining.remove(best_feat)

    return np.asarray(selected, dtype=int), mrmr_scores


def rank_features(
    method: str,
    x: np.ndarray,
    y_bin: np.ndarray,
    mrmr_lambda: float = 1.0,
    mutual_info_n_neighbors: int = 3,
    l1_selector_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if method == SelectorName.S2N:
        scores = score_s2n(x, y_bin)
    elif method == SelectorName.WELCH_T:
        scores = score_welch_t(x, y_bin)
    elif method == SelectorName.F_TEST:
        scores = score_f_test(x, y_bin)
    elif method == SelectorName.PEARSON:
        scores = score_pearson(x, y_bin)
    elif method == SelectorName.MUTUAL_INFO:
        scores = score_mutual_info(x, y_bin, n_neighbors=int(mutual_info_n_neighbors))
    elif method == SelectorName.L1_LOGREG:
        scores = score_l1_logreg(x, y_bin, c_value=float(l1_selector_c))
    elif method == SelectorName.MRMR:
        return rank_mrmr_features(x, y_bin, lambda_weight=float(mrmr_lambda))
    else:
        raise ValueError(f"Unsupported univariate selector method={method}")
    order = _rank_desc_with_index_tiebreak(scores)
    return order, scores
