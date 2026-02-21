from __future__ import annotations

import numpy as np


def _rank_desc_with_index_tiebreak(scores: np.ndarray) -> np.ndarray:
    idx = np.arange(scores.shape[0], dtype=int)
    return np.lexsort((idx, -scores))


def _fallback_relief_scores(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    # Deterministic ReliefF-like fallback when skrebate is unavailable.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n, p = x.shape
    w = np.zeros(p, dtype=float)
    for i in range(n):
        xi = x[i]
        dist = np.linalg.norm(x - xi, axis=1)
        same = np.where(y == y[i])[0]
        same = same[same != i]
        diff = np.where(y != y[i])[0]
        if same.size == 0 or diff.size == 0:
            continue
        k_hit = min(n_neighbors, same.size)
        k_miss = min(n_neighbors, diff.size)
        hit = same[np.argpartition(dist[same], k_hit - 1)[:k_hit]]
        miss = diff[np.argpartition(dist[diff], k_miss - 1)[:k_miss]]
        w += np.mean(np.abs(xi - x[miss]), axis=0)
        w -= np.mean(np.abs(xi - x[hit]), axis=0)
    w = w / max(n, 1)
    w[~np.isfinite(w)] = -np.inf
    return w


def relief_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    try:
        from skrebate import ReliefF

        rel = ReliefF(
            n_features_to_select=x.shape[1],
            n_neighbors=int(n_neighbors),
            n_jobs=-1,
        )
        rel.fit(x, y)
        scores = np.asarray(rel.feature_importances_, dtype=float)
    except Exception:
        scores = _fallback_relief_scores(x, y, n_neighbors=int(n_neighbors))

    scores[~np.isfinite(scores)] = -np.inf
    order = _rank_desc_with_index_tiebreak(scores)
    return order, scores

