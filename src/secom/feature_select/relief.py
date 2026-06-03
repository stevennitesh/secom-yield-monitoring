"""ReliefF feature ranking with a deterministic local fallback."""

from __future__ import annotations

import numpy as np


def _rank_desc_with_index_tiebreak(scores: np.ndarray) -> np.ndarray:
    """Rank higher scores first, with deterministic lower-index tie breaks."""
    feature_indices = np.arange(scores.shape[0], dtype=int)
    return np.lexsort((feature_indices, -scores))


def _fallback_relief_scores(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute deterministic ReliefF-like scores when skrebate is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n_rows, n_features = x.shape
    weights = np.zeros(n_features, dtype=float)

    for row_idx in range(n_rows):
        row = x[row_idx]
        distances = np.linalg.norm(x - row, axis=1)
        hit_candidates = np.where(y == y[row_idx])[0]
        hit_candidates = hit_candidates[hit_candidates != row_idx]
        miss_candidates = np.where(y != y[row_idx])[0]

        if hit_candidates.size == 0 or miss_candidates.size == 0:
            continue

        k_hit = min(n_neighbors, hit_candidates.size)
        k_miss = min(n_neighbors, miss_candidates.size)
        nearest_hits = hit_candidates[np.argpartition(distances[hit_candidates], k_hit - 1)[:k_hit]]
        nearest_misses = miss_candidates[np.argpartition(distances[miss_candidates], k_miss - 1)[:k_miss]]
        weights += np.mean(np.abs(row - x[nearest_misses]), axis=0)
        weights -= np.mean(np.abs(row - x[nearest_hits]), axis=0)

    weights = weights / max(n_rows, 1)
    weights[~np.isfinite(weights)] = -np.inf
    return weights


def relief_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ReliefF feature order and scores, falling back when skrebate is absent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    try:
        from skrebate import ReliefF

        estimator = ReliefF(
            n_features_to_select=x.shape[1],
            n_neighbors=int(n_neighbors),
            n_jobs=-1,
        )
        estimator.fit(x, y)
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    except Exception:
        # Keep the study runnable in minimal environments while preserving deterministic order.
        scores = _fallback_relief_scores(x, y, n_neighbors=int(n_neighbors))

    scores[~np.isfinite(scores)] = -np.inf
    order = _rank_desc_with_index_tiebreak(scores)
    return order, scores
