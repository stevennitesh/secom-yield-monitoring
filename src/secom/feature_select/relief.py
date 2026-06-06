"""ReliefF feature ranking with a deterministic local fallback."""

from __future__ import annotations

import numpy as np

from secom.feature_select._ranking import rank_desc_with_index_tiebreak, sanitize_scores


def _constant_feature_mask(x: np.ndarray) -> np.ndarray:
    """Identify columns that cannot carry ReliefF signal."""
    return np.std(np.asarray(x, dtype=float), axis=0, ddof=0) <= 0


def _sanitize_relief_scores(scores: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize invalid ReliefF scores and force constants to bottom rank."""
    sanitized = sanitize_scores(scores)
    sanitized[_constant_feature_mask(x)] = -np.inf
    return sanitized


def _class_neighbor_candidates(y: np.ndarray, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return same-class and opposite-class candidate indices for one row."""
    hit_candidates = np.flatnonzero(y == y[row_idx])
    hit_candidates = hit_candidates[hit_candidates != row_idx]
    miss_candidates = np.flatnonzero(y != y[row_idx])
    return hit_candidates, miss_candidates


def _nearest_candidates(distances: np.ndarray, candidates: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Select the nearest candidate indices without fully sorting all distances."""
    k = min(int(n_neighbors), candidates.size)
    return candidates[np.argpartition(distances[candidates], k - 1)[:k]]


def _fallback_relief_scores(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute deterministic ReliefF-like scores when skrebate is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n_rows, n_features = x.shape
    weights = np.zeros(n_features, dtype=float)

    for row_idx in range(n_rows):
        row = x[row_idx]
        distances = np.linalg.norm(x - row, axis=1)
        hit_candidates, miss_candidates = _class_neighbor_candidates(y, row_idx)

        if hit_candidates.size == 0 or miss_candidates.size == 0:
            continue

        nearest_hits = _nearest_candidates(distances, hit_candidates, n_neighbors)
        nearest_misses = _nearest_candidates(distances, miss_candidates, n_neighbors)
        weights += np.mean(np.abs(row - x[nearest_misses]), axis=0)
        weights -= np.mean(np.abs(row - x[nearest_hits]), axis=0)

    weights = weights / max(n_rows, 1)
    return _sanitize_relief_scores(weights, x)


def relief_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ReliefF feature order and scores, falling back when skrebate is absent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("ReliefF n_neighbors must be positive")
    try:
        from skrebate import ReliefF

        estimator = ReliefF(
            n_features_to_select=x.shape[1],
            n_neighbors=n_neighbors,
            n_jobs=-1,
        )
        estimator.fit(x, y)
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    except Exception:
        # Keep the study runnable in minimal environments while preserving deterministic order.
        scores = _fallback_relief_scores(x, y, n_neighbors=n_neighbors)

    scores = _sanitize_relief_scores(scores, x)
    order = rank_desc_with_index_tiebreak(scores)
    return order, scores
