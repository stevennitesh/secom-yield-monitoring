"""Shared score cleanup and deterministic ranking for feature selectors."""

from __future__ import annotations

import numpy as np


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Coerce undefined scores to the shared bottom-rank sentinel."""
    sanitized = np.asarray(scores, dtype=float).copy()
    sanitized[np.isnan(sanitized)] = -np.inf
    return sanitized


def rank_desc_with_index_tiebreak(scores: np.ndarray, feature_indices: np.ndarray | None = None) -> np.ndarray:
    """Rank higher scores first, with deterministic lower-index tie breaks."""
    tie_break_indices = (
        np.arange(scores.shape[0], dtype=int) if feature_indices is None else np.asarray(feature_indices)
    )
    return np.lexsort((tie_break_indices, -scores))
