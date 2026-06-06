"""Gram-Schmidt style feature ranking with deterministic tie handling."""

from __future__ import annotations

import numpy as np

from secom.config import EPS_SELECTOR
from secom.feature_select._ranking import rank_desc_with_index_tiebreak


def _constant_feature_mask(x: np.ndarray, eps: float) -> np.ndarray:
    """Identify pre-orthogonalization columns that cannot carry selector signal."""
    x = np.asarray(x, dtype=float)
    return np.std(x, axis=0, ddof=0) <= eps


def gram_schmidt_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    k: int,
    eps: float = EPS_SELECTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank up to ``k`` features by iterative correlation with label residuals."""
    k = int(k)
    if k <= 0:
        raise ValueError("Gram-Schmidt k must be positive")
    x_raw = np.asarray(x, dtype=float)
    x_work = x_raw.copy()
    y = np.asarray(y_bin, dtype=float)
    residual = y - np.mean(y)

    n_features = x_work.shape[1]
    remaining = list(range(n_features))
    selected: list[int] = []
    final_scores = np.full(n_features, -np.inf, dtype=float)

    constant_mask = _constant_feature_mask(x_raw, eps=eps)

    while remaining and len(selected) < k:
        residual_norm = np.linalg.norm(residual)
        if residual_norm < eps:
            break

        remaining_arr = np.asarray(remaining, dtype=int)
        scores = np.full(remaining_arr.shape[0], -np.inf, dtype=float)
        for local_idx, feature_idx in enumerate(remaining_arr):
            if constant_mask[feature_idx]:
                continue

            feature_vector = x_work[:, feature_idx]
            feature_norm = np.linalg.norm(feature_vector)
            score = abs(float(np.dot(feature_vector, residual))) / (feature_norm * residual_norm + eps)
            if not np.isfinite(score):
                score = -np.inf
            scores[local_idx] = score
            final_scores[feature_idx] = score

        local_order = rank_desc_with_index_tiebreak(scores, feature_indices=remaining_arr)
        best_feat = int(remaining_arr[local_order[0]])
        best_score = float(scores[local_order[0]])
        if not np.isfinite(best_score):
            break

        selected.append(best_feat)
        basis_vector = x_work[:, best_feat]
        basis_vector = basis_vector / (np.linalg.norm(basis_vector) + eps)

        for feature_idx in remaining:
            if feature_idx == best_feat:
                continue
            x_work[:, feature_idx] = (
                x_work[:, feature_idx] - np.dot(x_work[:, feature_idx], basis_vector) * basis_vector
            )
        residual = residual - np.dot(residual, basis_vector) * basis_vector
        remaining.remove(best_feat)

    selected_arr = np.asarray(selected, dtype=int)
    return selected_arr, final_scores
