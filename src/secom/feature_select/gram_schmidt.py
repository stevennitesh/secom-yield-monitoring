from __future__ import annotations

import numpy as np

from secom.config import EPS_SELECTOR


def gram_schmidt_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    k: int,
    eps: float = EPS_SELECTOR,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw = np.asarray(x, dtype=float)
    x_work = x_raw.copy()
    y = np.asarray(y_bin, dtype=float)
    residual = y - np.mean(y)

    p = x_work.shape[1]
    remaining = list(range(p))
    selected: list[int] = []
    final_scores = np.full(p, -np.inf, dtype=float)

    # Constant-feature pre-check applies on raw pre-orthogonalized representations.
    raw_norms = np.linalg.norm(x_raw, axis=0)
    constant_mask = raw_norms <= eps

    for idx in np.where(constant_mask)[0]:
        final_scores[int(idx)] = -np.inf

    while remaining and len(selected) < k:
        if np.linalg.norm(residual) < eps:
            break
        rem_arr = np.asarray(remaining, dtype=int)
        scores = np.full(rem_arr.shape[0], -np.inf, dtype=float)
        r_norm = np.linalg.norm(residual)
        for local_idx, feat_idx in enumerate(rem_arr.tolist()):
            if constant_mask[feat_idx]:
                continue
            xj = x_work[:, feat_idx]
            xj_norm = np.linalg.norm(xj)
            score = abs(float(np.dot(xj, residual))) / (xj_norm * r_norm + eps)
            if not np.isfinite(score):
                score = -np.inf
            scores[local_idx] = score
            final_scores[feat_idx] = score

        # Deterministic tie-break by smallest feature index.
        local_order = np.lexsort((rem_arr, -scores))
        best_feat = int(rem_arr[local_order[0]])
        best_score = float(scores[local_order[0]])
        if not np.isfinite(best_score):
            break

        selected.append(best_feat)
        q = x_work[:, best_feat]
        q = q / (np.linalg.norm(q) + eps)

        for feat_idx in remaining:
            if feat_idx == best_feat:
                continue
            x_work[:, feat_idx] = x_work[:, feat_idx] - np.dot(x_work[:, feat_idx], q) * q
        residual = residual - np.dot(residual, q) * q
        remaining.remove(best_feat)

    selected_arr = np.asarray(selected, dtype=int)
    if selected_arr.size == 0:
        idx = np.arange(p, dtype=int)
        fallback_order = np.lexsort((idx, -final_scores))
        return fallback_order, final_scores
    return selected_arr, final_scores
