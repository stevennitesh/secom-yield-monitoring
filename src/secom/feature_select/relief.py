"""Reference-equivalent ReliefF ranking with bounded reuse of training-only scores."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

import numpy as np

from secom.feature_select._ranking import rank_desc_with_index_tiebreak, sanitize_scores

_SCORE_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
RELIEF_CACHE_SIZE = 128


def relief_backend_name() -> str:
    """Choose the verified accelerator or an explicit upstream reference run."""
    backend = os.environ.get("SECOM_RELIEF_BACKEND", "vectorized")
    if backend not in ("vectorized", "reference"):
        raise ValueError("SECOM_RELIEF_BACKEND must be vectorized or reference")
    return backend


def relief_worker_count() -> int:
    """The accelerator needs one process; bound the optional reference worker pool."""
    workers = int(os.environ.get("SECOM_RELIEF_N_JOBS", str(min(4, os.cpu_count() or 1))))
    if workers < 1:
        raise ValueError("SECOM_RELIEF_N_JOBS must be a positive integer")
    return 1 if relief_backend_name() == "vectorized" else workers


def _constant_feature_mask(x: np.ndarray) -> np.ndarray:
    """Identify columns that cannot carry ReliefF signal."""
    return np.std(np.asarray(x, dtype=float), axis=0, ddof=0) <= 0


def _sanitize_relief_scores(scores: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize invalid ReliefF scores and force constants to bottom rank."""
    sanitized = sanitize_scores(scores)
    sanitized[_constant_feature_mask(x)] = -np.inf
    return sanitized


def relief_rank_features(
    x: np.ndarray,
    y_bin: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic ranks; dependency and fitting failures propagate to the workflow."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("ReliefF n_neighbors must be positive")
    from skrebate import ReliefF

    if relief_backend_name() == "vectorized":
        from secom.feature_select._relief_backend import VectorizedReliefF

        ReliefF = VectorizedReliefF
    workers = relief_worker_count()
    digest = hashlib.sha256(x.tobytes(order="C"))
    digest.update(y.tobytes(order="C"))
    key = (ReliefF, n_neighbors, workers, x.shape, y.shape, digest.digest())
    if key in _SCORE_CACHE:
        _SCORE_CACHE.move_to_end(key)
        order, scores = _SCORE_CACHE[key]
        return order.copy(), scores.copy()

    estimator = ReliefF(
        n_features_to_select=x.shape[1],
        n_neighbors=n_neighbors,
        n_jobs=workers,
    )
    estimator.fit(x, y)
    scores = np.asarray(estimator.feature_importances_, dtype=float)

    scores = _sanitize_relief_scores(scores, x)
    order = rank_desc_with_index_tiebreak(scores)
    _SCORE_CACHE[key] = (order.copy(), scores.copy())
    if len(_SCORE_CACHE) > RELIEF_CACHE_SIZE:
        _SCORE_CACHE.popitem(last=False)
    return order, scores
