"""Population stability index helpers for temporal drift checks."""

from __future__ import annotations

import numpy as np

from secom.config import EPS_PSI

_DECILE_QUANTILES = np.arange(0.1, 1.0, 0.1)


def _decile_edges(reference_values: np.ndarray) -> np.ndarray:
    """Build unique decile edges from non-missing development values."""
    non_missing = reference_values[~np.isnan(reference_values)]
    if non_missing.size == 0:
        return np.array([], dtype=float)

    quantiles = np.quantile(non_missing, _DECILE_QUANTILES)
    return np.unique(np.asarray(quantiles, dtype=float))


def _psi_bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values into DEV-defined PSI bins plus one missing-value bin."""
    arr = np.asarray(values, dtype=float)
    n_bins = len(edges) + 2
    if arr.size == 0:
        return np.zeros(n_bins, dtype=float)

    missing = np.isnan(arr)
    bin_indices = np.searchsorted(edges, arr, side="left")
    bin_indices = np.asarray(bin_indices, dtype=int)
    bin_indices[missing] = len(edges) + 1
    return np.bincount(bin_indices, minlength=n_bins).astype(float)


def psi_for_feature(dev_vals: np.ndarray, lock_vals: np.ndarray) -> float:
    """Return PSI between development and lockbox values using DEV-defined bins."""
    dev = np.asarray(dev_vals, dtype=float)
    lock = np.asarray(lock_vals, dtype=float)
    edges = _decile_edges(dev)

    # One bin per interval plus one explicit missing-value bin.
    dev_counts = _psi_bin_counts(dev, edges)
    lock_counts = _psi_bin_counts(lock, edges)

    p = dev_counts / max(dev.shape[0], 1)
    q = lock_counts / max(lock.shape[0], 1)
    psi = np.sum((p - q) * np.log((p + EPS_PSI) / (q + EPS_PSI)))
    return float(psi)
