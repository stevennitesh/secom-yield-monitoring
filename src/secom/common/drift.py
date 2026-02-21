from __future__ import annotations

import numpy as np

from secom.config import EPS_PSI


def psi_for_feature(dev_vals: np.ndarray, lock_vals: np.ndarray) -> float:
    dev = np.asarray(dev_vals, dtype=float)
    lock = np.asarray(lock_vals, dtype=float)
    dev_nm = dev[~np.isnan(dev)]
    if dev_nm.size > 0:
        q = np.quantile(dev_nm, np.arange(0.1, 1.0, 0.1))
        edges = np.unique(np.asarray(q, dtype=float))
    else:
        edges = np.array([], dtype=float)

    def bin_index(val: float) -> int:
        if np.isnan(val):
            return len(edges) + 1
        if edges.size == 0:
            return 0
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    n_bins = (len(edges) + 1) + 1
    dev_counts = np.zeros(n_bins, dtype=float)
    lock_counts = np.zeros(n_bins, dtype=float)
    for v in dev:
        dev_counts[bin_index(float(v))] += 1
    for v in lock:
        lock_counts[bin_index(float(v))] += 1

    p = dev_counts / max(dev.shape[0], 1)
    q = lock_counts / max(lock.shape[0], 1)
    psi = np.sum((p - q) * np.log((p + EPS_PSI) / (q + EPS_PSI)))
    return float(psi)
