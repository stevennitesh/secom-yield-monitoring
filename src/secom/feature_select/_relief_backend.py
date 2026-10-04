"""Exact binary, imputed-data scoring acceleration for pinned skrebate 0.8.4.

Distances and feature typing remain upstream. The NumPy operations below preserve
its neighbor ordering, mixed-feature ramp, class normalization and summation order.
This adapter deliberately supports only the finite binary inputs used by this study.
"""

from __future__ import annotations

import sys
from importlib.metadata import version

import numpy as np
from skrebate import ReliefF


class VectorizedReliefF(ReliefF):
    """Replace per-feature Python calls while retaining the reference calculation."""

    def fit(self, X, y, weights=None):
        """Fail explicitly if the pinned reference or supported input contract changes."""
        if version("skrebate") != "0.8.4":
            raise RuntimeError("Vectorized ReliefF requires the verified skrebate 0.8.4 reference")
        if not np.all(np.isfinite(X)) or len(np.unique(y)) != 2 or weights is not None:
            raise ValueError("Vectorized ReliefF requires finite, imputed features and binary, unweighted labels")
        return super().fit(X, y, weights=weights)

    def _find_neighbors(self, inst):
        """Select the same nearest hits/misses in the reference's combined order."""
        # Finite-input upstream distances are symmetric; preserve its default argsort ties.
        distances = self._distance_array[inst].copy()
        distances[inst] = sys.maxsize
        nearest = np.argsort(distances)
        hits = self._y[nearest] == self._y[inst]
        positions = np.concatenate(
            (np.flatnonzero(hits)[: self.n_neighbors], np.flatnonzero(~hits)[: self.n_neighbors])
        )
        return nearest[np.sort(positions)]

    def _run_algorithm(self):
        """Score all features together, with the reference's per-feature reductions."""
        if self._class_type != "binary":
            raise ValueError("Vectorized ReliefF supports only binary labels")
        x, y = self._X, self._y
        continuous = np.array([self.attr[h][0] == "continuous" for h in self._headers])
        columns = np.flatnonzero(continuous)
        categorical = np.flatnonzero(~continuous)
        ranges = np.array([self.attr[self._headers[j]][3] for j in columns])[:, None]
        stds = np.array([self.attr[self._headers[j]][4] for j in columns])[:, None]
        # Upstream stacks per-instance arrays in C order before summing across samples.
        # empty_like(x) would use F order for some real preprocessing outputs and change rounding.
        scores = np.empty(x.shape, dtype=float, order="C")
        for inst in range(len(x)):
            neighbors = self._find_neighbors(inst)
            hits = neighbors[y[neighbors] == y[inst]]
            misses = neighbors[y[neighbors] != y[inst]]
            contributions = []
            for group in (hits, misses):
                delta = np.zeros(x.shape[1])
                if len(group):
                    # Each feature's neighbors must be contiguous to match ramp_vec(...).sum().
                    values = np.ascontiguousarray(x[np.ix_(group, columns)].T)
                    raw = np.abs(x[inst, columns, None] - values)
                    diff = raw / ranges
                    if self.data_type == "mixed":
                        diff = np.where(raw > stds, 1.0, diff)
                    delta[columns] = diff.sum(axis=1) / len(group)
                    delta[categorical] = (x[np.ix_(group, categorical)] != x[inst, categorical]).sum(axis=0) / len(
                        group
                    )
                contributions.append(delta)
            scores[inst] = (-contributions[0] + contributions[1]) / float(len(x))
        return scores.sum(axis=0)
