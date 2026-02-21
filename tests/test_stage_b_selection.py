from __future__ import annotations

import numpy as np

from secom.config import ScalerName
from secom.pipeline import _select_best_inner_config


def test_stage_b_inner_selection_tiebreak_chain() -> None:
    rows = [
        {
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.81,
            "mean_inner_BER": 0.30,
        },
        {
            "k": 20,
            "C": 1.0,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.81,
            "mean_inner_BER": 0.30,
        },
        {
            "k": 10,
            "C": 0.1,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.805,
            "mean_inner_BER": 0.29,
        },
    ]
    best = _select_best_inner_config(rows)
    # Within 0.01 AUC window, lower BER row should win first.
    assert best["k"] == 10
    assert np.isclose(best["C"], 0.1)

