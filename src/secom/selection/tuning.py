"""Temporal selection tuning helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from secom.config import ScalerName

_FLOAT_TOLERANCE = 1e-12
_NEAR_BEST_AUC_BAND = 0.01
_ConfigKey = Callable[[dict[str, Any]], tuple[Any, ...]]


def _inner_config_simplicity_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    """Prefer smaller temporal configs after near-best AUC and BER ties."""
    nn = row.get("n_neighbors")
    nn_key = math.inf if nn is None else nn
    scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
    return (row["k"], row["C"], scaler_pref, nn_key)


def select_near_best_auc_config(
    config_rows: list[dict[str, Any]],
    *,
    simplicity_key: _ConfigKey,
    empty_message: str = "No configs to select",
) -> dict[str, Any]:
    """Choose a config by near-best AUC, BER, then a caller-supplied simplicity key."""
    if not config_rows:
        raise ValueError(empty_message)
    best_auc = max(float(row["mean_inner_ROC_AUC"]) for row in config_rows)
    near_best_auc = [
        row
        for row in config_rows
        if float(row["mean_inner_ROC_AUC"]) >= best_auc - _NEAR_BEST_AUC_BAND - _FLOAT_TOLERANCE
    ]
    min_ber = min(float(row["mean_inner_BER"]) for row in near_best_auc)
    tied_on_ber = [row for row in near_best_auc if np.isclose(float(row["mean_inner_BER"]), min_ber)]
    return min(tied_on_ber, key=simplicity_key)


def select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the temporal config by near-best AUC, BER, then deterministic simplicity."""
    return select_near_best_auc_config(config_rows, simplicity_key=_inner_config_simplicity_key)
