"""Temporal selection tuning helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from secom.config import ScalerName

_FLOAT_TOLERANCE = 1e-12
_ConfigKey = Callable[[dict[str, Any]], tuple[Any, ...]]


def _is_optional_missing(value: Any) -> bool:
    """Return whether an optional config value is null-like."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _inner_config_simplicity_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    """Prefer smaller inner configs after inner BER ties."""
    nn = row.get("n_neighbors")
    nn_key = math.inf if _is_optional_missing(nn) else float(nn)
    scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
    return (float(row["k"]), float(row["C"]), scaler_pref, nn_key)


def select_ber_config(
    config_rows: list[dict[str, Any]],
    *,
    simplicity_key: _ConfigKey,
    empty_message: str = "No configs to select",
) -> dict[str, Any]:
    """Minimize inner BER; break numerical ties by declared simplicity/order."""
    if not config_rows:
        raise ValueError(empty_message)
    min_ber = min(float(row["mean_inner_BER"]) for row in config_rows)
    tied_on_ber = [row for row in config_rows if abs(float(row["mean_inner_BER"]) - min_ber) <= _FLOAT_TOLERANCE]
    return min(tied_on_ber, key=simplicity_key)


def select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the temporal config by inner BER, then deterministic simplicity."""
    return select_ber_config(config_rows, simplicity_key=_inner_config_simplicity_key)
