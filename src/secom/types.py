from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from secom.cv import OuterFoldPlanResult


@dataclass(frozen=True)
class DataBundle:
    all_data: pd.DataFrame
    dev: pd.DataFrame
    lockbox: pd.DataFrame
    feature_columns: list[str]
    dev_with_weeks: pd.DataFrame
    fold_plan: OuterFoldPlanResult | None
    lane_b_feasible: bool
    lane_b_infeasible_reason: str | None


@dataclass(frozen=True)
class RoleConfig:
    role: str
    selector: str
    k: int
    c_value: float
    scaler: str
    n_neighbors: int | None

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "k": int(self.k),
            "C": float(self.c_value),
            "scaler": self.scaler,
            "n_neighbors": None if self.n_neighbors is None else int(self.n_neighbors),
        }


@dataclass
class FittedRoleModel:
    config: RoleConfig
    imputer: Any
    scaler: Any
    selected_local_idx: np.ndarray
    selected_global_idx: list[int]
    clf: Any
    dev_scores: np.ndarray
    scientific_threshold: float
    operational_threshold: float
    threshold_at_tnr90_dev: float
    tnr_at_tnr90_dev: float
    tpr_at_tnr90_dev: float
    feature_meta: list[Any]
