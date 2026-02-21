from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from secom.config import ScalerName


@dataclass(frozen=True)
class TransformedFeature:
    feature_index: int
    feature_type: str
    feature_name_or_source_col: str
    raw_index: int


def make_imputer(add_indicator: bool) -> SimpleImputer:
    return SimpleImputer(
        strategy="median",
        add_indicator=add_indicator,
        keep_empty_features=True,
    )


def make_scaler(name: str):
    if name == ScalerName.STANDARD:
        return StandardScaler(with_mean=True, with_std=True)
    if name == ScalerName.ROBUST:
        return RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        )
    raise ValueError(f"Unknown scaler: {name}")


def transformed_feature_metadata_from_imputer(
    imputer: SimpleImputer, raw_feature_count: int
) -> list[TransformedFeature]:
    out: list[TransformedFeature] = []
    for raw_idx in range(raw_feature_count):
        out.append(
            TransformedFeature(
                feature_index=raw_idx,
                feature_type="value",
                feature_name_or_source_col=f"X{raw_idx}",
                raw_index=raw_idx,
            )
        )

    if getattr(imputer, "indicator_", None) is not None:
        for raw_idx in imputer.indicator_.features_.tolist():
            out.append(
                TransformedFeature(
                    feature_index=raw_feature_count + int(raw_idx),
                    feature_type="missing_indicator",
                    feature_name_or_source_col=f"M{int(raw_idx)}",
                    raw_index=int(raw_idx),
                )
            )
    return out


def local_to_global_feature_indices(
    local_indices: np.ndarray,
    transformed_meta: list[TransformedFeature],
) -> list[int]:
    return [transformed_meta[int(i)].feature_index for i in local_indices.tolist()]


def build_feature_universe(raw_feature_count: int) -> list[TransformedFeature]:
    universe = []
    for raw_idx in range(raw_feature_count):
        universe.append(
            TransformedFeature(
                feature_index=raw_idx,
                feature_type="value",
                feature_name_or_source_col=f"X{raw_idx}",
                raw_index=raw_idx,
            )
        )
    for raw_idx in range(raw_feature_count):
        universe.append(
            TransformedFeature(
                feature_index=raw_feature_count + raw_idx,
                feature_type="missing_indicator",
                feature_name_or_source_col=f"M{raw_idx}",
                raw_index=raw_idx,
            )
        )
    return universe

