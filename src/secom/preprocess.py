"""Preprocessing helpers and transformed-feature metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from secom.config import ScalerName


@dataclass(frozen=True)
class TransformedFeature:
    """Metadata for one transformed feature column."""

    feature_index: int
    feature_type: str
    feature_name_or_source_col: str
    raw_index: int


def _value_feature(raw_idx: int) -> TransformedFeature:
    """Return report metadata for one raw value feature."""
    return TransformedFeature(
        feature_index=raw_idx,
        feature_type="value",
        feature_name_or_source_col=f"X{raw_idx}",
        raw_index=raw_idx,
    )


def _missing_indicator_feature(raw_idx: int, raw_feature_count: int) -> TransformedFeature:
    """Return report metadata for one imputed missingness indicator."""
    return TransformedFeature(
        feature_index=raw_feature_count + raw_idx,
        feature_type="missing_indicator",
        feature_name_or_source_col=f"M{raw_idx}",
        raw_index=raw_idx,
    )


def make_imputer(add_indicator: bool) -> SimpleImputer:
    """Create the median imputer used before feature selection."""
    return SimpleImputer(
        strategy="median",
        add_indicator=add_indicator,
        keep_empty_features=True,
    )


def make_scaler(name: str):
    """Create a configured scaler by study scaler name."""
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
    """Return transformed-column metadata for value columns plus fitted indicators."""
    out = [_value_feature(raw_idx) for raw_idx in range(raw_feature_count)]

    if getattr(imputer, "indicator_", None) is not None:
        # SimpleImputer exposes only indicators for raw columns that were missing at fit time.
        out.extend(
            _missing_indicator_feature(raw_idx=int(raw_idx), raw_feature_count=raw_feature_count)
            for raw_idx in imputer.indicator_.features_.tolist()
        )
    return out


def local_to_global_feature_indices(
    local_indices: np.ndarray,
    transformed_meta: list[TransformedFeature],
) -> list[int]:
    """Map transformed local column indices back to reportable feature indices."""
    return [transformed_meta[int(i)].feature_index for i in local_indices.tolist()]


def build_feature_universe(raw_feature_count: int) -> list[TransformedFeature]:
    """Return the full reportable value-plus-missing-indicator feature universe."""
    universe = [_value_feature(raw_idx) for raw_idx in range(raw_feature_count)]
    universe.extend(
        _missing_indicator_feature(raw_idx=raw_idx, raw_feature_count=raw_feature_count)
        for raw_idx in range(raw_feature_count)
    )
    return universe
