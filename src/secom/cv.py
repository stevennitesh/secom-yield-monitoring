"""Temporal splitting and outer-fold planning helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from secom.config import FoldPlanName, INNER_MIN_CLASS, LOCKBOX_FRAC


@dataclass(frozen=True)
class DevLockboxSplit:
    """Chronological DEV/LOCKBOX split metadata."""

    dev: pd.DataFrame
    lockbox: pd.DataFrame
    n_total_after_nat_drop: int
    n_dev: int
    n_lockbox: int


@dataclass(frozen=True)
class OuterFold:
    """One expanding-window temporal outer fold."""

    outer_fold: int
    train_start_week: int
    train_end_week: int
    test_start_week: int
    test_end_week: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_n: int
    test_n: int
    train_fails: int
    test_fails: int
    train_start_ts: pd.Timestamp
    train_end_ts: pd.Timestamp
    test_start_ts: pd.Timestamp
    test_end_ts: pd.Timestamp


@dataclass(frozen=True)
class OuterFoldPlanResult:
    """Selected temporal fold plan and its concrete folds."""

    plan_name: str
    folds: list[OuterFold]
    last_week: int


def split_dev_lockbox(df: pd.DataFrame, lockbox_frac: float = LOCKBOX_FRAC) -> DevLockboxSplit:
    """Split sorted data into DEV and the final chronological LOCKBOX fraction."""
    n = len(df)
    n_lockbox = int(np.floor(lockbox_frac * n))
    if n_lockbox <= 0 or n_lockbox >= n:
        raise ValueError(f"Invalid lockbox size {n_lockbox} for N={n}")
    dev = df.iloc[: n - n_lockbox].copy()
    lockbox = df.iloc[n - n_lockbox :].copy()
    return DevLockboxSplit(
        dev=dev,
        lockbox=lockbox,
        n_total_after_nat_drop=n,
        n_dev=len(dev),
        n_lockbox=len(lockbox),
    )


def add_dev_week_bins(dev: pd.DataFrame) -> pd.DataFrame:
    """Add one-indexed week labels relative to the first DEV timestamp."""
    out = dev.copy()
    t_min = out["timestamp"].min()
    delta_days = (out["timestamp"] - t_min).dt.total_seconds() / (24 * 3600)
    out["week_idx"] = np.floor(delta_days / 7.0).astype(int)
    out["week_label"] = out["week_idx"] + 1
    return out


def _make_outer_fold(
    dev: pd.DataFrame,
    outer_fold: int,
    train_weeks: tuple[int, int],
    test_weeks: tuple[int, int],
) -> OuterFold:
    """Create one outer fold from inclusive train/test week ranges."""
    train_mask = dev["week_label"].between(train_weeks[0], train_weeks[1], inclusive="both")
    test_mask = dev["week_label"].between(test_weeks[0], test_weeks[1], inclusive="both")
    train_idx = dev.index[train_mask].to_numpy(dtype=int)
    test_idx = dev.index[test_mask].to_numpy(dtype=int)

    train = dev.loc[train_idx]
    test = dev.loc[test_idx]
    if train.empty or test.empty:
        raise ValueError(f"Fold {outer_fold} empty split")

    return OuterFold(
        outer_fold=outer_fold,
        train_start_week=train_weeks[0],
        train_end_week=train_weeks[1],
        test_start_week=test_weeks[0],
        test_end_week=test_weeks[1],
        train_index=train_idx,
        test_index=test_idx,
        train_n=len(train),
        test_n=len(test),
        train_fails=int(np.sum(train["y_bin"].to_numpy() == 1)),
        test_fails=int(np.sum(test["y_bin"].to_numpy() == 1)),
        train_start_ts=train["timestamp"].min(),
        train_end_ts=train["timestamp"].max(),
        test_start_ts=test["timestamp"].min(),
        test_end_ts=test["timestamp"].max(),
    )


def choose_outer_fold_plan(dev_with_weeks: pd.DataFrame) -> OuterFoldPlanResult | None:
    """Fixed calendar-fraction blocks; never redesign periods using test labels."""
    if dev_with_weeks.empty:
        return None
    last_week = int(dev_with_weeks.week_label.max())
    boundaries = [int(np.floor(last_week * f)) for f in (0.5, 2 / 3, 5 / 6)] + [last_week]
    if boundaries[0] < 1 or len(set(boundaries)) != 4:
        return None
    try:
        folds = [
            _make_outer_fold(dev_with_weeks, i, (1, start), (start + 1, end))
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1)
        ]
    except ValueError:
        return None
    return OuterFoldPlanResult(FoldPlanName.PRIMARY_3FOLD, folds, last_week)


def fit_calibration_indices(n: int, calibration_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the last chronological 20% for calibration; no label-based boundary."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration fraction must be between zero and one")
    cut = int(np.floor((1 - calibration_fraction) * n))
    if cut < 2 or n - cut < 2:
        raise ValueError("insufficient chronological fit/calibration samples")
    return np.arange(cut), np.arange(cut, n)


def chronological_inner_splits(
    y: np.ndarray, calibration_fraction: float = 0.20
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three disjoint chronological validation blocks with expanding train prefixes."""
    n = len(y)
    boundaries = [int(np.floor(n * f)) for f in (0.5, 2 / 3, 5 / 6)] + [n]
    splits = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start < 4 or end <= start:
            continue
        fit, calibration = fit_calibration_indices(start, calibration_fraction)
        # These are validation labels inside the outer-training prefix, never outer evaluation labels.
        # Keep the fixed periods; omit an undefined BER objective rather than invent its absent-class rate.
        if all(len(np.unique(y[index])) == 2 for index in (fit, calibration, np.arange(start, end))):
            splits.append((np.arange(start), np.arange(start, end)))
    if len(splits) < 2:
        raise ValueError("fewer than two feasible chronological inner splits")
    return splits


def check_inner_cv_feasible(
    y: np.ndarray,
    min_class_count: int = INNER_MIN_CLASS,
) -> bool:
    """Return whether both classes have enough rows for inner CV."""
    y = np.asarray(y, dtype=int)
    n_fail = int(np.sum(y == 1))
    n_pass = int(np.sum(y == 0))
    return min(n_fail, n_pass) >= min_class_count


def temporal_feasibility_gate(
    dev: pd.DataFrame,
    plan: OuterFoldPlanResult | None,
    min_class_count: int = INNER_MIN_CLASS,
) -> tuple[bool, str | None]:
    """Return temporal workflow feasibility and a manifest-safe reason when blocked."""
    if plan is None:
        return (False, "insufficient_nonempty_calendar_blocks")
    try:
        training_sets = [dev.loc[fold.train_index, "y_bin"].to_numpy() for fold in plan.folds]
        training_sets.append(dev["y_bin"].to_numpy())
        for labels in training_sets:
            fit, calibration = fit_calibration_indices(len(labels))
            if min(np.bincount(labels[fit], minlength=2)) < min_class_count or len(np.unique(labels[calibration])) < 2:
                return False, "insufficient_classes_in_fit_or_calibration"
            chronological_inner_splits(labels[fit])
    except ValueError as exc:
        return False, str(exc)
    return (True, None)


def to_time_window_string(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    """Format an inclusive timestamp window for artifact rows."""
    return f"{start_ts.strftime('%Y-%m-%dT%H:%M:%S')}/{end_ts.strftime('%Y-%m-%dT%H:%M:%S')}"
