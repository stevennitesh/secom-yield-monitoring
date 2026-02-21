from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from secom.config import FoldPlanName, INNER_MIN_CLASS, LOCKBOX_FRAC, MIN_TEST_FAILS


@dataclass(frozen=True)
class DevLockboxSplit:
    dev: pd.DataFrame
    lockbox: pd.DataFrame
    n_total_after_nat_drop: int
    n_dev: int
    n_lockbox: int


@dataclass(frozen=True)
class OuterFold:
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
    plan_name: str
    folds: list[OuterFold]
    last_week: int


def split_dev_lockbox(df: pd.DataFrame, lockbox_frac: float = LOCKBOX_FRAC) -> DevLockboxSplit:
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


def _plan_windows(last_week: int) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    return {
        FoldPlanName.PRIMARY_3FOLD: [
            ((1, 5), (6, last_week)),
            ((1, 7), (8, last_week)),
            ((1, 9), (10, last_week)),
        ],
        FoldPlanName.FALLBACK_3FOLD: [
            ((1, 4), (5, last_week)),
            ((1, 6), (7, last_week)),
            ((1, 8), (9, last_week)),
        ],
        FoldPlanName.FALLBACK_2FOLD: [
            ((1, 6), (7, last_week)),
            ((1, 8), (9, last_week)),
        ],
    }


def _build_plan(dev: pd.DataFrame, plan_name: str, windows) -> list[OuterFold]:
    folds: list[OuterFold] = []
    for i, (train_w, test_w) in enumerate(windows, start=1):
        folds.append(
            _make_outer_fold(
                dev=dev,
                outer_fold=i,
                train_weeks=train_w,
                test_weeks=test_w,
            )
        )
    return folds


def choose_outer_fold_plan(
    dev_with_weeks: pd.DataFrame,
    min_test_fails: int = MIN_TEST_FAILS,
) -> OuterFoldPlanResult | None:
    last_week = int(dev_with_weeks["week_label"].max())
    plans = _plan_windows(last_week)
    for plan_name in [
        FoldPlanName.PRIMARY_3FOLD,
        FoldPlanName.FALLBACK_3FOLD,
        FoldPlanName.FALLBACK_2FOLD,
    ]:
        folds = _build_plan(dev_with_weeks, plan_name, plans[plan_name])
        if all(f.test_fails >= min_test_fails for f in folds):
            return OuterFoldPlanResult(plan_name=plan_name, folds=folds, last_week=last_week)
    return None


def check_inner_cv_feasible(
    y: np.ndarray,
    min_class_count: int = INNER_MIN_CLASS,
) -> bool:
    y = np.asarray(y, dtype=int)
    n_fail = int(np.sum(y == 1))
    n_pass = int(np.sum(y == 0))
    return min(n_fail, n_pass) >= min_class_count


def lane_b_feasibility_gate(
    dev: pd.DataFrame,
    plan: OuterFoldPlanResult | None,
    min_class_count: int = INNER_MIN_CLASS,
) -> tuple[bool, str | None]:
    if plan is None:
        return (False, "min_test_fails_lt_20_after_fallbacks")
    for fold in plan.folds:
        y_train = dev.loc[fold.train_index, "y_bin"].to_numpy()
        if not check_inner_cv_feasible(y_train, min_class_count=min_class_count):
            return (False, "min_class_count_lt_5_for_inner_cv")
    if not check_inner_cv_feasible(dev["y_bin"].to_numpy(), min_class_count=min_class_count):
        return (False, "min_class_count_lt_5_for_inner_cv")
    return (True, None)


def fold_plan_manifest_ranges(plan: OuterFoldPlanResult) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for fold in plan.folds:
        out.append(
            {
                "outer_fold": fold.outer_fold,
                "train_weeks": [fold.train_start_week, fold.train_end_week],
                "test_weeks": [fold.test_start_week, fold.test_end_week],
            }
        )
    return out


def to_time_window_string(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    return f"{start_ts.strftime('%Y-%m-%dT%H:%M:%S')}/{end_ts.strftime('%Y-%m-%dT%H:%M:%S')}"

