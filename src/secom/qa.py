from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import BenchmarkClassifier, ReplicationMode, SelectorName


def validate_benchmark_replication_artifacts(
    *,
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    full_fit_df: pd.DataFrame,
) -> None:
    for name, df, required in (
        ("benchmark_sweep", sweep_df, {"selector", "classifier", "replication_mode"}),
        ("benchmark_best_config", best_df, {"selector", "classifier", "replication_mode"}),
        ("benchmark_fold_metrics", fold_metrics_df, {"selector", "classifier", "replication_mode", "fold", "BER"}),
        ("benchmark_summary", summary_df, {"selector", "classifier", "replication_mode", "mean_BER"}),
        ("benchmark_ablation", ablation_df, {"selector", "classifier", "BER_reference", "BER_missing_indicator", "delta_BER"}),
        ("benchmark_full_fit_summary", full_fit_df, {"selector", "classifier", "replication_mode", "BER_full_dataset"}),
    ):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

    valid_classifiers = set(BenchmarkClassifier.ALL + BenchmarkClassifier.OPTIONAL_BENCHMARK)
    valid_replication_modes = {
        ReplicationMode.STRICT,
        ReplicationMode.WITH_MISSING_INDICATORS,
    }
    valid_selectors = set(SelectorName.ALL)

    for name, df in (
        ("benchmark_sweep", sweep_df),
        ("benchmark_best_config", best_df),
        ("benchmark_fold_metrics", fold_metrics_df),
        ("benchmark_summary", summary_df),
        ("benchmark_full_fit_summary", full_fit_df),
    ):
        bad_classifiers = set(df["classifier"].dropna().astype(str).unique()) - valid_classifiers
        if bad_classifiers:
            raise ValueError(f"{name}: invalid classifier values {sorted(bad_classifiers)}")
        bad_selectors = set(df["selector"].dropna().astype(str).unique()) - valid_selectors
        if bad_selectors:
            raise ValueError(f"{name}: invalid selector values {sorted(bad_selectors)}")
        bad_modes = set(df["replication_mode"].dropna().astype(str).unique()) - valid_replication_modes
        if bad_modes:
            raise ValueError(f"{name}: invalid replication_mode values {sorted(bad_modes)}")

    triplet_cols = ["selector", "classifier", "replication_mode"]
    expected_modes = {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS}

    def _mode_map(df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
        out: dict[tuple[str, str], set[str]] = {}
        for (selector, classifier), frame in df.groupby(["selector", "classifier"], dropna=False):
            out[(str(selector), str(classifier))] = set(frame["replication_mode"].dropna().astype(str).unique())
        return out

    for name, df in (
        ("benchmark_best_config", best_df),
        ("benchmark_summary", summary_df),
        ("benchmark_full_fit_summary", full_fit_df),
    ):
        mode_map = _mode_map(df)
        for key, modes in mode_map.items():
            if modes != expected_modes:
                raise ValueError(f"{name}: expected paired replication modes for {key}, got {sorted(modes)}")

    if best_df.duplicated(triplet_cols, keep=False).any():
        raise ValueError("benchmark_best_config: duplicate triplets")
    if summary_df.duplicated(triplet_cols, keep=False).any():
        raise ValueError("benchmark_summary: duplicate triplets")
    if full_fit_df.duplicated(triplet_cols, keep=False).any():
        raise ValueError("benchmark_full_fit_summary: duplicate triplets")

    fold_group_sizes = fold_metrics_df.groupby(triplet_cols, dropna=False)["fold"].nunique()
    if not np.all(fold_group_sizes.to_numpy(dtype=int) == 10):
        raise ValueError("benchmark_fold_metrics: each triplet must include exactly 10 folds")
    if fold_metrics_df["fold"].min() != 1 or fold_metrics_df["fold"].max() != 10:
        raise ValueError("benchmark_fold_metrics: fold values must be 1..10")
    if fold_metrics_df.duplicated([*triplet_cols, "fold"], keep=False).any():
        raise ValueError("benchmark_fold_metrics: duplicate (selector,classifier,replication_mode,fold) rows")

    summary_triplets = set(summary_df[triplet_cols].itertuples(index=False, name=None))
    best_triplets = set(best_df[triplet_cols].itertuples(index=False, name=None))
    full_fit_triplets = set(full_fit_df[triplet_cols].itertuples(index=False, name=None))
    fold_triplets = set(fold_metrics_df[triplet_cols].drop_duplicates().itertuples(index=False, name=None))
    if not (best_triplets == summary_triplets == full_fit_triplets == fold_triplets):
        raise ValueError("benchmark artifacts: inconsistent triplet coverage across benchmark outputs")

    if {"BER_reference", "BER_missing_indicator", "delta_BER"}.issubset(ablation_df.columns):
        diff = np.abs(
            ablation_df["delta_BER"]
            - (ablation_df["BER_reference"] - ablation_df["BER_missing_indicator"])
        )
        if np.any(diff > 1e-9):
            raise ValueError("benchmark_ablation: delta_BER mismatch")

    ablation_pairs = set(ablation_df[["selector", "classifier"]].drop_duplicates().itertuples(index=False, name=None))
    summary_pairs = set(summary_df[["selector", "classifier"]].drop_duplicates().itertuples(index=False, name=None))
    if ablation_pairs != summary_pairs:
        raise ValueError("benchmark_ablation: selector/classifier coverage mismatch vs summary")
