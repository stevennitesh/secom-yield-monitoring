from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import LaneAClassifier, ReplicationMode, SelectorName

_VALID_CLASSIFIERS = set(LaneAClassifier.ALL)
_VALID_REPLICATION_MODES = {
    ReplicationMode.STRICT,
    ReplicationMode.WITH_MISSING_INDICATORS,
}
_TUNING_TRACE_REQUIRED_COLS = {
    "selector",
    "classifier",
    "fold",
    "replication_mode",
    "chosen_alpha",
    "chosen_gamma",
    "chosen_C",
    "chosen_mrmr_lambda",
    "chosen_mutual_info_n_neighbors",
    "chosen_l1_selector_c",
    "threshold",
    "selector_tuning_scope",
}


def validate_lane_a_artifacts(
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    tuning_trace_df: pd.DataFrame,
    classifiers_run: list[str],
    selectors_run: list[str] | None = None,
) -> None:
    classifiers_run = sorted(set(classifiers_run))
    selectors_run = (
        sorted(set(SelectorName.ALL))
        if selectors_run is None
        else sorted(set(selectors_run))
    )
    expected_cls_set = set(classifiers_run)
    expected_selector_set = set(selectors_run)
    unknown_in_run = expected_cls_set - _VALID_CLASSIFIERS
    if unknown_in_run:
        raise ValueError(f"classifiers_run unknown values: {sorted(unknown_in_run)}")
    unknown_selectors = expected_selector_set - set(SelectorName.ALL)
    if unknown_selectors:
        raise ValueError(f"selectors_run unknown values: {sorted(unknown_selectors)}")

    for name, df in [
        ("summary", summary_df),
        ("ablation", ablation_df),
        ("strict", strict_df),
        ("mi", mi_df),
    ]:
        if "classifier" not in df.columns:
            raise ValueError(f"{name}: missing 'classifier' column")
        actual_cls_set = set(df["classifier"].dropna().astype(str).unique())
        if actual_cls_set != expected_cls_set:
            raise ValueError(f"{name}: classifier set {actual_cls_set} != expected {expected_cls_set}")
        if "selector" not in df.columns:
            raise ValueError(f"{name}: missing 'selector' column")
        actual_selector_set = set(df["selector"].dropna().astype(str).unique())
        if actual_selector_set != expected_selector_set:
            raise ValueError(
                f"{name}: selector set {actual_selector_set} != expected {expected_selector_set}"
            )

    bad_modes = set(summary_df["replication_mode"].dropna().astype(str).unique()) - _VALID_REPLICATION_MODES
    if bad_modes:
        raise ValueError(f"summary: invalid replication_mode values: {sorted(bad_modes)}")

    n_cls = len(classifiers_run)
    n_sel = len(selectors_run)
    if len(summary_df) != n_sel * n_cls * 2:
        raise ValueError(f"summary: expected {n_sel * n_cls * 2} rows, got {len(summary_df)}")
    if len(ablation_df) != n_sel * n_cls:
        raise ValueError(f"ablation: expected {n_sel * n_cls} rows, got {len(ablation_df)}")
    if len(strict_df) != n_sel * n_cls * 10:
        raise ValueError(f"strict: expected {n_sel * n_cls * 10} rows, got {len(strict_df)}")
    if len(mi_df) != n_sel * n_cls * 10:
        raise ValueError(f"mi: expected {n_sel * n_cls * 10} rows, got {len(mi_df)}")

    if LaneAClassifier.KRR_STRICT in expected_cls_set:
        mask = (
            (summary_df["classifier"] == LaneAClassifier.KRR_STRICT)
            & (summary_df["selector"] == SelectorName.F_TEST)
            & (summary_df["replication_mode"] == ReplicationMode.STRICT)
        )
        n = int(mask.sum())
        if n != 1:
            raise ValueError(f"Benchmark anchor row: expected 1, got {n}")

    missing_cols = _TUNING_TRACE_REQUIRED_COLS - set(tuning_trace_df.columns)
    if missing_cols:
        raise ValueError(f"tuning_trace missing columns: {sorted(missing_cols)}")

    n_tuned = len(expected_cls_set & {LaneAClassifier.KRR_BALANCED, LaneAClassifier.LOGREG})
    expected_trace_rows = n_sel * n_tuned * 2 * 10
    if len(tuning_trace_df) != expected_trace_rows:
        raise ValueError(f"tuning_trace: expected {expected_trace_rows} rows, got {len(tuning_trace_df)}")

    if n_tuned > 0:
        trace_cls = set(tuning_trace_df["classifier"].dropna().astype(str).unique())
        bad_trace_cls = trace_cls - {LaneAClassifier.KRR_BALANCED, LaneAClassifier.LOGREG}
        if bad_trace_cls:
            raise ValueError(f"tuning_trace: invalid classifier values: {sorted(bad_trace_cls)}")
        bad_scope = set(tuning_trace_df["selector_tuning_scope"].dropna().astype(str).unique()) - {
            "outer_train_fixed"
        }
        if bad_scope:
            raise ValueError(f"tuning_trace: invalid selector_tuning_scope values: {sorted(bad_scope)}")

    for (selector, classifier), grp in ablation_df.groupby(["selector", "classifier"]):
        expected = float(grp["BER_strict"].iloc[0]) - float(grp["BER_MI"].iloc[0])
        actual = float(grp["delta_BER"].iloc[0])
        if not np.isclose(actual, expected, atol=1e-9):
            raise ValueError(
                f"delta_BER mismatch ({selector},{classifier}): {actual} != {expected}"
            )
