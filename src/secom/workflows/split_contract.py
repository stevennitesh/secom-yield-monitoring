from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from secom.artifacts import ensure_reports_dir, write_csv, write_manifest
from secom.common.meta import git_commit_and_dirty, library_versions, strategy_sha256
from secom.config import ArtifactName, INNER_MIN_CLASS, LOCKBOX_FRAC, MIN_TEST_FAILS, SEED_PHASE3, SEEDS_PHASE2, SEEDS_STAGE_B
from secom.cv import (
    add_dev_week_bins,
    choose_outer_fold_plan,
    fold_plan_manifest_ranges,
    lane_b_feasibility_gate,
    split_dev_lockbox,
)
from secom.io import LoadedSecom, load_raw_secom, parse_sort_and_label
from secom.types import DataBundle


def run_split_contract(input_dir: Path, output_dir: Path, project_root: Path) -> DataBundle:
    reports = ensure_reports_dir(output_dir)
    loaded: LoadedSecom = load_raw_secom(input_dir)
    all_sorted = parse_sort_and_label(loaded.frame)

    split = split_dev_lockbox(all_sorted, lockbox_frac=LOCKBOX_FRAC)
    dev_weeks = add_dev_week_bins(split.dev)
    plan = choose_outer_fold_plan(dev_weeks, min_test_fails=MIN_TEST_FAILS)
    feasible, reason = lane_b_feasibility_gate(
        dev=dev_weeks, plan=plan, min_class_count=INNER_MIN_CLASS
    )

    split_meta = pd.DataFrame(
        [
            {
                "N_total_after_NaT_drop": split.n_total_after_nat_drop,
                "N_dev": split.n_dev,
                "N_lockbox": split.n_lockbox,
                "lockbox_rule": "last floor(0.15*N) rows after stable sort by (timestamp, raw_row_id)",
                "outer_fold_plan_used": None if plan is None else plan.plan_name,
                "lane_b_feasible": feasible,
                "lane_b_infeasible_reason": reason,
            }
        ]
    )
    write_csv(split_meta, reports / "split_metadata.csv")

    commit, dirty = git_commit_and_dirty(project_root)
    manifest = {
        "manifest_version": "1.0",
        "strategy_doc_path": "docs/final_end_to_end_report_strategy_merged.md",
        "strategy_doc_sha256": strategy_sha256(project_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
        "library_versions": library_versions(),
        "seed_policy": {
            "lane_a": [42],
            "stage_b": SEEDS_STAGE_B,
            "phase_2": SEEDS_PHASE2,
            "phase_3": [SEED_PHASE3],
        },
        "dev_lockbox_split": {
            "N_total_after_NaT_drop": split.n_total_after_nat_drop,
            "N_dev": split.n_dev,
            "N_lockbox": split.n_lockbox,
            "lockbox_rule": "last floor(0.15*N) rows after stable sort by (timestamp, raw_row_id)",
        },
        "outer_fold_plan_used": None if plan is None else plan.plan_name,
        "outer_fold_week_ranges": [] if plan is None else fold_plan_manifest_ranges(plan),
        "lane_b_feasible": feasible,
        "lane_b_infeasible_reason": reason,
        "challenger_available": None,
        "challenger_unavailable_reason": None,
        "frozen_primary": None,
        "frozen_challenger": None,
        "frozen_thresholds": None,
        "drift_gate_results": None,
        "empirical_ARL0_nan_reason": None,
    }
    write_manifest(manifest, reports / ArtifactName.MANIFEST)
    write_csv(dev_weeks, reports / "dev_sorted_with_weeks.csv")
    write_csv(split.lockbox, reports / "lockbox_sorted.csv")

    return DataBundle(
        all_data=all_sorted,
        dev=split.dev,
        lockbox=split.lockbox,
        feature_columns=loaded.feature_columns,
        dev_with_weeks=dev_weeks,
        fold_plan=plan,
        lane_b_feasible=feasible,
        lane_b_infeasible_reason=reason,
    )

