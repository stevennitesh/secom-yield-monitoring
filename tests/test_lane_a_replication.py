from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.config import SelectorName
from secom.workflows.lane_a import run_lane_a_replication
from secom.workflows.split_contract import run_split_contract


FAST_SELECTORS = [SelectorName.S2N, SelectorName.F_TEST]


@pytest.fixture(autouse=True)
def _fast_lane_a_tuning(monkeypatch) -> None:
    # Keep tests focused on wiring/contract checks, not exhaustive sweep cost.
    import secom.selection.tuning as tuning
    import secom.workflows.lane_a as lane_a

    monkeypatch.setattr(tuning, "LANE_A_KRR_ALPHA_GRID", [1.0], raising=False)
    monkeypatch.setattr(tuning, "LANE_A_KRR_GAMMA_GRID", [None], raising=False)
    monkeypatch.setattr(tuning, "LANE_A_LOGREG_C_GRID", [1.0], raising=False)
    monkeypatch.setattr(lane_a, "LANE_A_KRR_ALPHA_GRID", [1.0], raising=False)
    monkeypatch.setattr(lane_a, "LANE_A_KRR_GAMMA_GRID", [None], raising=False)
    monkeypatch.setattr(lane_a, "LANE_A_LOGREG_C_GRID", [1.0], raising=False)


PARAM_COLS = [
    "alpha",
    "gamma",
    "C",
    "n_neighbors",
]


def _read_lane_a_global_artifacts(out_dir: Path) -> dict[str, pd.DataFrame]:
    reports = out_dir / "reports"
    return {
        "sweep": pd.read_csv(reports / "lane_a_global_sweep.csv"),
        "best": pd.read_csv(reports / "lane_a_global_best_config.csv"),
        "fold": pd.read_csv(reports / "lane_a_global_fold_metrics.csv"),
        "summary": pd.read_csv(reports / "lane_a_global_summary.csv"),
        "ablation": pd.read_csv(reports / "lane_a_global_ablation.csv"),
        "full": pd.read_csv(reports / "lane_a_global_full_fit_summary.csv"),
    }


def test_lane_a_global_artifacts_and_pairing(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)
    run_lane_a_replication(bundle=bundle, output_dir=out_dir, selectors_run=FAST_SELECTORS)

    artifacts = _read_lane_a_global_artifacts(out_dir)
    sweep = artifacts["sweep"]
    best = artifacts["best"]
    fold = artifacts["fold"]
    summary = artifacts["summary"]
    ablation = artifacts["ablation"]
    full = artifacts["full"]

    expected_classifiers = {"krr", "logreg"}
    n_selectors = len(FAST_SELECTORS)
    n_triplets = n_selectors * len(expected_classifiers) * 2

    assert set(best["classifier"].unique()) == expected_classifiers
    assert set(fold["classifier"].unique()) == expected_classifiers
    assert set(summary["classifier"].unique()) == expected_classifiers
    assert set(ablation["classifier"].unique()) == expected_classifiers
    assert set(full["classifier"].unique()) == expected_classifiers

    assert len(best) == n_triplets
    assert len(summary) == n_triplets
    assert len(full) == n_triplets
    assert len(fold) == n_triplets * 10
    assert len(ablation) == n_selectors * len(expected_classifiers)
    assert len(sweep) >= n_triplets

    assert set(summary["replication_mode"].unique()) == {"strict", "with_missing_indicators"}
    assert set(fold["replication_mode"].unique()) == {"strict", "with_missing_indicators"}
    assert set(full["replication_mode"].unique()) == {"strict", "with_missing_indicators"}

    # One best row and exactly 10 folds per (selector, classifier, mode)
    assert best.groupby(["selector", "classifier", "replication_mode"]).size().eq(1).all()
    assert fold.groupby(["selector", "classifier", "replication_mode"])["fold"].nunique().eq(10).all()
    assert set(fold["fold"].unique()) == set(range(1, 11))

    # Best rows must exist in sweep by config tuple.
    merge_cols = ["selector", "classifier", "replication_mode", *PARAM_COLS]
    best_key = best[merge_cols].copy()
    sweep_key = sweep[merge_cols].drop_duplicates().copy()
    for col in PARAM_COLS:
        best_key[col] = best_key[col].astype(object).where(best_key[col].notna(), "__NA__")
        sweep_key[col] = sweep_key[col].astype(object).where(sweep_key[col].notna(), "__NA__")
    merged = best_key.merge(sweep_key, on=merge_cols, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()

    for _, row in ablation.iterrows():
        expected = float(row["BER_strict"]) - float(row["BER_MI"])
        assert np.isclose(float(row["delta_BER"]), expected, atol=1e-9)

    assert set(full["threshold_full_dataset_role"].unique()) == {"diagnostic_only"}
    assert np.isfinite(best["threshold_oof_global"]).all()
    assert np.isfinite(full["threshold_full_dataset"]).all()


def test_lane_a_logreg_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_logreg"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)
    run_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="logreg",
        selectors_run=FAST_SELECTORS,
    )

    artifacts = _read_lane_a_global_artifacts(out_dir)
    best = artifacts["best"]
    summary = artifacts["summary"]
    fold = artifacts["fold"]

    n_selectors = len(FAST_SELECTORS)
    assert len(best) == n_selectors * 2
    assert len(summary) == n_selectors * 2
    assert len(fold) == n_selectors * 2 * 10
    assert set(best["classifier"].unique()) == {"logreg"}
    assert set(summary["classifier"].unique()) == {"logreg"}


def test_lane_a_krr_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_krr"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)
    run_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="krr",
        selectors_run=FAST_SELECTORS,
    )

    artifacts = _read_lane_a_global_artifacts(out_dir)
    best = artifacts["best"]
    summary = artifacts["summary"]
    fold = artifacts["fold"]

    n_selectors = len(FAST_SELECTORS)
    assert len(best) == n_selectors * 2
    assert len(summary) == n_selectors * 2
    assert len(fold) == n_selectors * 2 * 10
    assert set(best["classifier"].unique()) == {"krr"}
    assert set(summary["classifier"].unique()) == {"krr"}


def test_lane_a_krr_strict_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_krr_strict"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)
    run_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="krr_strict",
        selectors_run=FAST_SELECTORS,
    )

    artifacts = _read_lane_a_global_artifacts(out_dir)
    best = artifacts["best"]
    summary = artifacts["summary"]
    assert len(best) == len(FAST_SELECTORS) * 2
    assert len(summary) == len(FAST_SELECTORS) * 2
    assert set(best["classifier"].unique()) == {"krr_strict"}
    assert set(summary["classifier"].unique()) == {"krr_strict"}


def test_lane_a_can_skip_relieff_for_faster_experiments(
    synthetic_input_dir, workspace_tmp_dir
) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_no_relieff"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)
    selectors = [s for s in FAST_SELECTORS if s != SelectorName.RELIEFF]
    run_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        selectors_run=selectors,
    )

    artifacts = _read_lane_a_global_artifacts(out_dir)
    for df in artifacts.values():
        assert SelectorName.RELIEFF not in set(df["selector"].unique())
