from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.config import SelectorName
from secom.pipeline import run_01_data_contract_and_split, run_02_lane_a_replication


FAST_SELECTORS = [SelectorName.S2N, SelectorName.F_TEST]


@pytest.fixture(autouse=True)
def _fast_lane_a_tuning(monkeypatch) -> None:
    # Keep tests focused on wiring/contract checks, not exhaustive sweep cost.
    import secom.pipeline as pipeline

    monkeypatch.setattr(pipeline, "LANE_A_KRR_BALANCED_ALPHA_GRID", [1.0], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_KRR_BALANCED_GAMMA_GRID", [None], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_LOGREG_C_GRID", [1.0], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_MRMR_LAMBDA_GRID", [1.0], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_MUTUAL_INFO_N_NEIGHBORS_GRID", [3], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_L1_SELECTOR_C_GRID", [1.0], raising=False)
    monkeypatch.setattr(pipeline, "LANE_A_KRR_BALANCED_INNER_SPLITS", 2, raising=False)


def test_lane_a_artifacts_and_pairing(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(bundle=bundle, output_dir=out_dir, selectors_run=FAST_SELECTORS)

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    expected_classifiers = {"krr_strict", "krr_balanced", "logreg"}
    n_selectors = len(FAST_SELECTORS)
    assert set(strict["classifier"]) == expected_classifiers
    assert set(mi["classifier"]) == expected_classifiers
    assert set(summary["classifier"]) == expected_classifiers
    assert set(ablation["classifier"]) == expected_classifiers

    assert len(strict) == n_selectors * 3 * 10
    assert len(mi) == n_selectors * 3 * 10
    assert len(ablation) == n_selectors * 3
    assert len(summary) == n_selectors * 3 * 2
    assert len(tuning) == n_selectors * 2 * 2 * 10
    assert set(tuning["classifier"]) == {"krr_balanced", "logreg"}
    assert set(tuning["selector_tuning_scope"].unique()) == {"outer_train_fixed"}
    assert "chosen_mrmr_lambda" in tuning.columns
    assert "chosen_mutual_info_n_neighbors" in tuning.columns
    assert "chosen_l1_selector_c" in tuning.columns
    mrmr_rows = tuning[tuning["selector"] == SelectorName.MRMR]
    mi_rows = tuning[tuning["selector"] == SelectorName.MUTUAL_INFO]
    l1_rows = tuning[tuning["selector"] == SelectorName.L1_LOGREG]
    if SelectorName.MRMR in FAST_SELECTORS:
        assert not mrmr_rows.empty
        assert np.isfinite(mrmr_rows["chosen_mrmr_lambda"]).all()
    if SelectorName.MUTUAL_INFO in FAST_SELECTORS:
        assert not mi_rows.empty
        assert np.isfinite(mi_rows["chosen_mutual_info_n_neighbors"]).all()
    if SelectorName.L1_LOGREG in FAST_SELECTORS:
        assert not l1_rows.empty
        assert np.isfinite(l1_rows["chosen_l1_selector_c"]).all()

    for selector in strict["selector"].unique():
        for classifier in expected_classifiers:
            s_folds = (
                strict.loc[
                    (strict["selector"] == selector) & (strict["classifier"] == classifier),
                    "fold",
                ]
                .sort_values()
                .tolist()
            )
            m_folds = (
                mi.loc[
                    (mi["selector"] == selector) & (mi["classifier"] == classifier),
                    "fold",
                ]
                .sort_values()
                .tolist()
            )
            assert s_folds == m_folds
            assert s_folds == list(range(1, 11))

    for _, row in ablation.iterrows():
        expected = float(row["BER_strict"]) - float(row["BER_MI"])
        assert np.isclose(float(row["delta_BER"]), expected, atol=1e-9)

    logreg_t = tuning[tuning["classifier"] == "logreg"]
    krrb_t = tuning[tuning["classifier"] == "krr_balanced"]
    assert np.isfinite(logreg_t["chosen_C"]).all()
    assert np.isfinite(krrb_t["chosen_alpha"]).all()
    assert np.isfinite(krrb_t["threshold"]).all()


def test_lane_a_notebook_classifier_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_logreg"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="logreg",
        selectors_run=FAST_SELECTORS,
    )

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    n_selectors = len(FAST_SELECTORS)
    assert len(strict) == n_selectors * 10
    assert len(mi) == n_selectors * 10
    assert len(ablation) == n_selectors
    assert len(summary) == n_selectors * 2
    assert len(tuning) == n_selectors * 2 * 10
    assert set(summary["classifier"].unique()) == {"logreg"}
    assert set(tuning["classifier"].unique()) == {"logreg"}
    assert np.isfinite(summary["mean_BER"]).all()


def test_lane_a_krr_balanced_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_krr_balanced"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="krr_balanced",
        selectors_run=FAST_SELECTORS,
    )

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    n_selectors = len(FAST_SELECTORS)
    assert len(strict) == n_selectors * 10
    assert len(mi) == n_selectors * 10
    assert len(ablation) == n_selectors
    assert len(summary) == n_selectors * 2
    assert len(tuning) == n_selectors * 2 * 10
    assert set(summary["classifier"].unique()) == {"krr_balanced"}
    assert set(tuning["classifier"].unique()) == {"krr_balanced"}
    assert set(summary["replication_mode"].unique()) == {"strict", "with_missing_indicators"}
    assert np.isfinite(summary["mean_BER"]).all()


def test_lane_a_krr_strict_mode_runs_with_empty_tuning_trace(
    synthetic_input_dir, workspace_tmp_dir
) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_krr_strict"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        lane_a_classifier="krr_strict",
        selectors_run=FAST_SELECTORS,
    )

    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")
    assert len(summary) == len(FAST_SELECTORS) * 2
    assert set(summary["classifier"].unique()) == {"krr_strict"}
    assert len(tuning) == 0


def test_lane_a_can_skip_relieff_for_faster_experiments(
    synthetic_input_dir, workspace_tmp_dir
) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_no_relieff"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    selectors = [s for s in FAST_SELECTORS if s != SelectorName.RELIEFF]
    run_02_lane_a_replication(
        bundle=bundle,
        output_dir=out_dir,
        selectors_run=selectors,
    )

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    assert SelectorName.RELIEFF not in set(strict["selector"].unique())
    assert SelectorName.RELIEFF not in set(mi["selector"].unique())
    assert SelectorName.RELIEFF not in set(ablation["selector"].unique())
    assert SelectorName.RELIEFF not in set(summary["selector"].unique())
    assert SelectorName.RELIEFF not in set(tuning["selector"].unique())

    n_selectors = len(selectors)
    assert len(strict) == n_selectors * 3 * 10
    assert len(mi) == n_selectors * 3 * 10
    assert len(ablation) == n_selectors * 3
    assert len(summary) == n_selectors * 3 * 2
    assert len(tuning) == n_selectors * 2 * 2 * 10
