from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secom.pipeline import run_01_data_contract_and_split, run_02_lane_a_replication


def test_lane_a_artifacts_and_pairing(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(bundle=bundle, output_dir=out_dir)

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    expected_classifiers = {"krr_strict", "krr_balanced", "logreg"}
    assert set(strict["classifier"]) == expected_classifiers
    assert set(mi["classifier"]) == expected_classifiers
    assert set(summary["classifier"]) == expected_classifiers
    assert set(ablation["classifier"]) == expected_classifiers

    assert len(strict) == 6 * 3 * 10
    assert len(mi) == 6 * 3 * 10
    assert len(ablation) == 6 * 3
    assert len(summary) == 6 * 3 * 2
    assert len(tuning) == 6 * 2 * 2 * 10
    assert set(tuning["classifier"]) == {"krr_balanced", "logreg"}
    assert set(tuning["selector_tuning_scope"].unique()) == {"outer_train_fixed"}

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
    run_02_lane_a_replication(bundle=bundle, output_dir=out_dir, lane_a_classifier="logreg")

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    assert len(strict) == 6 * 10
    assert len(mi) == 6 * 10
    assert len(ablation) == 6
    assert len(summary) == 6 * 2
    assert len(tuning) == 6 * 2 * 10
    assert set(summary["classifier"].unique()) == {"logreg"}
    assert set(tuning["classifier"].unique()) == {"logreg"}
    assert np.isfinite(summary["mean_BER"]).all()


def test_lane_a_krr_balanced_mode_runs(synthetic_input_dir, workspace_tmp_dir) -> None:
    out_dir = workspace_tmp_dir / "out_lane_a_krr_balanced"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle=bundle, output_dir=out_dir, lane_a_classifier="krr_balanced"
    )

    strict = pd.read_csv(out_dir / "reports" / "baseline_replication_strict.csv")
    mi = pd.read_csv(out_dir / "reports" / "baseline_replication_with_missing_indicators.csv")
    ablation = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")

    assert len(strict) == 6 * 10
    assert len(mi) == 6 * 10
    assert len(ablation) == 6
    assert len(summary) == 6 * 2
    assert len(tuning) == 6 * 2 * 10
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
        bundle=bundle, output_dir=out_dir, lane_a_classifier="krr_strict"
    )

    summary = pd.read_csv(out_dir / "reports" / "baseline_replication_summary.csv")
    tuning = pd.read_csv(out_dir / "reports" / "baseline_lane_a_tuning_trace.csv")
    assert len(summary) == 6 * 2
    assert set(summary["classifier"].unique()) == {"krr_strict"}
    assert len(tuning) == 0
