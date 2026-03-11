from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import ScalerName, SelectorName
from secom.selection.tuning import select_best_inner_config
from secom.workflows.lane_b import run_lane_b_stage_ab
from secom.workflows.split_contract import run_split_contract


def test_stage_b_inner_selection_tiebreak_chain() -> None:
    rows = [
        {
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.81,
            "mean_inner_BER": 0.30,
        },
        {
            "k": 20,
            "C": 1.0,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.81,
            "mean_inner_BER": 0.30,
        },
        {
            "k": 10,
            "C": 0.1,
            "scaler": ScalerName.STANDARD,
            "n_neighbors": None,
            "mean_inner_ROC_AUC": 0.805,
            "mean_inner_BER": 0.29,
        },
    ]
    best = select_best_inner_config(rows)
    # Within 0.01 AUC window, lower BER row should win first.
    assert best["k"] == 10
    assert np.isclose(best["C"], 0.1)


def test_stage_b_emits_splitwise_and_model_selection_contract(
    synthetic_input_dir,
    workspace_tmp_dir,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_stage_b"
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)

    import secom.workflows.lane_b as lane_b

    monkeypatch.setattr(lane_b, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(SelectorName, "STAGE_B", [SelectorName.S2N, SelectorName.F_TEST], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": ScalerName.STANDARD,
                "n_neighbors": None,
            }
        ]

    monkeypatch.setattr(lane_b, "build_stage_b_config_grid", _small_grid, raising=False)

    stage3 = run_lane_b_stage_ab(bundle=bundle, output_dir=out_dir)

    splitwise = pd.read_csv(out_dir / "reports" / "splitwise_timeaware_results.csv")
    stage_b_inner = pd.read_csv(out_dir / "reports" / "stage_b_inner_cv_results.csv")
    model_selection = pd.read_csv(out_dir / "reports" / "timeaware_model_selection.csv")

    assert stage3["lane_b_feasible"] is True

    required_splitwise_cols = {
        "selector",
        "outer_fold",
        "seed",
        "train_window",
        "test_window",
        "k",
        "C",
        "scaler",
        "n_neighbors",
        "threshold_policy",
        "outer_threshold",
        "n_test",
        "test_fails",
        "flagged_fraction",
        "BER",
        "True+",
        "True-",
    }
    assert required_splitwise_cols.issubset(splitwise.columns)
    assert splitwise["threshold_policy"].eq("outer_train_youden_ber_optimal").all()
    assert splitwise["flagged_fraction"].between(0.0, 1.0).all()
    assert (splitwise["n_test"] >= splitwise["test_fails"]).all()

    selected_counts = (
        stage_b_inner.groupby(["selector", "outer_fold", "seed"])["is_selected_config"].sum().astype(int)
    )
    assert (selected_counts == 1).all()

    assert {"selector", "n_folds", "n_seeds", "is_primary", "is_challenger"}.issubset(model_selection.columns)
    assert model_selection["is_primary"].sum() == 1


def test_stage_b_reuses_selector_pipeline_across_c_values(
    synthetic_input_dir,
    workspace_tmp_dir,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_stage_b_cache"
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)

    import secom.workflows.lane_b as lane_b

    monkeypatch.setattr(lane_b, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(SelectorName, "STAGE_B", [SelectorName.S2N], raising=False)
    monkeypatch.setattr(lane_b, "_stage_a_configs", lambda: [], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": c_value,
                "scaler": ScalerName.STANDARD,
                "n_neighbors": None,
            }
            for c_value in [0.01, 0.1, 1.0, 10.0]
        ]

    real_fit = lane_b.fit_selector_pipeline
    call_count = {"n": 0}

    def counting_fit(*args, **kwargs):
        call_count["n"] += 1
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(lane_b, "build_stage_b_config_grid", _small_grid, raising=False)
    monkeypatch.setattr(lane_b, "fit_selector_pipeline", counting_fit, raising=False)

    run_lane_b_stage_ab(bundle=bundle, output_dir=out_dir)

    expected_calls = len(bundle.fold_plan.folds) * 5 + len(bundle.fold_plan.folds)
    assert call_count["n"] == expected_calls
