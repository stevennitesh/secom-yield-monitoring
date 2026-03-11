from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from secom.artifacts import validate_schema_and_logic
from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold
from secom.config import SelectorName
from secom.workflows.freeze_lockbox import _manager_weekly_metrics, _safe_value_corrcoef
from secom.workflows.freeze_lockbox import run_freeze_lockbox
from secom.workflows.lane_b import run_lane_b_stage_ab
from secom.workflows.split_contract import run_split_contract
from secom.types import FittedRoleModel, RoleConfig


def test_operational_threshold_enforces_cap() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.05, 0.3, 0.4])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    weeks = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    t = operational_threshold(scores=scores, y_true=y, week_labels=weeks)
    preds = (scores >= t).astype(int)
    week_means = [preds[weeks == w].mean() for w in sorted(np.unique(weeks))]
    assert np.mean(week_means) <= 0.10 + 1e-9


def test_psi_feature_handles_missing_and_out_of_range() -> None:
    dev = np.array([1.0, 1.2, 1.1, np.nan, 0.9, 1.05])
    lock = np.array([3.0, 3.1, np.nan, 2.9, 3.2, 3.3])
    psi = psi_for_feature(dev, lock)
    assert np.isfinite(psi)
    assert psi >= 0.0


def test_manager_weekly_metrics_aggregates_weekly_counts() -> None:
    y = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.2, 0.8, 0.7, 0.4, 0.3], dtype=float)
    weeks = np.array([1, 1, 2, 2, 3, 3], dtype=int)

    metrics = _manager_weekly_metrics(
        y_true=y,
        scores=scores,
        threshold=0.5,
        week_labels=weeks,
    )

    assert metrics["dev_sample_count"] == 6.0
    assert metrics["dev_week_count"] == 3.0
    assert metrics["weekly_rate"] == 2.0
    assert np.isclose(metrics["predicted_flag_fraction"], 0.5)
    assert np.isclose(metrics["mean_weekly_flagged_wafers"], 1.0)
    assert np.isclose(metrics["mean_weekly_fail_captures"], 2.0 / 3.0)
    assert np.isclose(metrics["mean_weekly_fail_misses"], 1.0 / 3.0)


def test_safe_value_corrcoef_skips_constant_columns_without_runtime_warning() -> None:
    x = np.array(
        [
            [1.0, 0.1, 2.0],
            [1.0, 0.2, 1.5],
            [1.0, 0.3, 1.0],
            [1.0, 0.4, 0.5],
        ],
        dtype=float,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        corr = _safe_value_corrcoef(x)

    assert caught == []
    assert corr.shape == (3, 3)
    assert np.isnan(corr[0, 1])
    assert np.isnan(corr[0, 2])
    assert np.isfinite(corr[1, 2])


def test_phase2_freeze_reuses_selector_pipeline_across_c_values(monkeypatch) -> None:
    import secom.workflows.freeze_lockbox as freeze_lockbox

    monkeypatch.setattr(freeze_lockbox, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 4,
                "C": 0.1,
                "scaler": "StandardScaler",
                "n_neighbors": None,
            },
            {
                "selector": selector,
                "k": 4,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": None,
            },
        ]

    monkeypatch.setattr(freeze_lockbox, "build_stage_b_config_grid", _small_grid, raising=False)

    call_count = 0
    real_fit_selector_pipeline = freeze_lockbox.fit_selector_pipeline

    def _counted_fit_selector_pipeline(**kwargs):
        nonlocal call_count
        call_count += 1
        return real_fit_selector_pipeline(**kwargs)

    monkeypatch.setattr(
        freeze_lockbox,
        "fit_selector_pipeline",
        _counted_fit_selector_pipeline,
        raising=False,
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20, dtype=int)

    freeze_df, role_cfg = freeze_lockbox._phase2_freeze_for_role(
        role="primary",
        selector=SelectorName.S2N,
        x_dev=x,
        y_dev=y,
    )

    assert call_count == 5
    assert len(freeze_df) == 10
    assert set(freeze_df["C"]) == {0.1, 1.0}
    assert role_cfg.c_value in {0.1, 1.0}


def test_lockbox_eval_context_reuses_scoring_path() -> None:
    import secom.workflows.freeze_lockbox as freeze_lockbox

    class _CountingTransform:
        def __init__(self) -> None:
            self.calls = 0

        def transform(self, x):
            self.calls += 1
            return np.asarray(x, dtype=float)

    class _CountingModel:
        def __init__(self) -> None:
            self.calls = 0
            self.coef_ = np.array([[1.0]], dtype=float)

        def predict_proba(self, x):
            self.calls += 1
            p = np.clip(np.asarray(x, dtype=float)[:, 0], 0.0, 1.0)
            return np.column_stack([1.0 - p, p])

    model = FittedRoleModel(
        config=RoleConfig(
            role="primary",
            selector=SelectorName.S2N,
            k=4,
            c_value=1.0,
            scaler="StandardScaler",
            n_neighbors=None,
        ),
        imputer=_CountingTransform(),
        scaler=_CountingTransform(),
        selected_local_idx=np.array([0], dtype=int),
        selected_global_idx=[0],
        clf=_CountingModel(),
        dev_scores=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
        scientific_threshold=0.5,
        operational_threshold=0.4,
        threshold_at_tnr90_dev=0.5,
        tnr_at_tnr90_dev=0.9,
        tpr_at_tnr90_dev=0.8,
        feature_meta=[
            SimpleNamespace(
                feature_type="value",
                raw_index=0,
                feature_index=0,
                feature_name_or_source_col="x0",
            )
        ],
    )

    x_dev = np.array([[0.0], [1.0], [0.2], [0.7]], dtype=float)
    y_dev = np.array([0, 1, 0, 1], dtype=int)
    x_lock = np.array([[0.1], [0.9], [0.3], [0.8]], dtype=float)
    y_lock = np.array([0, 1, 0, 1], dtype=int)

    lock_ctx = freeze_lockbox._prepare_lockbox_eval_context(
        model=model,
        x_lock_raw=x_lock,
        y_lock=y_lock,
    )
    lock_df = freeze_lockbox._score_lockbox_for_role(
        model=model,
        y_lock=y_lock,
        lock_ctx=lock_ctx,
    )
    drift = freeze_lockbox._drift_gate_for_role(
        model=model,
        x_dev_raw=x_dev,
        y_dev=y_dev,
        x_lock_raw=x_lock,
        y_lock=y_lock,
        lock_ctx=lock_ctx,
    )

    assert model.imputer.calls == 1
    assert model.scaler.calls == 1
    assert model.clf.calls == 1
    assert len(lock_df) == 2
    assert drift["psi_feature_count"] == 1


def test_freeze_lockbox_emits_manager_outputs_and_passes_audit(
    synthetic_input_dir,
    workspace_tmp_dir,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_freeze_lockbox"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_split_contract(synthetic_input_dir, out_dir, project_root)

    import secom.workflows.freeze_lockbox as freeze_lockbox
    import secom.workflows.lane_b as lane_b

    monkeypatch.setattr(lane_b, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(freeze_lockbox, "SEEDS_PHASE2", [42], raising=False)
    monkeypatch.setattr(SelectorName, "ACTIVE", [SelectorName.S2N, SelectorName.F_TEST], raising=False)
    monkeypatch.setattr(SelectorName, "STAGE_B", [SelectorName.S2N, SelectorName.F_TEST], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": None,
            }
        ]

    monkeypatch.setattr(lane_b, "build_stage_b_config_grid", _small_grid, raising=False)
    monkeypatch.setattr(freeze_lockbox, "build_stage_b_config_grid", _small_grid, raising=False)

    stage3 = run_lane_b_stage_ab(bundle=bundle, output_dir=out_dir)
    assert stage3["lane_b_feasible"] is True

    freeze_result = run_freeze_lockbox(bundle=bundle, stage3=stage3, output_dir=out_dir)
    assert freeze_result["lane_b_feasible"] is True

    manager = pd.read_csv(out_dir / "reports" / "manager_facing_outputs.csv")
    assert {
        "role",
        "selector",
        "threshold_policy",
        "dev_sample_count",
        "dev_week_count",
        "weekly_rate",
        "predicted_flag_fraction",
        "mean_weekly_flagged_wafers",
        "mean_weekly_fail_captures",
        "mean_weekly_fail_misses",
        "stage_b_mean_flagged_fraction",
        "lockbox_flagged_fraction",
    }.issubset(manager.columns)
    assert {"scientific", "operational"}.issubset(set(manager["threshold_policy"]))
    assert manager["predicted_flag_fraction"].between(0.0, 1.0).all()
    assert manager["stage_b_mean_flagged_fraction"].between(0.0, 1.0).all()
    assert manager["lockbox_flagged_fraction"].between(0.0, 1.0).all()
    assert (manager["dev_week_count"] > 0).all()
    assert np.allclose(
        manager["weekly_rate"].to_numpy(dtype=float),
        manager["dev_sample_count"].to_numpy(dtype=float)
        / manager["dev_week_count"].to_numpy(dtype=float),
    )

    schema = validate_schema_and_logic(out_dir)
    assert schema.ok, schema.errors
