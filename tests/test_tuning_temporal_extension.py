"""Scientific separation and recomputable receipts for the bounded extension."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from secom.artifacts import read_manifest, write_csv, write_manifest
from secom.config import ArtifactName
from secom.metrics import find_ber_optimal_threshold
from secom.workflows import benchmark_common as common
from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_tuned import _select_best_tuned_config
from secom.workflows.benchmark_procedures import selected_candidate
from secom.workflows.calibration import calibration_diagnostics
from secom.workflows.temporal_krr import run_krr_comparator


def test_none_and_effective_gamma_share_one_fit_with_identical_scores_and_threshold(monkeypatch):
    common.reset_model_score_cache()
    actual = common.fit_benchmark_krr_model
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return actual(*args, **kwargs)

    monkeypatch.setattr(common, "fit_benchmark_krr_model", counted)
    x = np.random.default_rng(3).normal(size=(30, 10))
    y = np.tile([0, 1], 15)
    reference_default = actual(x, y, alpha=1, gamma=None).predict(x[:10])
    reference_numeric = actual(x, y, alpha=1, gamma=0.1).predict(x[:10])
    np.testing.assert_array_equal(reference_default, reference_numeric)
    assert find_ber_optimal_threshold(y[:10], reference_default) == find_ber_optimal_threshold(
        y[:10], reference_numeric
    )
    first = common.fit_classifier_scores("krr", x, y, x[:10], {"alpha": 1, "gamma": None})
    second = common.fit_classifier_scores("krr", x, y, x[:10], {"alpha": 1, "gamma": 0.1})
    assert len(calls) == 1
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert find_ber_optimal_threshold(y[:10], first[1]) == find_ber_optimal_threshold(y[:10], second[1])
    common.fit_classifier_scores("krr", x[:, :5], y, x[:10, :5], {"alpha": 1, "gamma": None})
    common.fit_classifier_scores("krr", x, y, x[:10], {"alpha": 100, "gamma": 0.1})
    assert len(calls) == 3


def test_infeasible_comparator_paths_keep_fixed_periods_and_explicit_unavailability():
    data, fold = _case()
    data.loc[84:119, "y_bin"] = 0
    search, predictions, metrics, calibration, diagnostics = run_krr_comparator(
        data=data, x_raw=data.attrs["raw_features"], folds=[fold], selectors=["F-test"]
    )
    assert search.empty and predictions.empty and not calibration and not diagnostics
    assert len(metrics) == 6 and not metrics.available.any()
    assert set(metrics.fold) == {1}
    assert set(metrics.unavailable_reason) == {"insufficient_classes_in_fit_or_calibration"}


def test_tuned_grid_and_actual_width_resolution_preserve_original_reference():
    original = common.classifier_param_grid("krr")
    tuned = common.tuned_classifier_param_grid("krr")
    assert len(original) == 12 and len(tuned) == 16
    assert {cfg["alpha"] for cfg in original} == {0.1, 1, 10}
    assert {cfg["alpha"] for cfg in tuned} == {0.1, 1, 10, 100}
    assert {cfg["gamma_multiplier"] for cfg in tuned} == {0.1, 0.2, 1, 2}
    cfg = {"alpha": 100, "gamma_multiplier": 0.2}
    assert common.effective_classifier_config(cfg, 10)["gamma"] == 0.02
    assert common.effective_classifier_config(cfg, 7)["gamma"] == 0.2 / 7
    assert cfg == {"alpha": 100, "gamma_multiplier": 0.2}
    assert common.tuned_classifier_param_grid("logreg") == common.classifier_param_grid("logreg")


def test_multiplier_lineage_treats_different_inner_outer_full_widths_as_same_candidate(workspace_tmp_dir):
    from secom.artifacts import _validate_selector_config_lineage

    config = {
        "selector": "F-test",
        "classifier": "krr",
        "replication_mode": "with_missing_indicators",
        "k": 40,
        "C": np.nan,
        "alpha": 100,
        "gamma_multiplier": 1,
        "n_neighbors": np.nan,
    }
    frames = {
        ArtifactName.BENCHMARK_TUNED_SEARCH: pd.DataFrame(
            [{**config, "fold": 1, "gamma": 1 / 10, "is_selected_config": True}]
        ),
        ArtifactName.BENCHMARK_TUNED_FOLD_METRICS: pd.DataFrame([{**config, "fold": 1, "gamma": 1 / 12}]),
        ArtifactName.BENCHMARK_TUNED_BEST_CONFIG: pd.DataFrame([{**config, "gamma": 1 / 14}]),
        ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY: pd.DataFrame([{**config, "gamma": 1 / 14}]),
    }
    errors = []
    _validate_selector_config_lineage(
        artifact_frames=frames, reports=workspace_tmp_dir, active_original=False, active_tuned=True, errors=errors
    )
    assert not errors
    frames[ArtifactName.BENCHMARK_TUNED_FOLD_METRICS].loc[0, "gamma_multiplier"] = 2
    _validate_selector_config_lineage(
        artifact_frames=frames, reports=workspace_tmp_dir, active_original=False, active_tuned=True, errors=errors
    )
    assert errors


def test_family_and_joint_exact_ties_prefer_budget_then_stronger_regularization():
    rows = [
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": k,
            "alpha": alpha,
            "gamma_multiplier": 1,
            "gamma": 1 / k,
            "C": np.nan,
            "n_neighbors": None,
            "mean_inner_BER": 0.25,
        }
        for k, alpha in ((20, 100), (10, 0.1), (10, 100))
    ]
    assert _select_best_tuned_config(rows) is rows[2]
    assert selected_candidate([(row, {}, np.array([])) for row in rows])[0] is rows[2]
    # Strictly better BER takes precedence over simplicity, even if the gain is small.
    rows[0]["mean_inner_BER"] -= 1e-8
    assert _select_best_tuned_config(rows) is rows[0]


def test_cross_family_exact_tie_has_fixed_shape_and_stable_order():
    common_cfg = {
        "classifier": "krr",
        "replication_mode": "strict",
        "k": 10,
        "alpha": 100,
        "gamma_multiplier": 1,
        "gamma": 0.1,
        "C": np.nan,
        "mean_inner_BER": 0.25,
    }
    rows = [
        {**common_cfg, "selector": "ReliefF", "n_neighbors": 5},
        {**common_cfg, "selector": "F-test", "n_neighbors": None},
    ]
    winner = selected_candidate([(row, {}, np.array([])) for row in rows])[0]
    assert winner in rows
    assert _select_best_tuned_config(rows) is winner


def test_calibration_low_counts_and_leave_one_failure_out_guards():
    row = calibration_diagnostics([0, 0, 1], [0.1, 0.3, 0.8], 0.8)
    assert row["fragile_calibration"] and not row["lofo_available"]
    assert row["BER_step_failure"] == 0.5 and row["BER_step_pass"] == 0.25
    assert np.isnan(row["lofo_threshold_min"])
    absent = calibration_diagnostics([0, 0], [0.1, 0.3], float("inf"))
    assert np.isnan(absent["BER_step_failure"]) and not absent["lofo_available"]
    row = calibration_diagnostics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.8], 0.4)
    assert row["lofo_available"] and row["lofo_threshold_min"] <= row["lofo_threshold_max"]
    assert row["lofo_flagged_fraction_min"] <= row["lofo_flagged_fraction_max"]
    assert row["diagnostic_semantics"] == "fixed_model_calibration_instability_only_not_CI"


def _case():
    data = pd.DataFrame(
        {
            "raw_row_id": np.arange(160),
            "y_bin": np.tile([0, 1], 80),
            "timestamp": pd.Timestamp("2008-01-01") + pd.to_timedelta(np.arange(160), unit="h"),
        }
    )
    data.attrs["raw_features"] = np.random.default_rng(73).normal(size=(160, 12))
    fold = SimpleNamespace(outer_fold=1, train_index=np.arange(120), test_index=np.arange(120, 160))
    return data, fold


def test_comparator_future_labels_do_not_select_configs_or_thresholds():
    data, fold = _case()
    first = run_krr_comparator(data=data, x_raw=data.attrs["raw_features"], folds=[fold], selectors=["F-test"])
    changed = data.copy()
    changed.loc[fold.test_index, "y_bin"] = 1 - changed.loc[fold.test_index, "y_bin"]
    second = run_krr_comparator(data=changed, x_raw=changed.attrs["raw_features"], folds=[fold], selectors=["F-test"])
    pd.testing.assert_frame_equal(first[0], second[0])
    columns = ["procedure", "sample_id", "config_id", "score", "threshold", "prediction"]
    pd.testing.assert_frame_equal(first[1][columns], second[1][columns])
    assert not np.array_equal(first[1].y_true, second[1].y_true)
    for procedure in first[1].procedure.unique():
        assert set(first[1].loc[first[1].procedure == procedure, "sample_id"]) == set(fold.test_index)


def test_thirty_percent_calibration_never_enters_config_selection_or_model_fit():
    data, fold = _case()
    first = run_krr_comparator(data=data, x_raw=data.attrs["raw_features"], folds=[fold], selectors=["F-test"])
    changed = data.copy()
    changed.loc[84:119, "y_bin"] = 1 - changed.loc[84:119, "y_bin"]
    second = run_krr_comparator(data=changed, x_raw=changed.attrs["raw_features"], folds=[fold], selectors=["F-test"])
    pd.testing.assert_frame_equal(
        first[0][first[0].calibration_fraction == 0.3].reset_index(drop=True),
        second[0][second[0].calibration_fraction == 0.3].reset_index(drop=True),
    )
    columns = ["procedure", "sample_id", "config_id", "score"]
    a, b = (item[1][item[1].calibration_fraction == 0.3][columns].reset_index(drop=True) for item in (first, second))
    pd.testing.assert_frame_equal(a, b)
    for row in first[0][first[0].calibration_fraction == 0.3].to_dict("records"):
        for column in ("fit_sample_ids", "calibration_sample_ids", "validation_sample_ids"):
            assert max(int(v) for v in row[column].split(",")) < 84
    for diagnostic in first[4]:
        if diagnostic["calibration_fraction"] == 0.3:
            assert max(int(v) for v in diagnostic["fit_sample_ids"].split(",")) < 84
            assert diagnostic["fit_n"] == 84 and diagnostic["calibration_n"] == 36


@pytest.mark.parametrize(
    "artifact",
    [
        ArtifactName.TEMPORAL_KRR_SEARCH,
        ArtifactName.TEMPORAL_KRR_PREDICTIONS,
        ArtifactName.TEMPORAL_KRR_METRICS,
        ArtifactName.TEMPORAL_CALIBRATION_SCORES,
        ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS,
    ],
)
def test_missing_new_artifact_is_hard_error(active_artifacts_output_dir, artifact):
    (active_artifacts_output_dir / "reports" / artifact).unlink()
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok and any(artifact in error for error in audit.errors)


@pytest.mark.parametrize(
    "artifact,column",
    [
        (ArtifactName.TEMPORAL_KRR_METRICS, "TP"),
        (ArtifactName.TEMPORAL_KRR_SEARCH, "mean_inner_BER"),
        (ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS, "calibration_fails"),
    ],
)
def test_recompute_detects_tamper_without_relying_on_hashes(active_artifacts_output_dir, artifact, column):
    path = active_artifacts_output_dir / "reports" / artifact
    frame = pd.read_csv(path)
    frame.loc[0, column] += 1
    write_csv(frame, path)
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok and any("comparator/calibration" in error for error in audit.errors)


def test_comparator_schema_and_hash_tamper_are_errors(active_artifacts_output_dir):
    reports = active_artifacts_output_dir / "reports"
    path = reports / ArtifactName.TEMPORAL_KRR_SEARCH
    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    import hashlib

    manifest["artifact_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    write_manifest(manifest, reports / ArtifactName.MANIFEST)
    write_csv(pd.read_csv(path).drop(columns="inner_BER"), path)
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok
    assert any("schema" in error for error in audit.errors)
    assert any("artifact hash mismatch" in error for error in audit.errors)


def test_header_only_unavailable_comparator_warns_without_invalidating_benchmarks(active_artifacts_output_dir):
    reports = active_artifacts_output_dir / "reports"
    for name in (ArtifactName.TEMPORAL_KRR_SEARCH, ArtifactName.TEMPORAL_KRR_PREDICTIONS):
        path = reports / name
        write_csv(pd.read_csv(path).iloc[:0], path)
    path = reports / ArtifactName.TEMPORAL_KRR_METRICS
    metrics = pd.read_csv(path)
    metrics["available"] = False
    metrics["unavailable_reason"] = "insufficient_classes_in_fit_or_calibration"
    write_csv(metrics, path)
    for name in (ArtifactName.TEMPORAL_CALIBRATION_SCORES, ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS):
        path = reports / name
        frame = pd.read_csv(path)
        write_csv(frame[~frame.procedure.str.startswith("krr_")], path)
    audit = run_study_audit(active_artifacts_output_dir)
    assert audit.ok, audit.errors
    assert any("unavailable" in warning for warning in audit.warnings)


def test_equal_inner_boundary_timestamps_preserve_disjoint_fixed_rows(active_artifacts_output_dir):
    path = active_artifacts_output_dir / "reports" / ArtifactName.TEMPORAL_KRR_SEARCH
    search = pd.read_csv(path)
    search.loc[0, "validation_start_timestamp"] = search.loc[0, "calibration_end_timestamp"]
    search.loc[0, "inner_timestamp_tie"] = True
    metrics_path = active_artifacts_output_dir / "reports" / ArtifactName.TEMPORAL_KRR_METRICS
    metrics = pd.read_csv(metrics_path)
    selected = search.iloc[0]
    metrics.loc[
        (metrics.fold == selected.fold)
        & (metrics.calibration_fraction == selected.calibration_fraction)
        & (metrics.candidate_id == selected.candidate_id),
        "n_inner_timestamp_ties",
    ] += 1
    write_csv(metrics, metrics_path)
    write_csv(search, path)
    audit = run_study_audit(active_artifacts_output_dir)
    assert audit.ok, audit.errors


@pytest.mark.parametrize("tamper", ["fit_id", "calibration_id", "chronology", "prediction_boundary"])
def test_lr_calibration_receipts_reject_evaluation_contamination_without_hash_failure(
    active_artifacts_output_dir, tamper
):
    """Consistent receipt tampering must still fail the scientific separation audit."""
    reports = active_artifacts_output_dir / "reports"
    honest = run_study_audit(active_artifacts_output_dir)
    assert honest.ok, honest.errors
    paths = {
        name: reports / name
        for name in (ArtifactName.TEMPORAL_CALIBRATION_SCORES, ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS)
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    evaluation = pd.read_csv(reports / ArtifactName.TEMPORAL_PREDICTIONS)
    fold = int(evaluation.fold.iloc[0])
    held_out = evaluation[evaluation.fold == fold]
    masks = {name: (frame.fold == fold) & (frame.procedure == "temporal_joint") for name, frame in frames.items()}
    if tamper == "fit_id":
        for name, frame in frames.items():
            ids = str(frame.loc[masks[name], "fit_sample_ids"].iloc[0]).split(",")
            ids[-1] = str(int(held_out.sample_id.iloc[0]))
            frame.loc[masks[name], "fit_sample_ids"] = ",".join(ids)
    elif tamper == "calibration_id":
        scores = frames[ArtifactName.TEMPORAL_CALIBRATION_SCORES]
        index = scores.index[masks[ArtifactName.TEMPORAL_CALIBRATION_SCORES]][0]
        scores.loc[index, "sample_id"] = held_out.sample_id.iloc[0]
    else:
        key = "calibration_end_timestamp" if tamper == "chronology" else "fit_end_timestamp"
        value = held_out.timestamp.min() if tamper == "chronology" else "2007-12-31 23:00:00"
        for name, frame in frames.items():
            frame.loc[masks[name], key] = value
    for name, frame in frames.items():
        write_csv(frame, paths[name])
    # Fixtures have no artifact hashes: rejection must come from ID/boundary
    # semantics, including when FIT receipt strings agree in both artifacts.
    result = run_study_audit(active_artifacts_output_dir)
    assert not result.ok
    assert not any("artifact hash mismatch" in error for error in result.errors)
    expected = (
        "FIT/calibration IDs overlap evaluation"
        if tamper in {"fit_id", "calibration_id"}
        else "FIT/calibration chronology overlaps evaluation"
        if tamper == "chronology"
        else "prediction boundary receipt mismatch"
    )
    assert any(f"LR outer calibration: {expected}" in error for error in result.errors), result.errors


@pytest.mark.parametrize(
    "artifact,column",
    [
        (ArtifactName.TEMPORAL_PREDICTIONS, "timestamp"),
        (ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS, "fit_end_timestamp"),
    ],
)
def test_lr_chronology_missing_schema_returns_hard_audit_error(active_artifacts_output_dir, artifact, column):
    """Chronology validation must reject missing schema without crashing the auditor."""
    path = active_artifacts_output_dir / "reports" / artifact
    write_csv(pd.read_csv(path).drop(columns=column), path)
    result = run_study_audit(active_artifacts_output_dir)
    assert not result.ok
    assert any(column in error for error in result.errors), result.errors
