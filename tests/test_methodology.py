"""Scientific separation regressions for the corrected study, using actual fits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from secom.cv import add_dev_week_bins, choose_outer_fold_plan, chronological_inner_splits, fit_calibration_indices
from secom.types import RoleConfig
from secom.workflows import benchmark_tuned as benchmark, temporal_robustness as temporal
from secom.workflows.benchmark_procedures import missingness_baseline, selected_candidate
from secom.workflows.audit import run_study_audit
from secom.config import ArtifactName


def _chronological_data(n=240):
    rng = np.random.default_rng(81)
    y = np.tile([0, 0, 1, 0, 1], n // 5)
    x = rng.normal(size=(n, 6))
    x[:, 0] += y
    x[::7, 2] = np.nan
    return x, y


def test_inner_benchmark_threshold_uses_held_out_scores_only(monkeypatch):
    views = [
        benchmark._InnerSelectorView(np.ones((4, 1)), np.array([0, 1, 0, 1]), np.ones((2, 1)), np.array([0, 1]))
        for _ in range(3)
    ]
    calls = []

    def score(**kwargs):
        calls.append(kwargs)
        return np.array([100, -100, 100, -100]), np.array([0.2, 0.8])

    monkeypatch.setattr(benchmark, "fit_classifier_scores", score)
    selected = benchmark._inner_cv_summary_for_config(
        classifier="logreg", classifier_config={"C": 1}, prepared_inner_views=views
    )
    assert selected["threshold_inner_oof"] == 0.8
    assert selected["mean_inner_BER"] == 0
    assert all(call["include_train_scores"] is False for call in calls)


def test_inner_selector_fits_only_inner_train_and_metadata_ignores_eval_missingness(monkeypatch):
    x, y = _chronological_data(60)
    calls = []
    original = benchmark.fit_selector_pipeline

    def record(**kwargs):
        calls.append((kwargs["x_train_raw"].copy(), kwargs["x_eval_raw"].copy()))
        return original(**kwargs)

    monkeypatch.setattr(benchmark, "fit_selector_pipeline", record)
    benchmark._prepare_inner_selector_views(
        x_outer_train_raw=x,
        y_outer_train=y,
        selector="F-test",
        add_indicator=True,
        selector_config={"k": 3, "n_neighbors": None},
    )
    assert len(calls) == 3
    assert all(len(train) == 40 and len(valid) == 20 for train, valid in calls)
    assert all(not set(train[:, 1]).intersection(valid[:, 1]) for train, valid in calls)


def test_joint_winner_is_inner_selected_even_when_outer_scores_favor_other_family():
    common = {
        "classifier": "logreg",
        "replication_mode": "strict",
        "k": 10,
        "C": 1,
        "alpha": np.nan,
        "gamma": np.nan,
        "n_neighbors": np.nan,
        "threshold_inner_oof": 0.5,
    }
    candidates = [
        ({**common, "selector": "S2N", "mean_inner_BER": 0.1}, {"BER": 0.9}, np.array([0.9])),
        ({**common, "selector": "F-test", "mean_inner_BER": 0.4}, {"BER": 0.0}, np.array([0.1])),
    ]
    assert selected_candidate(candidates)[0]["selector"] == "S2N"
    candidates[0][1]["BER"] = 1
    assert selected_candidate(candidates)[0]["selector"] == "S2N"


def test_missingness_baseline_no_missing_or_allconstant_masks_is_uninformative():
    x, y = _chronological_data(60)
    for data in (np.ones_like(x), np.full_like(x, np.nan)):
        scores, threshold = missingness_baseline(data[:50], y[:50], data[50:])
        np.testing.assert_array_equal(scores, np.full(10, 0.5))
        assert threshold == -np.inf


def test_calendar_periods_are_disjoint_and_invariant_to_test_label_changes():
    _x, y = _chronological_data()
    frame = pd.DataFrame({"timestamp": pd.date_range("2008-01-01", periods=len(y), freq="12h"), "y_bin": y})
    weeks = add_dev_week_bins(frame)
    plan = choose_outer_fold_plan(weeks)
    assert plan is not None
    seen = set()
    for fold in plan.folds:
        assert not seen.intersection(fold.test_index)
        assert fold.train_end_ts < fold.test_start_ts
        seen.update(fold.test_index)
    changed = weeks.copy()
    changed.loc[list(seen), "y_bin"] = 0
    altered = choose_outer_fold_plan(changed)
    for before, after in zip(plan.folds, altered.folds):
        np.testing.assert_array_equal(before.train_index, after.train_index)
        np.testing.assert_array_equal(before.test_index, after.test_index)
    for train, valid in chronological_inner_splits(y):
        fit, calibration = fit_calibration_indices(len(train))
        assert fit.max() < calibration.min() < valid.min()


def test_retained_temporal_model_uses_fit_prefix_and_eval_labels_cannot_change_freeze():
    x, y = _chronological_data()
    cfg = RoleConfig("primary", "F-test", 3, 1, "StandardScaler", None)
    model = temporal._fit_phase3_role_model(cfg, x[:200], y[:200], np.arange(200) // 14, 6)
    assert model.fit_n == 160
    assert model.calibration_indices.tolist() == list(range(160, 200))
    np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[:160], axis=0))
    expected = model.clf.predict_proba(
        model.scaler.transform(model.imputer.transform(x[160:200]))[:, model.selected_local_idx]
    )[:, 1]
    np.testing.assert_allclose(model.calibration_scores, expected)
    before = (model.scientific_threshold, model.operational_threshold, model.clf.coef_.copy())
    context = temporal._prepare_lockbox_eval_context(model=model, x_lock_raw=x[200:], y_lock=y[200:])
    changed = temporal._prepare_lockbox_eval_context(model=model, x_lock_raw=x[200:], y_lock=1 - y[200:])
    np.testing.assert_array_equal(context["lock_scores"], changed["lock_scores"])
    assert model.scientific_threshold == before[0] and model.operational_threshold == before[1]
    np.testing.assert_array_equal(model.clf.coef_, before[2])


def test_temporal_outer_threshold_does_not_use_eval_labels():
    x, y = _chronological_data()
    kwargs = dict(
        x_train_raw=x[:200],
        y_train=y[:200],
        x_eval_raw=x[200:],
        method="F-test",
        k=3,
        c_value=1,
        scaler_name="StandardScaler",
        n_neighbors=None,
        return_scores=True,
    )
    _metrics, threshold, scores = temporal._fit_eval_with_labels(**kwargs, y_eval=y[200:])
    _changed, altered_threshold, altered_scores = temporal._fit_eval_with_labels(**kwargs, y_eval=1 - y[200:])
    assert threshold == altered_threshold
    np.testing.assert_allclose(scores, altered_scores)


def test_mspc_source_and_thresholds_frozen_before_eval_labels():
    x, y = _chronological_data()
    kwargs = dict(
        x_train_pass=x[:160][y[:160] == 0], x_calibration=x[160:200], y_calibration=y[160:200], x_eval=x[200:]
    )
    actual = temporal._mspc_fit_and_score(**kwargs, y_eval=y[200:])
    altered = temporal._mspc_fit_and_score(**kwargs, y_eval=1 - y[200:])
    for key in ("calibration_selected_MSPC_source", "frozen_threshold", "T2_frozen_threshold", "Q_frozen_threshold"):
        assert actual[key] == altered[key]
    assert actual["source_selection_region"] == "held_out_calibration"
    assert "observed_mean_inter_alarm_spacing" in actual and "empirical_ARL0" not in actual


@pytest.mark.parametrize("field", ["prediction", "threshold", "sample_id"])
def test_audit_rejects_altered_procedure_receipts(benchmark_replication_case, workspace_tmp_dir, field):
    import shutil

    destination = workspace_tmp_dir / "tampered"
    shutil.copytree(benchmark_replication_case["out_dir"], destination)
    path = destination / "reports" / ArtifactName.BENCHMARK_PREDICTIONS
    frame = pd.read_csv(path)
    joint = frame[frame.procedure == "joint"].index
    if field == "sample_id":
        frame.loc[joint[0], field] = frame.loc[joint[1], field]
    elif field == "prediction":
        frame.loc[joint[0], field] = 1 - frame.loc[joint[0], field]
    else:
        frame.loc[joint[0], field] = 12345
    frame.to_csv(path, index=False)
    audit = run_study_audit(destination)
    assert not audit.ok and any("procedures" in error for error in audit.errors)


def test_low_failure_exact_intervals_and_retrospective_claims(temporal_artifacts_case):
    reports = temporal_artifacts_case["out_dir"] / "reports"
    frame = pd.read_csv(reports / ArtifactName.TEMPORAL_LOCKBOX)
    assert frame.TPR_exact_lower.between(0, 1).all()
    assert frame.TPR_exact_upper.between(0, 1).all()
    assert (frame.TPR_exact_lower <= frame["True+"]).all()
    assert (frame.TPR_exact_upper >= frame["True+"]).all()
    drift = pd.read_csv(reports / ArtifactName.TEMPORAL_DRIFT)
    assert not drift.confirmatory_claims_allowed.any()
    manifest = json.loads((reports / ArtifactName.MANIFEST).read_text())
    assert "retrospective_later_block_not_fresh_confirmatory_lockbox" in manifest["temporal_claim_restrictions"]


def test_temporal_inner_undefined_ber_does_not_become_selection_evidence():
    y = np.tile([0, 1], 120)
    y[200:] = 0
    splits = chronological_inner_splits(y)
    assert len(splits) == 2
    assert all(len(np.unique(y[valid])) == 2 for _, valid in splits)
    y[160:] = 0
    with pytest.raises(ValueError, match="fewer than two"):
        chronological_inner_splits(y)
