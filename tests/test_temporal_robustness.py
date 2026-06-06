"""End-to-end tests for temporal robustness artifact generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from secom.artifacts import read_manifest
from secom.config import ArtifactName, ScalerName, SelectorName, StudyStatus
from secom.types import FittedRoleModel, RoleConfig
from secom.workflows import temporal_robustness
from secom.workflows.audit import run_study_audit
from secom.workflows.manifest import write_benchmark_status, write_temporal_status
from tests.assertions import assert_artifacts_exist, assert_columns_include


def test_temporal_robustness_emits_temporal_artifacts_and_audit_is_non_blocking(
    temporal_artifacts_case: dict[str, object],
) -> None:
    """Temporal robustness should emit its artifact family without blocking primary claims."""
    out_dir = temporal_artifacts_case["out_dir"]
    result = temporal_artifacts_case["result"]

    assert result["temporal_robustness_status"] in {"passed", "warning"}

    reports = out_dir / "reports"
    assert_artifacts_exist(
        reports,
        [
            ArtifactName.TEMPORAL_SPLIT_METADATA,
            ArtifactName.TEMPORAL_SELECTOR_SCREENING,
            ArtifactName.TEMPORAL_MODEL_SELECTION,
            ArtifactName.TEMPORAL_INNER_CV,
            ArtifactName.TEMPORAL_FREEZE,
            ArtifactName.TEMPORAL_LOCKBOX,
            ArtifactName.TEMPORAL_DRIFT,
            ArtifactName.TEMPORAL_MSPC,
            ArtifactName.TEMPORAL_COST_CURVES,
            ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
            ArtifactName.MANIFEST,
        ],
    )

    selection_df = pd.read_csv(reports / ArtifactName.TEMPORAL_MODEL_SELECTION)
    assert_columns_include(selection_df, ["selector", "status", "is_primary", "is_challenger", "mean_BER"])

    audit = run_study_audit(out_dir)
    assert audit.ok, audit.errors


def test_temporal_failure_overwrites_stale_pass_manifest(workspace_tmp_dir: Path, monkeypatch) -> None:
    """Failed temporal reruns should preserve benchmark status while marking temporal failure."""
    out_dir = workspace_tmp_dir / "out"
    reports = out_dir / "reports"
    write_benchmark_status(
        manifest_path=reports / ArtifactName.MANIFEST,
        project_root=workspace_tmp_dir,
        original_status=StudyStatus.PASSED,
        tuned_status=StudyStatus.PASSED,
    )
    write_temporal_status(
        manifest_path=reports / ArtifactName.MANIFEST,
        project_root=workspace_tmp_dir,
        temporal_status=StudyStatus.PASSED,
    )
    fold = SimpleNamespace(
        train_index=np.asarray([0, 1, 2, 3], dtype=int),
        test_index=np.asarray([4, 5], dtype=int),
        outer_fold=1,
    )
    frame = pd.DataFrame(
        {
            "sensor_000": np.ones(6, dtype=float),
            "sensor_001": np.ones(6, dtype=float),
            "y_bin": [0, 0, 0, 1, 1, 1],
            "week_label": [1, 1, 2, 2, 3, 3],
        }
    )
    bundle = SimpleNamespace(
        all_data=frame,
        dev=frame,
        lockbox=frame,
        temporal_feasible=True,
        temporal_infeasible_reason=None,
        fold_plan=SimpleNamespace(folds=[fold]),
        dev_with_weeks=frame,
        feature_columns=["sensor_000", "sensor_001"],
    )

    monkeypatch.setattr(temporal_robustness, "_build_bundle", lambda _input_dir: bundle)
    monkeypatch.setattr(
        temporal_robustness,
        "_fit_eval_with_labels",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("forced temporal failure")),
    )

    with pytest.raises(RuntimeError, match="forced temporal failure"):
        temporal_robustness.run_temporal_robustness(
            input_dir=workspace_tmp_dir / "raw",
            output_dir=out_dir,
            selectors_run=[SelectorName.S2N],
        )

    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    assert manifest["primary_study_status"] == StudyStatus.PASSED
    assert manifest["benchmark_original_status"] == StudyStatus.PASSED
    assert manifest["benchmark_tuned_status"] == StudyStatus.PASSED
    assert manifest["temporal_robustness_status"] == StudyStatus.FAILED


def test_temporal_failure_manifest_write_does_not_mask_original_exception(
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Failure-manifest errors should not replace the original temporal exception."""

    def fail_run(**_kwargs):
        """Simulate the real temporal workflow failure."""
        raise RuntimeError("temporal root cause")

    def fail_manifest_write(**_kwargs):
        """Simulate a secondary failure while writing the manifest."""
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(temporal_robustness, "_run_temporal_robustness", fail_run)
    monkeypatch.setattr(temporal_robustness, "write_temporal_failure", fail_manifest_write)

    with pytest.raises(RuntimeError, match="temporal root cause"):
        temporal_robustness.run_temporal_robustness(
            input_dir=workspace_tmp_dir / "raw",
            output_dir=workspace_tmp_dir / "out",
        )


def test_temporal_selector_grids_match_stage_scope_and_reject_unknowns() -> None:
    """Temporal selector grids should keep screening and Stage-B scopes explicit."""
    stage_a = temporal_robustness._stage_a_configs([SelectorName.F_TEST, SelectorName.RELIEFF])

    assert stage_a == [
        {
            "selector": SelectorName.F_TEST,
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": None,
        },
        {
            "selector": SelectorName.RELIEFF,
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": 10,
        },
    ]

    f_test_grid = temporal_robustness.build_stage_b_config_grid(SelectorName.F_TEST)
    assert len(f_test_grid) == 24
    assert {row["k"] for row in f_test_grid} == {10, 20, 40}
    assert {row["C"] for row in f_test_grid} == {0.01, 0.1, 1.0, 10.0}
    assert {row["scaler"] for row in f_test_grid} == {ScalerName.STANDARD, ScalerName.ROBUST}
    assert {row["n_neighbors"] for row in f_test_grid} == {None}

    relief_grid = temporal_robustness.build_stage_b_config_grid(SelectorName.RELIEFF)
    assert len(relief_grid) == 72
    assert {row["n_neighbors"] for row in relief_grid} == {5, 10, 20}

    with pytest.raises(ValueError, match="Unknown selector"):
        temporal_robustness._stage_a_configs(["Bogus"])
    with pytest.raises(ValueError, match="Unknown selector"):
        temporal_robustness.build_stage_b_config_grid("Bogus")


def test_lockbox_context_uses_frozen_temporal_transforms_and_selected_indices() -> None:
    """Lockbox scoring should transform through the fitted DEV model without selector refit."""

    class RecordingImputer:
        """Record raw lockbox input while returning the fitted transformed feature matrix."""

        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def transform(self, x_raw: np.ndarray) -> np.ndarray:
            self.seen.append(np.asarray(x_raw, dtype=float))
            return np.asarray(
                [
                    [10.0, 100.0, 1000.0],
                    [20.0, 200.0, 2000.0],
                    [30.0, 300.0, 3000.0],
                ],
                dtype=float,
            )

    class RecordingScaler:
        """Record imputed lockbox input while returning frozen scaled features."""

        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def transform(self, x_imp: np.ndarray) -> np.ndarray:
            self.seen.append(np.asarray(x_imp, dtype=float))
            return np.asarray(x_imp, dtype=float) / 10.0

    class RecordingClassifier:
        """Record selected lockbox features before returning probabilities."""

        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def predict_proba(self, x_sel: np.ndarray) -> np.ndarray:
            self.seen.append(np.asarray(x_sel, dtype=float))
            return np.asarray(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.7, 0.3],
                ],
                dtype=float,
            )

    imputer = RecordingImputer()
    scaler = RecordingScaler()
    clf = RecordingClassifier()
    model = FittedRoleModel(
        config=RoleConfig(
            role="primary",
            selector=SelectorName.S2N,
            k=1,
            c_value=1.0,
            scaler=ScalerName.STANDARD,
            n_neighbors=None,
        ),
        imputer=imputer,
        scaler=scaler,
        selected_local_idx=np.asarray([1], dtype=int),
        selected_global_idx=[1],
        clf=clf,
        dev_scores=np.asarray([0.1, 0.8, 0.3], dtype=float),
        scientific_threshold=0.5,
        operational_threshold=0.6,
        threshold_at_tnr90_dev=0.5,
        tnr_at_tnr90_dev=1.0,
        tpr_at_tnr90_dev=0.5,
        feature_meta=[],
    )
    x_lock_raw = np.asarray([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    y_lock = np.asarray([0, 1, 0], dtype=int)

    lock_ctx = temporal_robustness._prepare_lockbox_eval_context(
        model=model,
        x_lock_raw=x_lock_raw,
        y_lock=y_lock,
    )

    assert np.array_equal(imputer.seen[0], x_lock_raw, equal_nan=True)
    assert np.array_equal(
        scaler.seen[0], np.asarray([[10.0, 100.0, 1000.0], [20.0, 200.0, 2000.0], [30.0, 300.0, 3000.0]])
    )
    assert np.array_equal(clf.seen[0], np.asarray([[10.0], [20.0], [30.0]]))
    assert lock_ctx["lock_scores"].tolist() == [0.1, 0.8, 0.3]


def test_temporal_selector_eval_failure_names_selector_context() -> None:
    """Temporal selector-prep failures should identify selector, k, scaler, and neighbors."""
    with pytest.raises(
        RuntimeError,
        match="temporal selector failure.*selector=Gram-Schmidt.*k=2.*scaler=StandardScaler",
    ):
        temporal_robustness._prepare_selector_eval_view(
            x_train_raw=np.ones((4, 3), dtype=float),
            y_train=np.asarray([0, 0, 1, 1], dtype=int),
            x_eval_raw=np.ones((2, 3), dtype=float),
            y_eval=np.asarray([0, 1], dtype=int),
            method=SelectorName.GRAM_SCHMIDT,
            k=2,
            scaler_name=ScalerName.STANDARD,
            add_indicator=False,
            n_neighbors=None,
        )


def test_temporal_role_fit_failure_names_role_selector_context() -> None:
    """Full-DEV role fitting should fail before model fit when selector yields no features."""
    role_cfg = RoleConfig(
        role="primary",
        selector=SelectorName.GRAM_SCHMIDT,
        k=2,
        c_value=1.0,
        scaler=ScalerName.STANDARD,
        n_neighbors=None,
    )

    with pytest.raises(RuntimeError, match="temporal role selector failure.*role=primary.*selector=Gram-Schmidt"):
        temporal_robustness._fit_phase3_role_model(
            role_cfg=role_cfg,
            x_dev_raw=np.ones((6, 3), dtype=float),
            y_dev=np.asarray([0, 0, 0, 1, 1, 1], dtype=int),
            week_labels=np.asarray([1, 1, 2, 2, 3, 3], dtype=int),
            raw_feature_count=3,
        )


def test_temporal_selector_summary_uses_coherent_modal_config_tuple() -> None:
    """Temporal modal config fields should come from one winning config tuple."""
    outer_eval_df = pd.DataFrame(
        [
            {
                "selector": SelectorName.S2N,
                "outer_fold": 1,
                "seed": 42,
                "k": 10,
                "C": 10.0,
                "scaler": ScalerName.STANDARD,
                "n_neighbors": np.nan,
                "BER": 0.30,
                "True+": 0.60,
                "True-": 0.80,
            },
            {
                "selector": SelectorName.S2N,
                "outer_fold": 2,
                "seed": 42,
                "k": 10,
                "C": 1.0,
                "scaler": ScalerName.ROBUST,
                "n_neighbors": np.nan,
                "BER": 0.20,
                "True+": 0.70,
                "True-": 0.90,
            },
            {
                "selector": SelectorName.S2N,
                "outer_fold": 3,
                "seed": 42,
                "k": 20,
                "C": 1.0,
                "scaler": ScalerName.STANDARD,
                "n_neighbors": np.nan,
                "BER": 0.25,
                "True+": 0.65,
                "True-": 0.85,
            },
        ]
    )

    summary = temporal_robustness._summarize_temporal_selector_results(
        outer_eval_df=outer_eval_df,
        deciding_outer_fold=3,
    )

    row = summary[0]
    assert row["modal_k"] == 10
    assert row["modal_C"] == 1.0
    assert row["modal_scaler"] == ScalerName.ROBUST
    assert np.isnan(row["modal_n_neighbors"])


def test_temporal_role_selection_uses_deciding_fold_before_simplicity() -> None:
    """Primary role tie-breaks should use temporal evidence before config simplicity."""
    selector_stats = [
        {
            "selector": SelectorName.S2N,
            "mean_BER": 0.20,
            "mean_True+": 0.70,
            "mean_True-": 0.80,
            "modal_k": 10,
            "modal_C": 0.01,
            "modal_scaler": ScalerName.STANDARD,
            "modal_n_neighbors": np.nan,
            "vote_outer_BER": 0.30,
            "vote_outer_True+": 0.60,
        },
        {
            "selector": SelectorName.F_TEST,
            "mean_BER": 0.20,
            "mean_True+": 0.70,
            "mean_True-": 0.80,
            "modal_k": 40,
            "modal_C": 10.0,
            "modal_scaler": ScalerName.ROBUST,
            "modal_n_neighbors": np.nan,
            "vote_outer_BER": 0.10,
            "vote_outer_True+": 0.90,
        },
    ]

    primary, challenger = temporal_robustness._choose_temporal_roles(selector_stats)

    assert primary == SelectorName.F_TEST
    assert challenger == SelectorName.S2N
