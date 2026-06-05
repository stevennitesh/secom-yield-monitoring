"""End-to-end tests for temporal robustness artifact generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import ArtifactName, ScalerName, SelectorName
from secom.types import FittedRoleModel, RoleConfig
from secom.workflows import temporal_robustness
from secom.workflows.audit import run_study_audit
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
