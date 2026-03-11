from __future__ import annotations

from pathlib import Path

import pandas as pd

from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_replication import run_benchmark_replication


def test_benchmark_replication_emits_primary_artifacts_and_passes_audit(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
) -> None:
    out_dir = workspace_tmp_dir / "out_benchmark_replication"
    result = run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )

    assert result["primary_study_status"] == StudyStatus.PASSED

    reports = out_dir / "reports"
    expected = [
        ArtifactName.BENCHMARK_SWEEP,
        ArtifactName.BENCHMARK_BEST_CONFIG,
        ArtifactName.BENCHMARK_FOLD_METRICS,
        ArtifactName.BENCHMARK_SUMMARY,
        ArtifactName.BENCHMARK_ABLATION,
        ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
        ArtifactName.FEATURE_STABILITY,
        ArtifactName.FEATURE_REPORT,
        ArtifactName.MANIFEST,
    ]
    for name in expected:
        assert (reports / name).exists(), name

    summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_SUMMARY)
    assert {"selector", "classifier", "replication_mode", "mean_BER"}.issubset(summary_df.columns)

    feature_report_df = pd.read_csv(reports / ArtifactName.FEATURE_REPORT)
    assert {
        "selector",
        "classifier",
        "replication_mode",
        "feature_index",
        "feature_type",
        "selection_frequency",
        "conditional_effect_magnitude",
        "expected_contribution",
    }.issubset(feature_report_df.columns)

    audit = run_study_audit(out_dir)
    assert audit.ok, audit.errors
    assert audit.claim_restrictions == []


def test_benchmark_replication_feature_report_aligns_with_requested_classifier(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
) -> None:
    out_dir = workspace_tmp_dir / "out_benchmark_replication_krr_only"
    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )

    feature_report_df = pd.read_csv(out_dir / "reports" / ArtifactName.FEATURE_REPORT)
    assert set(feature_report_df["classifier"].dropna().astype(str).unique()) == {"krr"}
