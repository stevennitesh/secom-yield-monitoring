from __future__ import annotations

import json
from pathlib import Path

from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from tests.assertions import assert_artifacts_exist


def test_end_to_end_active_studies_share_output_dir(
    active_artifacts_output_dir: Path,
) -> None:
    audit = run_study_audit(active_artifacts_output_dir)
    manifest = json.loads((active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST).read_text(encoding="utf-8"))

    assert manifest["primary_study_status"] == StudyStatus.PASSED
    assert manifest["benchmark_original_status"] == StudyStatus.PASSED
    assert manifest["benchmark_tuned_status"] == StudyStatus.PASSED
    assert manifest["temporal_robustness_status"] in {StudyStatus.PASSED, StudyStatus.WARNING}
    assert audit.ok, audit.errors

    reports = active_artifacts_output_dir / "reports"
    assert_artifacts_exist(
        reports,
        [
            ArtifactName.BENCHMARK_SUMMARY,
            ArtifactName.BENCHMARK_TUNED_SUMMARY,
            ArtifactName.FEATURE_REPORT,
            ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
            ArtifactName.TEMPORAL_MODEL_SELECTION,
            ArtifactName.TEMPORAL_LOCKBOX,
            ArtifactName.MANIFEST,
        ],
    )
