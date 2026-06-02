from __future__ import annotations

import pandas as pd

from secom.config import ArtifactName
from secom.workflows.audit import run_study_audit


def test_temporal_robustness_emits_temporal_artifacts_and_audit_is_non_blocking(
    temporal_artifacts_case: dict[str, object],
) -> None:
    out_dir = temporal_artifacts_case["out_dir"]
    result = temporal_artifacts_case["result"]

    assert result["temporal_robustness_status"] in {"passed", "warning"}

    reports = out_dir / "reports"
    expected = [
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
    ]
    for name in expected:
        assert (reports / name).exists(), name

    selection_df = pd.read_csv(reports / ArtifactName.TEMPORAL_MODEL_SELECTION)
    assert {"selector", "status", "is_primary", "is_challenger", "mean_BER"}.issubset(selection_df.columns)

    audit = run_study_audit(out_dir)
    assert audit.ok, audit.errors
