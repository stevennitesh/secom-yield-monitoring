from __future__ import annotations

from pathlib import Path

import pandas as pd

from secom.config import ArtifactName
from secom.workflows.audit import run_study_audit
from secom.workflows.temporal_robustness import run_temporal_robustness


def test_temporal_robustness_emits_temporal_artifacts_and_audit_is_non_blocking(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_temporal_robustness"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    result = run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

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
