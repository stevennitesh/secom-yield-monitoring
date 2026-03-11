from __future__ import annotations

from pathlib import Path

from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows.temporal_robustness import run_temporal_robustness


def test_end_to_end_active_studies_share_output_dir(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_end_to_end_study"

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

    benchmark_result = run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    temporal_result = run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )
    audit = run_study_audit(out_dir)

    assert benchmark_result["primary_study_status"] == StudyStatus.PASSED
    assert temporal_result["temporal_robustness_status"] in {StudyStatus.PASSED, StudyStatus.WARNING}
    assert audit.ok, audit.errors

    reports = out_dir / "reports"
    for name in [
        ArtifactName.BENCHMARK_SUMMARY,
        ArtifactName.FEATURE_REPORT,
        ArtifactName.TEMPORAL_MODEL_SELECTION,
        ArtifactName.TEMPORAL_LOCKBOX,
        ArtifactName.MANIFEST,
    ]:
        assert (reports / name).exists(), name
