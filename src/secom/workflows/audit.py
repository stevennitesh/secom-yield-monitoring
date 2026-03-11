from __future__ import annotations

import json
from pathlib import Path

from secom.artifacts import (
    ValidationResult,
    load_artifact_frames,
    validate_required_artifacts,
    validate_schema_and_logic,
)
from secom.config import ArtifactName, StudyStatus


def run_study_audit(output_dir: Path) -> ValidationResult:
    reports = output_dir / "reports"
    manifest_path = reports / ArtifactName.MANIFEST
    if not manifest_path.exists():
        return ValidationResult(
            ok=False,
            errors=[f"missing artifact: {ArtifactName.MANIFEST}"],
            warnings=[],
            claim_restrictions=[],
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_status = str(manifest.get("primary_study_status", StudyStatus.NOT_RUN))
    temporal_status = str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN))

    artifact_frames = load_artifact_frames(output_dir)
    errors = validate_required_artifacts(
        output_dir=output_dir,
        primary_status=primary_status,
        temporal_status=temporal_status,
    )
    schema = validate_schema_and_logic(
        output_dir=output_dir,
        artifact_frames=artifact_frames,
        manifest=manifest,
    )

    merged_errors = list(dict.fromkeys(errors + schema.errors))
    merged_warnings = list(dict.fromkeys(schema.warnings))
    merged_restrictions = list(dict.fromkeys(schema.claim_restrictions))
    return ValidationResult(
        ok=len(merged_errors) == 0,
        errors=merged_errors,
        warnings=merged_warnings,
        claim_restrictions=merged_restrictions,
    )
