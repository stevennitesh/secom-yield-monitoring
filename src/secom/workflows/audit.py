"""Workflow wrapper for artifact-level study validation."""

from __future__ import annotations

from pathlib import Path
import hashlib

from secom.artifacts import (
    ValidationResult,
    load_artifact_frames,
    read_manifest,
    validate_required_artifacts,
    validate_schema_and_logic,
)
from secom.config import ArtifactName, StudyStatus


def run_study_audit(output_dir: Path) -> ValidationResult:
    """Validate generated study artifacts and merge schema, logic, and claim findings."""
    reports = output_dir / "reports"
    manifest_path = reports / ArtifactName.MANIFEST
    if not manifest_path.exists():
        return ValidationResult(
            ok=False,
            errors=[f"missing artifact: {ArtifactName.MANIFEST}"],
            warnings=[],
            claim_restrictions=[],
        )

    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError, UnicodeError) as exc:
        return ValidationResult(
            ok=False,
            errors=[f"{ArtifactName.MANIFEST}: cannot read manifest: {exc}"],
            warnings=[],
            claim_restrictions=[],
        )
    primary_status = str(manifest.get("primary_study_status", StudyStatus.NOT_RUN))
    original_status = str(manifest.get("benchmark_original_status", StudyStatus.NOT_RUN))
    tuned_status = str(manifest.get("benchmark_tuned_status", StudyStatus.NOT_RUN))
    temporal_status = str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN))

    # Required artifacts depend on manifest status, while schema checks inspect available frames.
    errors = validate_required_artifacts(
        output_dir=output_dir,
        primary_status=primary_status,
        benchmark_original_status=original_status,
        benchmark_tuned_status=tuned_status,
        temporal_status=temporal_status,
    )
    artifact_frames = load_artifact_frames(output_dir, errors=errors)
    schema = validate_schema_and_logic(
        output_dir=output_dir,
        artifact_frames=artifact_frames,
        manifest=manifest,
    )

    hashes = manifest.get("artifact_sha256", {})
    if not isinstance(hashes, dict):
        errors.append("run_manifest.json: artifact_sha256 must be an object")
    else:
        for name, expected in hashes.items():
            if Path(name).name != name or not name.endswith(".csv"):
                errors.append(f"run_manifest.json: invalid artifact hash name {name}")
                continue
            path = reports / name
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            except OSError as exc:
                errors.append(f"run_manifest.json: cannot read hashed artifact {name}: {exc}")
                continue
            if actual != expected or actual is None:
                errors.append(f"run_manifest.json: artifact hash mismatch for {name}")
    merged_errors = list(dict.fromkeys(errors + schema.errors))
    merged_warnings = list(dict.fromkeys(schema.warnings))
    merged_restrictions = list(dict.fromkeys(schema.claim_restrictions))
    return ValidationResult(
        ok=len(merged_errors) == 0,
        errors=merged_errors,
        warnings=merged_warnings,
        claim_restrictions=merged_restrictions,
    )
