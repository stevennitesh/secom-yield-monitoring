"""Tests for portable provenance, verified acquisition and evidence export boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from secom.common.meta import source_tree_identity, strategy_sha256
from secom.config import ArtifactName
from secom.evidence import export_public_snapshot
from secom.provenance import begin_full_study, dataset_profile, finish_full_study
from secom.workflows.audit import run_study_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_identity_is_portable_and_detects_source_changes(workspace_tmp_dir: Path) -> None:
    source = workspace_tmp_dir / "src/secom/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value = 1\n")
    initial = source_tree_identity(workspace_tmp_dir)
    source.write_bytes(b"value = 1\r\n")
    assert source_tree_identity(workspace_tmp_dir) == initial
    source.write_bytes(b"value = 2\n")
    assert source_tree_identity(workspace_tmp_dir)["sha256"] != initial["sha256"]


def test_dataset_profile_records_observed_inputs_and_hashes(synthetic_input_dir: Path) -> None:
    profile = dataset_profile(synthetic_input_dir)
    assert profile["n_samples"] == 260
    assert profile["n_features"] == 12
    assert profile["n_passes"] + profile["n_fails"] == 260
    assert profile["invalid_timestamps"] == 0
    assert not profile["matches_reference_files"]
    for name, item in profile["files"].items():
        assert item["sha256"] == hashlib.sha256((synthetic_input_dir / name).read_bytes()).hexdigest()


def test_full_study_rejects_reusing_existing_artifacts(workspace_tmp_dir: Path) -> None:
    reports = workspace_tmp_dir / "reports"
    reports.mkdir()
    marker = reports / "previous.csv"
    marker.write_text("keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output directory"):
        begin_full_study(workspace_tmp_dir / "absent", workspace_tmp_dir, PROJECT_ROOT, None)
    assert marker.read_text(encoding="utf-8") == "keep\n"


def _bind_fixture_provenance(output_dir: Path, input_dir: Path) -> bytes:
    """Attach explicit synthetic provenance to a schema-valid artifact fixture."""
    path = output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        dataset=dataset_profile(input_dir),
        source_tree=source_tree_identity(PROJECT_ROOT),
        study_spec_sha256=strategy_sha256(PROJECT_ROOT),
        resolved_packages={"test-fixture": "synthetic"},
        execution={"started_at_utc": "2026-10-03T00:00:00+00:00"},
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    finish_full_study(output_dir, {"total_modeling": 0.0})
    return path.read_bytes()


def test_export_preserves_manifest_and_produces_independently_auditable_archive(
    active_artifacts_output_dir: Path,
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
) -> None:
    original = _bind_fixture_provenance(active_artifacts_output_dir, synthetic_input_dir)
    destination = workspace_tmp_dir / "public"
    export_public_snapshot(active_artifacts_output_dir, destination, PROJECT_ROOT)
    assert (destination / "evidence/run_manifest.json").read_bytes() == original
    receipt = json.loads((destination / "evidence/audit_receipt.json").read_text(encoding="utf-8"))
    assert receipt["audit_ok"]
    assert receipt["claim_restrictions"]
    assert not list(destination.rglob("*.csv"))
    assert not list(destination.rglob("*.zip"))
    for name, sha in receipt["files_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha
    extracted = workspace_tmp_dir / "extracted"
    local_archive = active_artifacts_output_dir / "evidence/study_artifacts.zip"
    assert hashlib.sha256(local_archive.read_bytes()).hexdigest() == receipt["local_archive_sha256"]
    with ZipFile(local_archive) as archive:
        assert "reports/feature_stability.csv" in archive.namelist()
        assert "source/src/secom/reporting.py" in archive.namelist()
        assert not any(name.startswith("data/") or name.endswith("secom.data") for name in archive.namelist())
        archive.extractall(extracted)
    assert run_study_audit(extracted).ok
    assert source_tree_identity(extracted / "source") == source_tree_identity(PROJECT_ROOT)


def test_export_removes_legacy_bulk_and_preserves_manual_files(
    active_artifacts_output_dir: Path, synthetic_input_dir: Path, workspace_tmp_dir: Path
) -> None:
    _bind_fixture_provenance(active_artifacts_output_dir, synthetic_input_dir)
    destination = workspace_tmp_dir / "public"
    evidence = destination / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "benchmark_summary.csv").write_text("old generated data", encoding="utf-8")
    (evidence / "study_artifacts.zip").write_bytes(b"old generated archive")
    manual = evidence / "review_notes.txt"
    manual.write_text("keep reviewer notes", encoding="utf-8")
    export_public_snapshot(active_artifacts_output_dir, destination, PROJECT_ROOT)
    assert not (evidence / "benchmark_summary.csv").exists()
    assert not (evidence / "study_artifacts.zip").exists()
    assert manual.read_text(encoding="utf-8") == "keep reviewer notes"


def test_changed_csv_is_rejected_before_public_export(
    active_artifacts_output_dir: Path,
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
) -> None:
    _bind_fixture_provenance(active_artifacts_output_dir, synthetic_input_dir)
    path = active_artifacts_output_dir / "reports/benchmark_summary.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok
    assert any("artifact hash mismatch" in error for error in audit.errors)
    destination = workspace_tmp_dir / "public"
    with pytest.raises(ValueError, match="Cannot export failed audit"):
        export_public_snapshot(active_artifacts_output_dir, destination, PROJECT_ROOT)
    assert not destination.exists()


def test_changed_source_identity_is_rejected_before_public_export(
    active_artifacts_output_dir: Path,
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
) -> None:
    _bind_fixture_provenance(active_artifacts_output_dir, synthetic_input_dir)
    path = active_artifacts_output_dir / "reports/run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_tree"]["sha256"] = "different"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        export_public_snapshot(active_artifacts_output_dir, workspace_tmp_dir / "public", PROJECT_ROOT)


def test_fetch_verifies_archive_without_extracting_extra_members(workspace_tmp_dir: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    from scripts import fetch_secom

    payloads = {"secom.data": b"1 2\n", "secom_labels.data": b'-1 "01/01/2008 12:00:00"\n'}
    monkeypatch.setattr(
        fetch_secom, "SECOM_FILE_SHA256", {name: hashlib.sha256(body).hexdigest() for name, body in payloads.items()}
    )
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for name, body in payloads.items():
            archive.writestr(name, body)
        archive.writestr("../unexpected.txt", b"ignore")
    monkeypatch.setattr(fetch_secom, "urlopen", lambda *_args, **_kwargs: BytesIO(archive_bytes.getvalue()))
    output = workspace_tmp_dir / "inputs"
    fetch_secom.fetch_secom(output)
    assert set(path.name for path in output.iterdir()) == set(payloads)
    monkeypatch.setattr(fetch_secom, "urlopen", lambda *_args, **_kwargs: pytest.fail("verified data must be reused"))
    fetch_secom.fetch_secom(output)
    (output / "secom.data").write_bytes(b"different local data")
    with pytest.raises(ValueError, match="differs from the reference"):
        fetch_secom.fetch_secom(output)
    assert (output / "secom.data").read_bytes() == b"different local data"


@pytest.mark.parametrize("extension", ["csv", "json", "md"])
def test_git_checkout_preserves_evidence_receipt_bytes(workspace_tmp_dir: Path, extension: str) -> None:
    """Git normalization must not invalidate hashes after publishing or cloning evidence."""
    artifact = workspace_tmp_dir / f"artifact.{extension}"
    artifact.write_bytes(b"public evidence\r\nexact bytes\r\n")
    command = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "hash-object"]
    raw_hash = subprocess.check_output(command + ["--no-filters", str(artifact.resolve())], cwd=PROJECT_ROOT)
    checkout_hash = subprocess.check_output(
        command + [f"--path=docs/results/evidence/example.{extension}", str(artifact.resolve())], cwd=PROJECT_ROOT
    )
    assert checkout_hash == raw_hash
