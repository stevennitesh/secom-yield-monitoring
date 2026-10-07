"""Small public report export with complete evidence retained in ignored run storage."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from secom.artifacts import read_manifest
from secom.common.meta import source_tree_identity, strategy_sha256
from secom.config import ArtifactName, StudyStatus
from secom.provenance import sha256_file
from secom.html_report import write_html_report
from secom.reporting import write_final_report, write_report_skeleton
from secom.workflows.audit import run_study_audit

# Explicit editorial owners only. Model code, metrics, pins, scientific specs 01–05,
# config and workflow/audit owners cannot pass this exception to normal export.
PRESENTATION_SOURCE_PATHS = frozenset(
    {
        "src/secom/reporting.py",
        "src/secom/report_figures.py",
        "src/secom/report_language.py",
        "src/secom/html_report.py",
        "src/secom/evidence.py",
        "scripts/export_results.py",
        "scripts/run_final_report.py",
        "scripts/run_html_report.py",
        "docs/spec/06-report-structure.md",
        "docs/spec/07-artifact-contracts.md",
        "docs/spec/08-audit-and-claim-semantics.md",
    }
)


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_presentation_source(output_dir: Path, project_root: Path) -> tuple[dict, dict, list[str], Path]:
    """Bind unchanged results to exact archived execution before accepting editorial drift."""
    reports = output_dir / "reports"
    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    for field in ("dataset", "source_tree", "execution", "resolved_packages", "artifact_sha256"):
        if not manifest.get(field):
            raise ValueError(f"Complete-run provenance is missing: {field}")
    if manifest.get("primary_study_status") != StudyStatus.PASSED or manifest.get("temporal_robustness_status") not in (
        StudyStatus.PASSED,
        StudyStatus.WARNING,
    ):
        raise ValueError("Presentation refresh requires complete benchmark and temporal studies")
    if set(manifest["artifact_sha256"]) != {p.name for p in reports.glob("*.csv")}:
        raise ValueError("CSV hash inventory does not cover the complete artifact set")
    for name, sha in manifest["artifact_sha256"].items():
        if sha256_file(reports / name) != sha:
            raise ValueError(f"Changed executed CSV: {name}")
    archive = output_dir / "evidence/study_artifacts.zip"
    if not archive.is_file():
        raise ValueError("Exact executed source archive is required for presentation refresh")
    recorded = manifest["source_tree"]
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Execution archive contains duplicate member names")
        if bundle.read("reports/" + ArtifactName.MANIFEST) != (reports / ArtifactName.MANIFEST).read_bytes():
            raise ValueError("Execution archive manifest differs from the preserved run manifest")
        source_names = {
            name.removeprefix("source/") for name in names if name.startswith("source/") and not name.endswith("/")
        }
        if source_names != set(recorded["files"]) | {"README.md", "Makefile"}:
            raise ValueError("Execution archive source inventory differs from its manifest")
        archived_hashes = {
            name: _hash_bytes(bundle.read("source/" + name).replace(b"\r\n", b"\n")) for name in recorded["files"]
        }
        digest = hashlib.sha256()
        for name, sha in sorted(archived_hashes.items()):
            digest.update(f"{name}\0{sha}\n".encode())
        if archived_hashes != recorded["files"] or digest.hexdigest() != recorded["sha256"]:
            raise ValueError("Execution archive source does not match the executed source identity")
        spec = hashlib.sha256()
        for name in sorted(
            name for name in recorded["files"] if name.startswith("docs/spec/") and name.split("/")[-1][:2].isdigit()
        ):
            content = bundle.read("source/" + name).replace(b"\r\n", b"\n")
            spec.update(name.encode() + b"\0" + str(len(content)).encode() + b"\0" + content + b"\0")
        if spec.hexdigest() != manifest["study_spec_sha256"]:
            raise ValueError("Execution archive study specifications do not match the executed spec identity")
        for name, sha in manifest["artifact_sha256"].items():
            digest = hashlib.sha256()
            with bundle.open("reports/" + name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sha:
                raise ValueError(f"Execution archive CSV differs from the manifest: {name}")
    current = source_tree_identity(project_root)
    changed = sorted(
        name
        for name in set(recorded["files"]) | set(current["files"])
        if recorded["files"].get(name) != current["files"].get(name)
    )
    forbidden = set(changed) - PRESENTATION_SOURCE_PATHS
    if forbidden:
        raise ValueError("Presentation refresh rejects scientific source changes: " + ", ".join(sorted(forbidden)))
    return manifest, current, changed, archive


def refresh_presentation(
    output_dir: Path, presentation_output_dir: Path, project_root: Path, *, destination: Path | None = None
) -> Path:
    """Render saved evidence into a fresh editorial run; preserve its execution source."""
    source, target = output_dir.resolve(), presentation_output_dir.resolve()
    paths = [source, target] + ([destination.resolve()] if destination is not None else [])
    for i, path in enumerate(paths):
        for other in paths[i + 1 :]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Presentation, source run and public destination must not overlap")
    if target.exists():
        raise ValueError("Presentation refresh requires a fresh output directory")
    manifest, rendering, changed, executed_archive = _verify_presentation_source(output_dir, project_root)
    audit = run_study_audit(output_dir)
    if not audit.ok:
        raise ValueError("Cannot refresh failed audit: " + "; ".join(audit.errors))
    # All source/artifact/schema checks precede the first output mutation.
    reports = target / "reports"
    reports.mkdir(parents=True)
    shutil.copy2(source / "reports" / ArtifactName.MANIFEST, reports / ArtifactName.MANIFEST)
    write_final_report(source, reports_destination=reports)
    write_report_skeleton(source, reports_destination=reports)
    provenance = {
        "presentation_only": True,
        "source_run": str(output_dir),
        "presentation_run": str(presentation_output_dir),
        "source_tree_sha256": manifest["source_tree"]["sha256"],
        "rendering_source": rendering,
        "changed_rendering_paths": changed,
        "executed_archive_path": str(executed_archive),
        "executed_archive_sha256": sha256_file(executed_archive),
        "rendering_reader_files_sha256": {
            name: _hash_bytes((project_root / name).read_bytes().replace(b"\r\n", b"\n"))
            for name in ("README.md", "Makefile")
        },
    }
    evidence = target / "evidence"
    evidence.mkdir()
    archive = evidence / "study_artifacts.zip"
    with ZipFile(executed_archive) as old, ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in old.namelist():
            if name.startswith("source/"):
                bundle.writestr(name, old.read(name))
        for path in sorted((source / "reports").glob("*.csv")):
            bundle.write(path, "reports/" + path.name)
        for path in sorted(reports.rglob("*")):
            if path.is_file():
                bundle.write(path, "reports/" + path.relative_to(reports).as_posix())
        for name in rendering["files"]:
            bundle.writestr("presentation_source/" + name, (project_root / name).read_bytes().replace(b"\r\n", b"\n"))
        for name in provenance["rendering_reader_files_sha256"]:
            bundle.writestr("presentation_source/" + name, (project_root / name).read_bytes().replace(b"\r\n", b"\n"))
        bundle.writestr("presentation_provenance.json", json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="secom-presentation-") as scratch:
        staging = Path(scratch) / "public"
        staging.mkdir()
        shutil.copy2(reports / ArtifactName.FINAL_REPORT, staging / ArtifactName.FINAL_REPORT)
        shutil.copytree(reports / "figures", staging / "figures")
        (staging / "evidence").mkdir()
        shutil.copy2(reports / ArtifactName.MANIFEST, staging / "evidence" / ArtifactName.MANIFEST)
        receipt = {
            **provenance,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_ok": audit.ok,
            "errors": audit.errors,
            "warnings": audit.warnings,
            "claim_restrictions": audit.claim_restrictions,
            "rendering_source_sha256": rendering["sha256"],
            "local_archive_path": str(archive),
            "local_archive_sha256": sha256_file(archive),
            "files_sha256": {
                p.relative_to(staging).as_posix(): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()
            },
        }
        receipt_text = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        (staging / "evidence/audit_receipt.json").write_text(receipt_text, encoding="utf-8")
        (evidence / "audit_receipt.json").write_text(receipt_text, encoding="utf-8")
        write_html_report(staging)
        # Reader guide is authored deliberately outside generated publication bytes.
        if destination is not None:
            _publish_snapshot(staging, destination, set(manifest["artifact_sha256"]) | {"study_artifacts.zip"})
    return (destination if destination is not None else reports) / ArtifactName.FINAL_REPORT


def _publish_snapshot(staging: Path, destination: Path, generated_bulk: set[str]) -> None:
    """Publish a complete directory, retaining the previous snapshot until the rename succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.with_name(f".{destination.name}.previous")
    marker = previous / "publication.json"
    ownership = {"destination": str(destination.resolve())}
    old_snapshot = previous / "snapshot"
    # A process interrupted between directory renames leaves the old snapshot here.
    if previous.exists():
        if previous.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != ownership:
            raise ValueError(f"Refusing to overwrite unrelated publication backup: {previous}")
        if not destination.exists():
            old_snapshot.rename(destination)
        shutil.rmtree(previous)
    # Ordinary mkdir inherits the destination parent's usable access. On Windows,
    # TemporaryDirectory creates an owner-only directory which survives the rename.
    scratch = destination.parent / f".{destination.name}-{uuid.uuid4().hex}"
    scratch.mkdir()
    try:
        candidate = scratch / "public"
        if destination.exists():
            if destination.is_symlink() or any(path.is_symlink() for path in destination.rglob("*")):
                raise ValueError("Public snapshot must not contain symbolic links")
            shutil.copytree(destination, candidate)
        else:
            candidate.mkdir()
        for name in generated_bulk:
            old_path = candidate / "evidence" / name
            if old_path.is_file():
                old_path.unlink()
        shutil.copytree(staging, candidate, dirs_exist_ok=True)
        had_previous = destination.exists()
        if had_previous:
            previous.mkdir()
            marker.write_text(json.dumps(ownership), encoding="utf-8")
            destination.rename(old_snapshot)
        try:
            candidate.rename(destination)
        except BaseException:
            if had_previous:
                old_snapshot.rename(destination)
                shutil.rmtree(previous)
            raise
        if had_previous:
            shutil.rmtree(previous)
    finally:
        shutil.rmtree(scratch)


def export_public_snapshot(
    output_dir: Path, destination: Path, project_root: Path, *, presentation_output_dir: Path | None = None
) -> Path:
    """Export report, figures and receipts; keep bulky evidence in the audited run."""
    if presentation_output_dir is not None:
        return refresh_presentation(output_dir, presentation_output_dir, project_root, destination=destination)
    run_path, snapshot_path = output_dir.resolve(), destination.resolve()
    if run_path.is_relative_to(snapshot_path) or snapshot_path.is_relative_to(run_path):
        raise ValueError("Public snapshot destination must not overlap the source run directory")
    audit = run_study_audit(output_dir)
    if not audit.ok:
        raise ValueError("Cannot export failed audit: " + "; ".join(audit.errors))
    reports = output_dir / "reports"
    manifest_path = reports / ArtifactName.MANIFEST
    manifest = read_manifest(manifest_path)
    if manifest.get("primary_study_status") != StudyStatus.PASSED:
        raise ValueError("Public evidence requires both benchmark studies to pass")
    if manifest.get("temporal_robustness_status") not in (StudyStatus.PASSED, StudyStatus.WARNING):
        raise ValueError("Public evidence requires a completed temporal stress test")
    for field in ("dataset", "source_tree", "execution", "resolved_packages", "artifact_sha256"):
        if not manifest.get(field):
            raise ValueError(f"Complete-run provenance is missing: {field}")
    if manifest["source_tree"] != source_tree_identity(project_root):
        raise ValueError("Current study source differs from the run; restore that source or run a fresh study")
    if manifest["study_spec_sha256"] != strategy_sha256(project_root):
        raise ValueError("Current study spec differs from the executed study")
    expected_csvs = {path.name for path in reports.glob("*.csv")}
    if set(manifest["artifact_sha256"]) != expected_csvs:
        raise ValueError("CSV hash inventory does not cover the complete artifact set")
    write_final_report(output_dir)
    with tempfile.TemporaryDirectory(prefix="secom-evidence-") as scratch:
        staging = Path(scratch) / "public"
        evidence = staging / "evidence"
        evidence.mkdir(parents=True)
        shutil.copy2(reports / ArtifactName.FINAL_REPORT, staging / ArtifactName.FINAL_REPORT)
        shutil.copytree(reports / "figures", staging / "figures")
        shutil.copy2(manifest_path, evidence / ArtifactName.MANIFEST)
        archive = Path(scratch) / "study_artifacts.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
            for path in sorted(reports.rglob("*")):
                if path.is_file() and (
                    path.suffix in (".csv", ".png") or path.name in (ArtifactName.MANIFEST, ArtifactName.FINAL_REPORT)
                ):
                    bundle.write(path, "reports/" + path.relative_to(reports).as_posix())
            for name in manifest["source_tree"]["files"]:
                bundle.writestr("source/" + name, (project_root / name).read_bytes().replace(b"\r\n", b"\n"))
            for name in ("README.md", "Makefile"):
                bundle.writestr("source/" + name, (project_root / name).read_bytes().replace(b"\r\n", b"\n"))
        local_archive = output_dir / "evidence" / archive.name
        local_archive.parent.mkdir(parents=True, exist_ok=True)
        inventory = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path != archive
        }
        receipt = {
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(output_dir),
            "audit_ok": audit.ok,
            "errors": audit.errors,
            "warnings": audit.warnings,
            "claim_restrictions": audit.claim_restrictions,
            "source_tree_sha256": manifest["source_tree"]["sha256"],
            "local_archive_path": str(local_archive),
            "local_archive_sha256": sha256_file(archive),
            "files_sha256": inventory,
        }
        (evidence / "audit_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_html_report(staging)
        readme = f"""# Results Snapshot

Open [the offline HTML report](index.html) in a browser, or read [the canonical technical report](final_report.md). This small public snapshot was exported from `{output_dir}` after the complete artifact audit passed. Git tracks both report formats, six figures, the unchanged execution manifest, the scientific audit receipt and separate [HTML rendering provenance](html_provenance.json). Detailed CSVs and the full source/artifact ZIP remain in ignored run storage.

The source execution records base Git commit `{manifest["git_commit"]}`, dirty status `{manifest["git_dirty"]}` and exact source content hash `{manifest["source_tree"]["sha256"]}`. A dirty run includes uncommitted source changes; it does not pretend the base commit contains them. The unchanged [run manifest](evidence/run_manifest.json) records input hashes, resolved dependencies, study settings, timing and artifact hashes. The publication revision is the repository commit that eventually contains this snapshot.

- Primary benchmark status: `{manifest["primary_study_status"]}`.
- Temporal stress-test status: `{manifest["temporal_robustness_status"]}`.
- Temporal claim restrictions: `{", ".join(audit.claim_restrictions) or "none"}`.
- [Audit receipt](evidence/audit_receipt.json): errors, warnings, restrictions and exported-file hashes.
- Complete CSVs stay in `{output_dir}/reports/`.
- The full artifact/source archive is `{local_archive}`: all CSVs, manifest, report, figures and exact study source, without raw data. It is a local file, not a GitHub download; the public receipt records its hash.

## Reproduce and Audit

Use Python 3.11 or 3.12. See the root README for portable installation. Fetch verified data with `python scripts/fetch_secom.py`, then run into a fresh directory:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/reproduction --classifiers krr,logreg --progress --strict
python scripts/run_audit.py --output-dir runs/reproduction --strict
python scripts/export_results.py --output-dir runs/reproduction
```

The full study performs many repeated fits; this snapshot's manifest records the measured modeling duration and thread settings. Export saves the full archive under `runs/reproduction/evidence/study_artifacts.zip`. To audit that local archive without training, unzip it and run its `source/scripts/run_audit.py --output-dir <extracted-directory> --strict`. To reproduce the exact dirty source, install from the archive's `source/` directory and run its scripts. Original fixed-budget and tuned nested joint-procedure estimates retain separate claim scopes from exploratory family diagnostics and retrospective chronological logistic-regression stress evidence.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        previous_archive = Path(scratch) / "previous_study_artifacts.zip"
        had_archive = local_archive.is_file()
        if had_archive:
            shutil.copy2(local_archive, previous_archive)
        try:
            shutil.copy2(archive, local_archive)
            _publish_snapshot(staging, destination, expected_csvs | {"study_artifacts.zip"})
        except BaseException:
            # A cleanup error can be raised after the complete directory was published.
            # In that case its new receipt already references the new ZIP; keep both.
            try:
                published = (destination / "evidence/audit_receipt.json").read_bytes() == (
                    evidence / "audit_receipt.json"
                ).read_bytes()
            except OSError:
                published = False
            if not published:
                if had_archive:
                    shutil.copy2(previous_archive, local_archive)
                else:
                    local_archive.unlink(missing_ok=True)
            raise
    return destination / ArtifactName.FINAL_REPORT
