"""Small public report export with complete evidence retained in ignored run storage."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from secom.artifacts import read_manifest
from secom.common.meta import source_tree_identity, strategy_sha256
from secom.config import ArtifactName, StudyStatus
from secom.provenance import sha256_file
from secom.reporting import write_final_report
from secom.workflows.audit import run_study_audit


def export_public_snapshot(output_dir: Path, destination: Path, project_root: Path) -> Path:
    """Export report, figures and receipts; keep bulky evidence in the audited run."""
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
        shutil.copy2(archive, local_archive)
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
            "local_archive_sha256": sha256_file(local_archive),
            "files_sha256": inventory,
        }
        (evidence / "audit_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        readme = f"""# Results Snapshot

Read [final_report.md](final_report.md) first. This small public snapshot was exported from `{output_dir}` after the complete artifact audit passed. Git tracks the report, its figures, the unchanged execution manifest and the audit receipt. Detailed CSVs and the full source/artifact ZIP remain in ignored run storage.

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

The full study performs many repeated fits; this snapshot's manifest records the measured modeling duration and thread settings. Export saves the full archive under `runs/reproduction/evidence/study_artifacts.zip`. To audit that local archive without training, unzip it and run its `source/scripts/run_audit.py --output-dir <extracted-directory> --strict`. To reproduce the exact dirty source, install from the archive's `source/` directory and run its scripts. Original non-nested results, tuned family-wise estimates and chronological logistic-regression diagnostics retain separate claim scopes.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        # Remove only named generated bulk from the previous export; preserve manual files.
        for name in expected_csvs | {"study_artifacts.zip"}:
            old_path = destination / "evidence" / name
            if old_path.is_file():
                if old_path.resolve().parent != (destination / "evidence").resolve():
                    raise ValueError("Refusing to remove evidence outside the snapshot directory")
                old_path.unlink()
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                target = destination / path.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return destination / ArtifactName.FINAL_REPORT
