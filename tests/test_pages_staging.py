"""Static publication must preserve evidence and reject a stale or changed HTML report."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from secom.html_report import CORE_FILES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/results"


def stage(source, destination):
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/stage_pages.py"),
            "--input-dir",
            str(source),
            "--output-dir",
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pages_publishes_only_verified_saved_files(workspace_tmp_dir):
    snapshot = workspace_tmp_dir / "snapshot"
    shutil.copytree(SOURCE, snapshot)
    (snapshot / "private-notes.txt").write_text("must stay local")
    (snapshot / "old-archive.zip").write_bytes(b"do not publish")
    output = workspace_tmp_dir / "site"
    result = stage(snapshot, output)
    assert result.returncode == 0, result.stderr
    published = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    assert published == set(CORE_FILES) | {
        "evidence/audit_receipt.json",
        "index.html",
        "html_provenance.json",
        ".nojekyll",
    }
    for name in published - {".nojekyll"}:
        assert (output / name).read_bytes() == (snapshot / name).read_bytes()
    assert (output / ".nojekyll").read_bytes() == b""


@pytest.mark.parametrize(
    "failure", ["core", "html", "inputs", "execution", "model_fit", "audit", "inventory", "receipt_identity"]
)
def test_pages_rejects_invalid_publication_before_creating_output(workspace_tmp_dir, failure):
    snapshot = workspace_tmp_dir / "snapshot"
    shutil.copytree(SOURCE, snapshot)
    path = snapshot / "html_provenance.json"
    provenance = json.loads(path.read_text())
    if failure == "core":
        (snapshot / "final_report.md").write_text("changed evidence")
    elif failure == "html":
        (snapshot / "index.html").write_text("stale or modified HTML")
    elif failure == "inputs":
        provenance["inputs_sha256"].pop("final_report.md")
    elif failure == "execution":
        provenance["scientific_source_tree_sha256"] = "different execution"
    elif failure == "model_fit":
        provenance["models_fitted"] = 1
    else:
        receipt_path = snapshot / "evidence/audit_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if failure == "audit":
            receipt["audit_ok"] = False
        elif failure == "inventory":
            receipt["files_sha256"]["../private"] = "untrusted"
        else:
            receipt["source_tree_sha256"] = "different execution"
        receipt_path.write_text(json.dumps(receipt))
    path.write_text(json.dumps(provenance))
    output = workspace_tmp_dir / "site"
    result = stage(snapshot, output)
    assert result.returncode == 1 and "ERROR:" in result.stderr
    assert not output.exists()


def test_pages_preserves_existing_output_and_rejects_overlap(workspace_tmp_dir):
    output = workspace_tmp_dir / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    assert stage(SOURCE, output).returncode == 1
    assert marker.read_text() == "preserve"
    result = stage(SOURCE, SOURCE / "nested")
    assert result.returncode == 1 and "overlap" in result.stderr
    assert not (SOURCE / "nested").exists()
