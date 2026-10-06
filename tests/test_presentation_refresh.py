"""Editorial publication preserves executed evidence and rejects scientific drift."""

from __future__ import annotations

import hashlib
import json
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from secom.common.meta import source_tree_identity
from secom.evidence import refresh_presentation
from tests.test_provenance_and_export import PROJECT_ROOT, _bind_fixture_provenance


@pytest.fixture
def executed_presentation_fixture(active_artifacts_output_dir, synthetic_input_dir, workspace_tmp_dir):
    """Archive a fabricated, audited run and copy its source for isolated mutation tests."""
    _bind_fixture_provenance(active_artifacts_output_dir, synthetic_input_dir)
    root = workspace_tmp_dir / "render-root"
    identity = source_tree_identity(PROJECT_ROOT)
    for name in [*identity["files"], "README.md", "Makefile"]:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, destination)
    evidence = active_artifacts_output_dir / "evidence"
    evidence.mkdir()
    with ZipFile(evidence / "study_artifacts.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in [*identity["files"], "README.md", "Makefile"]:
            bundle.writestr("source/" + name, (root / name).read_bytes().replace(b"\r\n", b"\n"))
        for path in (active_artifacts_output_dir / "reports").glob("*"):
            if path.is_file():
                bundle.write(path, "reports/" + path.name)
    return active_artifacts_output_dir, root, workspace_tmp_dir / "editorial", workspace_tmp_dir / "public"


def _inventory(directory):
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_editorial_difference_preserves_execution_and_archives_both_sources(executed_presentation_fixture):
    old, root, fresh, public = executed_presentation_fixture
    before = _inventory(old)
    path = root / "src/secom/reporting.py"
    path.write_bytes(path.read_bytes() + b"\n# isolated editorial test change\n")
    public.mkdir()
    (public / "reader-notes.md").write_text("preserve manual guide", encoding="utf-8")
    refresh_presentation(old, fresh, root, destination=public)
    assert _inventory(old) == before
    assert not list((fresh / "reports").glob("*.csv"))
    manifest = (old / "reports/run_manifest.json").read_bytes()
    assert (fresh / "reports/run_manifest.json").read_bytes() == manifest
    assert (public / "evidence/run_manifest.json").read_bytes() == manifest
    assert (public / "reader-notes.md").read_text() == "preserve manual guide"
    receipt = json.loads((public / "evidence/audit_receipt.json").read_text())
    assert receipt["presentation_only"] and receipt["audit_ok"]
    assert receipt["changed_rendering_paths"] == ["src/secom/reporting.py"]
    assert receipt["source_tree_sha256"] == json.loads(manifest)["source_tree"]["sha256"]
    assert receipt["rendering_source"] == source_tree_identity(root)
    assert receipt["rendering_source_sha256"] != receipt["source_tree_sha256"]
    archive = fresh / "evidence/study_artifacts.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["local_archive_sha256"]
    for name, sha in receipt["files_sha256"].items():
        assert hashlib.sha256((public / name).read_bytes()).hexdigest() == sha
    with ZipFile(archive) as bundle, ZipFile(old / "evidence/study_artifacts.zip") as executed:
        for name in executed.namelist():
            if name.startswith("source/") or name.endswith(".csv"):
                assert bundle.read(name) == executed.read(name)
        assert bundle.read("reports/run_manifest.json") == manifest
        assert bundle.read("presentation_source/src/secom/reporting.py") == path.read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(bundle.read("presentation_source/README.md")).hexdigest()
            == receipt["rendering_reader_files_sha256"]["README.md"]
        )
        assert (
            json.loads(bundle.read("presentation_provenance.json"))["rendering_source"] == receipt["rendering_source"]
        )
        assert not any(name.startswith("data/") or name.endswith(".npy") for name in bundle.namelist())


@pytest.mark.parametrize(
    "name",
    [
        "src/secom/models.py",
        "src/secom/config.py",
        "requirements.txt",
        "docs/spec/02-benchmark-replication-study.md",
        "docs/spec/README.md",
    ],
)
def test_scientific_difference_rejected_before_mutation(executed_presentation_fixture, name):
    old, root, fresh, public = executed_presentation_fixture
    before = _inventory(old)
    path = root / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="rejects scientific source changes"):
        refresh_presentation(old, fresh, root, destination=public)
    assert not fresh.exists() and not public.exists()
    assert _inventory(old) == before


def test_changed_csv_rejected_before_mutation(executed_presentation_fixture):
    old, root, fresh, public = executed_presentation_fixture
    path = old / "reports/benchmark_procedure_summary.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed executed CSV"):
        refresh_presentation(old, fresh, root, destination=public)
    assert not fresh.exists() and not public.exists()


@pytest.mark.parametrize(
    "member", ["source/src/secom/models.py", "reports/run_manifest.json", "reports/benchmark_predictions.csv"]
)
def test_tampered_execution_archive_rejected_before_mutation(executed_presentation_fixture, member):
    old, root, fresh, public = executed_presentation_fixture
    path = old / "evidence/study_artifacts.zip"
    with ZipFile(path) as bundle:
        contents = {name: bundle.read(name) for name in bundle.namelist()}
    contents[member] += b"\nchanged\n"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as bundle:
        for name, content in contents.items():
            bundle.writestr(name, content)
    with pytest.raises(ValueError, match="Execution archive"):
        refresh_presentation(old, fresh, root, destination=public)
    assert not fresh.exists() and not public.exists()


@pytest.mark.parametrize("target_kind", ["existing", "inside_source", "public_overlap"])
def test_nonfresh_or_overlapping_editorial_output_rejected(executed_presentation_fixture, target_kind):
    old, root, fresh, public = executed_presentation_fixture
    if target_kind == "existing":
        fresh.mkdir()
    elif target_kind == "inside_source":
        fresh = old / "editorial"
    else:
        public = fresh / "public"
    before = _inventory(old)
    with pytest.raises(ValueError, match="fresh|overlap"):
        refresh_presentation(old, fresh, root, destination=public)
    assert _inventory(old) == before
