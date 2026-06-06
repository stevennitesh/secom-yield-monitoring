"""Tests for script entrypoint behavior and top-level orchestration."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from secom.config import ArtifactName, StudyStatus
from secom.workflows.full_study import run_full_study


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run_script_help(script_name: str) -> subprocess.CompletedProcess[str]:
    """Run one repository script's help path."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _script_module(module_name: str):
    """Import a script module through the same path users execute."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


def test_benchmark_bundle_help_matches_active_classifier_defaults() -> None:
    """Benchmark bundle help should match the KRR default used by both layers."""
    result = _run_script_help("run_benchmark_replication.py")
    normalized_stdout = " ".join(result.stdout.split())

    assert result.returncode == 0
    assert "Defaults to krr for both original and tuned benchmark layers" in normalized_stdout
    assert "original runs all benchmark classifiers" not in result.stdout


def test_script_help_paths_are_quiet() -> None:
    """Script help should parse without importing plotting/reporting side effects."""
    for script_name in (
        "run_audit.py",
        "run_benchmark_replication.py",
        "run_benchmark_tuned.py",
        "run_final_report.py",
    ):
        result = _run_script_help(script_name)

        assert result.returncode == 0, script_name
        assert result.stderr == "", script_name


def test_report_and_audit_cli_defaults_target_full_study_run(monkeypatch) -> None:
    """Audit and report commands should default to the canonical full-study output directory."""
    for module_name in ("run_audit", "run_final_report", "run_report_skeleton"):
        module = _script_module(module_name)
        monkeypatch.setattr(sys, "argv", [f"{module_name}.py"])

        assert module.parse_args().output_dir == "runs/full_study"


def test_full_study_writes_canonical_report_after_passing_audit(
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Full-study orchestration should produce the canonical report, not the scaffold."""
    import secom.workflows.full_study as full_study

    final_report = workspace_tmp_dir / "reports" / ArtifactName.FINAL_REPORT

    monkeypatch.setattr(
        full_study,
        "run_benchmark_replication",
        lambda **_kwargs: {
            "benchmark_original_status": StudyStatus.PASSED,
            "benchmark_tuned_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        },
    )
    monkeypatch.setattr(
        full_study,
        "run_temporal_robustness",
        lambda **_kwargs: {"temporal_robustness_status": StudyStatus.WARNING},
    )
    monkeypatch.setattr(
        full_study,
        "run_study_audit",
        lambda **_kwargs: SimpleNamespace(ok=True, errors=[], warnings=[], claim_restrictions=[]),
    )
    monkeypatch.setattr(full_study, "write_final_report", lambda *, output_dir: final_report, raising=False)

    result = run_full_study(input_dir=workspace_tmp_dir / "raw", output_dir=workspace_tmp_dir)

    assert result["report_path"] == str(final_report)
    assert final_report.name == ArtifactName.FINAL_REPORT


def test_full_study_skips_report_when_audit_fails(
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Full-study orchestration should not render canonical claims after failed audit."""
    import secom.workflows.full_study as full_study

    report_calls = []

    monkeypatch.setattr(
        full_study,
        "run_benchmark_replication",
        lambda **_kwargs: {
            "benchmark_original_status": StudyStatus.PASSED,
            "benchmark_tuned_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        },
    )
    monkeypatch.setattr(
        full_study,
        "run_temporal_robustness",
        lambda **_kwargs: {"temporal_robustness_status": StudyStatus.WARNING},
    )
    monkeypatch.setattr(
        full_study,
        "run_study_audit",
        lambda **_kwargs: SimpleNamespace(ok=False, errors=["missing artifact"], warnings=[], claim_restrictions=[]),
    )
    monkeypatch.setattr(
        full_study, "write_final_report", lambda *, output_dir: report_calls.append(output_dir), raising=False
    )

    result = run_full_study(input_dir=workspace_tmp_dir / "raw", output_dir=workspace_tmp_dir)

    assert result["report_path"] is None
    assert report_calls == []
