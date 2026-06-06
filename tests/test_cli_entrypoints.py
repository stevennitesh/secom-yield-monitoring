"""Tests for script entrypoint behavior and top-level orchestration."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from secom.artifacts import ensure_reports_dir, write_manifest
from secom.config import ArtifactName, StudyStatus
from secom.workflows.full_study import run_full_study
from secom.workflows.manifest import initial_study_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUN_SCRIPT_NAMES = tuple(script.name for script in sorted(SCRIPTS_DIR.glob("run_*.py")))


def _manifest_with_statuses(statuses: dict[str, object]) -> dict[str, object]:
    """Build a complete study manifest with caller-supplied status overrides."""
    manifest = initial_study_manifest(PROJECT_ROOT)
    manifest.update(statuses)
    return manifest


def _run_script_help(script_name: str) -> subprocess.CompletedProcess[str]:
    """Run one repository script's help path."""
    env = os.environ.copy()
    env.pop("MPLCONFIGDIR", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), "--help"],
        cwd=PROJECT_ROOT,
        env=env,
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
    for script_name in RUN_SCRIPT_NAMES:
        result = _run_script_help(script_name)

        assert result.returncode == 0, script_name
        assert result.stderr == "", script_name


def test_report_and_audit_cli_defaults_target_full_study_run(monkeypatch) -> None:
    """Audit and report commands should default to the canonical full-study output directory."""
    for module_name in ("run_audit", "run_final_report", "run_report_skeleton"):
        module = _script_module(module_name)
        monkeypatch.setattr(sys, "argv", [f"{module_name}.py"])

        assert module.parse_args().output_dir == "runs/full_study"


def test_final_report_cli_prints_structured_audit_failure(monkeypatch, capsys) -> None:
    """Expected report audit failures should not surface as Python tracebacks."""
    import secom.reporting as reporting

    module = _script_module("run_final_report")
    monkeypatch.setattr(sys, "argv", ["run_final_report.py"])

    def fail_report(*_args, **_kwargs) -> None:
        raise RuntimeError("Cannot render final report because study audit failed: missing artifact")

    monkeypatch.setattr(reporting, "write_final_report", fail_report)

    with pytest.raises(SystemExit) as raised:
        module.main()
    captured = capsys.readouterr()

    assert raised.value.code == 1
    assert "ERROR: Cannot render final report because study audit failed: missing artifact" in captured.out
    assert captured.err == ""


def test_benchmark_bundle_cli_prints_failed_manifest_after_workflow_exception(
    workspace_tmp_dir: Path,
    monkeypatch,
    capsys,
) -> None:
    """Benchmark CLI crashes should still report persisted study statuses."""
    module = _script_module("run_benchmark_replication")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark_replication.py", "--output-dir", str(workspace_tmp_dir)],
    )

    def fail_benchmark(*_args, **_kwargs) -> None:
        reports = ensure_reports_dir(workspace_tmp_dir)
        write_manifest(
            _manifest_with_statuses(
                {
                    "primary_study_status": StudyStatus.FAILED,
                    "benchmark_original_status": StudyStatus.FAILED,
                    "benchmark_tuned_status": StudyStatus.NOT_RUN,
                }
            ),
            reports / ArtifactName.MANIFEST,
        )
        raise RuntimeError("benchmark crashed")

    monkeypatch.setattr(module, "run_benchmark_replication", fail_benchmark)

    with pytest.raises(SystemExit) as raised:
        module.main()
    captured = capsys.readouterr()

    assert raised.value.code == 1
    assert "PRIMARY_STUDY_STATUS: failed" in captured.out
    assert "BENCHMARK_ORIGINAL_STATUS: failed" in captured.out
    assert "BENCHMARK_TUNED_STATUS: not_run" in captured.out
    assert "WORKFLOW_ERROR: benchmark: benchmark crashed" in captured.out


def test_temporal_cli_prints_failed_manifest_after_workflow_exception(
    workspace_tmp_dir: Path,
    monkeypatch,
    capsys,
) -> None:
    """Temporal CLI crashes should still report persisted temporal status."""
    module = _script_module("run_temporal_robustness")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_temporal_robustness.py", "--output-dir", str(workspace_tmp_dir)],
    )

    def fail_temporal(*_args, **_kwargs) -> None:
        reports = ensure_reports_dir(workspace_tmp_dir)
        write_manifest(
            _manifest_with_statuses(
                {
                    "temporal_robustness_status": StudyStatus.FAILED,
                    "temporal_claim_restrictions": ["high_shift_blocks_lockbox_superiority_claim"],
                }
            ),
            reports / ArtifactName.MANIFEST,
        )
        raise RuntimeError("temporal crashed")

    monkeypatch.setattr(module, "run_temporal_robustness", fail_temporal)

    with pytest.raises(SystemExit) as raised:
        module.main()
    captured = capsys.readouterr()

    assert raised.value.code == 1
    assert "TEMPORAL_ROBUSTNESS_STATUS: failed" in captured.out
    assert "CLAIM_RESTRICTION: high_shift_blocks_lockbox_superiority_claim" in captured.out
    assert "WORKFLOW_ERROR: temporal: temporal crashed" in captured.out


def test_standalone_benchmark_clis_report_only_layer_status(monkeypatch, capsys) -> None:
    """Standalone benchmark commands should not print aggregate primary status."""
    cases = (
        (
            "run_original_replication",
            "run_original_benchmark_replication",
            {"benchmark_original_status": StudyStatus.PASSED, "primary_study_status": StudyStatus.NOT_RUN},
            "BENCHMARK_ORIGINAL_STATUS: passed",
        ),
        (
            "run_benchmark_tuned",
            "run_tuned_benchmark_replication",
            {"benchmark_tuned_status": StudyStatus.PASSED, "primary_study_status": StudyStatus.NOT_RUN},
            "BENCHMARK_TUNED_STATUS: passed",
        ),
    )
    for module_name, workflow_name, result, expected_line in cases:
        module = _script_module(module_name)
        monkeypatch.setattr(sys, "argv", [f"{module_name}.py"])
        monkeypatch.setattr(module, workflow_name, lambda *_args, **_kwargs: result)

        module.main()
        captured = capsys.readouterr()

        assert expected_line in captured.out
        assert "PRIMARY_STUDY_STATUS" not in captured.out


def test_temporal_cli_strict_fails_when_temporal_study_is_not_run(monkeypatch, capsys) -> None:
    """Strict temporal CLI runs should fail when the requested temporal study cannot execute."""
    module = _script_module("run_temporal_robustness")
    monkeypatch.setattr(sys, "argv", ["run_temporal_robustness.py", "--strict"])
    monkeypatch.setattr(
        module,
        "run_temporal_robustness",
        lambda *_args, **_kwargs: {"temporal_robustness_status": StudyStatus.NOT_RUN},
    )

    with pytest.raises(SystemExit) as raised:
        module.main()

    assert raised.value.code == 1
    assert "TEMPORAL_ROBUSTNESS_STATUS: not_run" in capsys.readouterr().out


def test_temporal_cli_strict_allows_temporal_warnings(monkeypatch, capsys) -> None:
    """Strict temporal CLI runs should keep warning-level temporal evidence nonblocking."""
    module = _script_module("run_temporal_robustness")
    monkeypatch.setattr(sys, "argv", ["run_temporal_robustness.py", "--strict"])
    monkeypatch.setattr(
        module,
        "run_temporal_robustness",
        lambda *_args, **_kwargs: {
            "temporal_robustness_status": StudyStatus.WARNING,
            "claim_restrictions": ["high_shift_blocks_lockbox_superiority_claim"],
        },
    )

    module.main()
    captured = capsys.readouterr()

    assert "TEMPORAL_ROBUSTNESS_STATUS: warning" in captured.out
    assert "CLAIM_RESTRICTION: high_shift_blocks_lockbox_superiority_claim" in captured.out


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


def test_full_study_renders_report_after_temporal_failure_when_audit_allows_it(
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Temporal workflow failures should stay warning-level when benchmark evidence is valid."""
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

    def fail_temporal(**kwargs) -> None:
        reports = ensure_reports_dir(kwargs["output_dir"])
        write_manifest(
            _manifest_with_statuses(
                {
                    "primary_study_status": StudyStatus.PASSED,
                    "benchmark_original_status": StudyStatus.PASSED,
                    "benchmark_tuned_status": StudyStatus.PASSED,
                    "temporal_robustness_status": StudyStatus.FAILED,
                    "temporal_claim_restrictions": [],
                }
            ),
            reports / ArtifactName.MANIFEST,
        )
        raise RuntimeError("temporal crashed")

    monkeypatch.setattr(full_study, "run_temporal_robustness", fail_temporal)
    monkeypatch.setattr(
        full_study,
        "run_study_audit",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            errors=[],
            warnings=["temporal robustness status indicates failure"],
            claim_restrictions=[],
        ),
    )
    monkeypatch.setattr(full_study, "write_final_report", lambda *, output_dir: final_report, raising=False)

    result = run_full_study(input_dir=workspace_tmp_dir / "raw", output_dir=workspace_tmp_dir)

    assert result["temporal"]["temporal_robustness_status"] == StudyStatus.FAILED
    assert result["report_path"] == str(final_report)
    assert result["workflow_errors"] == [{"step": "temporal", "error": "temporal crashed"}]


def test_full_study_stops_after_benchmark_failure_and_skips_report(
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Benchmark failures should remain hard blockers for full-study reporting."""
    import secom.workflows.full_study as full_study

    temporal_calls = []
    report_calls = []

    def fail_benchmark(**kwargs) -> None:
        reports = ensure_reports_dir(kwargs["output_dir"])
        write_manifest(
            _manifest_with_statuses(
                {
                    "primary_study_status": StudyStatus.FAILED,
                    "benchmark_original_status": StudyStatus.FAILED,
                    "benchmark_tuned_status": StudyStatus.NOT_RUN,
                    "temporal_robustness_status": StudyStatus.NOT_RUN,
                    "temporal_claim_restrictions": [],
                }
            ),
            reports / ArtifactName.MANIFEST,
        )
        raise RuntimeError("benchmark crashed")

    monkeypatch.setattr(full_study, "run_benchmark_replication", fail_benchmark)
    monkeypatch.setattr(full_study, "run_temporal_robustness", lambda **_kwargs: temporal_calls.append(_kwargs))
    monkeypatch.setattr(
        full_study,
        "run_study_audit",
        lambda **_kwargs: SimpleNamespace(
            ok=False,
            errors=["primary study status indicates failure"],
            warnings=[],
            claim_restrictions=[],
        ),
    )
    monkeypatch.setattr(
        full_study, "write_final_report", lambda *, output_dir: report_calls.append(output_dir), raising=False
    )

    result = run_full_study(input_dir=workspace_tmp_dir / "raw", output_dir=workspace_tmp_dir)

    assert result["benchmark"]["primary_study_status"] == StudyStatus.FAILED
    assert result["temporal"]["temporal_robustness_status"] == StudyStatus.NOT_RUN
    assert result["report_path"] is None
    assert result["workflow_errors"] == [{"step": "benchmark", "error": "benchmark crashed"}]
    assert temporal_calls == []
    assert report_calls == []


def test_full_study_cli_prints_workflow_errors(monkeypatch, capsys) -> None:
    """The full-study CLI should surface child workflow exceptions in structured output."""
    module = _script_module("run_full_study")
    monkeypatch.setattr(sys, "argv", ["run_full_study.py"])
    monkeypatch.setattr(
        module,
        "run_full_study",
        lambda *_args: {
            "benchmark": {"primary_study_status": StudyStatus.PASSED},
            "benchmark_original_status": StudyStatus.PASSED,
            "benchmark_tuned_status": StudyStatus.PASSED,
            "temporal": {"temporal_robustness_status": StudyStatus.FAILED},
            "audit": SimpleNamespace(ok=True, errors=[], warnings=["temporal failed"], claim_restrictions=[]),
            "report_path": "runs/full_study/reports/final_report.md",
            "workflow_errors": [{"step": "temporal", "error": "temporal crashed"}],
        },
    )

    module.main()
    captured = capsys.readouterr()

    assert "WORKFLOW_ERROR: temporal: temporal crashed" in captured.out
    assert "FINAL_REPORT: runs/full_study/reports/final_report.md" in captured.out
