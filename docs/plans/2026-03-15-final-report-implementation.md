# Final Report Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a separate report-generation command that produces a polished `final_report.md`, six figure PNGs, and an optional PDF from existing study artifacts.

**Architecture:** Keep `src/secom/reporting.py` as the orchestration layer that loads artifacts, computes narrative summaries, and writes markdown. Add a small plotting helper module so figure code does not overwhelm markdown assembly, then expose the workflow through a new `scripts/run_final_report.py` entry point. Preserve artifact-driven behavior: the report reads only the manifest and existing CSV outputs, and surfaces missing data explicitly.

**Tech Stack:** Python, pandas, numpy, matplotlib, pytest

---

> Repo policy note: do not commit unless the user explicitly asks. The “commit” step in each task is intentionally replaced with a verification checkpoint.

### Task 1: Add the canonical final report entry point and artifact names

**Files:**
- Modify: `src/secom/config.py`
- Modify: `src/secom/reporting.py`
- Create: `scripts/run_final_report.py`
- Test: `tests/test_final_report.py`

**Step 1: Write the failing test**

```python
def test_final_report_is_generated_from_active_artifacts(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report"
    run_benchmark_replication(input_dir=synthetic_input_dir, output_dir=out_dir)
    run_temporal_robustness(input_dir=synthetic_input_dir, output_dir=out_dir)

    report_path = write_final_report(out_dir)

    assert report_path.name == ArtifactName.FINAL_REPORT
    assert report_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_final_report.py::test_final_report_is_generated_from_active_artifacts`
Expected: FAIL because `write_final_report` and `ArtifactName.FINAL_REPORT` do not exist yet.

**Step 3: Write minimal implementation**

- Add `ArtifactName.FINAL_REPORT = "final_report.md"`.
- Add `write_final_report(output_dir: Path, *, export_pdf: bool = False) -> Path` in `src/secom/reporting.py`.
- Add `scripts/run_final_report.py` mirroring the existing script pattern:

```python
from secom.reporting import write_final_report

...
out = write_final_report(Path(args.output_dir), export_pdf=args.export_pdf)
print(out)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_final_report.py::test_final_report_is_generated_from_active_artifacts`
Expected: PASS

**Step 5: Verification checkpoint**

Run: `python -m pytest -q tests/test_final_report.py`
Expected: the new file-level test suite is green for the implemented subset.

### Task 2: Replace scaffold placeholders with finished prose and provenance

**Files:**
- Modify: `src/secom/reporting.py`
- Test: `tests/test_final_report.py`
- Keep compatible: `tests/test_report_skeleton.py`

**Step 1: Write the failing tests**

```python
def test_final_report_contains_finished_narrative_sections(...):
    report_path = write_final_report(out_dir)
    text = report_path.read_text(encoding="utf-8")

    assert "## What I Built" in text
    assert "## Provenance Appendix" in text
    assert "Summarize the SECOM benchmark context" not in text
    assert "Describe the full-dataset replication protocol" not in text


def test_final_report_surfaces_required_industrialization_gaps(...):
    text = write_final_report(out_dir).read_text(encoding="utf-8")

    assert "No downstream decision or action outcome data" in text
    assert "Single-dataset evidence only" in text
    assert "deployment decision objectives and cost accounting" in text
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_final_report.py -k "finished_narrative_sections or industrialization_gaps"`
Expected: FAIL because the renderer still emits scaffold-style placeholders and incomplete industrialization framing.

**Step 3: Write minimal implementation**

Implement helper functions in `src/secom/reporting.py` to render:

- Executive summary in polished prose
- `## What I Built`
- Artifact-backed benchmark and tuned benchmark narratives
- Full industrialization gap language required by `docs/spec/05-industrialization-gap-analysis.md`
- `## Provenance Appendix` from `run_manifest.json`

Keep `write_report_skeleton` available if still needed by existing tests, but make `write_final_report` the canonical output path.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_final_report.py -k "finished_narrative_sections or industrialization_gaps"`
Expected: PASS

**Step 5: Regression checkpoint**

Run: `python -m pytest -q tests/test_report_skeleton.py`
Expected: PASS if skeleton behavior is intentionally preserved; otherwise update that test file in the same task to reflect the new canonical report surface and re-run until green.

### Task 3: Add figure generation for the six required visuals

**Files:**
- Modify: `pyproject.toml`
- Create: `src/secom/report_figures.py`
- Modify: `src/secom/reporting.py`
- Test: `tests/test_final_report.py`

**Step 1: Write the failing test**

```python
def test_final_report_writes_expected_figure_files(...):
    write_final_report(out_dir)

    figures_dir = out_dir / "reports" / "figures"
    assert (figures_dir / "benchmark_comparison.png").exists()
    assert (figures_dir / "tuned_vs_original_delta.png").exists()
    assert (figures_dir / "feature_stability.png").exists()
    assert (figures_dir / "temporal_drift.png").exists()
    assert (figures_dir / "lockbox_vs_mspc.png").exists()
    assert (figures_dir / "workload_cost_framing.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_final_report.py::test_final_report_writes_expected_figure_files`
Expected: FAIL because no figures are generated yet.

**Step 3: Write minimal implementation**

- Add `matplotlib` to `pyproject.toml` dependencies.
- Create `src/secom/report_figures.py` with small helpers such as:

```python
def write_benchmark_comparison_figure(..., output_path: Path) -> None: ...
def write_tuned_delta_figure(..., output_path: Path) -> None: ...
def write_feature_stability_figure(..., output_path: Path) -> None: ...
def write_temporal_drift_figure(..., output_path: Path) -> None: ...
def write_lockbox_vs_mspc_figure(..., output_path: Path) -> None: ...
def write_workload_cost_figure(..., output_path: Path) -> None: ...
```

- Use the non-interactive Agg backend.
- Call those helpers from `write_final_report` and reference the generated PNGs in markdown.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_final_report.py::test_final_report_writes_expected_figure_files`
Expected: PASS

**Step 5: Verification checkpoint**

Run: `python -m pytest -q tests/test_final_report.py -k figure`
Expected: PASS

### Task 4: Add optional PDF export and report the fallback cleanly

**Files:**
- Modify: `scripts/run_final_report.py`
- Modify: `src/secom/reporting.py`
- Test: `tests/test_final_report.py`

**Step 1: Write the failing tests**

```python
def test_final_report_pdf_export_is_optional_when_tool_missing(..., monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    report_path = write_final_report(out_dir, export_pdf=True)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "PDF export skipped" in text or (out_dir / "reports" / "final_report.pdf").exists() is False
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_final_report.py -k optional_when_tool_missing`
Expected: FAIL because PDF export handling does not exist yet.

**Step 3: Write minimal implementation**

- Add a small export helper that checks `shutil.which("pandoc")`.
- If `--export-pdf` is requested and `pandoc` is available, render `final_report.pdf`.
- If the tool is missing, do not fail markdown generation; instead emit a clear console message and/or append a short note in the report provenance section.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_final_report.py -k optional_when_tool_missing`
Expected: PASS

**Step 5: Verification checkpoint**

Run: `python -m pytest -q tests/test_final_report.py`
Expected: PASS

### Task 5: Update operator docs and run the adjacent verification suite

**Files:**
- Modify: `README.md`
- Modify: `scripts/run_report_skeleton.py` only if you decide to deprecate it or redirect users to the final report command
- Test: `tests/test_final_report.py`
- Test: `tests/test_report_skeleton.py`
- Test: `tests/test_study_audit.py`

**Step 1: Write the failing doc-oriented test or assertion**

If you want README coverage, add a small test that checks the CLI help text or just treat this as a docs task and verify manually.

**Step 2: Update documentation**

Document:

- the new `python scripts/run_final_report.py --output-dir runs/full_study` command
- `final_report.md` as the canonical report artifact
- `reports/figures/` as generated outputs
- optional PDF export behavior

**Step 3: Run adjacent verification**

Run: `python -m pytest -q tests/test_final_report.py tests/test_report_skeleton.py tests/test_study_audit.py`
Expected: PASS

**Step 4: Run a realistic end-to-end report generation smoke check**

Run: `python scripts/run_final_report.py --output-dir runs/full_study`
Expected: prints the path to `runs/full_study/reports/final_report.md` and writes the figure files under `runs/full_study/reports/figures/`

**Step 5: Final verification checkpoint**

Run: `python -m pytest -q tests/test_final_report.py tests/test_report_skeleton.py tests/test_study_audit.py`
Expected: PASS with no new failures before reporting completion.
