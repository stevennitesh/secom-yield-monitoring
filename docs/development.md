# Development and Evidence Workflow

[README](../README.md#run-locally) owns setup and quick verification; this guide owns study and publication commands. [AGENTS](../AGENTS.md) owns scope and guardrails; [specifications](spec/README.md) own scientific requirements.

## Code Map

Paths below are relative to `src/secom/` and `tests/`.

| Responsibility | Implementation | Focused tests |
| --- | --- | --- |
| Input validation and provenance | `io.py`, `qa.py`, `provenance.py` | `test_io.py`, `test_provenance_and_export.py` |
| Training-only transformations and selection | `preprocess.py`, `selection/engine.py`, `feature_select/` | `test_selection.py`, `test_univariate_selectors.py`, `test_relief_backend.py` |
| Model scores, thresholds and metrics | `models.py`, `metrics.py`, `common/thresholds.py` | `test_models.py`, `test_metrics_threshold_optimization.py` |
| Shared original/tuned grids and bounded score reuse | `workflows/benchmark_*.py` | `test_benchmark_replication.py`, `test_model_score_cache.py`, `test_study_preparation_reuse.py` |
| Chronology, KRR comparison and calibration | `cv.py`, `selection/tuning.py`, `workflows/temporal_*.py`, `workflows/calibration.py` | `test_temporal_robustness.py`, `test_tuning_temporal_extension.py` |
| Artifact contracts and audit | `config.py`, `artifacts.py`, `workflows/audit.py` | `test_artifact_contracts.py`, `test_study_audit.py` |
| Report language, figures and offline HTML | `reporting.py`, `report_language.py`, `report_figures.py`, `html_report.py` | `test_final_report.py`, `test_report_figures.py`, `test_report_skeleton.py`, `test_html_report.py` |
| Full workflow, metadata and export | `workflows/full_study.py`, `common/meta.py`, `evidence.py` | `test_cli_entrypoints.py`, `test_end_to_end_study.py`, `test_metadata.py`, `test_provenance_and_export.py` |
| Verified static publication | `scripts/stage_pages.py`, `.github/workflows/pages.yml` (repository root) | `test_pages_staging.py` |

Thin entrypoints live in `scripts/`. Use configuration and the auditor for active artifact schemas, rather than inferring them from historical CSVs.

## Choose Checks by the Change

Use the active environment; Windows can call `.venv/Scripts/python.exe` directly.

| Changed surface | Verification |
| --- | --- |
| Documentation | Links/anchors, commands against CLI help, evidence wording and `git diff --check`; no training |
| Study code | Relevant tests above; substantial changes also require the full checks below and supported entrypoint smoke checks |
| Identity or export | Metadata, provenance/export and presentation rejection tests |
| Runtime | Measured equivalence at [the performance owner](performance.md); broader timing only when justified by the optimization |

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
```

Make is optional. [CI](../.github/workflows/ci.yml) runs explicit checks on Ubuntu Python 3.11/3.12 and Windows Python 3.12. Local checks do not establish hosted success; synthetic CLI tests do not establish real-data study results.

## Operations and Their Effects

Use the [study command table](#checks-and-study-commands).

| Operation | Effect and boundary |
| --- | --- |
| Full study | Fits models, audits and renders; requires authorization, a fresh directory, both classifiers and `--progress --strict` |
| Strict artifact audit | Reads a completed run; temporal warnings/restrictions can remain after a pass |
| Report regeneration | Fits no models but replaces generated report/figure bytes; use deliberately |
| Public export | Deliberate requested snapshot replacement; requires audited artifacts and matching current source/spec identity |

The full workflow rejects nonempty `reports/`. Preserve completed and interrupted runs. CSVs and the complete source/artifact ZIP stay under ignored `runs/`; raw data stay under ignored `data/`. The curated snapshot keeps the technical report, six figures, manifest and audit receipt, plus an offline HTML companion and its own rendering provenance. Manual public files are retained. The HTML supports browser printing; optional technical-report PDF export requires an external Pandoc/PDF environment.

## Reproduce the Study

After the [local setup](../README.md#run-locally), fetch the verified dataset and use a fresh output directory:

```bash
python scripts/fetch_secom.py --output-dir data/raw
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --classifiers krr,logreg --progress --strict
python scripts/run_audit.py --output-dir runs/full_study --strict
```

Read `runs/full_study/reports/final_report.md` when it finishes. The latest full run took about 24 minutes on one Windows machine; this is an observed budget, not a hardware-independent guarantee. Running the study does not publish its results.

## Checks and Study Commands

Commands assume the environment is active. Benchmark entrypoints default to kernel ridge; the full-study command above includes both models.

| Task | Command |
| --- | --- |
| Full test suite | `python -m pytest -q` |
| Lint | `python -m ruff check src tests scripts` |
| Format check | `python -m ruff format --check src tests scripts` |
| Reference benchmark | `python scripts/run_original_replication.py --input-dir data/raw --output-dir runs/original_replication --strict` |
| Tuned benchmark | `python scripts/run_benchmark_tuned.py --input-dir data/raw --output-dir runs/benchmark_tuned --strict` |
| Both benchmarks | `python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --strict` |
| Chronological study | `python scripts/run_temporal_robustness.py --input-dir data/raw --output-dir runs/temporal_robustness --strict` |
| Regenerate its report | `python scripts/run_final_report.py --output-dir runs/full_study` |
| Deliberately replace the public snapshot | `python scripts/export_results.py --output-dir runs/full_study` |

## Offline HTML Report

Open `docs/results/index.html` in a browser. Styles, charts and the technical-report/manifest/audit downloads are embedded; no server or network is needed for the main report. The adjacent `html_provenance.json` records the verified input and renderer hashes separately from scientific execution.

To rebuild from the curated evidence without fitting models, use a fresh destination:

```bash
python scripts/run_html_report.py --input-dir docs/results --output-dir runs/html_review
```

The builder rejects changed core hashes, failed audits, overlapping paths and reused destinations. Normal and editorial public exports rebuild HTML from their new staged evidence automatically. For an authorized HTML-only refresh, validate the fresh folder and copy only `index.html` and `html_provenance.json` into `docs/results/`; preserve all eight core files. This does not refresh the scientific run or its historical Markdown rendering identity.

## Hosted Report

The main reader report is [hosted on GitHub Pages](https://stevennitesh.github.io/secom-yield-monitoring/). Its source remains `docs/results/index.html`; the hosted and offline versions use identical bytes.

The [Pages workflow](../.github/workflows/pages.yml) deploys from `main` when report or publication files change, or when manually dispatched on `main`. Pages uses the GitHub Actions publishing source. Before uploading, the standard-library staging command verifies the scientific receipt, all core input hashes and the HTML's separate input/output provenance:

```bash
python scripts/stage_pages.py --input-dir docs/results --output-dir .tmp/pages-review
```

Use a fresh destination. Only the HTML, its provenance, canonical Markdown, six figures, manifest and audit receipt are copied, plus an empty `.nojekyll` marker. The workflow fits no models, regenerates no evidence and uploads no raw data, local run archives or working notes. The repository's About website points to the same report.

## Documentation Changes and Historical Provenance

| Included in execution identity | Outside that identity |
| --- | --- |
| LF-normalized `pyproject.toml`, `requirements.txt`, Python under `src/secom/` and `scripts/`, all Markdown under `docs/spec/` | AGENTS, root README, general documentation, tests and CI |

The eight ordered specifications also have a separate study hash. Even prose edits there change identity. Preserve historical reports, CSVs, manifests and receipts; a passing audit does not make an older run current-source evidence. Never weaken identity checks or relabel a manifest.

To audit historical evidence without training, extract its exact local archive and run:

```bash
python <extracted-directory>/source/scripts/run_audit.py --output-dir <extracted-directory> --strict
```

New scientific source/spec changes require a fresh authorized study for updated evidence. The narrow editorial exception below preserves the original execution.

## Presentation-Only Refresh

When clearer report/charts are requested without new experiments, choose an **unused** presentation destination:

```bash
python scripts/export_results.py --output-dir runs/20261004_tuned_temporal_full_01 --presentation-output-dir runs/reader_refresh
```

| Before output mutation | Required check |
| --- | --- |
| Scientific evidence | Unchanged CSV hashes and byte-identical manifest against the executed archive |
| Archived execution | Exact source inventory and ordered spec identity |
| Current differences | Only enumerated rendering/export owners and specs06/07/08 |
| Destination | Fresh and disjoint from existing evidence |

Model code, scientific specs01-05, dependencies, metrics, grids and splits cannot use this exception. Normal export still rejects source mismatch. The report-only CLI supports a fresh presentation destination without publishing.

The local editorial archive streams existing CSVs and retains original source under `source/`, current renderer under `presentation_source/`, and separate provenance. It excludes raw data and persisted matrices. Verify focused report/figure/export/metadata/CLI tests, lint/format, saved-result agreement, all six images, links and predecessor hashes. Preserve manual files and old runs/archives.

[Current evidence](results/README.md#evidence) records execution and rendering status; [engineering checks](engineering.md) summarize implemented safeguards.

If later scientific-spec prose prevents a presentation refresh, use an isolated copy of the execution archive's source and overlay only the permitted editorial owners. Run that copy's exporter so the loaded code matches its recorded rendering identity. Keep the current scientific specifications and the historical execution unchanged; do not broaden the exception or relabel a manifest.
