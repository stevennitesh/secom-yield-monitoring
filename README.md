# SECOM Yield Monitoring

[![CI](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml)

A reproducible benchmark study of semiconductor test pass/fail prediction and chronological robustness.

This project turns the public [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) into a reproducible Python study pipeline. It predicts recorded test pass/fail outcomes from high-dimensional manufacturing sensor data, compares benchmark and tuned models under leakage-controlled evaluation, and generates audit-ready artifacts plus a final markdown report.

The repo is designed as a defensible ML engineering case study. It emphasizes reproducibility, careful evaluation, transparent artifacts, and clear limits on what can and cannot be claimed from a public benchmark dataset.

## At A Glance

| Area | Summary |
| --- | --- |
| Domain | Semiconductor manufacturing yield monitoring |
| Business question | Can sensor data predict recorded pass/fail labels, and how robust is that association under chronological shift? |
| Dataset | UCI SECOM: 1,567 manufacturing examples, 590 measurement columns (591 in source metadata), missing values, and 104 failures |
| ML task | Imbalanced binary classification for pass/fail risk scoring |
| Primary metric | Balanced Error Rate (BER), supported by fail recall, pass specificity, ROC AUC, PR AUC, MCC, and F2 |
| Study design | Benchmark replication first, tuned benchmark second, temporal robustness as a separate stress test |
| Deliverable | Python package and CLI pipeline that produce metrics, manifests, audits, figures, and a final report |
| Stack | Python 3.11+, pandas, NumPy, SciPy, scikit-learn, skrebate, pytest, Ruff, GitHub Actions |

## What This Project Demonstrates

This repository is meant to show practical data science and ML engineering judgment:

- turning messy public manufacturing data into validated modeling inputs
- handling severe class imbalance where missed failures and false alarms have different business costs
- comparing feature-selection methods on a high-dimensional sensor problem
- preserving benchmark comparability before introducing tuned improvements
- avoiding leakage in preprocessing, feature selection, cross-validation, and model evaluation
- generating reproducible artifacts instead of one-off notebook results
- separating research evidence from production-readiness claims

## Why This Problem Matters

Yield loss creates direct margin pressure in semiconductor manufacturing. Failed units waste capacity, delayed detection slows root-cause analysis, and noisy alerts consume engineering time.

A useful monitoring model therefore has to balance two competing risks:

- catching enough true failures to support earlier investigation
- keeping false alarms low enough that the alert workload remains credible

The SECOM dataset is a useful public benchmark dataset because it exposes that tradeoff in a realistic setting: many sensor variables, few failures, missing values, and a benchmark target that makes modeling claims checkable.

## Study Design

The project is organized around four evidence layers.

| Layer | What It Does | Why It Matters |
| --- | --- | --- |
| Original benchmark replication | Reproduces the literature-style 40-feature benchmark family. | Provides a literature-style reference comparison, with the published classifier discrepancy disclosed. |
| Tuned benchmark | Tunes selectors and classifiers under the same benchmark framing. | Tests whether a better risk-scoring model is possible without changing the study target. |
| Temporal robustness | Runs chronological development and lockbox stress tests. | Checks whether signal quality changes when later production entities differ from earlier ones. |
| Industrialization-gap analysis | Documents missing deployment evidence. | Makes clear what cost, workflow, governance, and monitoring decisions would be needed before production use. |

The key design choice is claim separation. Temporal robustness can restrict operational confidence, but it does not automatically invalidate the original benchmark replication or tuned benchmark comparison.

## Current Results

A fresh full study on October 3, 2026 produced the [report](docs/results/final_report.md), [audit receipt](docs/results/evidence/audit_receipt.json) and [execution manifest](docs/results/evidence/run_manifest.json). Git keeps this small snapshot and its figures; detailed CSVs and the full artifact/source archive remain under ignored `runs/`. The artifact audit passed; temporal warnings and claim restrictions remain active. The manifest records approximately 20.9 minutes of modeling with the `vectorized` ReliefF backend and 1 scoring process(es). See [runtime measurements and equivalence checks](docs/performance.md).

Lower BER is better: it averages the failure miss rate and pass false-alarm rate. An all-pass rule has BER 0.500 despite 93.36% accuracy.

| Evidence Layer | Observed Result | Interpretation |
| --- | --- | --- |
| Original benchmark | `ReliefF` / `krr` / `with_missing_indicators`, mean BER `0.292`, TPR `0.627`, TNR `0.788` | Configurations are selected from the same non-nested CV sweep used to report them. |
| Tuned benchmark | `F-test` / `logreg` / `with_missing_indicators`, mean BER `0.309`, TPR `0.685`, TNR `0.697` | Hyperparameters are tuned within each family using nested CV; the leading family is selected retrospectively. |
| Temporal stress test | `ReliefF` with balanced logistic regression; DEV mean BER `0.471` | A separate chronological study, rather than out-of-time validation of the exact KRR model. |
| Temporal claim status | `HIGH_SHIFT`; 1 active restriction(s) | Lockbox results are descriptive and do not establish operational superiority. |

The primary scientific lockbox threshold catches 4 of 9 failures among 235 samples, with 0 false positives and 5 missed failures. The small failure count limits precision. Matched-TNR90 comparisons use retrospective ROC thresholds chosen with lockbox labels; they are diagnostic, not frozen operating performance.

The reference comparison also has a source discrepancy: [UCI](https://archive.ics.uci.edu/dataset/179/secom) describes the published 40-feature table as kernel ridge, while [Table 1 of the original paper](https://proceedings.mlr.press/v6/mccann10a/mccann10a.pdf) labels it Naive Bayes. The report keeps those values as reference context and qualifies exact replication claims.

Feature evidence separates selection stability from full-data logistic-regression coefficients. Prominent missing indicators M112/M247/M385/M519 have identical missingness masks, with an October missing rate near 89%; that association may reflect collection regime or time. Early warning, causal process drivers and intervention benefit remain unestablished.

Git dirty status is preserved as `True`. The execution manifest identifies the base commit plus exact source hashes, and the archive contains that source. A passing artifact audit is local study validation; the updated hosted CI workflow still needs a published commit to run.

## Key Engineering Choices

- **Benchmark-first workflow:** replication is treated as a separate evidence layer, not overwritten by later tuning.
- **Leakage-controlled modeling:** imputation, scaling, feature selection, and model fitting stay inside the evaluation flow.
- **Imbalance-aware metrics:** BER is the primary metric because pass/fail classes are highly imbalanced.
- **Audit-friendly outputs:** runs produce manifests, artifact checks, figures, and report files for traceability.
- **Explicit claim boundaries:** the generated report distinguishes benchmark findings, temporal stress-test warnings, and production gaps.

## How To Review This Repo

For a quick project review:

1. Read this README for the problem framing, study structure, and claim boundaries.
2. Open `docs/results/final_report.md` for the generated result narrative and figures.
3. Inspect `src/secom/workflows/benchmark_replication.py` and `src/secom/workflows/benchmark_tuned.py` for orchestration.
4. Inspect `src/secom/selection/engine.py`, `src/secom/metrics.py`, and `src/secom/io.py` for the core ML and data-quality logic.
5. Read `tests/test_benchmark_replication.py`, `tests/test_metrics_threshold_optimization.py`, and `tests/test_io.py` for representative regression coverage.
6. Run the direct Python checks under Run Locally, or optional `make check`, to verify Ruff linting, Ruff formatting, and pytest.

For a deeper technical review:

- read the active study specs under `docs/spec/`
- inspect the curated evidence snapshot under `docs/results/`
- run the full study pipeline and compare the regenerated `runs/full_study/reports/final_report.md`
- inspect `docs/results/evidence/run_manifest.json` and the audit output for artifact provenance
- compare the benchmark, tuned benchmark, and temporal robustness sections without merging their claims

## Repository Map

| Path | What It Contains |
| --- | --- |
| `src/secom/` | Study package for parsing, preprocessing, feature selection, metrics, workflows, audits, and reporting |
| `scripts/` | CLI entry points for each study layer and report-generation path |
| `tests/` | Regression tests for parsing, metrics, selectors, workflows, audit rules, and report output |
| `docs/spec/` | Canonical study contract, artifact schemas, report structure, and claim semantics |
| `docs/plans/` | Historical implementation plans for the report design |
| `docs/results/` | Small public snapshot: final report, figures, execution manifest, and audit receipt |
| `runs/` | Generated active study outputs; intentionally gitignored so results can be regenerated cleanly |

## Run Locally

Use Python 3.11 or 3.12. Installation and checks work directly through Python on Windows, Linux and macOS; Make targets are optional shortcuts.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python scripts/fetch_secom.py --output-dir data/raw
```

The fetch command obtains the [official UCI archive](https://archive.ics.uci.edu/static/public/179/secom.zip), verifies the two modeling files against fixed SHA-256 values and reuses existing verified files. It refuses to overwrite different local data. SECOM is attributed to McCann and Johnston through [UCI, DOI 10.24432/C54305](https://archive.ics.uci.edu/dataset/179/secom), under the dataset's CC BY 4.0 license. This is dataset attribution, not a repository software license.

The raw files are `data/raw/secom.data` and `data/raw/secom_labels.data`. Observed shape is 1,567 × 590; UCI and the original paper describe 591 features. Rows are production entities of unspecified physical unit, and pre-outcome measurement availability is unproven.

The `data/` directory is intentionally gitignored. The repository stores the study code and contracts, not the external dataset files.

Run the full study into a fresh output directory. The evidence run includes KRR plus a balanced logistic-regression association comparator; the default without `--classifiers` remains KRR. CLI commands default `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before numeric imports, while preserving explicit environment settings. ReliefF uses a NumPy scoring adapter verified against pinned skrebate 0.8.4, in one process, with a bounded cache of identical training rankings. Set `SECOM_RELIEF_BACKEND=reference` to run the upstream implementation; `SECOM_RELIEF_N_JOBS` controls its worker count (default at most four). The manifest records the chosen backend, duration and effective thread settings. `--progress` shows benchmark and temporal stages. See [runtime measurements](docs/performance.md).

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --classifiers krr,logreg --progress --strict
```

When the full-study audit passes, the canonical generated report is written to:

```text
runs/full_study/reports/final_report.md
```

The checked-in public evidence snapshot lives at:

```text
docs/results/final_report.md
```

After reviewing a completed run, export the audited report, figures and small provenance receipts with:

```bash
python scripts/export_results.py --output-dir runs/full_study
```

The exporter preserves execution provenance and rejects changed source or mismatched artifacts. Full CSVs stay in ignored run storage, and the complete source/artifact ZIP is saved to `runs/full_study/evidence/study_artifacts.zip`. The public snapshot excludes those large files.

## Useful Commands

The commands below assume the virtual environment is active.

| Task | Command |
| --- | --- |
| Original benchmark replication | `python scripts/run_original_replication.py --input-dir data/raw --output-dir runs/original_replication --strict` |
| Tuned benchmark | `python scripts/run_benchmark_tuned.py --input-dir data/raw --output-dir runs/benchmark_tuned --strict` |
| Benchmark bundle | `python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --strict` |
| Temporal robustness study | `python scripts/run_temporal_robustness.py --input-dir data/raw --output-dir runs/temporal_robustness --strict` |
| Audit generated artifacts | `python scripts/run_audit.py --output-dir runs/full_study --strict` |
| Regenerate final report | `python scripts/run_final_report.py --output-dir runs/full_study` |
| Optional PDF export | `python scripts/run_final_report.py --output-dir runs/full_study --export-pdf` |
| Lint only | `make lint` |
| Format check only | `make format-check` |
| Tests only | `make test` |

## Generated Artifacts

Active outputs are written under `runs/<study>/reports/`.

Key generated files include:

- `run_manifest.json` for provenance
- `final_report.md` for the canonical generated report
- `figures/*.png` for report figures
- `benchmark_*` artifacts for the original benchmark layer
- `benchmark_tuned_*` artifacts for the tuned benchmark layer
- `temporal_*` artifacts for the temporal robustness layer
- audit entries classified as `ERROR`, `WARNING`, or `CLAIM_RESTRICTION`

`final_report_skeleton.md` may also be generated as a scaffold/debugging aid, but `final_report.md` is the report artifact to review.

The repository also keeps a curated public snapshot under `docs/results/` so readers can review the latest committed evidence without regenerating the full study.

## Limitations And Next Data Needed

This project is intentionally careful about claim boundaries. The current study can compare benchmark methods and stress-test temporal robustness, but production deployment would still require:

- downstream action and outcome data showing whether alerts changed process decisions
- explicit cost ratios for missed failures, false alarms, inspection time, and scrap or rework
- operating-point approval from process engineering or business owners
- monitoring for data drift, calibration drift, and alert workload
- validation on additional products, tools, or manufacturing time periods

Those gaps are part of the analysis, not hidden caveats.

## Technical Reference

The active study contract lives in `docs/spec/`:

1. `docs/spec/01-study-goal.md`
2. `docs/spec/02-benchmark-replication-study.md`
3. `docs/spec/03-feature-stability-and-interpretation.md`
4. `docs/spec/04-temporal-robustness-study.md`
5. `docs/spec/05-industrialization-gap-analysis.md`
6. `docs/spec/06-report-structure.md`
7. `docs/spec/07-artifact-contracts.md`
8. `docs/spec/08-audit-and-claim-semantics.md`
