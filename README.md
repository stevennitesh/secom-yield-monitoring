# SECOM Yield Monitoring

[![CI](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml)

Production-style machine learning study for semiconductor yield risk monitoring.

This project turns the public [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) into a reproducible Python study pipeline. It predicts wafer pass/fail outcomes from high-dimensional manufacturing sensor data, compares benchmark and tuned models under leakage-controlled evaluation, and generates audit-ready artifacts plus a final markdown report.

The repo is designed as a defensible ML engineering case study. It emphasizes reproducibility, careful evaluation, transparent artifacts, and clear limits on what can and cannot be claimed from a public benchmark dataset.

## At A Glance

| Area | Summary |
| --- | --- |
| Domain | Semiconductor manufacturing yield monitoring |
| Business question | Can sensor data identify wafers at elevated failure risk early enough to prioritize engineering review? |
| Dataset | UCI SECOM: 1,567 manufacturing examples, 591 process features, missing values, and 104 failures |
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

The SECOM dataset is a useful public portfolio problem because it exposes that tradeoff in a realistic setting: many sensor variables, few failures, missing values, and a benchmark target that makes modeling claims checkable.

## Study Design

The project is organized around four evidence layers.

| Layer | What It Does | Why It Matters |
| --- | --- | --- |
| Original benchmark replication | Reproduces the literature-style 40-feature benchmark family. | Establishes that the project can match the known comparison target before claiming improvements. |
| Tuned benchmark | Tunes selectors and classifiers under the same benchmark framing. | Tests whether a better risk-scoring model is possible without changing the study target. |
| Temporal robustness | Runs chronological development and lockbox stress tests. | Checks whether signal quality changes when future wafers differ from historical wafers. |
| Industrialization-gap analysis | Documents missing deployment evidence. | Makes clear what cost, workflow, governance, and monitoring decisions would be needed before production use. |

The key design choice is claim separation. Temporal robustness can restrict operational confidence, but it does not automatically invalidate the original benchmark replication or tuned benchmark comparison.

## Current Results

A regenerated full-study run on June 6, 2026 produced the canonical report at `runs/full_study/reports/final_report.md`. Lower BER is better because it averages the error rate across the failure and pass classes.

| Evidence Layer | Headline Result | How To Read It |
| --- | --- | --- |
| Original benchmark replication | Best row: `ReliefF` + `krr` with missing indicators, mean BER `0.292` | The benchmark protocol finds a credible yield-risk signal in the SECOM measurements. |
| Tuned benchmark | Best row: `ReliefF` + `krr` in strict mode, mean BER `0.319` | This is the more conservative benchmark estimate because hyperparameters are selected inside nested cross-validation. |
| Temporal robustness | Primary chronological candidate: `ReliefF`, mean BER `0.471` | The future-looking stress test is much harder than the benchmark setting. |
| Temporal claim status | `HIGH_SHIFT` drift gate with one active claim restriction | Lockbox results remain useful diagnostics, but not confirmatory proof of operational superiority. |

Plain-language interpretation:

- The benchmark studies support the core project claim: SECOM sensor data contains usable signal for yield-risk modeling.
- The tuned benchmark is intentionally stricter than the original replication, so its slightly worse BER is not a regression; it is a more conservative estimate.
- The temporal study warns that future wafers look materially different from the development period. The development failure rate was `7.13%`, while the lockbox failure rate was `3.83%`; the score-distribution KS p-value was `3.79e-08`, max PSI was `5.125`, and median PSI was `0.569`.
- Because of that shift, the report does not claim production readiness. It reports the lockbox evidence as descriptive stress-test evidence and keeps deployment requirements explicit.

## Key Engineering Choices

- **Benchmark-first workflow:** replication is treated as a separate evidence layer, not overwritten by later tuning.
- **Leakage-controlled modeling:** imputation, scaling, feature selection, and model fitting stay inside the evaluation flow.
- **Imbalance-aware metrics:** BER is the primary metric because pass/fail classes are highly imbalanced.
- **Audit-friendly outputs:** runs produce manifests, artifact checks, figures, and report files for traceability.
- **Explicit claim boundaries:** the generated report distinguishes benchmark findings, temporal stress-test warnings, and production gaps.

## How To Review This Repo

For a fast hiring review:

1. Read this README for the problem framing, study structure, and claim boundaries.
2. Inspect `src/secom/workflows/benchmark_replication.py` and `src/secom/workflows/benchmark_tuned.py` for orchestration.
3. Inspect `src/secom/selection/engine.py`, `src/secom/metrics.py`, and `src/secom/io.py` for the core ML and data-quality logic.
4. Read `tests/test_benchmark_replication.py`, `tests/test_metrics_threshold_optimization.py`, and `tests/test_io.py` for representative regression coverage.
5. Run `make check` after local setup to verify Ruff linting, Ruff formatting, and pytest.

For a deeper technical review:

- read the active study specs under `docs/spec/`
- run the full study pipeline and open `runs/full_study/reports/final_report.md`
- inspect `runs/full_study/reports/run_manifest.json` and the audit output for artifact provenance
- compare the benchmark, tuned benchmark, and temporal robustness sections without merging their claims

## Repository Map

| Path | What It Contains |
| --- | --- |
| `src/secom/` | Study package for parsing, preprocessing, feature selection, metrics, workflows, audits, and reporting |
| `scripts/` | CLI entry points for each study layer and report-generation path |
| `tests/` | Regression tests for parsing, metrics, selectors, workflows, audit rules, and report output |
| `docs/spec/` | Canonical study contract, artifact schemas, report structure, and claim semantics |
| `docs/plans/` | Historical implementation plans for the report design |
| `runs/` | Generated active study outputs; intentionally gitignored so results can be regenerated cleanly |

## Run Locally

Create a Python 3.11+ virtual environment and install the project. Use any supported interpreter; replace `python3.11` with `python3.12` if that is your local version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

Run the standard local gate:

```bash
make check
```

Download the UCI SECOM data and place the raw files under `data/raw/`:

```text
data/raw/secom.data
data/raw/secom_labels.data
```

The `data/` directory is intentionally gitignored. The repository stores the study code and contracts, not the external dataset files.

Run the full study:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --strict
```

When the full-study audit passes, the canonical generated report is written to:

```text
runs/full_study/reports/final_report.md
```

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
