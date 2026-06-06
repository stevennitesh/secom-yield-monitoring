# SECOM Yield Monitoring Study

A benchmark-first machine learning case study for semiconductor yield risk monitoring.

The project uses the UCI SECOM dataset to predict pass/fail manufacturing outcomes from high-dimensional sensor data, replicate the original public benchmark, improve the benchmark with tuned models, and separate research evidence from operational claims. The emphasis is not only model accuracy; it is disciplined evaluation, reproducible artifacts, and clear business interpretation.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Business problem | Identify wafers at elevated failure risk so process teams can prioritize investigation and reduce yield loss. |
| Dataset | [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom): 1,567 manufacturing examples, 591 real-valued process features, missing values, and 104 failures. |
| Modeling challenge | Severe class imbalance, many noisy or redundant sensor signals, missingness, and limited failure examples. |
| Primary metric | Balanced Error Rate (BER), supported by True+ / fail recall, True- / pass specificity, ROC AUC, PR AUC, MCC, and F2. |
| Study style | Original benchmark replication first, tuned benchmark second, temporal robustness as a separate stress test. |
| Main deliverable | A reproducible Python study pipeline that generates benchmark artifacts, audit results, figures, and a final markdown report. |

## What This Demonstrates

This repo is designed to show practical data science and ML engineering judgment:

- converting messy public manufacturing data into a validated modeling dataset
- handling imbalanced classification where missed failures and false alarms have different business costs
- comparing feature-selection methods on a high-dimensional sensor problem
- preserving benchmark comparability before introducing tuned improvements
- avoiding leakage in preprocessing, feature selection, cross-validation, and model evaluation
- producing audit-friendly artifacts instead of one-off notebook results
- stating industrialization limits clearly instead of overstating production readiness

## Why This Problem Matters

Manufacturing yield problems create direct margin pressure: failed units waste production capacity, late detection slows root-cause analysis, and noisy alerts consume engineering time. A useful monitoring model must therefore balance two risks:

- catching enough true failures to support early intervention
- keeping false alarms low enough that investigation workload remains credible

The SECOM dataset is a good portfolio problem because it forces that tradeoff in a realistic setting: many sensor variables, few failures, missing values, and a public benchmark that makes the modeling claims checkable.

## Study Design

The project is organized around four evidence layers.

| Layer | Purpose | Business Question |
| --- | --- | --- |
| Original benchmark replication | Reproduce the literature-style 40-feature benchmark family. | Can the project match the known baseline before claiming improvements? |
| Tuned benchmark | Tune selectors and classifiers under the same benchmark framing. | Does tuning improve the risk-scoring model without changing the comparison target? |
| Temporal robustness | Stress-test the workflow with chronological development and lockbox splits. | Would the signal hold up when future wafers differ from historical wafers? |
| Industrialization-gap analysis | Document what is still missing for deployment. | What decisions, cost data, monitoring, and governance would be required before production use? |

The key design choice is separation of claims. Temporal robustness can limit operational confidence, but it does not erase the value of the original benchmark replication or tuned benchmark comparison.

## Modeling Approach

The active benchmark pipeline includes:

- raw file validation for ragged rows, label values, timestamps, and missing labels
- deterministic row ordering by timestamp and raw row id
- leakage-controlled imputation, optional missingness indicators, scaling, and feature selection
- feature selectors: S2N, pooled T-test, F-test, Pearson, ReliefF, and Gram-Schmidt
- classifiers: kernel ridge regression and logistic regression
- nested/tuned benchmark paths with cached selector transformations for runtime efficiency
- bootstrap uncertainty summaries for fold-level metrics
- manifest, audit, and report generation for provenance and claim control

## Repository Map

| Path | What It Contains |
| --- | --- |
| `src/secom/` | Production-style study package: parsing, preprocessing, feature selection, metrics, workflows, audits, and reporting. |
| `scripts/` | CLI entry points for running each study layer and generating reports. |
| `tests/` | Regression tests for parsing, metrics, selectors, benchmark workflows, audit rules, and report output. |
| `docs/spec/` | Canonical study contract, artifact schemas, report structure, and claim semantics. |
| `docs/plans/` | Historical implementation plans for the report design. |
| `reports/` | Historical tracked artifacts from the earlier pre-reframe lane; useful context, not the active output contract. |
| `runs/` | Generated active study outputs. This directory is intentionally gitignored so results can be regenerated cleanly. |

## How To Review This Repo

For a quick hiring review:

1. Read this README for the problem framing and project scope.
2. Inspect `src/secom/workflows/benchmark_replication.py` and `src/secom/workflows/benchmark_tuned.py` for the study orchestration.
3. Inspect `src/secom/selection/engine.py`, `src/secom/metrics.py`, and `src/secom/io.py` for the core ML/data-quality logic.
4. Read `tests/test_benchmark_replication.py`, `tests/test_metrics_threshold_optimization.py`, and `tests/test_io.py` for the regression surface.
5. Run `make PYTHON=.venv/bin/python check` to verify lint, formatting, and tests.

For a deeper technical review, read the active specs in this order:

1. `docs/spec/01-study-goal.md`
2. `docs/spec/02-benchmark-replication-study.md`
3. `docs/spec/03-feature-stability-and-interpretation.md`
4. `docs/spec/04-temporal-robustness-study.md`
5. `docs/spec/05-industrialization-gap-analysis.md`
6. `docs/spec/06-report-structure.md`
7. `docs/spec/07-artifact-contracts.md`
8. `docs/spec/08-audit-and-claim-semantics.md`

## Setup

Create and populate a local virtual environment:

```bash
python -m venv .venv
make PYTHON=.venv/bin/python install
```

Run the standard local gate:

```bash
make PYTHON=.venv/bin/python check
```

Useful focused commands:

```bash
make PYTHON=.venv/bin/python lint
make PYTHON=.venv/bin/python format-check
make PYTHON=.venv/bin/python format
make PYTHON=.venv/bin/python test
make PYTHON=.venv/bin/python coverage
```

## Data

Download the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) and place the raw files under `data/raw/`:

```text
data/raw/secom.data
data/raw/secom_labels.data
```

The `data/` directory is intentionally gitignored. The repository stores the study code and contracts, not the external dataset files.

## Running The Study

Original benchmark replication:

```bash
python scripts/run_original_replication.py --input-dir data/raw --output-dir runs/original_replication --strict
```

Use this command for the UCI original 40-feature benchmark comparison. The original replication default uses KRR with the UCI selector family, including pooled T-test and Pearson.

Tuned benchmark:

```bash
python scripts/run_benchmark_tuned.py --input-dir data/raw --output-dir runs/benchmark_tuned --strict
```

The tuned benchmark defaults to KRR for the primary improved-benchmark pass. To explicitly run the full classifier family:

```bash
python scripts/run_benchmark_tuned.py --input-dir data/raw --output-dir runs/benchmark_tuned --classifiers krr,logreg --strict
```

Benchmark study bundle:

```bash
python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --strict
```

The bundle uses original-replication defaults for the original layer and tuned KRR defaults for the tuned layer. To force both classifiers through both layers:

```bash
python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --classifiers krr,logreg --strict
```

Secondary temporal study:

```bash
python scripts/run_temporal_robustness.py --input-dir data/raw --output-dir runs/temporal_robustness --strict
```

Full study bundle:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --strict
```

When the full-study audit passes, this command also writes the canonical
`runs/full_study/reports/final_report.md`.

Audit-generated artifacts:

```bash
python scripts/run_audit.py --output-dir runs/full_study --strict
```

Regenerate the canonical markdown report from an existing audited run:

```bash
python scripts/run_final_report.py --output-dir runs/full_study
```

Optional PDF export:

```bash
python scripts/run_final_report.py --output-dir runs/full_study --export-pdf
```

## Output Model

Generated active outputs are written under `runs/<study>/reports/`.

Original benchmark artifacts use the `benchmark_*` family plus:

- `feature_stability.csv`
- `feature_report.csv`

Tuned benchmark artifacts use the `benchmark_tuned_*` family plus:

- `benchmark_tuned_feature_stability.csv`
- `benchmark_tuned_feature_report.csv`

Temporal robustness artifacts use the `temporal_*` family.

Shared report outputs include:

- `run_manifest.json`
- `final_report.md`
- `final_report_skeleton.md`
- `figures/*.png`

`final_report.md` is the canonical generated report artifact. `final_report_skeleton.md` remains available as a scaffold/debugging aid.

Audit output distinguishes:

- `ERROR`
- `WARNING`
- `CLAIM_RESTRICTION`

## Limitations And Next Data Needed

This project is intentionally careful about claim boundaries. The current study can compare benchmark methods and stress-test temporal robustness, but production deployment would still require:

- downstream action/outcome data showing whether alerts changed process decisions
- explicit cost ratios for missed failures, false alarms, inspection time, and scrap/rework
- operating-point approval from process engineering or business owners
- monitoring for data drift, calibration drift, and alert workload
- validation on additional products, tools, or manufacturing time periods

Those gaps are part of the analysis, not hidden caveats.
