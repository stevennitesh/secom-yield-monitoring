# SECOM Yield Monitoring

[![CI](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/stevennitesh/secom-yield-monitoring/actions/workflows/ci.yml)

**Can manufacturing measurements identify failures, and do those patterns hold in later samples?**

A reproducible Python study of 1,567 semiconductor production records, 590 anonymous measurements and 104 recorded failures from [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom). It compares a fixed-budget benchmark, bounded tuning and transfer to later samples.

**[Read the study report →](https://stevennitesh.github.io/secom-yield-monitoring/)**

The report brings together the methods, six charts, findings, limitations and downloadable technical evidence.

## Main Finding

On the same ten held-out folds, tuning lowered mean **balanced error from 31.43% to 30.86%**. Balanced error averages missed-failure and false-alert rates; lower is better.

Across 104 failures and 1,463 passes, the tuned procedure produced **74 fewer false alerts and caught four fewer failures**: false alerts fell from 445 to 371, while failures caught fell from 70 to 66.

![Complete reference and tuned procedures: balanced error, failures caught and false alerts](docs/results/figures/benchmark_comparison.png)

Rates average test folds; counts pool predictions. Whiskers show observed fold ranges, not confidence intervals.

Chronological tests showed weak transfer and fragile calibration. The final later block is retrospective. Anonymous measurements with unknown collection timing do not establish early warning or production readiness.

## What Was Built

- Nested evaluation with training-only preprocessing and feature selection.
- Chronological fitting, separate threshold calibration and later-sample evaluation.
- Vectorized ReliefF scoring and bounded caches, with [measured equivalence and speed](docs/performance.md).
- Saved predictions, input/source hashes, artifact audits and reproducible reporting.

**Stack:** Python · NumPy/pandas · SciPy · scikit-learn/skrebate · Matplotlib · pytest/Ruff.

The custom work covers study orchestration, selection, calibration, acceleration and evidence checks. Model estimators come from scikit-learn; ReliefF builds on skrebate.

## Run Locally

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Quick verification uses synthetic fixtures and needs no dataset download:

```bash
python -m pytest -q tests/test_relief_backend.py tests/test_model_score_cache.py
```

For dataset acquisition, the complete study and audit commands, see [Reproduce the study](docs/development.md#reproduce-the-study). Raw data and full runs stay gitignored.

[Documentation map](docs/README.md) | [Developer guide](docs/development.md) | [Study specifications](docs/spec/README.md)

Dataset: McCann and Johnston (2008), UCI, DOI 10.24432/C54305, CC BY 4.0. Software: [MIT](LICENSE).
