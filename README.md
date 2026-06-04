# SECOM Benchmark-First Study

This repository is organized around four explicit study layers:

1. Original benchmark replication as the faithful literature-style study
2. Tuned benchmark as the improved benchmark study
3. Temporal robustness as a secondary stress-test study
4. Industrialization-gap analysis as a required interpretation layer

The active canonical spec lives under:

- `docs/spec/`

The rebuilt active surface now supports:

1. original benchmark replication artifact generation
2. tuned benchmark artifact generation
3. temporal robustness artifact generation
4. categorized study audit output
5. polished markdown final-report generation from active artifacts
6. optional PDF export from the generated markdown report

The intent is deliberate:

1. show the original benchmark faithfully
2. show how much a tuned version improves it
3. show what still breaks under temporal stress
4. show what a real industrial study would still require

## Development

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

## Active Entry Points

Original replication:

```bash
python scripts/run_original_replication.py --input-dir data/raw --output-dir runs/original_replication --strict
```

Use this command for the UCI original 40-feature benchmark comparison. The original replication default includes the UCI selector family, including Pearson.

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

Audit:

```bash
python scripts/run_audit.py --output-dir runs/benchmark_replication --strict
```

Full study bundle:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --strict
```

Report scaffold from existing artifacts:

```bash
python scripts/run_report_skeleton.py --output-dir runs/full_study
```

Canonical final report from existing artifacts:

```bash
python scripts/run_final_report.py --output-dir runs/full_study
```

Optional PDF export:

```bash
python scripts/run_final_report.py --output-dir runs/full_study --export-pdf
```

## Current Output Model

Original replication artifacts use the `benchmark_*` family plus:

- `feature_stability.csv`
- `feature_report.csv`

Tuned benchmark artifacts use the `benchmark_tuned_*` family plus:

- `benchmark_tuned_feature_stability.csv`
- `benchmark_tuned_feature_report.csv`

Shared outputs:

- `run_manifest.json`
- `final_report.md`
- `final_report_skeleton.md`
- `figures/*.png`

Secondary temporal artifacts use `temporal_*` names.

`final_report.md` is the canonical generated report artifact. `final_report_skeleton.md` remains available as a scaffold/debugging aid.

Audit output distinguishes:

- `ERROR`
- `WARNING`
- `CLAIM_RESTRICTION`

The final generated report is designed to read like a professional study draft:

- original replication first
- tuned benchmark second
- original vs tuned comparison third
- feature stability and interpretation fourth
- temporal robustness fifth
- industrialization gaps and conclusions stated explicitly

## Canonical Reading Order

1. `docs/spec/01-study-goal.md`
2. `docs/spec/02-benchmark-replication-study.md`
3. `docs/spec/03-feature-stability-and-interpretation.md`
4. `docs/spec/04-temporal-robustness-study.md`
5. `docs/spec/05-industrialization-gap-analysis.md`
6. `docs/spec/06-report-structure.md`
7. `docs/spec/07-artifact-contracts.md`
8. `docs/spec/08-audit-and-claim-semantics.md`
