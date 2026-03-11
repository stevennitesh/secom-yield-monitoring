# SECOM Benchmark-First Study

This repository is organized around three explicit study layers:

1. Benchmark replication as the primary scientific study
2. Temporal robustness as a secondary stress-test study
3. Industrialization-gap analysis as a required interpretation layer

The active canonical spec lives under:

- `docs/spec/`

The rebuilt active surface now supports:

1. benchmark replication artifact generation
2. temporal robustness artifact generation
3. categorized study audit output
4. markdown report-skeleton generation from active artifacts

## Active Entry Points

Primary study:

```bash
python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --strict
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

## Current Output Model

Primary benchmark artifacts use `benchmark_*` names plus:

- `feature_stability.csv`
- `feature_report.csv`
- `run_manifest.json`
- `final_report_skeleton.md`

Secondary temporal artifacts use `temporal_*` names.

Audit output distinguishes:

- `ERROR`
- `WARNING`
- `CLAIM_RESTRICTION`

## Canonical Reading Order

1. `docs/spec/01-study-goal.md`
2. `docs/spec/02-benchmark-replication-study.md`
3. `docs/spec/03-feature-stability-and-interpretation.md`
4. `docs/spec/04-temporal-robustness-study.md`
5. `docs/spec/05-industrialization-gap-analysis.md`
6. `docs/spec/06-report-structure.md`
7. `docs/spec/07-artifact-contracts.md`
8. `docs/spec/08-audit-and-claim-semantics.md`
