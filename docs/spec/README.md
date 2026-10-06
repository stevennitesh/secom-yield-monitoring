# Active Study Spec

This directory owns the active canonical specification for the implemented SECOM benchmark-first study. For repository orientation and maintenance, start with [the context index](../README.md).

## Source Hierarchy

1. Files in `docs/spec/` are canonical for accepted study meaning.
2. Active code and tests must be brought into alignment with these files.
3. Generated snapshots preserve the source/spec identity actually executed; later prose clarification does not retroactively change their manifests. See [the evidence workflow](../development.md#documentation-changes-and-historical-provenance).

## Reading Order

1. [01 Study Goal](01-study-goal.md)
2. [02 Benchmark Replication Study](02-benchmark-replication-study.md)
3. [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
4. [04 Temporal Robustness Study](04-temporal-robustness-study.md)
5. [05 Industrialization Gap Analysis](05-industrialization-gap-analysis.md)
6. [06 Report Structure](06-report-structure.md)
7. [07 Artifact Contracts](07-artifact-contracts.md)
8. [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)

## Methodology Revision

The active specs require nested BER-first joint benchmark procedures, inner-OOF thresholds, same-fold baselines/predictions, and independent chronological FIT/calibration/future separation. [The context index](../README.md) and [results snapshot](../results/README.md) own current execution and export status. Code validation and real-data scientific evidence are separate; snapshots retain the source/spec identity actually executed.

The active bounded extension adds improved tuned KRR coverage, equivalent-gamma cache reuse, a DEV-only KRR input-mode/calibration-window comparator, and calibration fragility diagnostics. It does not extend the production or dataset scope.
