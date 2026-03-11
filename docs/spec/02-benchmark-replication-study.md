# 02 Benchmark Replication Study

## Scope

This file defines the primary scientific study.

## Purpose

The benchmark replication study answers:

1. Can SECOM pass/fail yield be predicted with literature-style supervised pipelines?
2. Which selector/classifier combinations are strongest under replication-style evaluation?
3. Does adding missing-indicator features change the result materially?

## Core Design

1. Use the full available dataset after timestamp parsing and `NaT` removal.
2. Use stratified replication-style cross-validation as the primary evaluation.
3. Perform preprocessing and feature selection inside training folds only.
4. Treat missing-indicator ablation as a mandatory paired comparison.
5. Use the benchmark replication study as the main basis for conclusions.

## Required Outputs

1. Config sweep results
2. Best config per selector/classifier/ablation mode
3. Fold-level performance
4. Summary performance with uncertainty
5. Missing-indicator ablation results
6. Full-fit summary for interpretive use

## Claim Rule

Claims about replication success, selector comparison, classifier comparison, and missing-indicator benefit come from this study, not from the temporal stress-test study.

## See Also

- [01 Study Goal](01-study-goal.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
