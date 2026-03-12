# 02 Benchmark Replication Study

## Scope

This file defines the two benchmark studies that anchor the project.

## Purpose

The benchmark study layer answers:

1. Can SECOM pass/fail yield be predicted with literature-style supervised pipelines under an original replication protocol?
2. How much do the results improve when the same selector family is tuned under a stricter nested benchmark design?
3. Does adding missing-indicator features change the result materially in both studies?

## Core Design

1. Use the full available dataset after timestamp parsing and `NaT` removal.
2. Run an original replication study that keeps the literature-style fixed-budget selector comparison.
3. Run a tuned benchmark study that uses nested tuning and AUC-first inner selection before final thresholded BER reporting.
4. Perform preprocessing and feature selection inside training folds only.
5. Treat missing-indicator ablation as a mandatory paired comparison in both studies.
6. Use the benchmark study layer as the main basis for project conclusions.

## Required Outputs

1. Original replication search, best-config, fold, summary, ablation, and full-fit outputs
2. Tuned benchmark search, selected-config, fold, summary, ablation, and full-fit outputs
3. Feature-stability and feature-report outputs for both studies

## Claim Rule

Claims about replication success, selector comparison, classifier comparison, tuned improvement, and missing-indicator benefit come from the benchmark study layer, not from the temporal stress-test study.

## See Also

- [01 Study Goal](01-study-goal.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
