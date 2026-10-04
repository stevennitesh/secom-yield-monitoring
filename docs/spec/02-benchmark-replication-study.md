# 02 Benchmark Replication Study

## Scope

This file defines the two benchmark studies that anchor the project.

## Purpose

The benchmark study layer answers:

1. Can SECOM pass/fail yield be predicted with literature-style supervised pipelines under an original replication protocol?
2. How much do the results improve when the same selector family is tuned under a stricter nested benchmark design?
3. Does adding missing-indicator features change the result materially in both studies?

## Core Design

1. Use the full available dataset after timestamp validation; reject unparseable timestamps without silently dropping rows.
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

## Protocol and Reference Limits

The original anchor uses 10 shuffled stratified folds (seed 42), 40 features, train-fitted median imputation/scaling/selection and train-fitted BER thresholds. Classifier configurations are selected on the same sweep used to report results; this is non-nested and can be optimistic. The tuned study uses three inner stratified folds within each of ten outer folds, searches k in 10/20/40 and selects by inner ROC AUC with BER tie-breaking. Selector/classifier/mode families are evaluated separately; the leading family is a post-hoc descriptive minimum, not a separately validated champion.

Report paired missing-indicator deltas as strict BER minus missing-indicator BER (positive is improvement). Fold-bootstrap uncertainty is descriptive because folds share training observations. Full-data fits are interpretation diagnostics only.

UCI describes its published reference table as kernel ridge; Table 1 of McCann and Johnston (2010) labels it Naive Bayes. Preserve this discrepancy and avoid claiming exact classifier replication. Local data contain 590 measurement columns despite the 591-feature metadata. Inputs represent production entities with test timestamps; no physical unit or pre-outcome availability is established.

ReliefF uses the pinned skrebate 0.8.4 distance and feature-typing implementation. The default NumPy adapter preserves its binary, imputed-data neighbor ordering, mixed-feature ramp, normalization and summation order. Exact score/rank equivalence against the upstream implementation is required for runtime optimizations; folds, seeds, feature budgets and search grids are unchanged. Import or fitting failures are errors, never permission to substitute a different algorithm. Rankings may be reused only for byte-identical training measurements and labels, shape, neighbor count and backend identity; the bounded cache returns copies and never stores validation labels. The adapter runs in one process. An explicit `SECOM_RELIEF_BACKEND=reference` run uses upstream scoring with at most four workers by default, adjustable through positive `SECOM_RELIEF_N_JOBS`. The backend and actual worker count are recorded in the full-run manifest. The full evidence run includes KRR and a balanced logistic-regression comparator; focused KRR-only entrypoints remain supported.

Benchmark score vectors may be reused only when the classifier configuration, training measurements and labels, evaluation measurements, shapes/dtypes/strides and fitting implementation match exactly. The in-memory cache is bounded to 4,096 entries, returns independent copies, stores no raw arrays or validation labels, and starts cold for a complete timed study. Every family still performs its specified parameter selection and evaluation; reuse eliminates only identical fits. The full-run manifest records cache hits, misses and retained score bytes.

## See Also

- [01 Study Goal](01-study-goal.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
