# 04 Temporal Robustness Study

## Scope

This file defines the secondary temporal stress-test study.

## Purpose

The temporal robustness study answers:

1. How do benchmark-selected ideas behave under stricter future-looking evaluation?
2. How sensitive are results to temporal drift and threshold-freeze discipline?
3. What changes when the problem is treated more like a deployment stress test?

## Core Design

1. Use a chronological DEV/LOCKBOX split.
2. Use time-aware folds inside DEV.
3. Perform diagnostic selector screening, temporal model selection, config freeze, full-DEV refit, threshold freeze, lockbox evaluation, drift gate, and MSPC comparison.
4. Treat all outputs from this study as secondary evidence relative to the benchmark replication study.

## Interpretation Rules

1. Strong temporal robustness supports confidence in the primary study.
2. Weak temporal robustness does not automatically invalidate the primary study.
3. Drift restrictions and lockbox claim restrictions are scoped to this study unless explicitly elevated.
4. Operational framing from this study is informative, but not a substitute for real deployment evidence.

## Model and Threshold Semantics

The temporal workflow evaluates balanced logistic regression. It is a separate chronological stress test, not an out-of-time validation of the exact selected KRR benchmark model. Scientific and illustrative operational thresholds are fitted on DEV scores and frozen before lockbox evaluation. They are not independently calibrated operating points.

Matched-TNR90 supervised/MSPC comparisons select ROC thresholds retrospectively using evaluation labels. Label them diagnostic and descriptive; do not present them as frozen-threshold operating performance. Report lockbox sample/failure counts and TP/FP/TN/FN alongside rates. Small failure counts and drift limit precision and superiority claims.

## See Also

- [01 Study Goal](01-study-goal.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
- [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)
