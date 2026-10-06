# 01 Study Goal

## Scope

This file owns the accepted scientific objective and study scope of the SECOM research project.

The deliverable is a concise, defensible ML engineering case study backed by a reproducible implementation and audited evidence. Favor clear results, methodological judgment, and measured engineering improvements. Production services, deployment infrastructure, additional datasets, and expanded experiment suites are outside the current scope unless explicitly requested.

## Canonical Goal

The project has four ordered goals:

1. Produce a literature-inspired fixed-budget SECOM benchmark reproduction with nested selection and explicit reference limits.
2. Produce a tuned benchmark extension that keeps the selector family but compares a tuned feature budget using the same nested BER selection and held-out samples.
3. Evaluate temporal robustness under stricter deployment-like conditions as secondary stress-test evidence.
4. State clearly what this dataset does and does not support for real industrial deployment.

## Precedence

1. Original benchmark replication is the first primary scientific study.
2. Tuned benchmark is the second primary scientific study.
3. Temporal robustness is secondary evidence.
4. Industrialization-gap analysis is required report content, not optional commentary.
5. Project-level conclusions must separate what came from original replication, tuned benchmark, and temporal stress testing.

## Required Outcome

Deliver a report that:

1. reproduces defensible benchmark-style performance and feature-selection results,
2. shows how those results behave under temporal stress,
3. avoids overclaiming what the dataset cannot support,
4. demonstrates strong engineering judgment about industrial applicability.

The benchmark-first pipeline and public evidence snapshot are implemented. Further work should address a concrete presentation, correctness, reproducibility, or performance gap. Industrialization gaps describe evidence needed for stronger future claims; they do not require building those capabilities for this research project. Proposed review-budget or alternative-dataset studies do not replace this contract without an explicit scope decision.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [05 Industrialization Gap Analysis](05-industrialization-gap-analysis.md)
- [06 Report Structure](06-report-structure.md)

## Evidence Execution Status

Executed-run and validated-snapshot status live in [the context index](../README.md) and [the results snapshot](../results/README.md). Each run preserves the source/spec identity actually executed; implementation tests and real-data scientific evidence are distinct. A new real-data run and snapshot export require explicit authorization. Historical evidence is interpreted against its recorded source.

The active bounded extension adds improved tuned KRR coverage, equivalent-gamma cache reuse, a DEV-only KRR input-mode/calibration-window comparator, and calibration fragility diagnostics. It does not extend the production or dataset scope.
