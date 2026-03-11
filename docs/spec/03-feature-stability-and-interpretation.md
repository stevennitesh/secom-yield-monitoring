# 03 Feature Stability and Interpretation

## Scope

This file defines how feature stability and feature interpretation are computed and reported for the primary study.

## Required Quantities

Report, at minimum:

1. selection frequency
2. transformed-feature identity
3. conditional effect magnitude
4. expected contribution
5. cluster grouping for highly correlated raw value features

## Interpretation Rules

1. Feature outputs are prioritization aids, not causal proof.
2. Missing-indicator features and value features must remain distinguishable.
3. Clusters should reduce redundant emphasis on highly correlated value features.
4. Stability should be interpreted across benchmark resamples, not only from a single final fit.

## Primary Role

Feature stability and interpretation are part of the primary study and should be presented alongside benchmark replication results, not as a secondary appendix.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [05 Industrialization Gap Analysis](05-industrialization-gap-analysis.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
