# 03 Feature Stability and Interpretation

## Scope

This file defines how feature stability and feature interpretation are computed and reported for the benchmark study layer.

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
5. Original and tuned benchmark studies should each retain their own feature-stability and feature-report outputs.

## Primary Role

Feature stability and interpretation are part of the benchmark study layer and should be presented alongside both original-replication and tuned-benchmark results, not as a secondary appendix.

## Operational Definitions

Selection frequency is the fraction of outer training folds selecting a transformed feature for a specific study/selector/classifier/mode. Do not pool incompatible configurations or mix this frequency with coefficient units in charts.

For balanced logistic regression, conditional effect magnitude is the absolute coefficient from the full-data fit in the selected, scaled feature space. Expected contribution is frequency times that magnitude, a heuristic ranking rather than an expected economic benefit or causal effect. For RBF KRR these coefficient quantities are unavailable; keep them null and display stability. Include a logistic-regression association comparator in the full evidence run.

Feature names Xn and Mn denote zero-based source column n and its missingness indicator. Report identical missingness masks and monthly missing rates for prominent selected indicators as full-sample descriptive context. These may reflect collection regime/time; multiple correlated indicators are not independent causes. Raw-value correlation clusters likewise support interpretation only.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [05 Industrialization Gap Analysis](05-industrialization-gap-analysis.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
