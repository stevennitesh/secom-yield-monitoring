# 06 Report Structure

## Scope

This file defines the required structure of the final active report.

## Required Sections

1. Executive summary
2. Dataset and study scope
3. Original replication design
4. Original replication search summary
5. Original replication results
6. Tuned benchmark design
7. Tuned benchmark search summary
8. Tuned benchmark results
9. Original vs tuned benchmark comparison
10. Feature stability and interpretation
11. Temporal robustness stress test
12. Industrialization gaps
13. Conclusions and next data requirements

## Metric Policy

### Headline Metrics

The main report narrative must prioritize these benchmark-study metrics:

1. `BER`
2. `TPR` / `True+`
3. `TNR` / `True-`
4. uncertainty summaries for the mean metric values
5. missing-indicator ablation deltas
6. selector/classifier comparison outcomes
7. feature-stability summaries from the benchmark studies

### Secondary Robustness Metrics

These belong in the temporal robustness section and must not be presented as the main project result:

1. temporal `BER`, `TPR`, `TNR`
2. lockbox `BER`, `TPR`, `TNR`
3. prevalence shift
4. KS score-shift test results
5. PSI summaries
6. matched-`TNR=90%` supervised vs MSPC comparison

### Illustrative Industry Metrics

These may be shown for operational framing, but must be labeled illustrative or exploratory rather than production-validated:

1. weekly flagged wafers
2. weekly fail captures and misses
3. `predicted_flag_fraction`
4. workload framing
5. cost curves
6. operating-point recommendations

### De-emphasized Metrics

The following may appear as supporting diagnostics, appendix material, or tables, but should not drive the main conclusions:

1. `ROC_AUC`
2. `PR_AUC`
3. `MCC`
4. `F2`

## Ordering Rules

1. Original replication results must appear before tuned benchmark results.
2. Original replication design must explain:
   1. fixed feature budget,
   2. literature-style selector/classifier comparison,
   3. in-fold preprocessing and selection,
   4. missing-indicator paired comparison,
   5. final thresholded reporting via `BER`, `TPR`, and `TNR`.
3. Original replication search summary must show:
   1. evaluated selector/classifier/mode combinations,
   2. the fixed search space,
   3. selected configurations per selector/classifier/mode.
4. Tuned benchmark design must explain:
   1. nested CV,
   2. tuned selector parameters,
   3. tuned classifier parameters,
   4. `ROC_AUC` as the threshold-free inner objective,
   5. final thresholded reporting via `BER`, `TPR`, and `TNR`.
5. Tuned benchmark search summary must show:
   1. search-space coverage,
   2. selected-config counts,
   3. modal selected configurations per selector/classifier/mode.
6. Temporal robustness must explain:
   1. chronological DEV/LOCKBOX split,
   2. time-aware folds,
   3. selector screening,
   4. temporal model selection,
   5. config and threshold freeze,
   6. lockbox evaluation,
   7. drift gating,
   8. MSPC comparison.
7. Temporal model selection summary must show:
   1. primary selector,
   2. challenger availability,
   3. selector ranking,
   4. modal configurations for the selected selectors.
8. Drift and claim restrictions must appear before MSPC, workload, or cost interpretation.
9. Both benchmark studies must appear before temporal stress-test results.
10. Feature interpretation must be attached to the benchmark studies, not treated only as an operational appendix.
11. Industrialization gaps must be explicit and substantive.
12. Conclusions must separate:
   1. what was replicated,
   2. what improved under tuning,
   3. what was stress-tested,
   4. what remains unsupported.

## Narrative Rule

The report must make it obvious which findings are:

1. original benchmark conclusions,
2. tuned benchmark conclusions,
3. secondary robustness observations,
4. non-claimable or unsupported for real deployment.

Operational framing must not be written as if the dataset already supports production-readiness claims.

## See Also

- [01 Study Goal](01-study-goal.md)
- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [05 Industrialization Gap Analysis](05-industrialization-gap-analysis.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
- [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)
