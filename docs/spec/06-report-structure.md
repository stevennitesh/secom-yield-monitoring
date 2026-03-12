# 06 Report Structure

## Scope

This file defines the required structure of the final active report.

## Required Sections

1. Executive summary
2. Dataset and study scope
3. Original replication design
4. Original replication results
5. Tuned benchmark design
6. Tuned benchmark results
7. Feature stability and interpretation
8. Temporal robustness stress test
9. Industrialization gaps
10. Conclusions and next data requirements

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
2. Both benchmark studies must appear before temporal stress-test results.
3. Feature interpretation must be attached to the benchmark studies, not treated only as an operational appendix.
4. Industrialization gaps must be explicit and substantive.
5. Conclusions must separate:
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
