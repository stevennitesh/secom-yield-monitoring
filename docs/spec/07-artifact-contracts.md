# 07 Artifact Contracts

## Scope

This file defines the artifact families for the rebuilt study.

## Benchmark Study Artifact Families

The original replication study should produce:

1. benchmark sweep
2. benchmark best config
3. benchmark fold metrics
4. benchmark summary
5. benchmark ablation
6. benchmark full-fit summary
7. feature stability
8. feature report

The tuned benchmark study should produce:

1. benchmark tuned search
2. benchmark tuned best config
3. benchmark tuned fold metrics
4. benchmark tuned summary
5. benchmark tuned ablation
6. benchmark tuned full-fit summary
7. benchmark tuned feature stability
8. benchmark tuned feature report

These artifacts are canonical for:

1. original replication benchmark metrics,
2. tuned benchmark metrics,
3. uncertainty summaries,
4. ablation results,
5. selector/classifier comparison,
6. feature-stability interpretation.

## Secondary Study Artifact Family

The temporal robustness study should produce:

1. temporal split metadata
2. temporal selector screening
3. temporal model selection
4. temporal inner-CV results
5. temporal freeze results
6. temporal lockbox results
7. temporal drift summary
8. temporal MSPC summary
9. temporal cost curves
10. temporal manager-facing outputs

These artifacts are canonical for:

1. temporal stress-test metrics,
2. drift and claim-restriction evidence,
3. lockbox comparison outputs,
4. illustrative workload and cost framing.

## Metric Tier Mapping

### Headline Metric Sources

Headline benchmark metrics should come from the original and tuned benchmark artifact families, especially:

1. benchmark summary
2. benchmark fold metrics
3. benchmark tuned summary
4. benchmark tuned fold metrics
5. benchmark ablation
6. benchmark tuned ablation
7. feature stability
8. benchmark tuned feature stability
9. feature report
10. benchmark tuned feature report

### Secondary Metric Sources

Temporal robustness metrics should come from the secondary artifact family, especially:

1. temporal model selection
2. temporal lockbox results
3. temporal drift summary
4. temporal MSPC summary

### Illustrative Metric Sources

Illustrative industry-facing metrics should come from:

1. temporal cost curves
2. temporal manager-facing outputs

These outputs must not be treated as production-validated operating metrics.

## Manifest Rule

The run manifest must distinguish between:

1. primary-study status,
2. benchmark original status,
3. benchmark tuned status,
4. temporal-study status,
5. temporal claim restrictions,
6. industrialization notes or gaps where applicable.

## Naming Rule

Artifact naming should follow the new study structure rather than the legacy lane numbering or old workflow ordering.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)
