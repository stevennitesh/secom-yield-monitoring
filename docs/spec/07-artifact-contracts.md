# 07 Artifact Contracts

## Scope

This file defines the artifact families for the rebuilt study.

## Primary Study Artifact Family

The benchmark replication study should produce:

1. benchmark sweep
2. benchmark best config
3. benchmark fold metrics
4. benchmark summary
5. benchmark ablation
6. benchmark full-fit summary
7. feature stability
8. feature report

These artifacts are canonical for:

1. headline benchmark metrics,
2. uncertainty summaries,
3. ablation results,
4. selector/classifier comparison,
5. feature-stability interpretation.

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

Headline benchmark metrics should come from the primary artifact family only, especially:

1. benchmark summary
2. benchmark fold metrics
3. benchmark ablation
4. feature stability
5. feature report

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
2. temporal-study status,
3. temporal claim restrictions,
4. industrialization notes or gaps where applicable.

## Naming Rule

Artifact naming should follow the new study structure rather than the legacy lane numbering or old workflow ordering.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)
