---
spec_id: "11"
title: "Manager-Facing Outputs"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 11"
applies_to:
  - "final report"
keywords:
  - report outputs
  - weekly flagged wafers
  - operational framing
---
# 11 Manager-Facing Outputs

## Normative Outputs

1. Weekly flagged wafers at scientific and operational thresholds.
2. Weekly fail capture and miss counts.
3. Review workload estimates and recommended alert policy.
4. Stable top features grouped into high-correlation clusters.
5. Supervised vs MSPC at matched `TNR=90%`.
6. Operational framing of workload:
   1. `weekly_rate = DEV_sample_count / DEV_week_count`,
   2. `DEV_week_count = count_unique(week_idx)` over DEV (weeks with at least one wafer in DEV),
   3. `predicted_flag_fraction` from full-DEV post-freeze predictions (report separately for scientific vs operational thresholds, for each available frozen model),
   4. Stage B mean flagged fraction shown as robustness diagnostic,
   5. lockbox flagged fraction shown as holdout observation.

## Canonical Sources for Outputs

1. Weekly flagged wafers, fail capture / miss counts, and workload framing are sourced from `reports/manager_facing_outputs.csv`.
2. Workload and alert-policy framing is supported by:
   1. the operational-threshold policy in Section 06,
   2. `reports/manager_facing_outputs.csv` for `predicted_flag_fraction`, `weekly_rate`, and weekly per-threshold summaries,
   3. `reports/operational_cost_curves.csv` for cost tradeoffs.
3. Stable top features grouped into high-correlation clusters are sourced from `reports/feature_report.csv`, using the clustering and feature-identity rules from Section 08.2.
4. Supervised-vs-MSPC matched `TNR=90%` comparison is sourced from:
   1. `reports/final_lockbox_result.csv` for supervised `TPR_at_TNR90`,
   2. `reports/mspc_baseline.csv` for MSPC `best_MSPC_TPR_at_TNR90`.

## Claim Boundary

1. These outputs are report-facing and may be shown descriptively even when claim restrictions apply.
2. Any lockbox superiority interpretation or recommendation based on the matched `TNR=90%` comparison remains subject to:
   1. the drift-gate rules in Section 07,
   2. the claim restrictions in Section 13.

## Language Rule

1. Report feature relationships as prioritization associations, not causal proofs.

## See Also

- [06 Threshold Policy](06-threshold-policy.md)
- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
- [08.2 Feature Stability and Identity](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md)
- [13 Claim Policy](13-claim-policy.md)
