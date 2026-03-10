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

## Language Rule

1. Report feature relationships as prioritization associations, not causal proofs.
