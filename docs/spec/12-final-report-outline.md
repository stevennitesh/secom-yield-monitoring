---
spec_id: "12"
title: "Final Report Outline"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 12"
applies_to:
  - "final report"
keywords:
  - report outline
  - limitations
---
# 12 Final Report Outline

## Required Outline

1. Executive summary.
2. Problem context and data realities (imbalance, missingness, temporal shift).
3. Lane A replication results (`strict` and `+MI`).
4. Lane B selection results (Stage A diagnostic + Stage B multi-seed ranking + Phase 2 freeze).
5. Lockbox results for available frozen models and thresholds.
6. Operational thresholding and workload/cost tradeoffs.
7. MSPC comparison at matched `TNR=90%`.
8. Feature-actionability and clustered feature-prioritization section.
9. Limitations and deployment caveats.

## Canonical Sources by Section

1. Executive summary:
   1. synthesize the report-ready outputs from Section 11,
   2. do not exceed the claim boundaries from Section 13.
2. Problem context and data realities:
   1. derive data-partition and temporal-validation context from Sections 03 and 04.2.1.
3. Lane A replication results:
   1. source from Section 04.1 and the Lane A artifacts in Section 10.2.
4. Lane B selection results:
   1. source from Sections 04.2.2, 04.2.3, 05, and the Stage B artifacts in Section 10.3.
5. Lockbox results:
   1. source from Sections 06 and 07,
   2. use the frozen-threshold and lockbox artifacts in Section 10.4 and the drift-gate outputs in Section 10.5.
6. Operational thresholding and workload/cost tradeoffs:
   1. source from Section 06,
   2. use `reports/manager_facing_outputs.csv` and `reports/operational_cost_curves.csv`.
7. MSPC comparison at matched `TNR=90%`:
   1. source from Sections 07 and 08.1,
   2. use `reports/final_lockbox_result.csv` and `reports/mspc_baseline.csv`.
8. Feature-actionability and clustered feature-prioritization:
   1. source from Sections 08.2 and 11,
   2. use `reports/feature_report.csv`,
   3. preserve the non-causal interpretation rule.
9. Limitations and deployment caveats:
   1. use the mandatory limitations in this file,
   2. ensure they temper the executive summary and comparison sections.

## Claim Boundary

1. Report sections covering lockbox results, threshold recommendations, and MSPC comparison may present descriptive outputs even when claim restrictions apply.
2. Any superiority interpretation or recommendation based on lockbox or matched-`TNR=90%` comparisons remains subject to:
   1. the drift-gate rules in Section 07,
   2. the claim policy in Section 13.

## Mandatory Limitations Content

1. Single dataset, single process context, limited time window.
2. Temporal non-i.i.d. behavior and drift risk.
3. Anonymous features and non-causal interpretation.
4. Revalidation required under baseline process shift.
5. Historical-window workload estimates are not guaranteed bounds.

## See Also

- [11 Manager-Facing Outputs](11-manager-facing-outputs.md)
- [13 Claim Policy](13-claim-policy.md)
