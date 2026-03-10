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
7. MSPC comparison and feature-actionability section.
8. Limitations and deployment caveats.

## Mandatory Limitations Content

1. Single dataset, single process context, limited time window.
2. Temporal non-i.i.d. behavior and drift risk.
3. Anonymous features and non-causal interpretation.
4. Revalidation required under baseline process shift.
5. Historical-window workload estimates are not guaranteed bounds.
