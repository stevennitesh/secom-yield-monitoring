---
spec_id: "06"
title: "Threshold Policy"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §6"
applies_to:
  - "Lane A"
  - "Lane B"
used_by_runbooks:
  - "../report_strategy/runbooks/02_lane_a_replication.md"
  - "../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
keywords:
  - thresholds
  - ber optimal
  - operational threshold
  - tnr90
---
# 06 Threshold Policy

## Scope

Thresholds are finalized after Phase 2 freeze and Phase 3 full-DEV refit.

## Lane A-Specific Threshold Rules

1. For Lane A config selection/reporting, use `threshold_oof_global` derived from pooled OOF predictions/labels of the selected config.
2. Lane A full-data threshold (`threshold_full_dataset`) is diagnostic/deployment-reference only and is not used for Lane A tuning claims.
3. Lane B threshold rules in this section remain unchanged (`inner-train`, `outer-train`, and post-freeze full-DEV rules as specified).

## Supervised Scoring and Classification Rule

1. Score is predicted fail probability `p_hat = Pr(y_bin=1)` from the frozen pipeline.
2. Predict fail iff `p_hat >= threshold_value` (else predict pass).
3. Threshold candidate set for searches: sorted unique scores in the relevant training slice plus two sentinels (`-inf` flags all; `+inf` flags none).

## Thresholds to Report for Each Available Frozen Model

1. Scientific threshold: BER-optimal threshold on full-DEV in-sample predictions.
   1. if multiple thresholds achieve the same minimum BER, choose the one with higher TPR;
   2. if still tied, choose the lowest threshold value.
   This same BER-optimal threshold definition (including tie-breaks) applies whenever this protocol says "derive BER-optimal threshold" on any training slice (inner-train, outer-train, full DEV).
2. Operational threshold: pre-registered review-capacity threshold.

## Operational Constants and Rules

1. Review-capacity cap: mean weekly flagged fraction on DEV must be `<=10%`.
2. Mean weekly flagged fraction uses the same dataset-anchored 7-day bins as [04.2.1 Outer Time-Aware Folds](04-two-lane-plan/04.2.1-outer-time-aware-folds.md).
3. Mean weekly flagged fraction definition is fixed (unweighted by week size):
   1. for each week `w`: `frac_w = flagged_w / wafers_w`,
   2. `mean_weekly_flagged_fraction = mean_w(frac_w)`,
   3. do not use sample-weighted averaging across weeks,
   4. compute `mean_w(...)` over weeks that appear in DEV (weeks with `wafers_w > 0`); do not inject empty weeks.
4. Operational threshold selection rule: choose the highest-TPR threshold satisfying the `<=10%` cap.
   1. if multiple thresholds have the same highest TPR under the cap, choose the lowest threshold value.
5. Secondary matched comparison point: `TNR=90%`.
6. `TNR=90%` extraction rule: choose the lowest threshold with `TNR >= 0.90`; if tied in TNR, choose higher TPR; if still tied, choose the lowest threshold value.
7. `TNR=90%` reporting quantities (per evaluation slice; reporting only, not a frozen threshold):
   1. compute `threshold_at_TNR90` by applying the extraction rule above to that slice's `(y_bin, p_hat)` pairs,
   2. compute `TNR_at_TNR90` and `TPR_at_TNR90` on that same slice at `threshold_at_TNR90`,
   3. this matched point is used only for the supervised-vs-MSPC comparison at matched `TNR=90%`; it must never be used to choose the frozen scientific/operational thresholds.
8. Cost-ratio sensitivity set: `FN:FP in {1,2,5,10,20}`.
9. Lockbox is never used to choose thresholds.

## See Also

- [04.1 Lane A Replication](04-two-lane-plan/04.1-lane-a-replication.md)
- [04.2.5 Phase 3 Final Refit and Threshold Derivation](04-two-lane-plan/04.2.5-phase-3-final-refit-and-thresholds.md)
- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md)
