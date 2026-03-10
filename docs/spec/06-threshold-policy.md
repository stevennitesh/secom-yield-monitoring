---
spec_id: "06"
title: "Threshold Policy"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 6"
applies_to:
  - "Lane A"
  - "Lane B"
keywords:
  - thresholds
  - ber optimal
  - operational threshold
  - tnr90
---
# 06 Threshold Policy

## Scope

This section defines:

1. Lane A thresholding for replication reporting,
2. the shared BER-optimal threshold rule,
3. Lane B frozen-model threshold policy after Phase 2 freeze and Phase 3 full-DEV refit.

## Lane A Thresholding Rules

1. Lane A thresholding uses a model-specific scalar `score`:
   1. for `logreg`, `score = Pr(y_bin=1)` from the classifier,
   2. for `krr` and `krr_strict`, `score` is the raw regression-model output.
2. For Lane A config selection/reporting, use `threshold_oof_global` derived from pooled OOF `(y_bin, score)` pairs of the selected config.
3. Lane A predicts fail iff `score >= threshold_value`.
4. Lane A threshold candidate set for searches is the sorted unique `score` values in the relevant evaluation slice plus two sentinels (`-inf` flags all; `+inf` flags none).
5. Lane A full-data threshold (`threshold_full_dataset`) is diagnostic/deployment-reference only and is not used for Lane A tuning claims.
6. Derive `threshold_full_dataset` from the full-dataset in-sample `(y_bin, score)` pairs of the selected-config full-data refit using the shared BER-optimal threshold rule in this section.

## Shared BER-Optimal Threshold Rule

1. When this protocol says "derive BER-optimal threshold", compute the threshold on the relevant `(y_true, score)` or `(y_true, p_hat)` pairs by:
   1. evaluating the sorted unique candidate values from that slice plus two sentinels (`-inf` flags all; `+inf` flags none),
   2. choosing the threshold with minimum BER,
   3. if multiple thresholds achieve the same minimum BER, choosing the one with higher TPR,
   4. if still tied, choosing the lowest threshold value.
2. This same BER-optimal threshold rule applies to:
   1. Lane A `threshold_oof_global`,
   2. Lane A `threshold_full_dataset`,
   3. Lane B inner-train BER thresholds,
   4. Lane B outer-train BER thresholds,
   5. Lane B scientific threshold on full DEV.

## Lane B Frozen-Model Scoring Rule

1. For Lane B frozen models, score is predicted fail probability `p_hat = Pr(y_bin=1)` from the frozen pipeline.
2. Predict fail iff `p_hat >= threshold_value` (else predict pass).
3. Lane B threshold candidate set for searches is the sorted unique `p_hat` values in the relevant slice plus two sentinels (`-inf` flags all; `+inf` flags none).

## Lane B Thresholds to Report for Each Available Frozen Model

1. Scientific threshold: BER-optimal threshold on full-DEV in-sample predictions.
2. Operational threshold: pre-registered review-capacity threshold.

## Lane B Operational Constants and Rules

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

## Decision Outputs

1. Lane A threshold outputs:
   1. `reports/lane_a_global_best_config.csv` records `threshold_oof_global` for each selected config,
   2. `reports/lane_a_global_fold_metrics.csv` records the same fixed `threshold_oof_global` used for fold-level reporting,
   3. `reports/lane_a_global_full_fit_summary.csv` records `threshold_full_dataset` and marks it as `diagnostic_only`.
2. Lane B threshold outputs:
   1. `reports/final_lockbox_result.csv` is the canonical row-based output for frozen `threshold_value` and reporting-only `threshold_at_TNR90`,
   2. `reports/run_manifest.json` is the canonical run-level output for frozen thresholds via `frozen_thresholds`.

## See Also

- [04.1 Lane A Replication](04-two-lane-plan/04.1-lane-a-replication.md)
- [04.2.5 Phase 3 Final Refit and Threshold Derivation](04-two-lane-plan/04.2.5-phase-3-final-refit-and-thresholds.md)
- [10.2 Lane A Artifacts](10-required-artifacts-and-schemas/10.2-lane-a-artifacts.md)
- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md)
