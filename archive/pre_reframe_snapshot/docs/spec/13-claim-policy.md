---
spec_id: "13"
title: "Claim Policy"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 13"
applies_to:
  - "final report"
keywords:
  - claims
  - superiority
  - drift gate
  - benchmark
---
# 13 Claim Policy

## Global Precedence

1. Drift gate status governs lockbox claim eligibility per frozen supervised model.
2. If a frozen supervised model's drift gate is `HIGH_SHIFT`, superiority claims are disallowed for that model.
3. When superiority claims are disallowed, the underlying report outputs may still be shown descriptively, but recommendation language must not imply a blocked superiority conclusion.

## Allowed Claims

1. Replication missing-indicator improvement is allowed only when `reports/lane_a_global_ablation.csv` shows `delta_BER = BER_strict - BER_MI` with a 95% CI entirely above `0` for the relevant `(selector, classifier)`.
2. Time-aware selection is more deployment-realistic than random CV.
3. Thresholding was set without lockbox tuning.
4. Supervised advantage over MSPC only if, on the lockbox slice, the supervised model's `TPR_at_TNR90` (from `reports/final_lockbox_result.csv`) exceeds MSPC's `best_MSPC_TPR_at_TNR90` (from the `eval_scope='lockbox'` row in `reports/mspc_baseline.csv`) and that model's drift gate is not `HIGH_SHIFT`.
5. If Lane B BER is worse than Lane A BER, describe this as expected under stricter temporal validation.
6. If MSPC matches or exceeds supervised at `TNR=90%`, report this as a valid finding.
7. Benchmark-improvement claim versus `33.5%` BER is allowed only when the `classifier='krr', selector='F-test', replication_mode='strict'` mean BER 95% CI upper bound is below `0.335` (CI from `reports/lane_a_global_summary.csv` using percentile bootstrap on Lane A fold BER; `n_boot=1000`, seed `42`), and both anchor rows `(F-test,krr,strict)` and `(F-test,krr,with_missing_indicators)` exist.
8. Primary head-to-head benchmark claim versus `33.5%` must use `classifier='krr', selector='F-test', replication_mode='strict'` BER; other selectors/classifiers are supportive evidence only.

## Forbidden Claims

1. Causality from feature importance.
2. Cross-fab or long-horizon generalization without new validation.
3. Lockbox superiority after any post-lockbox tuning.
4. Supervised superiority when MSPC is equal or better at matched `TNR=90%`.
5. Any superiority claim when drift gate is `HIGH_SHIFT`.

## Evidence Sources

1. Replication missing-indicator improvement claims:
   1. `reports/lane_a_global_ablation.csv`.
2. Benchmark-improvement claims versus `33.5%`:
   1. `reports/lane_a_global_summary.csv`,
   2. required benchmark anchor rows as specified in this section.
3. Time-aware-validation and no-lockbox-tuning claims:
   1. Sections 04.2 and 06,
   2. `reports/run_manifest.json` for registered split, freeze, and threshold context.
4. Lockbox supervised-vs-MSPC comparison claims:
   1. `reports/final_lockbox_result.csv`,
   2. `reports/mspc_baseline.csv`,
   3. `reports/drift_gate_summary.csv` or `reports/run_manifest.json` via `drift_gate_results`.
5. Lane B versus Lane A narrative framing:
   1. `reports/lane_a_global_summary.csv`,
   2. `reports/timeaware_model_selection.csv`.

## Recommendation Boundary

1. Workload and alert-policy recommendations may be reported descriptively from the pre-registered threshold and cost-analysis outputs.
2. If lockbox superiority claims are blocked by drift status or other claim restrictions, recommendations must not say or imply that a supervised model is superior to MSPC or otherwise superior on the blocked evidence path.
3. Recommendations based on matched `TNR=90%` lockbox comparisons remain subject to the same drift-gate and claim restrictions as the underlying superiority claim.

## See Also

- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
- [11 Manager-Facing Outputs](11-manager-facing-outputs.md)
- [12 Final Report Outline](12-final-report-outline.md)
