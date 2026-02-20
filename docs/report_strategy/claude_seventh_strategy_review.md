# Claude Seventh Strategy Review — Final Implementation Readiness

## Purpose

Final top-to-bottom pass of `final_end_to_end_report_strategy.md` after all
round-6 issues and Codex's additional `predicted_flag_fraction` fix were
applied. This review checks for any remaining implementation-blocking issues
before coding begins.

---

## Round 6 Resolutions — Confirmed

1. §5.2 workload annotation corrected to `(360 non-ReliefF + 1080 ReliefF-
   neighbor branch)`. ✓
2. Union alarm rule added to §9.3.6 and §11.2:
   `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)`. ✓
3. `predicted_flag_fraction` source for expected workload pinned to full-DEV
   post-freeze predictions at the frozen threshold; Stage B mean outer-test
   fraction repositioned as robustness diagnostic. ✓

---

## Full Document Checklist

### §1 End Goal
Three deliverables (replication, deployment model, operational impact) map
directly to Lane A (§4.1), Lane B (§4.2–§6), and §10. No orphaned goals. ✓

### §2 Validation Rules
No-leakage and no-lockbox-tuning rules are unconditional. Config freeze
definition is precise (immediately after §6 logic on Stage B results). Drift
gate cross-references §8. ✓

### §3 Data Partition
NaT drop → sort → lockbox = last floor(0.15 × N) samples → DEV = first 85%.
N defined as post-NaT count. Lockbox is by sample count, not time duration. ✓

### §4.1 Lane A
KernelRidge with `y_krr = 2*y_bin - 1` for {-1,+1} encoding; decision
threshold at 0.0 is valid on this encoding. Paired ablation CI pinned
(percentile bootstrap, n_boot=1000). ✓

### §4.2 Lane B
- Week bins anchored to `t_min_dev`, 1-based labels mapped to `week_idx = N-1`.
- Overlapping test windows explained (anchored expanding-window backtesting,
  not i.i.d. CV).
- Fallback chain: primary → deterministic fallback → 2-fold fallback →
  Lane B infeasible. All deterministic, no ambiguity.
- Stage A: outer-folds-only, fixed settings, no inner CV.
- Stage B: inner StratifiedKFold (n=5, ROC-AUC), outer BER via Youden
  from outer-train only.
- Finalist promotion: top 2 + optional 3rd within 0.02 BER of 2nd. ✓

### §5 Model-Selection Search Space
Pearson ≡ F-test mathematically; F-test only promoted to Stage B. Backfill
rule for de-dup edge case. Stage A fixed settings all specified. Stage B
grid complete (k, C, scaler; missing_mode fixed; classifier fixed).
ReliefF n_neighbors sweep {5, 10, 20}. Workload totals correct:
720 (base), 1440 (ReliefF in top-2 = 360 + 1080), 1800 (3 selectors with
ReliefF). ✓

### §6 Selection Logic
Lexicographic: minimize BER → near-equal (0.02) → True+ → smaller k.
Challenger BER floor ≤ 0.40. Challenger by highest True-. Overlapping folds
framed as time-backtest evidence, not i.i.d. ✓

### §7 Threshold Policy
Post-freeze. Youden (in-sample full DEV) + 10% weekly cap (in-sample full DEV).
In-sample calibration acknowledged. Week bins for flagged-fraction computation
cross-referenced to §4.2 anchored bins. TNR=90% rule: lowest threshold with
TNR ≥ 0.90. ✓

### §8 Lockbox Protocol
Train frozen configs on full DEV → apply frozen thresholds → evaluate once →
no retuning.

Drift gate criteria: prevalence shift, KS p-value on predicted probabilities,
max PSI on top-10 value features (ranked by absolute logistic coefficient,
"value" = original sensor columns only). PASS / CAUTION / HIGH_SHIFT.
HIGH_SHIFT → no superiority claims.

MSPC companion: fit on full DEV pass wafers → score lockbox → T2/Q
TPR_at_TNR90 for §13.4 claim. ✓

### §9 Metrics Policy
Lane A: BER/True+/True- + CI. Lane B: fold-wise + mean±std (no formal CI
from 3 folds). Lockbox: point estimates. Overlapping backtests interpreted
as robustness diagnostics. MSPC union alarm rule defined. fold_jaccard_
stability defined as per-feature pairwise selection agreement. Feature cluster
on imputed original features only (exclude indicator columns). ✓

### §10 Manager Outputs
Weekly flagged, fail capture, workload, stable features, MSPC comparison at
TNR=90%, recommended alert policy (both thresholds + expected weekly wafers).
Weekly rate = DEV_sample_count / (max(week_idx) + 1). predicted_flag_fraction
from full-DEV post-freeze at frozen threshold. ARL0/alarm_rate in §10.10
(union alarm stream). ✓

### §11 Artifacts
12 artifacts, all with minimum column specifications.

| Artifact | Source Step | Key Spec Check |
|---|---|---|
| baseline_replication_strict.csv | Lane A | selector, fold, BER, True+, True-, n_train, n_test, n_test_fails ✓ |
| baseline_replication_with_missing_indicators.csv | Lane A | same ✓ |
| baseline_missing_indicator_ablation.csv | Lane A | selector, BER_strict, BER_MI, delta_BER, CI bounds, n_boot ✓ |
| timeaware_selector_screening.csv | Stage A | method, mean_BER, min_test_fails, promoted_to_stage_b ✓ |
| timeaware_model_selection.csv | Stage B | config_id, method, k, C, scaler, is_primary, is_challenger ✓ |
| splitwise_timeaware_results.csv | Stage B | config_id, fold_index, test_fails, BER, threshold_value ✓ |
| final_lockbox_result.csv | Lockbox §8 | config_id, threshold_policy, BER, True+, True-, ROC_AUC, lockbox_n, lockbox_fails ✓ |
| mspc_baseline.csv | MSPC §9.3.6 + §8.10 | eval_scope, fold_index='LOCKBOX' sentinel, union alarm, ARL0 NaN rule, TPR_at_TNR90 ✓ |
| operational_cost_curves.csv | §7 | cost_ratio, 4 config columns, baselines ✓ |
| feature_report.csv | §9.3.10 | selection_frequency, conditional_effect_magnitude, expected_contribution, fold_jaccard_stability, cluster_id ✓ |
| run_manifest.json | Throughout | SHA-256 hash with sorted keys and 6 sig fig floats ✓ |
| drift_gate_summary.csv | §8.5 | all gate inputs + drift_gate_status + lockbox_claims_allowed ✓ |

### §12 Report Outline
8-section structure covers all deliverables. Limitations section has 5
pre-specified required items (single dataset, temporal non-i.i.d., anonymous
features, recalibration needed, threshold estimates are historical). ✓

### §13 Claim Policy
Global precedence rule: HIGH_SHIFT blocks lockbox superiority claims.
Can claim supervised > MSPC only with non-HIGH_SHIFT drift gate (item 4).
Can claim MSPC-positive finding only with non-HIGH_SHIFT drift gate (item 6).
Cannot claim under HIGH_SHIFT (item 5). Benchmark CI: percentile bootstrap on
F-test Replication-Strict BER, upper bound < 0.335 required. ✓

---

## End-to-End Pipeline Trace

| Step | Protocol Source | Output |
|---|---|---|
| Drop NaT, sort, partition | §3 | DEV, lockbox |
| Lane A: 10-fold KRR ablation | §4.1 | artifacts 1–3 |
| Stage A: fixed-settings sweep | §4.2, §5.2 | artifact 4 |
| Stage B: finalist promotion → nested tuning | §4.2, §5, §6 | artifacts 5–6 |
| Config freeze | §2, §6 | frozen primary + challenger |
| Threshold derivation (full DEV in-sample) | §7 | two thresholds per config |
| Lockbox evaluation | §8 | artifact 7 |
| Drift gate | §8.5 | artifact 12 |
| MSPC evaluation (per fold + lockbox) | §9.3.6, §8.10 | artifact 8 |
| Feature stability | §9.3.10 | artifact 10 |
| Operational outputs | §7, §10 | artifacts 9, manager outputs |
| run_manifest.json | §2, §5, §11 | artifact 11 |

All steps are sequentially dependent and non-circular. Every artifact has a
producing step and a consuming section. ✓

---

## Issues Found

None.

All round-6 issues resolved. No new implementation-blocking gaps identified
in this pass.

---

## Verdict

**Implementation-ready.**

The document is fully specified across 13 sections and 12 required artifacts.
Over seven review rounds, every ambiguity that could produce diverging
implementations has been closed:

- Data partition: sample-count lockbox, NaT handling, t_min_dev anchoring
- Lane A: KernelRidge {-1,+1} encoding
- Lane B: overlapping window explanation, finalist count pinned, fallback
  chain deterministic, inner/outer threshold policies separated
- MSPC: per-fold discipline, lockbox companion, union alarm rule,
  ARL0/alarm_rate source
- Threshold: week-bin cross-reference, in-sample transparency,
  predicted_flag_fraction from full-DEV post-freeze
- Claims: drift gate overrides all lockbox superiority claims, benchmark
  selector locked, CI method pinned
- Artifacts: 12 schemas complete with sentinel values and NaN rules

No further review is required. Implementation may begin.
