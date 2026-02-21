# Runbook 05: Artifact QA and Claim Gate Checks

Canonical source: `docs/final_end_to_end_report_strategy_merged.md` (Sections 10, 11, 12, 13, 14).

## Objective

Validate artifact completeness, schema correctness, and claim eligibility before report finalization.

## Inputs

1. All generated `reports/*` artifacts.
2. Canonical merged strategy and pre-registration checklist.

## Outputs

1. Pass/fail QA log.
2. Final claim eligibility decision table.

## Artifact QA Sequence

1. Confirm required artifact set for the current feasibility mode:
   1. Lane B feasible: full set (1-17)
   2. Lane B infeasible: Lane A artifacts + manifest rules only
   3. If `challenger_available=false`, enforce fallback artifact behavior:
      1. no challenger rows in row-grain artifacts
      2. challenger columns allowed only where canonical spec says write `NA`.
2. Validate row grain and uniqueness constraints:
   1. selected config uniqueness in Stage B
   2. frozen config uniqueness in Phase 2
   3. in `final_lockbox_result.csv`, for each role, verify `threshold_at_TNR90`, `TNR_at_TNR90`, and `TPR_at_TNR90` are identical across the `scientific` and `operational` rows.
3. Validate schema-level type/value constraints:
   1. enums (`selector`, `scaler`, `threshold_policy`, `eval_scope`, `model_scope`, `replication_mode`, `feature_type`)
   2. nullable rules (`n_neighbors` for non-ReliefF)
   3. index conventions (`outer_fold`, `inner_fold`, `fold`)
   4. `mspc_baseline.csv` `fold_index` type/value rule:
      always string; outer folds serialized as `"1"`, `"2"`, ... and lockbox row as `"LOCKBOX"`.
   5. `mspc_baseline.csv` scope coverage:
      1. one row per evaluated outer fold with `eval_scope='outer_fold'`
      2. exactly one lockbox row with `eval_scope='lockbox'`.
   6. artifact-specific `threshold_policy` enums:
      1. `splitwise_timeaware_results.csv`: only `outer_train_youden_ber_optimal`
      2. `final_lockbox_result.csv`: only `scientific` and `operational`.
   7. transformed feature identity consistency:
      `feature_index` uses the same 0-based scheme in both `feature_stability_by_seed.csv` and `feature_report.csv`.
4. Validate time-window formatting (`start_ts/end_ts`, timezone-naive, no `Z`).
5. Validate metric formulas and pinned implementations:
   1. BER definition
   2. ROC-AUC via `roc_auc_score`
   3. PR-AUC via `average_precision_score` (not trapezoid PR)
   4. MCC via `matthews_corrcoef`
   5. F2 via `fbeta_score(beta=2, zero_division=0)`
6. Validate drift-gate columns and mappings:
   1. PSI behavior including empty value-feature case
   2. `lockbox_claims_allowed` mapping from drift status
7. Validate run manifest required keys, hash normalization, and infeasibility fields.
   1. config-hash object form must include exactly keys `{selector, k, C, scaler, n_neighbors}`
   2. for non-ReliefF configs, `n_neighbors` must be present as JSON `null` (not omitted).

## Claim Gate Sequence

1. Benchmark claim (`33.5%`) gate:
   1. only `Replication-Strict F-test`
   2. CI upper bound < 0.335
2. Lockbox superiority gate:
   1. drift status not `HIGH_SHIFT` for that model
   2. supervised `TPR_at_TNR90` (lockbox) > MSPC `best_MSPC_TPR_at_TNR90` (lockbox)
3. If MSPC is equal or better at matched `TNR=90%`, force neutral/negative claim wording.

## Final Report Readiness Checklist

1. Manager-facing outputs in Section 11 are all populated.
2. Limitations section includes all mandatory caveats.
3. Every claim in the draft report is traceable to artifact fields and claim-policy gates.

## Exit Criteria

1. No missing required artifacts for the active feasibility mode.
2. No schema violations.
3. No claim that violates Section 13 constraints.
