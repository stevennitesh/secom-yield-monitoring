---
spec_id: "05"
title: "Stage B Method Ranking and Challenger Rules"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 5"
applies_to:
  - "Lane B"
keywords:
  - ranking
  - challenger
  - tie break
---
# 05 Stage B Method Ranking and Challenger Rules

## Scope

This section converts Stage B outer-test results into the frozen primary method and optional challenger method.

## Primary Model Selection

1. Rank methods by lowest mean outer BER across all `(outer_fold x seed)` results.
   Mean is the unweighted arithmetic mean over tuple-level outer-test values (each `(outer_fold, seed)` counts equally).
2. Tie-break 1: lower standard deviation of per-fold BER means (temporal stability).
   Definition: for each `outer_fold=f`, compute `mu_f = mean_seed(BER_{f,seed})`; then `std_per_fold_BER_means = std_f(mu_f)` with `ddof=1`.
3. Tie-break 2: higher mean True+ across `(outer_fold x seed)` (unweighted over tuples).
4. Tie-break 3:
   1. smaller modal `k` (if no unique mode, choose the smallest tied `k`),
   2. if still tied: smaller modal `C` (if no unique mode, choose the smallest tied `C`),
   3. if still tied: `StandardScaler` over `RobustScaler`,
   4. if still tied: deciding vote from `(seed=42, final outer fold in the active outer-fold plan)`:
      the active outer-fold plan is recorded in `reports/run_manifest.json` via `outer_fold_plan_used` and `outer_fold_week_ranges`;
      because the plan uses expanding windows, the final outer fold is the largest outer training window by construction
       1. lower outer-test BER,
       2. if still tied: higher outer-test True+,
       3. if still tied: lexicographically smaller selector name.

## Challenger Selection

1. Candidate pool: non-primary methods.
2. Eligibility: mean BER `<= 0.40`.
3. Among eligible methods, select highest mean True- across `(outer_fold x seed)` (unweighted over tuples).
4. Challenger tie-breaks:
   1. lower mean BER,
   2. lower std of per-fold BER means,
   3. lexicographically smaller selector name.
5. Challenger goes through Phase 2 and Phase 3 if eligible.

## No-Eligible-Challenger Fallback

1. Set `challenger_available=false`.
2. Run Phase 2 and Phase 3 for primary only.
3. `reports/final_lockbox_result.csv`: omit challenger rows (primary rows only).
4. `reports/operational_cost_curves.csv`: keep challenger columns but write challenger values as `NA`.
5. `reports/hyperparameter_freeze_results.csv`: primary-role rows only (no challenger rows).
6. Write challenger outputs as `NA` only in column-based artifacts that include challenger fields; do not create challenger rows in row-based artifacts.
7. In `reports/run_manifest.json`, write `challenger_available=false` and `challenger_unavailable_reason='no_eligible_method_under_BER_0.40'`.

## Decision Outputs

1. `reports/timeaware_model_selection.csv` is the canonical row-based output of the ranking decision:
   1. `is_primary=true` marks the winning primary method,
   2. `is_challenger=true` marks the selected challenger when one exists,
   3. if no challenger is eligible, no row in this artifact may have `is_challenger=true`.
2. `reports/run_manifest.json` is the canonical run-level output of the ranking decision:
   1. it records the active outer-fold plan used to interpret the deciding-vote tie-break,
   2. it records `challenger_available`,
   3. when `challenger_available=false`, it records `challenger_unavailable_reason`.

## See Also

- [04.2 Lane B Selection and Freeze Overview](04-two-lane-plan/04.2-lane-b-selection-and-freeze-overview.md)
- [04.2.3 Stage B Method Selection](04-two-lane-plan/04.2.3-stage-b-method-selection.md)
- [04.2.4 Phase 2 Hyperparameter Freeze](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md)
- [10.3 Stage B Selection Artifacts](10-required-artifacts-and-schemas/10.3-stage-b-selection-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md)
