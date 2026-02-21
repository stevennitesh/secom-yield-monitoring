# Runbook 01: Data Contract and DEV/LOCKBOX Split

Canonical source: `docs/final_end_to_end_report_strategy_merged.md` (Sections 2, 3, 4.2.1, 10, 14).

## Objective

Create deterministic dataset partitions and fold definitions with no ambiguity and no leakage.

## Inputs

1. Raw SECOM feature matrix and labels/time columns.
2. Canonical strategy document.

## Outputs

1. Deterministic sorted dataset with `raw_row_id`.
2. `DEV` and `LOCKBOX` partition metadata.
3. Lane B outer fold plan selection (`primary_3fold`, `fallback_3fold`, or `fallback_2fold`) or explicit Lane B infeasibility.

## Procedure

1. Parse timestamps as day-first `DD/MM/YYYY HH:MM:SS` with `errors='coerce'`.
2. Drop `NaT` rows.
3. Add `raw_row_id` as the original 0-based raw file index.
4. Stable sort by `(timestamp asc, raw_row_id asc)`.
5. Build lockbox by count:
   1. `N = row_count_after_NaT_drop`
   2. `N_lockbox = floor(0.15 * N)`
   3. `LOCKBOX = last N_lockbox rows`
   4. `DEV = first (N - N_lockbox) rows`
6. Build DEV week bins:
   1. `t_min_dev = min(timestamp in DEV)`
   2. `week_idx = floor((timestamp - t_min_dev)/7days)`
   3. `week_label = week_idx + 1`
   4. `last_week = max(week_label in DEV)`; all fold ranges below are inclusive and `last` means this value.
7. Evaluate fold plan in deterministic order:
   1. primary 3-fold:
      1. Fold 1: train weeks 1-5, test weeks 6-last
      2. Fold 2: train weeks 1-7, test weeks 8-last
      3. Fold 3: train weeks 1-9, test weeks 10-last
   2. fallback 3-fold:
      1. Fold 1: train weeks 1-4, test weeks 5-last
      2. Fold 2: train weeks 1-6, test weeks 7-last
      3. Fold 3: train weeks 1-8, test weeks 9-last
   3. fallback 2-fold:
      1. Fold A: train weeks 1-6, test weeks 7-last
      2. Fold B: train weeks 1-8, test weeks 9-last
   4. if `fallback_2fold` is selected, enforce artifact labeling:
      1. Fold A -> `outer_fold=1`
      2. Fold B -> `outer_fold=2`
      in all fold-level artifacts.
8. Enforce per-outer-test minimum fails `>=20`.
9. Enforce inner-CV feasibility gate for any future `StratifiedKFold(n_splits=5)` usage:
   1. class presence and minimum class count in every outer fold's outer-train slice
   2. `min(n_fail, n_pass) >= 5`
   3. before Phase 2, re-check the full DEV slice also satisfies class presence and `min(n_fail, n_pass) >= 5`.
10. If infeasible after fallback or inner-CV gate failure, set Lane B infeasible and stop Lane B path.

## Mandatory Recorded Fields

1. `N_total_after_NaT_drop`, `N_dev`, `N_lockbox`
2. Lockbox rule string
3. `outer_fold_plan_used`
4. `outer_fold_week_ranges`
5. `lane_b_feasible`
6. `lane_b_infeasible_reason` when infeasible
   1. use exact reason string `min_class_count_lt_5_for_inner_cv` for inner-CV feasibility failures.

## Exit Criteria

1. Re-running split logic yields identical row membership and fold windows.
2. No overlap between DEV and LOCKBOX.
3. Fold windows match canonical definitions.

## Failure Cases

1. No valid timestamps after parsing -> hard fail.
2. Any nondeterministic sort or tie handling -> hard fail.
3. Missing feasibility metadata in manifest -> hard fail.
