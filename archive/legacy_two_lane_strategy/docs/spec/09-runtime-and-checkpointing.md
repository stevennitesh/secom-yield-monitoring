---
spec_id: "09"
title: "Runtime and Checkpointing"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 9"
applies_to:
  - "Lane B"
keywords:
  - runtime
  - checkpointing
  - workload
---
# 09 Runtime and Checkpointing

## Stage B Expected Workload

1. Non-ReliefF selectors: `4 x 5 seeds x 3 outer folds x 24 configs x 5 inner folds = 7200`.
2. ReliefF selector: `1 x 5 x 3 x 72 configs x 5 = 5400`.
3. Total Stage B inner fits: `12600`.
4. If 2-fold outer fallback is active, multiply Stage B inner-fit counts above by `2/3`.

## Phase 2 Expected Workload

1. Per role, non-ReliefF winner: `24 configs x 25 = 600`.
2. Per role, ReliefF winner: `72 configs x 25 = 1800`.
3. Total Phase 2 for primary+challenger: `1200` to `3600`.
4. If no challenger: `600` to `1800`.
5. If Lane B runs with 2 outer folds, Phase-2 freeze workload is unchanged (Phase 2 is full-DEV repeated inner CV).

## Advisory Runtime Guidance

1. Expected wall time is advisory, not contractual: ~30 minutes to 4+ hours depending on hardware and ReliefF implementation.

## Checkpoint Units

1. Stage B checkpoint unit: `(selector, outer_fold, seed)`.
2. Phase 2 checkpoint unit: `(role, selector, config, seed, inner_fold)`.

## Completion Rule

1. A checkpoint unit is complete only when its row-level outputs have been durably written to the canonical artifacts for that unit.
2. Stage B unit completion requires the rows for that `(selector, outer_fold, seed)` to be present in:
   1. `reports/stage_b_inner_cv_results.csv`,
   2. `reports/splitwise_timeaware_results.csv`,
   3. `reports/feature_stability_by_seed.csv`.
3. Phase 2 unit completion requires the row for that `(role, selector, config, seed, inner_fold)` to be present in `reports/hyperparameter_freeze_results.csv`.
4. Summary artifacts must not be treated as checkpoint evidence on their own; they are downstream aggregations and may be regenerated from row-level artifacts.

## Resume Rule

1. On resume, a completed checkpoint unit may be skipped only if its required row-level outputs are already present and consistent with the active run configuration.
2. Consistency must be evaluated against the active run metadata recorded in `reports/run_manifest.json`, including at least:
   1. seed policy,
   2. split boundaries / outer-fold plan,
   3. search-grid and frozen-config context where applicable.
3. If a checkpoint unit is partial, missing any required row-level output, or inconsistent with the active run metadata, that unit must be recomputed from scratch.

## Decision Outputs

1. Row-level artifact outputs:
   1. Stage B checkpoint progress is materially evidenced by `reports/stage_b_inner_cv_results.csv`, `reports/splitwise_timeaware_results.csv`, and `reports/feature_stability_by_seed.csv`,
   2. Phase 2 checkpoint progress is materially evidenced by `reports/hyperparameter_freeze_results.csv`.
2. Run-level output:
   1. `reports/run_manifest.json` is the canonical run-level output for checkpoint and restart metadata,
   2. any persisted checkpoint/restart metadata in the manifest must remain consistent with the materialized row-level artifacts.

## See Also

- [04.2.3 Stage B Method Selection](04-two-lane-plan/04.2.3-stage-b-method-selection.md)
- [04.2.4 Phase 2 Hyperparameter Freeze](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md)
- [10.3 Stage B Selection Artifacts](10-required-artifacts-and-schemas/10.3-stage-b-selection-artifacts.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md)
