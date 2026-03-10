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

## Execution Controls

1. Expected wall time: ~30 minutes to 4+ hours depending on hardware and ReliefF implementation.
2. Checkpointing is required:
   1. Stage B checkpoint unit: `(selector, outer_fold, seed)`.
   2. Phase 2 checkpoint unit: `(role, selector, config, seed, inner_fold)`.

## See Also

- [04.2.3 Stage B Method Selection](04-two-lane-plan/04.2.3-stage-b-method-selection.md)
- [04.2.4 Phase 2 Hyperparameter Freeze](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md)
