---
spec_id: "04"
title: "Two-Lane Plan"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §4"
applies_to:
  - "Lane A"
  - "Lane B"
used_by_runbooks:
  - "../../report_strategy/runbooks/01_data_contract_and_split.md"
  - "../../report_strategy/runbooks/02_lane_a_replication.md"
  - "../../report_strategy/runbooks/03_lane_b_stage_a_stage_b.md"
  - "../../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
keywords:
  - lane a
  - lane b
  - stage a
  - stage b
  - freeze
---
# 04 Two-Lane Plan

## Scope

The project has two lanes:

1. Lane A for benchmark-faithful replication and ablation.
2. Lane B for deployment-realistic, time-aware model selection, freeze, and lockbox evaluation.

## Canonical Modules

- [04.1 Lane A Replication](04.1-lane-a-replication.md)
- [04.2.1 Outer Time-Aware Folds](04.2.1-outer-time-aware-folds.md)
- [04.2.2 Stage A Diagnostic](04.2.2-stage-a-diagnostic.md)
- [04.2.3 Stage B Method Selection](04.2.3-stage-b-method-selection.md)
- [04.2.4 Phase 2 Hyperparameter Freeze](04.2.4-phase-2-hyperparameter-freeze.md)
- [04.2.5 Phase 3 Final Refit and Threshold Derivation](04.2.5-phase-3-final-refit-and-thresholds.md)
- [04.3 Selector Implementation Contract](04.3-selector-implementation-contract.md)
