---
spec_id: "14"
title: "Pre-Registration Checklist"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 14"
applies_to:
  - "all"
keywords:
  - preregistration
  - checklist
  - registration
---
# 14 Pre-Registration Checklist

Use this as the compact pre-registration surface for the canonical strategy. Each item links back to the owning rule family instead of duplicating the full rule text.

## Data, Labels, and Split Policy

1. [Timestamp parsing contract](03-lane-b-data-partition-contract.md).
2. [Lane B lockbox holdout rule](03-lane-b-data-partition-contract.md).
3. [Label contract (`y_bin` positive class and `True+` / `True-` conventions)](03-lane-b-data-partition-contract.md).
4. [Outer fold plan, fallback hierarchy, fail minimum, and inner-CV feasibility gate](04-two-lane-plan/04.2.1-outer-time-aware-folds.md).
5. [Seed policy by stage](02-non-negotiable-validation-and-freeze-rules.md).

## Lane A Replication

1. [Lane A replication scope, pairing, classifier policy, and global OOF thresholding](04-two-lane-plan/04.1-lane-a-replication.md).
2. [ReliefF implementation determinism](02-non-negotiable-validation-and-freeze-rules.md) and [04.3 Selector Implementation Contract](04-two-lane-plan/04.3-selector-implementation-contract.md).
3. [Benchmark-claim lock rule for the `33.5%` head-to-head claim](13-claim-policy.md).

## Lane B Selection and Freeze

1. [Stage A role and fixed settings](04-two-lane-plan/04.2.2-stage-a-diagnostic.md).
2. [Stage B selector set and Pearson/F-test de-dup behavior](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
3. [Stage B and Phase 2 explicit search grids](04-two-lane-plan/04.2.3-stage-b-method-selection.md) and [04.2.4 Phase 2 Hyperparameter Freeze](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md).
4. [Inner scoring and tie-break rule](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
5. [Inner BER threshold derivation rule (inner-train only)](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
6. [Outer-test threshold derivation rule (outer-train only)](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
7. [Stage B method ranking and deterministic tie-break chain](05-stage-b-method-ranking-and-challenger.md).
8. [Challenger eligibility and deterministic tie-break chain](05-stage-b-method-ranking-and-challenger.md).
9. [No-eligible-challenger fallback behavior and artifact rules](05-stage-b-method-ranking-and-challenger.md) and [10.1 Common Artifact Conventions](10-required-artifacts-and-schemas/10.1-common-artifact-conventions.md).
10. [Phase 2 freeze rule for each available role](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md).
11. [Phase 3 refit and threshold-derivation rule](04-two-lane-plan/04.2.5-phase-3-final-refit-and-thresholds.md).
12. [Unweighted tuple-mean convention for Stage B method ranking metrics](05-stage-b-method-ranking-and-challenger.md).

## Thresholds, Drift, and Claims

1. [Supervised thresholding score, inequality rule, and candidate set](06-threshold-policy.md).
2. [Deterministic threshold tie-break rules for scientific, operational-cap, and `TNR=90%` extraction points](06-threshold-policy.md).
3. [Operational threshold constants (`<=10%` mean weekly cap, `TNR=90%`, cost-ratio set)](06-threshold-policy.md).
4. [PSI computation rule](07-lockbox-and-drift-gate.md).
5. [Drift gate criteria and claim restrictions](07-lockbox-and-drift-gate.md) and [13 Claim Policy](13-claim-policy.md).
6. [Reporting metric computation pins](08-metrics-and-feature-identity/08.1-metric-definitions.md).

## Features, Artifacts, and Runtime

1. [Lane B transformed feature identity contract](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
2. [Feature clustering rule](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
3. [Multi-seed feature-stability aggregation rule over `(outer_fold, seed)` tuple units](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
4. [Artifact-set conventions and infeasible-mode signaling](10-required-artifacts-and-schemas/10.1-common-artifact-conventions.md).
5. [Lane A artifact contract](10-required-artifacts-and-schemas/10.2-lane-a-artifacts.md).
6. [Stage B artifact contract](10-required-artifacts-and-schemas/10.3-stage-b-selection-artifacts.md).
7. [Freeze, lockbox, and MSPC artifact contract](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md).
8. [Manager-facing weekly/workload artifact contract](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md).
9. [Run-manifest deterministic hash policy, required keys, and checkpoint metadata stance](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md).
10. [Runtime, checkpoint-unit, and resume rules](09-runtime-and-checkpointing.md).
