---
spec_id: "14"
title: "Pre-Registration Checklist"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §14"
applies_to:
  - "all"
used_by_runbooks:
  - "../report_strategy/runbooks/01_data_contract_and_split.md"
  - "../report_strategy/runbooks/02_lane_a_replication.md"
  - "../report_strategy/runbooks/03_lane_b_stage_a_stage_b.md"
  - "../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
  - "../report_strategy/runbooks/05_artifacts_and_claim_checks.md"
keywords:
  - preregistration
  - checklist
  - audit
---
# 14 Pre-Registration Checklist

Use this as the compact audit checklist. Each item links back to the canonical rule family instead of duplicating the full rule text.

1. [Lane B lockbox holdout rule](03-lane-b-data-partition-contract.md).
2. [Outer fold plan, fallback hierarchy, fail minimum, and inner-CV feasibility gate](04-two-lane-plan/04.2.1-outer-time-aware-folds.md).
3. [Seed policy by stage](02-non-negotiable-validation-and-freeze-rules.md).
4. [Stage A role and fixed settings](04-two-lane-plan/04.2.2-stage-a-diagnostic.md).
5. [Stage B selector set and Pearson/F-test de-dup behavior](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
6. [Stage B and Phase 2 explicit search grids](04-two-lane-plan/04.2.3-stage-b-method-selection.md) and [04.2.4 Phase 2 Hyperparameter Freeze](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md).
7. [Inner scoring and tie-break rule](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
8. [Inner BER threshold derivation rule (inner-train only)](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
9. [Outer-test threshold derivation rule (outer-train only)](04-two-lane-plan/04.2.3-stage-b-method-selection.md).
10. [Stage B method ranking and deterministic tie-break chain](05-stage-b-method-ranking-and-challenger.md).
11. [Challenger eligibility and deterministic tie-break chain](05-stage-b-method-ranking-and-challenger.md).
12. [No-eligible-challenger fallback behavior and artifact rules](05-stage-b-method-ranking-and-challenger.md) and [10.1 Common Artifact Conventions](10-required-artifacts-and-schemas/10.1-common-artifact-conventions.md).
13. [Phase 2 freeze rule for each available role](04-two-lane-plan/04.2.4-phase-2-hyperparameter-freeze.md).
14. [Phase 3 refit and threshold-derivation rule](04-two-lane-plan/04.2.5-phase-3-final-refit-and-thresholds.md).
15. [Operational threshold constants (`<=10%` mean weekly cap, `TNR=90%`, cost-ratio set)](06-threshold-policy.md).
16. [Drift gate criteria and claim restrictions](07-lockbox-and-drift-gate.md) and [13 Claim Policy](13-claim-policy.md).
17. [Required artifacts, row grains, and uniqueness constraints](10-required-artifacts-and-schemas/README.md).
18. [Run-manifest deterministic hash policy and required keys](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md).
19. [ReliefF implementation determinism](02-non-negotiable-validation-and-freeze-rules.md) and [04.3 Selector Implementation Contract](04-two-lane-plan/04.3-selector-implementation-contract.md).
20. [Multi-seed feature-stability aggregation rule over `(outer_fold, seed)` tuple units](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
21. [Benchmark-claim lock rule for the `33.5%` head-to-head claim](13-claim-policy.md).
22. [Deterministic threshold tie-break rules for scientific, operational-cap, and `TNR=90%` extraction points](06-threshold-policy.md).
23. [Label contract (`y_bin` positive class and `True+` / `True-` conventions)](03-lane-b-data-partition-contract.md).
24. [Lane B transformed feature identity contract](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
25. [PSI computation rule](07-lockbox-and-drift-gate.md).
26. [Supervised thresholding score, inequality rule, and candidate set](06-threshold-policy.md).
27. [Lane A replication scope, pairing, classifier policy, and global OOF thresholding](04-two-lane-plan/04.1-lane-a-replication.md).
28. [Feature clustering rule](08-metrics-and-feature-identity/08.2-feature-stability-and-identity.md).
29. [Timestamp parsing contract](03-lane-b-data-partition-contract.md).
30. [Selector implementation contract](04-two-lane-plan/04.3-selector-implementation-contract.md).
31. [Unweighted tuple-mean convention for Stage B method ranking metrics](05-stage-b-method-ranking-and-challenger.md).
32. [Reporting metric computation pins](08-metrics-and-feature-identity/08.1-metric-definitions.md).
