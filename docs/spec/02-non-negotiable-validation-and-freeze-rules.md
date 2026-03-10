---
spec_id: "02"
title: "Non-Negotiable Validation and Freeze Rules"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §2"
applies_to:
  - "Lane A"
  - "Lane B"
used_by_runbooks:
  - "../report_strategy/runbooks/01_data_contract_and_split.md"
  - "../report_strategy/runbooks/05_artifacts_and_claim_checks.md"
keywords:
  - validation
  - freeze policy
  - leakage
  - randomness
---
# 02 Non-Negotiable Validation and Freeze Rules

## Scope

These rules apply across the project and must not be overridden by local workflow convenience.

## Normative Rules

1. No leakage: every preprocessing, feature-selection, and model-fitting step is fit on training data only for each split.
2. No lockbox tuning in Lane B: Lane B lockbox is touched once after model and threshold freeze.
3. Drift gate is mandatory before any lockbox superiority claim.
4. Feature interpretations are associative, not causal.
5. Randomness policy is pre-registered and mandatory:
   1. Lane A uses `StratifiedKFold(..., shuffle=True, random_state=42)` (fixed folds).
   2. Stage B uses seed set `{42, 11, 23, 37, 59}` for inner `StratifiedKFold(..., shuffle=True, random_state=seed)`.
   3. Phase 2 uses seed set `{42, 11, 23, 37, 59}` for repeated inner `StratifiedKFold(..., shuffle=True, random_state=seed)`.
   4. Phase 3 final refit uses seed `42` where applicable (for components that accept `random_state`).
   5. ReliefF implementation is deterministic in this project (`skrebate.ReliefF` has no `random_state` parameter); any variation arises from CV split assignment, not selector RNG.
6. Freeze policy is two-step:
   1. Method freeze: after Stage B method ranking.
   2. Hyperparameter freeze: after Phase 2 repeated inner CV on full DEV.
7. After freeze, no changes to:
   1. method set,
   2. search grid,
   3. tie-break logic,
   4. threshold policy.

## See Also

- [05 Stage B Method Ranking and Challenger Rules](05-stage-b-method-ranking-and-challenger.md)
- [06 Threshold Policy](06-threshold-policy.md)
- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
