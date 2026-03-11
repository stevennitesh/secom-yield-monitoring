# Agreed Changes: All-Selector Multi-Seed Strategy Update (Revised v2)

## Purpose

This document captures the currently agreed changes for the next strategy revision.
It is a planning artifact only (no implementation code changes).

## Agreed High-Level Architecture

1. Stage A remains mandatory, but only as a fixed-settings diagnostic (no elimination gate).
2. Stage B performs multi-seed nested CV for method selection across all distinct selectors.
3. Phase 2 (hyperparameter freeze) is run on full DEV for both roles when challenger is available:
   - primary method,
   - challenger method.
4. Phase 3 (final refit) trains frozen primary and, when available, frozen challenger using seed `42`.
5. Only these frozen Phase-3 refit models are lockbox-eligible.

## Agreed Selector Scope

1. Stage A includes all six baseline selectors:
   - `S2N`, `Welch-t`, `F-test`, `Pearson`, `ReliefF`, `Gram-Schmidt`.
2. Stage B applies Pearson/F-test de-dup:
   - keep `Pearson` in Stage A for comparability and narrative,
   - advance only `F-test` from that equivalent pair into Stage B.
3. Stage B selector set is therefore:
   - `S2N`, `Welch-t`, `F-test`, `ReliefF`, `Gram-Schmidt`.

## Agreed Seed and Randomness Policy

1. Multi-seed set is fixed to `{42, 11, 23, 37, 59}`.
2. Stage A is deterministic:
   - `ReliefF` must use `random_state=42` in Stage A.
3. Stage B uses all five seeds in the set above.
4. Phase 2 uses the same pre-registered seed set `{42, 11, 23, 37, 59}`.
5. Phase 3 final refit uses seed `42` only.

## Stage A (Diagnostic Only)

1. Fixed settings, outer folds only, no inner CV.
2. Stage A fixed settings are pinned in this document:
   - `k=40`,
   - `C=1.0`,
   - scaler=`RobustScaler`,
   - missing mode=`values + indicators`,
   - classifier=`LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`,
   - ReliefF-only setting: `n_neighbors=10`.
3. Threshold policy in Stage A is fixed to BER-optimal threshold on outer-train (Youden) applied unchanged to outer-test.
4. Purpose: report untuned baseline behavior for all selectors.
5. Stage A results are not used to eliminate or promote selectors.

## Agreed Stage B / Phase 2 Search Grid (Explicit)

1. For non-ReliefF selectors (`S2N`, `Welch-t`, `F-test`, `Gram-Schmidt`):
   - `k in {10, 20, 40}`,
   - `C in {0.01, 0.1, 1.0, 10.0}`,
   - scaler in `{StandardScaler, RobustScaler}`.
2. For ReliefF:
   - same `k`, `C`, and scaler grid as above,
   - plus `n_neighbors in {5, 10, 20}`.
3. Missing mode is fixed to `values + indicators` (not a grid dimension).
4. Classifier is fixed to `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)` in both Stage B and Phase 2.

## Stage B (Method Selection, Multi-Seed Nested CV)

For each `(selector, outer_fold, seed)`:

1. Run inner `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.
2. Select best config `(k, C, scaler[, n_neighbors for ReliefF])` by:
   - primary inner sort: highest mean inner `ROC-AUC`,
   - tie-break for near-equal configs: lowest mean inner out-of-fold BER among configs within `0.01` AUC of the best config.
   - if still tied, apply deterministic config tie-break: smaller `k`, then smaller `C`, then `StandardScaler` over `RobustScaler`, then smaller `n_neighbors` (when present).
3. Inner BER computation rule (mandatory):
   - derive BER-optimal threshold (Youden) on inner-train only,
   - apply that threshold unchanged to inner-validation,
   - do not use inner-validation labels to set threshold.
4. Outer-test evaluation rule (mandatory):
   - using the inner-selected config for `(selector, outer_fold, seed)`, derive BER-optimal threshold (Youden) on that outer fold's outer-train slice only,
   - apply that threshold unchanged to that outer fold's outer-test slice to compute BER/True+/True-.

Method ranking rule (Stage B output):

1. Primary ranking: lowest mean outer BER across all `(outer_fold x seed)` results.
2. Tie-break 1: lowest standard deviation of per-fold BER means (temporal stability).
3. Tie-break 2: highest mean True+ across `(outer_fold x seed)`.
4. Tie-break 3 (deterministic complexity chain):
   - 3a: smaller modal `k` across `(outer_fold x seed)` winners,
   - 3b: if tied, smaller modal `C`,
   - 3c: if tied, prefer `StandardScaler` over `RobustScaler`,
   - 3d: if tied, use `(seed=42, largest outer training window)` as deciding vote:
     choose lower outer-test BER; if tied, choose higher outer-test True+; if still tied, choose lexicographically smaller selector name.

Challenger definition:

1. Candidate pool: non-primary methods.
2. Eligibility: mean BER `<= 0.40`.
3. Selection: highest mean True- among eligible candidates, where mean True- is computed across all `(outer_fold x seed)` results.
   - deterministic challenger tie-breaks (if mean True- ties):
     1) lower mean BER,
     2) lower std of per-fold BER means,
     3) lexicographically smaller selector name.
4. Challenger also goes through Phase 2 freeze and Phase 3 final refit.

No-eligible-challenger fallback (pre-registered):

1. If no non-primary method satisfies mean BER `<= 0.40`, challenger is declared unavailable for that run.
2. Phase 2 and Phase 3 run for primary only.
3. `final_lockbox_result.csv` contains primary rows only (scientific + operational).
4. `operational_cost_curves.csv` challenger columns are written as `NA`.
5. `run_manifest.json` follows Artifact 8 policy; in this fallback it must set `challenger_available=false` with reason `no_eligible_method_under_BER_0.40`.
6. Phase 2 runtime in this fallback is primary-only (`600` to `1,800` fits depending on selector).
7. `hyperparameter_freeze_results.csv` contains primary-role rows only (no challenger rows).

## Phase 2: Hyperparameter Freeze on Full DEV

Run this phase separately for primary and challenger methods.

1. Input: selected method from Stage B (primary or challenger role).
2. Run repeated inner CV on full DEV:
   - use the same pre-registered seed set as Stage B: `{42, 11, 23, 37, 59}`,
   - 5 seeds x 5 folds = 25 inner observations per config.
3. Config selection rule:
   - first criterion: highest mean inner `ROC-AUC`,
   - tie-break: lowest mean inner out-of-fold BER within `0.01` AUC of best.
   - if still tied, apply deterministic config tie-break: smaller `k`, then smaller `C`, then `StandardScaler` over `RobustScaler`, then smaller `n_neighbors` (when present).
   - this same rule applies to both roles (`primary` and `challenger`).
4. Inner BER computation rule is identical to Stage B:
   - threshold from inner-train only, then applied to inner-validation.
5. Output: frozen `(selector, k, C, scaler[, n_neighbors])` for each role.

## Phase 3: Final Refit and Lockbox Eligibility

1. Train frozen primary config on full DEV with seed `42`.
2. Train frozen challenger config on full DEV with seed `42`, if challenger is available.
3. Derive scientific and operational thresholds on full-DEV in-sample predictions for each available frozen model (primary, and challenger if available).
4. Only these frozen Phase-3 models are lockbox-eligible supervised models.

## Runtime and Execution Controls

1. Stage B remains approximately `12,600` inner fits under current grid assumptions.
2. Phase 2 adds:
   - `600` to `1,800` inner fits per role (non-ReliefF vs ReliefF winner),
   - `1,200` to `3,600` total for primary+challenger.
3. Expected wall-clock range: ~30 minutes to 4+ hours depending on hardware and ReliefF implementation.
4. Checkpointing is required for both Stage B and Phase 2:
   - Stage B unit: `(selector, outer_fold, seed)`,
   - Phase 2 unit: `(role, selector, config, seed, inner_fold)`.

## Agreed Artifact Changes

Index format convention used by all artifacts below:

- `outer_fold`: 1-based integer index (`1, 2, ..., n_outer_folds`).
- `inner_fold`: 1-based integer index (`1, 2, ..., n_inner_folds`).

1. `reports/splitwise_timeaware_results.csv`
   - one row per `(selector, outer_fold, seed)` storing the inner-selected winner config and outer-test metrics,
   - `train_window` / `test_window` format: ISO-8601 interval string `start_ts/end_ts` in UTC,
   - minimum columns:
      `selector`, `outer_fold`, `seed`, `train_window`, `test_window`, `k`, `C`, `scaler`, `n_neighbors` (nullable),
      `threshold_policy` (`outer_train_youden_ber_optimal`), `outer_threshold`, `test_fails`, `BER`, `True+`, `True-`.
2. `reports/stage_b_inner_cv_results.csv` (new)
   - one row per `(selector, outer_fold, seed, k, C, scaler, n_neighbors)`,
   - include explicit config columns: `k`, `C`, `scaler`, `n_neighbors` (nullable for non-ReliefF),
   - include mean inner `ROC-AUC`, mean inner BER, and `is_selected_config` (boolean) for auditability of inner selection.
   - uniqueness constraint: for each `(selector, outer_fold, seed)`, exactly one row must have `is_selected_config=True`.
3. `reports/timeaware_selector_screening.csv`
   - remains Stage A fixed-settings diagnostic summary (no promotion fields).
4. `reports/timeaware_model_selection.csv`
   - Stage B method-level summary aggregated over `(outer_fold x seed)`,
   - minimum columns:
      `selector`, `n_folds`, `n_seeds`, `mean_BER`, `std_BER`, `std_per_fold_BER_means`,
      `mean_True+`, `std_True+`, `mean_True-`, `std_True-`, `is_primary`, `is_challenger`,
   - config-level columns are intentionally not included here (`config_id`, `k`, `C`, `scaler`, `n_neighbors`) because inner-selected configs vary by `(outer_fold, seed)`.
   - no-challenger fallback behavior: include only observed selector rows; if challenger is unavailable, no row has `is_challenger=True`.
5. `reports/seed_stability_summary.csv` (new)
   - one row per `(selector, seed)` with minimum columns:
     `selector`, `seed`, `mean_outer_BER`, `std_outer_BER`,
     `mean_outer_True+`, `mean_outer_True-`,
     `modal_k`, `modal_C`, `modal_scaler`, `modal_n_neighbors` (nullable), `n_outer_folds`.
   - modal tie rule (deterministic):
     - if no unique mode for `modal_k`, choose the smallest tied `k`,
     - if no unique mode for `modal_C`, choose the smallest tied `C`,
     - if no unique mode for scaler, prefer `StandardScaler` over `RobustScaler`,
     - if no unique mode for `modal_n_neighbors`, choose the smallest tied value.
6. `reports/feature_stability_by_seed.csv` (new)
   - row scope is full transformed-feature universe per `(selector, seed, outer_fold)`:
     every candidate transformed feature appears exactly once with a boolean `selected` flag.
   - one row per `(selector, seed, outer_fold, feature_id)` with minimum columns:
      `selector`, `seed`, `outer_fold`, `feature_id`, `feature_type` (`value`/`missing_indicator`), `selected` (boolean).
   - feature identity join-key rule: `feature_id` must match `feature_index` used by `feature_report.csv`.
   - cross-seed Jaccard is computed from this table as a derived analysis artifact (not hand-entered).
7. `reports/hyperparameter_freeze_results.csv` (new)
   - one row per `(role, selector, k, C, scaler, n_neighbors, seed, inner_fold)`,
   - full-DEV Phase-2 repeated-inner-CV results for both roles with minimum columns:
      `role` (`primary`/`challenger`), `selector`, `k`, `C`, `scaler`, `n_neighbors` (nullable),
      `seed`, `inner_fold`, `inner_ROC_AUC`, `inner_BER`,
      `mean_inner_ROC_AUC_by_config`, `mean_inner_BER_by_config`,
      `is_frozen_config` (boolean).
   - uniqueness constraint: for each `role`, exactly one unique config `(selector, k, C, scaler, n_neighbors)` has `is_frozen_config=True` across all `seed x inner_fold` rows.
8. `run_manifest.json` (updated)
   - must record the multi-seed set used for Stage B and Phase 2: `{42, 11, 23, 37, 59}`,
   - must always record `challenger_available` (`true`/`false`); if `false`, also record reason `no_eligible_method_under_BER_0.40`,
   - must record Phase-2 frozen config identity hashes for each available role (`primary` always, `challenger` when available),
   - hash spec (deterministic): `SHA-256` of UTF-8 JSON canonical serialization with sorted keys and compact separators `(",", ":")` (no extra whitespace).

## Pre-Registration Checklist

1. Seed set: `{42, 11, 23, 37, 59}`.
2. Stage A ReliefF determinism: `random_state=42`.
3. Stage A role: diagnostic only (no elimination).
4. Pearson/F-test de-dup behavior in Stage B (F-test only).
5. Inner metric rule: `ROC-AUC` primary with BER tie-break within `0.01` AUC of best.
6. Deterministic inner-config tie-break chain after ROC-AUC/BER tie: smaller `k`, then smaller `C`, then `StandardScaler` over `RobustScaler`, then smaller `n_neighbors` (when present).
7. Inner BER threshold rule: threshold from inner-train only, never from inner-validation.
8. Outer-test evaluation rule: BER-optimal threshold (Youden) from outer-train only, applied unchanged to outer-test.
9. Stage B method ranking rule and full deterministic tie-break chain.
10. Challenger definition, eligibility rule (`BER <= 0.40`), and challenger deterministic tie-break chain.
11. Phase 2 full-DEV repeated-inner-CV freeze rule for both primary and challenger.
12. Phase 2 seed-set rule: use `{42, 11, 23, 37, 59}`.
13. Final refit seed policy: `42` only.
14. Checkpointing requirement for Stage B and Phase 2 units.
15. No-eligible-challenger fallback policy and artifact behavior (`NA` challenger outputs + manifest flag).
16. Modal tie rule for `seed_stability_summary.csv` fields.
17. Stage A fixed settings (`k=40`, `C=1.0`, scaler=`RobustScaler`, missing mode=`values + indicators`, ReliefF `n_neighbors=10`).
18. Stage B/Phase 2 explicit grid (`k`, `C`, scaler, and ReliefF `n_neighbors`) and fixed classifier settings.
