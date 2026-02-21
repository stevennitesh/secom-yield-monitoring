# Runbook 04: Phase 2 Freeze, Phase 3 Refit, Lockbox Evaluation

Canonical source: `docs/final_end_to_end_report_strategy_merged.md` (Sections 4.2.4, 4.2.5, 6, 7, 8, 10, 13, 14).

## Objective

Freeze role-specific configurations on full DEV, refit final models, derive thresholds, and score lockbox once.

## Inputs

1. Primary (and optional challenger) methods from Stage B.
2. Full DEV slice.
3. Seed set `{42, 11, 23, 37, 59}` for Phase 2 repeated inner CV.

## Outputs

1. `reports/hyperparameter_freeze_results.csv`
2. `reports/final_lockbox_result.csv`
3. `reports/mspc_baseline.csv`
4. `reports/drift_gate_summary.csv`

## Phase 2 (Hyperparameter Freeze)

1. Run per available role:
   1. primary always
   2. challenger only when eligible
2. For each role:
   1. repeated inner CV on full DEV: 5 seeds x 5 folds = 25 observations/config
   2. fixed pipeline/settings are pinned:
      1. `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`
      2. scaler is a Phase 2 search dimension from the pre-registered grid (it is not frozen yet):
         1. `StandardScaler(with_mean=True, with_std=True)`
         2. `RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0,75.0))`
      3. `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`
      4. ReliefF uses deterministic `skrebate.ReliefF(..., n_jobs=-1)` (no `random_state`).
   3. same inner selection criteria/tie-break chain as Stage B
3. Freeze one config per role.
4. Ensure `is_frozen_config=True` uniqueness per role.

## Phase 3 (Final Refit and Threshold Freeze)

1. Refit each available frozen role config on full DEV once.
2. Derive thresholds on full-DEV in-sample predictions:
   1. scientific threshold (BER-optimal with tie-break rules)
   2. operational threshold (`<=10%` mean weekly flagged fraction using anchored 7-day bins)
      where the weekly mean is computed only over weeks with `wafers_w > 0` (exclude empty weeks).

## Lockbox Scoring

1. Score lockbox once per available role using frozen model.
2. Evaluate at scientific and operational thresholds.
3. Derive reporting-only matched point on the lockbox slice (not on DEV):
   1. `threshold_at_TNR90`
   2. `TNR_at_TNR90`
   3. `TPR_at_TNR90`
   using the canonical `TNR=90%` extraction rule.
4. Keep lockbox untouched for any tuning choice.

## Drift Gate

1. Compute per role:
   1. prevalence shift
   2. KS score-shift p-value
   3. PSI summary (`max_PSI`, `median_PSI`, `psi_feature_count`)
2. Assign drift status (`PASS`, `CAUTION`, `HIGH_SHIFT`).
3. Set `lockbox_claims_allowed` per fixed mapping.

## MSPC Companion

1. Outer-fold MSPC evaluation is mandatory:
   1. for each outer fold, fit MSPC on that outer-train pass subset only,
   2. score the corresponding outer-test slice,
   3. use union alarm stream: `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)`,
   4. write one row per outer fold with `eval_scope='outer_fold'`.
2. Lockbox MSPC companion evaluation is mandatory:
   1. fit MSPC on full-DEV pass wafers only,
   2. score lockbox once,
   3. use the same union alarm stream for lockbox metrics,
   4. write one row with `eval_scope='lockbox'`.
3. Report `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, `best_MSPC_TPR_at_TNR90`, `best_MSPC_source` for both outer-fold and lockbox rows.

## Exit Criteria

1. Exactly one lockbox pass (no post-lockbox tuning).
2. Thresholds and drift gate values are present per available role.
3. Supervised vs MSPC matched-point fields exist for claim checks.
4. In `reports/final_lockbox_result.csv`, for each role, `threshold_at_TNR90`, `TNR_at_TNR90`, and `TPR_at_TNR90` are identical across that role's `scientific` and `operational` rows.
5. `reports/mspc_baseline.csv` includes all required scopes:
   1. one row for each evaluated outer fold (`eval_scope='outer_fold'`)
   2. one lockbox row (`eval_scope='lockbox'`).
