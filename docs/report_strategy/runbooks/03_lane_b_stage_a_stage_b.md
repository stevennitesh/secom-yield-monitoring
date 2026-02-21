# Runbook 03: Lane B Stage A and Stage B

Canonical source: `docs/final_end_to_end_report_strategy_merged.md` (Sections 4.2.2, 4.2.3, 4.3, 5, 8, 9, 10, 14).

## Objective

Run diagnostic Stage A and multi-seed nested-CV Stage B to select primary/challenger methods under time-aware evaluation.

## Inputs

1. Lane B feasible DEV folds from Runbook 01.
2. Seed set `{42, 11, 23, 37, 59}` for Stage B inner CV.

## Outputs

1. `reports/timeaware_selector_screening.csv`
2. `reports/splitwise_timeaware_results.csv`
3. `reports/stage_b_inner_cv_results.csv`
4. `reports/timeaware_model_selection.csv`
5. `reports/seed_stability_summary.csv`
6. `reports/feature_stability_by_seed.csv`

## Stage A (Diagnostic Only)

1. Run all 6 selectors.
2. Fixed settings:
   1. `k=40`
   2. `C=1.0`
   3. `RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0,75.0))`
   4. `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`
   5. `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`
   6. ReliefF `n_neighbors=10`
   7. threshold policy `outer_train_youden_ber_optimal`:
      derive BER-optimal threshold on outer-train and apply unchanged to outer-test.
3. No inner CV. No elimination gate.

## Stage B (Method Selection)

1. Selector scope after de-dup:
   1. `S2N`, `Welch-t`, `F-test`, `ReliefF`, `Gram-Schmidt`.
2. For each `(selector, outer_fold, seed)`:
   1. inner CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
   2. grid:
      1. non-ReliefF: `k in {10,20,40}`, `C in {0.01,0.1,1.0,10.0}`, scaler in `{StandardScaler, RobustScaler}`
      2. ReliefF adds `n_neighbors in {5,10,20}`
   3. fixed preprocessing/classifier:
      1. `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`
      2. scaler is from the grid with pinned params:
         1. `StandardScaler(with_mean=True, with_std=True)`
         2. `RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0,75.0))`
      3. `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`
      4. ReliefF uses `skrebate.ReliefF(..., n_jobs=-1)` deterministically (no `random_state`).
3. Inner config selection order:
   1. max mean inner ROC-AUC
   2. among configs within 0.01 AUC: min mean inner BER
   3. deterministic ties: smaller `k`, smaller `C`, `StandardScaler`, smaller `n_neighbors`
4. Inner BER for each fold:
   1. fit on inner-train
   2. derive threshold on inner-train only
   3. apply unchanged to inner-validation
5. Outer evaluation:
   1. refit chosen config on outer-train
   2. derive threshold on outer-train only
   3. apply unchanged to outer-test
6. Persist tuple-level outputs and selected config flags.
7. Build `reports/feature_stability_by_seed.csv` with full transformed-feature-universe scope:
   1. for each `(selector, seed, outer_fold)`, emit one row per transformed `feature_index` (value and missing-indicator features),
   2. write `selected=1` for selected features and `selected=0` for all non-selected features (do not omit non-selected rows),
   3. for missing indicators not emitted in a tuple, write `selected=0` for their corresponding `feature_index` values.

## Method Ranking and Challenger

1. Rank methods by lowest mean BER across `(outer_fold x seed)` tuples using the unweighted arithmetic mean (each tuple counts equally).
2. Apply tie-break chain exactly as specified in Section 5.
3. Challenger:
   1. non-primary methods
   2. eligibility mean BER `<=0.40`
   3. select by highest mean `True-` with deterministic tie-breaks
4. If no eligible challenger, apply fallback policy and mark `challenger_available=false`.

## Selector Contract Guardrail

1. Selector behavior must conform exactly to Section 4.3 of the canonical strategy:
   1. formulas and eps constants (`1e-12` where specified),
   2. undefined-score handling,
   3. Gram-Schmidt zero-norm pre-check scope,
   4. deterministic feature tie-break by transformed `feature_index`.

## Exit Criteria

1. Stage B outputs complete for every feasible `(selector, outer_fold, seed)`.
2. Exactly one `is_selected_config=True` per `(selector, outer_fold, seed)`.
3. Ranking and challenger decisions reproducible from CSVs alone.
4. `feature_stability_by_seed.csv` is complete at full-universe row grain (not selected-only row grain).
