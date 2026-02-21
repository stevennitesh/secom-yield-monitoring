# SECOM Final End-to-End Report Strategy (Merged Canonical)

## Status

This document merges:

1. `docs/final_end_to_end_report_strategy.md`
2. `docs/report_strategy/agreed_all_selector_multiseed_changes.md`

This is the canonical strategy for implementation. If any older strategy text conflicts with this document, this document takes precedence.

---

## 1) End Goal

Deliver a report that is strong for a semiconductor hiring manager and defensible to technical reviewers:

1. Reproduce legacy-style SECOM baselines for direct comparability.
2. Select a deployment-realistic supervised model under strict time-aware validation.
3. Quantify operational impact (fails caught, alarm burden, threshold workload, cost sensitivity), not just model accuracy.
4. Compare supervised monitoring to an MSPC baseline at matched operating conditions.

---

## 2) Non-Negotiable Validation and Freeze Rules

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

---

## 3) Lane B Data Partition Contract (DEV/LOCKBOX)

1. Timestamp parsing contract is fixed:
   1. interpret dataset timestamps as day-first strings in format `DD/MM/YYYY HH:MM:SS`,
   2. parse with `errors='coerce'` (unparseable -> `NaT`),
   3. treat parsed timestamps as UTC-naive (no timezone conversion is applied).
2. Drop rows with unparseable timestamps (`NaT`) before sorting.
3. Add deterministic row identity before sorting:
   1. define `raw_row_id` as the 0-based row index from the raw SECOM file as read (before dropping `NaT`).
4. Sort all remaining rows by timestamp ascending.
   1. if timestamps tie, break ties by `raw_row_id` ascending,
   2. sorting must be stable.
5. Reserve lockbox by sample count, not time-span duration:
   1. `LOCKBOX = last floor(0.15 * N)` samples after sorting.
   2. `DEV = first N - floor(0.15 * N)` samples.
6. `N` is the row count after dropping `NaT`.
7. Any prior lockbox results are diagnostic only and non-final.
8. Label contract is fixed:
   1. raw dataset label `y_raw in {-1,+1}` where `-1=pass` and `+1=fail`,
   2. `y_bin = 1` for fail (positive class) and `0` for pass (negative class).
9. Metric conventions are fixed:
   1. `True+` means TPR/sensitivity on the fail class (`y_bin=1`),
   2. `True-` means TNR/specificity on the pass class (`y_bin=0`).
10. Lane-scoping note:
   1. Lane B uses the `DEV/LOCKBOX` partition defined above for all time-aware validation and freeze.
   2. Lane A replication intentionally uses the full dataset (`DEV+LOCKBOX`) for benchmark comparability and does not change Lane B's lockbox discipline.

---

## 4) Two-Lane Plan

## 4.1 Lane A: Replication (benchmark comparability)

Purpose: reproduce literature-style behavior as directly as possible.

Lane A has two mandatory runs:

1. `Replication-Strict` (no missing indicators),
2. `Replication+MI` (with missing indicators).

Protocol:

1. Data scope: full dataset (`DEV+LOCKBOX`), i.e., all `N` rows after dropping `NaT` and sorting by `(timestamp, raw_row_id)` per Section 3.
2. Selectors: `S2N`, `Welch-t`, `F-test`, `Pearson`, `ReliefF`, `Gram-Schmidt`.
3. Feature count: `k=40` fixed.
4. Split: 10-fold stratified CV with `shuffle=True, random_state=42`.
   Use identical CV folds for `Replication-Strict` and `Replication+MI` (paired ablation).
5. Imputer: `SimpleImputer(strategy='median', keep_empty_features=True)` with `add_indicator` set per ablation mode.
6. Scaler: `StandardScaler(with_mean=True, with_std=True)`.
7. Pipeline order: imputer -> scaler -> selector -> classifier.
8. Classifier: `KernelRidge(kernel='rbf', alpha=1.0, gamma=None)`.
9. Label encoding for Kernel Ridge: `y_krr = 2*y_bin - 1` in `{-1,+1}`.
   Selector scoring uses `y_bin` (0/1) labels.
10. Decision threshold (Lane A strict): derive BER-optimal threshold on each fold's training split only (using that fold's KRR training scores), then apply that fixed threshold unchanged to the corresponding fold test split.
11. Lane A thresholding is fold-train-only (no use of fold test labels for threshold selection).
12. Metrics: BER, True+, True-.
13. ReliefF settings (Lane A): `n_neighbors=10`.

Replication ablation (mandatory):

1. `Replication-Strict`: median imputation with `add_indicator=False`.
2. `Replication+MI`: median imputation with `add_indicator=True`.
3. Everything else identical between the two runs.
4. Report paired BER deltas with 95% percentile bootstrap CI for the mean delta across folds (`n_boot=1000`, seed `42`):
   1. compute per-fold `delta_BER = BER_strict_fold - BER_MI_fold` on paired folds,
   2. bootstrap resample fold indices with replacement,
   3. recompute `mean(delta_BER)` per bootstrap draw,
   4. take the 2.5th and 97.5th percentiles of the bootstrap mean distribution.

Interpretation:

1. `Replication-Strict` is the benchmark-faithful comparison.
2. `Replication+MI` isolates value from missing indicators.

## 4.2 Lane B: Deployment-realistic selection and freeze

Purpose: select the final supervised monitoring model under temporal validation.

### 4.2.1 Outer time-aware folds on DEV (used by Stage A and Stage B)

1. Build dataset-anchored 7-day bins on DEV:
   1. Define `t_min_dev = min(timestamp)` over DEV (after the DEV/LOCKBOX split in Section 3).
   2. `week_idx = floor((timestamp - t_min_dev) / 7 days)`.
   3. Bins are anchored to `t_min_dev`, not ISO/calendar weeks.
   4. Week labels are 1-based in reporting (`week N` maps to `week_idx=N-1`).
   5. Define `last_week_idx = max(week_idx)` over DEV and `last_week = last_week_idx + 1`; in the fold definitions below, `last` means `last_week`.
   6. Week ranges like "weeks a-b" are inclusive over 1-based week labels; empty weeks (no samples) are allowed and simply contribute zero rows.
2. Primary 3-fold anchored expanding windows:
   1. Fold 1: train weeks 1-5, test weeks 6-last.
   2. Fold 2: train weeks 1-7, test weeks 8-last.
   3. Fold 3: train weeks 1-9, test weeks 10-last.
3. Minimum required fail count in each outer test window: `20`.
4. Deterministic fallback if any primary fold violates fail minimum:
   1. Fold 1: train weeks 1-4, test weeks 5-last.
   2. Fold 2: train weeks 1-6, test weeks 7-last.
   3. Fold 3: train weeks 1-8, test weeks 9-last.
5. If fallback still violates fail minimum, use deterministic 2-fold fallback:
   1. Fold A: train weeks 1-6, test weeks 7-last.
   2. Fold B: train weeks 1-8, test weeks 9-last.
   3. Artifact labeling rule: if the 2-fold fallback is used, write Fold A as `outer_fold=1` and Fold B as `outer_fold=2` in all fold-level artifacts.
6. If 2-fold fallback still violates fail minimum, Lane B is infeasible and only Lane A claims are allowed.
7. Inner-CV feasibility gate (mandatory for Stage B and Phase 2):
   1. For any slice split by `StratifiedKFold(n_splits=5, ...)`, require both classes present and `min(n_fail, n_pass) >= 5`.
   2. This must hold for every outer fold's outer-train slice; if violated, Lane B is infeasible and only Lane A claims are allowed.
   3. Record `lane_b_infeasible_reason='min_class_count_lt_5_for_inner_cv'` in `reports/run_manifest.json`.
8. Outer-fold fail counts are mandatory in fold-level artifacts.

### 4.2.2 Stage A (diagnostic only, no elimination)

1. Stage A runs on all six baseline selectors.
2. Stage A has no inner CV and no promotion gate.
3. Stage A fixed settings are mandatory:
   1. `k=40`,
   2. `C=1.0`,
   3. scaler=`RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0,75.0))`,
   4. imputer=`SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)` (values + missing-only indicators),
   5. pipeline order: imputer -> scaler -> selector -> classifier,
   6. classifier=`LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`,
   7. ReliefF `n_neighbors=10`,
   8. threshold policy=`outer_train_youden_ber_optimal` (derive on outer-train, apply unchanged to outer-test).
4. Stage A output is narrative baseline only.

### 4.2.3 Stage B (all-selector multi-seed nested CV for method selection)

Stage B selector scope (after de-dup):

1. `S2N`
2. `Welch-t`
3. `F-test`
4. `ReliefF`
5. `Gram-Schmidt`

De-dup rule:

1. Keep `Pearson` in Stage A only.
2. Use `F-test` only in Stage B from the `Pearson/F-test` equivalent pair.

For each `(selector, outer_fold, seed)`:

1. Run inner `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.
2. Search grid:
   1. Non-ReliefF: `k in {10,20,40}`, `C in {0.01,0.1,1.0,10.0}`, `scaler in {StandardScaler, RobustScaler}`.
   2. ReliefF: same grid plus `n_neighbors in {5,10,20}`.
3. Fixed settings inside Stage B:
   1. imputer=`SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)` (values + missing-only indicators),
   2. pipeline order: imputer -> scaler -> selector -> classifier,
   3. classifier=`LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`,
   4. ReliefF note: `skrebate.ReliefF` is deterministic and has no `random_state` parameter; seed only affects inner-fold assignment.
   5. scaler parameter defaults are pinned:
      1. `StandardScaler(with_mean=True, with_std=True)`,
      2. `RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0,75.0))`.
4. Inner config selection:
   1. first criterion: highest mean inner ROC-AUC (computed on inner-validation using `p_hat = Pr(y_bin=1)` via `sklearn.metrics.roc_auc_score(y_bin, p_hat)`),
      where "mean inner" is the unweighted arithmetic mean over the 5 inner validation folds,
   2. second criterion: lowest mean inner BER among configs within `0.01` AUC of best,
      where mean inner BER is the unweighted arithmetic mean over the 5 inner validation folds (each fold BER computed with that fold's inner-train-derived threshold),
   3. if still tied: smaller `k`,
   4. if still tied: smaller `C`,
   5. if still tied: `StandardScaler` over `RobustScaler`,
   6. if still tied: smaller `n_neighbors` where applicable.
5. Inner BER computation rule:
   1. for each inner fold and candidate config, fit the pipeline on inner-train,
   2. derive Youden/BER-optimal threshold from inner-train predictions only,
   3. apply that fixed threshold unchanged to inner-validation predictions from the same inner-train fit,
   4. never use inner-validation labels to set threshold.
6. Outer-test evaluation rule:
   1. take the inner-selected config for `(selector, outer_fold, seed)`,
   2. refit the pipeline on that outer fold's outer-train slice using that config,
   3. derive BER-optimal threshold from outer-train predictions only,
   4. apply that fixed threshold unchanged to outer-test predictions from the same outer-train refit for BER/True+/True-.

### 4.2.4 Phase 2 (hyperparameter freeze on full DEV)

1. Run separately for each available role:
   1. primary always,
   2. challenger only if eligible.
2. Input: selected method from Stage B.
3. Repeated inner CV on full DEV:
   1. seeds `{42,11,23,37,59}`,
   2. per seed, use `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`,
   3. 25 inner observations per config (unweighted; each `(seed, inner_fold)` counts equally).
4. Fixed settings and pipeline order in Phase 2 are identical to Stage B:
   1. imputer=`SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)` (values + missing-only indicators),
   2. classifier=`LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`,
   3. pipeline order: imputer -> scaler -> selector -> classifier,
   4. use the same grid and the same inner selection rule as Stage B.
5. Output: one frozen config per available role: `(selector, k, C, scaler[, n_neighbors])`.

### 4.2.5 Phase 3 (final refit and threshold derivation on full DEV)

1. Refit frozen primary config on full DEV with seed `42`.
2. Refit frozen challenger config on full DEV with seed `42` if available.
3. Derive scientific and operational thresholds from full-DEV in-sample predictions for each available frozen model.
4. Only these Phase-3 frozen models are lockbox-eligible supervised models.

---

## 4.3 Selector Implementation Contract (Both Lanes)

This section pins selector behavior so two correct implementations produce the same rankings and selected feature sets.

Common input contract:

1. All selectors operate on the transformed feature matrix after preprocessing:
   1. imputer output (values and any emitted missing-only indicators),
   2. then scaler output (`StandardScaler` or `RobustScaler`),
   3. then selector ranks/selects features using `y_bin` labels.
2. Selector scoring is fit on training data only for the current split.
3. Undefined scores are handled deterministically:
   1. if a feature is constant on the selector's training slice (zero variance / zero norm) in its *raw pre-orthogonalized representation*, its score is set to `-inf`,
   2. any `NaN`/`inf` score is set to `-inf`.
   3. Gram-Schmidt scope note: do not apply the "zero norm -> -inf" rule to intermediate orthogonalized vectors; use the eps-regularized formulas below (which naturally yield scores near 0 for near-zero-norm residualized features).
4. Feature tie-break rule (rank/selection determinism):
   1. if two features have equal score, prefer smaller transformed `feature_index` as defined in Section 8,
   2. when selecting the top `k`, break ties at the cutoff using the same rule.

Selector definitions (binary labels, per feature `j`):

1. `S2N` (signal-to-noise):
   1. `score_j = |mu_fail - mu_pass| / (sd_fail + sd_pass + eps)`,
   2. where `mu_*` are class means on the selector training slice, `sd_*` are sample standard deviations with `ddof=1`, `eps=1e-12`.
2. `Welch-t`:
   1. `score_j = |t_j|` where
      `t_j = (mu_fail - mu_pass) / sqrt(sd_fail^2/n_fail + sd_pass^2/n_pass + eps)`,
      using the same `mu_*`, `sd_*`, and `eps` definitions as above.
3. `F-test`:
   1. `score_j` is the ANOVA F-statistic for binary `y_bin` (equivalent to `sklearn.feature_selection.f_classif` scores),
   2. rank by descending `score_j`.
4. `Pearson` (Stage A only):
   1. `score_j = |corr(X_j, y_bin)|` (Pearson correlation), rank descending,
   2. note: for binary labels, Pearson and F-test induce the same ranking up to monotone transforms; Pearson is excluded from Stage B by design.
5. `ReliefF`:
   1. use `skrebate.ReliefF(n_features_to_select=k, n_neighbors=n_neighbors, n_jobs=-1)` (deterministic; no `random_state`),
   2. rank by descending `feature_importances_` (treat any undefined importance as `-inf`),
   3. tie-break by smaller `feature_index`.
6. `Gram-Schmidt` (redundancy-aware greedy selection):
   1. use `eps=1e-12` (same constant as other selectors in this document),
   2. let `r = y_bin - mean(y_bin)` (centered target), and let `X` be the selector training matrix,
   3. repeat until `k` features selected or `||r|| < eps`:
      1. score each remaining feature `j` by `score_j = |<X_j, r>| / (||X_j|| * ||r|| + eps)`,
      2. select the feature with maximum `score_j` (tie-break by smaller `feature_index`),
      3. normalize `q = X_j / (||X_j|| + eps)`,
      4. orthogonalize: for each remaining feature `m`, set `X_m = X_m - <X_m, q> * q`,
      5. update residual: `r = r - <r, q> * q`.

---

## 5) Stage B Method Ranking and Challenger Rules

Primary model selection (Stage B output):

1. Rank methods by lowest mean outer BER across all `(outer_fold x seed)` results.
   Mean is the unweighted arithmetic mean over tuple-level outer-test values (each `(outer_fold, seed)` counts equally).
2. Tie-break 1: lower standard deviation of per-fold BER means (temporal stability).
   Definition: for each `outer_fold=f`, compute `mu_f = mean_seed(BER_{f,seed})`; then `std_per_fold_BER_means = std_f(mu_f)` with `ddof=1`.
3. Tie-break 2: higher mean True+ across `(outer_fold x seed)` (unweighted over tuples).
4. Tie-break 3:
   1. smaller modal `k` (if no unique mode, choose the smallest tied `k`),
   2. if still tied: smaller modal `C` (if no unique mode, choose the smallest tied `C`),
   3. if still tied: `StandardScaler` over `RobustScaler`,
   4. if still tied: deciding vote from `(seed=42, largest outer training window)`:
      (largest outer training window = outer fold with the greatest `train_n`; if tied, choose the one with the latest `train_end` timestamp)
       1. lower outer-test BER,
       2. if still tied: higher outer-test True+,
       3. if still tied: lexicographically smaller selector name.

Challenger selection:

1. Candidate pool: non-primary methods.
2. Eligibility: mean BER `<= 0.40`.
3. Among eligible methods, select highest mean True- across `(outer_fold x seed)` (unweighted over tuples).
4. Challenger tie-breaks:
   1. lower mean BER,
   2. lower std of per-fold BER means,
   3. lexicographically smaller selector name.
5. Challenger goes through Phase 2 and Phase 3 if eligible.

No-eligible-challenger fallback:

1. Set `challenger_available=false`.
2. Run Phase 2 and Phase 3 for primary only.
3. `reports/final_lockbox_result.csv`: omit challenger rows (primary rows only).
4. `reports/operational_cost_curves.csv`: keep challenger columns but write challenger values as `NA`.
5. `reports/hyperparameter_freeze_results.csv`: primary-role rows only (no challenger rows).
6. Write challenger outputs as `NA` only in column-based artifacts that include challenger fields; do not create challenger rows in row-based artifacts.
7. Record fallback reason: `no_eligible_method_under_BER_0.40`.

---

## 6) Threshold Policy (Post-Freeze)

Thresholds are finalized after Phase 2 freeze and Phase 3 full-DEV refit.

Scoring and classification rule (supervised models):

- Score is predicted fail probability `p_hat = Pr(y_bin=1)` from the frozen pipeline.
- Predict fail iff `p_hat >= threshold_value` (else predict pass).
- Threshold candidate set for searches: sorted unique scores in the relevant training slice plus two sentinels (`-inf` flags all; `+inf` flags none).

Thresholds to report for each available frozen model:

1. Scientific threshold: BER-optimal threshold on full-DEV in-sample predictions.
   1. if multiple thresholds achieve the same minimum BER, choose the one with higher TPR;
   2. if still tied, choose the lowest threshold value.
   This same BER-optimal threshold definition (including tie-breaks) applies whenever this protocol says "derive BER-optimal threshold" on any training slice (inner-train, outer-train, full DEV).
2. Operational threshold: pre-registered review-capacity threshold.

Operational constants and rules:

1. Review-capacity cap: mean weekly flagged fraction on DEV must be `<=10%`.
2. Mean weekly flagged fraction uses the same dataset-anchored 7-day bins as Section 4.2.
3. Mean weekly flagged fraction definition is fixed (unweighted by week size):
   1. for each week `w`: `frac_w = flagged_w / wafers_w`,
   2. `mean_weekly_flagged_fraction = mean_w(frac_w)`,
   3. do not use sample-weighted averaging across weeks.
   4. compute `mean_w(...)` over weeks that appear in DEV (weeks with `wafers_w > 0`); do not inject empty weeks.
4. Operational threshold selection rule: choose the highest-TPR threshold satisfying the `<=10%` cap.
   1. if multiple thresholds have the same highest TPR under the cap, choose the lowest threshold value.
5. Secondary matched comparison point: `TNR=90%`.
6. `TNR=90%` extraction rule: choose the lowest threshold with `TNR >= 0.90`; if tied in TNR, choose higher TPR; if still tied, choose the lowest threshold value.
7. `TNR=90%` reporting quantities (per evaluation slice; reporting only, not a frozen threshold):
   1. compute `threshold_at_TNR90` by applying the extraction rule above to that slice's `(y_bin, p_hat)` pairs,
   2. compute `TNR_at_TNR90` and `TPR_at_TNR90` on that same slice at `threshold_at_TNR90`.
   3. This matched point is used only for the supervised-vs-MSPC comparison at matched `TNR=90%`; it must never be used to choose the frozen scientific/operational thresholds.
8. Cost-ratio sensitivity set: `FN:FP in {1,2,5,10,20}`.
9. Lockbox is never used to choose thresholds.

---

## 7) Lockbox Protocol and Drift Gate

Scope:

1. Applies only if Lane B is feasible.

Procedure:

1. Score lockbox once using frozen primary (and frozen challenger if available).
2. Use frozen scientific and operational thresholds.
3. No retuning after lockbox.

Mandatory drift gate before interpreting lockbox superiority:

1. Prevalence shift: `abs(lockbox_fail_rate - DEV_fail_rate)`.
2. Score shift: per frozen supervised model, two-sided KS test on predicted fail probabilities `p_hat = Pr(y_bin=1)` from the Phase-3 frozen model (`DEV` vs `lockbox`).
   Use `scipy.stats.ks_2samp(dev_scores, lockbox_scores, alternative='two-sided', mode='auto').pvalue` with `alpha=0.01`.
3. Feature shift: PSI on each frozen supervised model's top-10 selected value features.
4. PSI feature scope: original value features only (exclude missing-indicator features).
5. PSI top-10 rule (per model): rank that model's selected value features by absolute scaled logistic coefficient (post-scaler) from its full-DEV frozen fit; use top 10 or all if fewer.
   Write `psi_feature_count` as the number of value features used for PSI (an integer in `{0,1,...,10}`).
   If the frozen model selects zero value features (only missing indicators), the PSI feature set is empty and `max_PSI=0.0` and `median_PSI=0.0` for that model (PSI criterion PASS by definition).
6. PSI computation rule (per feature):
   1. if DEV has at least one non-missing value for the feature, compute 9 interior bin edges as DEV non-missing quantiles at `{0.10,0.20,...,0.90}` (edges computed on DEV only); otherwise use an empty edge set,
   2. if the quantile edges are not strictly increasing (duplicate edges), collapse duplicates by taking unique sorted edges; bins are formed from the remaining edges,
   3. define non-missing bins as open-ended extremes from the (possibly collapsed) unique edge set `{e1<e2<...<em}`:
      `(-inf, e1]`, `(e1, e2]`, ..., `(em, +inf)` so non-missing lockbox values outside the DEV range map to the first or last bin.
      If DEV has no non-missing values for the feature, use a single non-missing bin `(-inf, +inf)` (so `p_nonmissing=0` on DEV),
   4. add one extra bin for missing values (NaNs) (if any),
   5. let `p_i` be the fraction of all `N_DEV` samples in bin `i` and let `q_i` be the fraction of all `N_lockbox` samples in bin `i` (fractions sum to 1 across bins),
   6. use `eps=1e-6` and compute `PSI = sum_i (p_i - q_i) * ln((p_i+eps)/(q_i+eps))`,
   7. compute `max_PSI` and `median_PSI` across the top-10 feature set.
7. Drift status:
   1. `PASS`: prevalence shift `<0.02` and KS `p>=0.01` and `max_PSI<0.30`,
   2. `CAUTION`: exactly one criterion violated,
   3. `HIGH_SHIFT`: two or more criteria violated.
8. If `HIGH_SHIFT`, do not make lockbox superiority claims.
   This applies per frozen supervised model (primary and challenger can differ).
9. `lockbox_claims_allowed` mapping is fixed:
   1. `PASS` and `CAUTION` write `lockbox_claims_allowed=true`,
   2. `HIGH_SHIFT` writes `lockbox_claims_allowed=false`.

Lockbox MSPC companion evaluation:

1. Fit MSPC on full-DEV pass wafers only.
2. Freeze autoscaler, PCA, and UCLs from DEV pass wafers.
3. Score lockbox once and report:
   1. `T2_TPR_at_TNR90`,
   2. `Q_TPR_at_TNR90`,
   3. `best_MSPC_TPR_at_TNR90`.
4. MSPC implementation spec (autoscaling, PCA component selection, UCL formulas, and contributions) is pinned to `docs/report_strategy/improvement_plan.md`.

---

## 8) Metrics Policy

Lane A reporting:

1. BER, True+, True-.
2. Mean, std, and 95% CI across folds.
3. Lane A CI method is fixed: 95% percentile bootstrap CI for the mean metric across folds (`n_boot=1000`, seed `42`):
   1. bootstrap resample the 10 fold indices with replacement,
   2. recompute the unweighted mean metric across the resampled folds,
   3. take the 2.5th and 97.5th percentiles of the bootstrap mean distribution.

Lane B and lockbox reporting:

1. BER, True+, True-.
2. ROC-AUC, PR-AUC.
3. MCC, F2.
4. Lane B: fold-wise values and mean+/-std.
5. Lockbox: point estimates only.
6. Metric computation definitions (reporting only, not used for selection unless explicitly stated elsewhere):
   1. `ROC_AUC = sklearn.metrics.roc_auc_score(y_true=y_bin, y_score=p_hat)`.
   2. `PR_AUC = sklearn.metrics.average_precision_score(y_true=y_bin, y_score=p_hat)` (average precision, not trapezoid PR integral).
   3. `MCC = sklearn.metrics.matthews_corrcoef(y_true=y_bin, y_pred=y_pred)`.
   4. `F2 = sklearn.metrics.fbeta_score(y_true=y_bin, y_pred=y_pred, beta=2, pos_label=1, zero_division=0)`.

Core metric definition (used throughout):

1. `BER = 0.5*(FNR + FPR)`.
2. With `True+ = TPR` and `True- = TNR`, `BER = 1 - 0.5*(True+ + True-)`.

Secondary diagnostics:

1. Brier and calibration diagnostics.
2. Weekly alarm burden and fail capture/miss counts.
3. Cost curves.
4. MSPC baseline comparison.
5. Feature stability and redundancy diagnostics.
6. MSPC fold discipline for outer folds:
   1. fit MSPC on each outer-train pass subset only,
   2. score corresponding outer-test set,
   3. union alarm stream rule: `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)`.
7. Brier dual baselines per evaluation slice prevalence `p`:
   1. all-pass baseline=`p`,
   2. prevalence-constant baseline=`p*(1-p)`.
8. Feature stability minimum definitions (computed per selector over its own `(outer_fold, seed)` tuples; do not pool across selectors):
   1. `selection_frequency`:
      fraction of `(outer_fold, seed)` tuples where the feature is selected;
      selection indicators `s_j(t)` are taken from the outer-train refit of the Stage B inner-selected config for each tuple `t=(outer_fold, seed)`,
      denominator is total tuple count (`n_folds x n_seeds`, e.g., `15` for 3 folds x 5 seeds),
   2. `conditional_effect_magnitude = |coef_j|` from the Phase-3 frozen primary full-DEV refit (absolute coefficient on the scaled feature, i.e., post-scaler),
   3. `expected_contribution = selection_frequency * conditional_effect_magnitude`,
   4. `fold_jaccard_stability` (co-selection fraction over tuple pairs):
      for feature `j` with binary selected indicator `s_j(t)` on tuple `t=(outer_fold, seed)`,
      compute
      `fold_jaccard_stability_j = [sum_{a<b} 1{s_j(a)=1 and s_j(b)=1}] / C(M,2)`,
      where `M = n_folds x n_seeds`,
      and note: column name retains historical label `fold_jaccard_stability`, but metric definition is co-selection fraction (not classical Jaccard |A intersect B| / |A union B|),
   5. `cluster_id`.
9. Feature cluster definition is fixed:
   1. build pairwise Pearson correlation matrix on full DEV post-freeze using the Phase-3 frozen primary's imputer (fit on full DEV) applied to impute raw value features only (exclude missing indicators and all lockbox rows),
   2. if any imputed value feature has zero variance on DEV, its Pearson correlations are undefined; treat as no edges (singleton cluster),
   3. build graph edges where `|corr| >= 0.95`,
   4. define `cluster_id` as connected-component membership in this graph.
10. Lane B transformed feature identity contract (used by `reports/feature_stability_by_seed.csv` and `reports/feature_report.csv`):
   1. Let `P` be the raw SECOM feature count (for this dataset, `P=590` value features).
      Raw value feature indices `j` are 0-based (so `j in {0,1,...,P-1}`).
   2. Value feature identity for raw column `j` is:
      1. `feature_type='value'`,
      2. `feature_index=j`,
      3. `feature_name_or_source_col='X{j}'`.
   3. Missing-indicator feature identity for raw column `j` is:
      1. `feature_type='missing_indicator'`,
      2. `feature_index=P+j`,
      3. `feature_name_or_source_col='M{j}'`.
   4. `SimpleImputer(add_indicator=True)` emits missing indicators for a fold only when a feature is missing in that fold's training slice ("missing-only").
      When emitted, indicator columns correspond to raw feature indices in ascending order of the underlying `indicator_features` list; map each emitted indicator for raw feature `j` to `feature_index=P+j` and `feature_name_or_source_col='M{j}'`.
      For feature-stability accounting, treat any indicator not emitted in a given `(outer_fold, seed)` tuple as `selected=0` for that tuple.

---

## 9) Runtime and Checkpointing

Stage B expected workload:

1. Non-ReliefF selectors: `4 x 5 seeds x 3 outer folds x 24 configs x 5 inner folds = 7200`.
2. ReliefF selector: `1 x 5 x 3 x 72 configs x 5 = 5400`.
3. Total Stage B inner fits: `12600`.
4. If 2-fold outer fallback is active, multiply Stage B inner-fit counts above by `2/3`.

Phase 2 expected workload:

1. Per role, non-ReliefF winner: `24 configs x 25 = 600`.
2. Per role, ReliefF winner: `72 configs x 25 = 1800`.
3. Total Phase 2 for primary+challenger: `1200` to `3600`.
4. If no challenger: `600` to `1800`.
5. If Lane B runs with 2 outer folds, Phase-2 freeze workload is unchanged (Phase 2 is full-DEV repeated inner CV).

Execution controls:

1. Expected wall time: ~30 minutes to 4+ hours depending on hardware and ReliefF implementation.
2. Checkpointing is required:
   1. Stage B checkpoint unit: `(selector, outer_fold, seed)`.
   2. Phase 2 checkpoint unit: `(role, selector, config, seed, inner_fold)`.

---

## 10) Required Artifacts and Schemas

Index conventions for all fold-based artifacts:

1. `outer_fold`: 1-based integer.
2. `inner_fold`: 1-based integer.
3. Lane A `fold`: 1-based integer in `{1,2,...,10}`.
4. `train_window` and `test_window`: ISO-8601 interval string `start_ts/end_ts` using the dataset's timezone-naive timestamps (no conversion; no timezone suffix).
   Treat dataset timestamps as UTC-naive (no conversion) and format timestamps as `YYYY-MM-DDTHH:MM:SS` (do not append `Z`).
   Define `start_ts = min(timestamp)` in the slice and `end_ts = max(timestamp)` in the slice (inclusive).
5. String label conventions are mandatory:
   1. `selector` values must exactly match the selector names used in Section 4.
   2. `scaler` values must be exactly `StandardScaler` or `RobustScaler`.
6. All `std_*` columns are sample standard deviations with `ddof=1` over the same units used for the corresponding `mean_*` columns.
7. Fail-count column conventions are fixed:
   1. `n_test_fails`, `test_fails`, and `lockbox_fails` all mean `count(y_bin==1)` in the corresponding evaluation slice.
   2. `min_test_fails` means `min(test_fails)` across the outer folds used in that run.

Required artifact set:

1. `reports/baseline_replication_strict.csv`
2. `reports/baseline_replication_with_missing_indicators.csv`
3. `reports/baseline_missing_indicator_ablation.csv`
4. `reports/baseline_replication_summary.csv`
5. `reports/timeaware_selector_screening.csv`
6. `reports/splitwise_timeaware_results.csv`
7. `reports/stage_b_inner_cv_results.csv`
8. `reports/timeaware_model_selection.csv`
9. `reports/seed_stability_summary.csv`
10. `reports/feature_stability_by_seed.csv`
11. `reports/hyperparameter_freeze_results.csv`
12. `reports/final_lockbox_result.csv`
13. `reports/mspc_baseline.csv`
14. `reports/operational_cost_curves.csv`
15. `reports/feature_report.csv`
16. `reports/drift_gate_summary.csv`
17. `reports/run_manifest.json`

Lane B infeasible artifact policy (if Section 4.2 infeasibility rules trigger: outer-test fail minimum or inner-CV feasibility gate):

1. Required outputs become:
   1. Lane A artifacts only: items 1-4,
   2. `reports/run_manifest.json` with `lane_b_feasible=false` and reason.
2. Lane B artifacts (items 5-16) are not required in this case.
3. If emitted, Lane B artifacts must be clearly marked `LANE_B_INFEASIBLE` with metric fields set to `NA`.

Minimum schema requirements:

1. `reports/baseline_replication_strict.csv` and `reports/baseline_replication_with_missing_indicators.csv`:
   1. `selector`, `fold`, `BER`, `True+`, `True-`, `n_train`, `n_test`, `n_test_fails`.
2. `reports/baseline_missing_indicator_ablation.csv`:
   1. `selector`, `BER_strict`, `BER_MI`, `delta_BER`, `CI_lower`, `CI_upper`, `n_boot`.
   2. sign convention is fixed: `delta_BER = BER_strict - BER_MI` (positive means missing indicators improved BER).
   3. `CI_lower` and `CI_upper` correspond to the same signed delta definition.
   4. `CI_lower` and `CI_upper` are the 95% percentile paired bootstrap CI for the mean `delta_BER` across folds (`n_boot=1000`, seed `42`):
      resample fold indices with replacement, recompute the unweighted mean `delta_BER` over resampled folds, then take the 2.5th and 97.5th percentiles.
3. `reports/baseline_replication_summary.csv`:
   1. one row per `(selector, replication_mode)` where `replication_mode in {'strict','with_missing_indicators'}`,
   2. `selector`, `replication_mode`, `n_folds`, `n_boot`, `boot_seed`,
   3. `mean_BER`, `std_BER`, `CI_lower_BER`, `CI_upper_BER`,
   4. `mean_True+`, `std_True+`, `CI_lower_True+`, `CI_upper_True+`,
   5. `mean_True-`, `std_True-`, `CI_lower_True-`, `CI_upper_True-`,
   6. CI method is fixed: 95% percentile bootstrap CI for the mean metric across folds (`n_boot=1000`, seed `42`), consistent with the Lane A CI rule in Section 8.
4. `reports/timeaware_selector_screening.csv`:
   1. `selector`, `n_splits`, `mean_BER`, `std_BER`, `mean_True+`, `mean_True-`, `min_test_fails`.
5. `reports/splitwise_timeaware_results.csv`:
     1. one row per `(selector, outer_fold, seed)`,
     2. `selector`, `outer_fold`, `seed`, `train_window`, `test_window`, `k`, `C`, `scaler`, `n_neighbors` (nullable for non-ReliefF selectors),
     3. `threshold_policy`, `outer_threshold`, `test_fails`, `BER`, `True+`, `True-`.
     4. allowed `threshold_policy` values are fixed: `outer_train_youden_ber_optimal`.
6. `reports/stage_b_inner_cv_results.csv`:
   1. one row per `(selector, outer_fold, seed, k, C, scaler, n_neighbors)` where `n_neighbors` is nullable for non-ReliefF selectors,
   2. `selector`, `outer_fold`, `seed`, `k`, `C`, `scaler`, `n_neighbors` (nullable for non-ReliefF selectors),
   3. `mean_inner_ROC_AUC`, `mean_inner_BER`, `is_selected_config`.
      `mean_inner_ROC_AUC` and `mean_inner_BER` are the unweighted arithmetic mean over the 5 inner validation folds for that `(selector, outer_fold, seed, config)`,
   4. uniqueness constraint: exactly one `is_selected_config=True` row per `(selector, outer_fold, seed)`.
7. `reports/timeaware_model_selection.csv`:
   1. method-level summary over `(outer_fold x seed)`,
   2. `selector`, `n_folds`, `n_seeds`, `mean_BER`, `std_BER`, `std_per_fold_BER_means`,
   3. `mean_True+`, `std_True+`, `mean_True-`, `std_True-`, `is_primary`, `is_challenger`,
   4. config columns (`k`, `C`, `scaler`, `n_neighbors`) are intentionally excluded because inner-selected configs vary by `(outer_fold, seed)`,
   5. no-challenger fallback behavior: if challenger is unavailable, no row has `is_challenger=true`.
   6. `std_per_fold_BER_means` definition is fixed: for each `outer_fold=f`, compute `mu_f = mean_seed(BER_{f,seed})`; then take `std_f(mu_f)` with `ddof=1`.
   7. `std_BER`, `std_True+`, and `std_True-` are sample standard deviations with `ddof=1` computed over the tuple-level values across all `(outer_fold, seed)` tuples for that selector.
8. `reports/seed_stability_summary.csv`:
   1. one row per `(selector, seed)`,
   2. `selector`, `seed`, `mean_outer_BER`, `std_outer_BER`, `mean_outer_True+`, `mean_outer_True-`,
   3. `modal_k`, `modal_C`, `modal_scaler`, `modal_n_neighbors` (nullable for non-ReliefF selectors), `n_outer_folds`,
   4. mode tie rule: smaller tied `k`, smaller tied `C`, `StandardScaler` over `RobustScaler`; if `selector=='ReliefF'`, then smaller tied `n_neighbors` else `modal_n_neighbors=NA`.
   5. `mean_outer_*` and `std_outer_BER` are computed across outer folds for that fixed `seed` (unweighted over folds; `std_outer_BER` uses `ddof=1`).
9. `reports/feature_stability_by_seed.csv`:
    1. row scope is full transformed-feature universe for each `(selector, seed, outer_fold)`,
    2. one row per `(selector, seed, outer_fold, feature_index)`,
    3. `selector`, `seed`, `outer_fold`, `feature_index`, `feature_type`, `selected` where `feature_type in {'value','missing_indicator'}`,
    4. `feature_index` uses the same 0-based transformed-feature indexing scheme from Section 8; it is not a foreign-key constraint because `reports/feature_stability_by_seed.csv` includes features beyond the frozen primary model.
    5. `selected=1` iff the feature is among the `k` features selected by the outer-train refit of the Stage B inner-selected config for that `(selector, seed, outer_fold)`; else `0`.
10. `reports/hyperparameter_freeze_results.csv`:
     1. one row per `(role, selector, k, C, scaler, n_neighbors, seed, inner_fold)` where `n_neighbors` is nullable for non-ReliefF selectors,
     2. `role`, `selector`, `k`, `C`, `scaler`, `n_neighbors` (nullable for non-ReliefF selectors), `seed`, `inner_fold` where `role in {'primary','challenger'}`,
     3. `inner_ROC_AUC`, `inner_BER`, `mean_inner_ROC_AUC_by_config`, `mean_inner_BER_by_config`, `is_frozen_config`,
     4. `mean_inner_*_by_config` are the unweighted arithmetic mean over the 25 `(seed, inner_fold)` observations for that unique `(role, selector, k, C, scaler, n_neighbors)`,
     5. uniqueness constraint: for each `role`, exactly one unique `(selector, k, C, scaler, n_neighbors)` has `is_frozen_config=True` across all `seed x inner_fold` rows.
11. `reports/final_lockbox_result.csv`:
      1. one row per available `(role, threshold_policy)` pair,
      2. `role`, `selector`, `threshold_policy`, `threshold_value`, `BER`, `True+`, `True-`, `ROC_AUC`, `PR_AUC`, `MCC`, `F2`, `lockbox_n`, `lockbox_fails`,
         plus reporting-only matched-point columns: `threshold_at_TNR90`, `TNR_at_TNR90`, `TPR_at_TNR90` (computed on the lockbox slice using the Section 6 `TNR=90%` extraction rule),
      3. allowed `threshold_policy` values are fixed: `scientific` and `operational`.
      4. `threshold_at_TNR90`, `TNR_at_TNR90`, and `TPR_at_TNR90` are properties of the model on the lockbox slice and must be identical for the `scientific` and `operational` rows of the same `role`.
12. `reports/mspc_baseline.csv`:
    1. `fold_index`, `eval_scope`, `T2_AUC`, `Q_AUC`, `alarm_rate`, `empirical_ARL0`,
    2. `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, `best_MSPC_TPR_at_TNR90`, `best_MSPC_source`,
    3. union alarm rule for `alarm_rate` and `empirical_ARL0`:
       `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)` and `alarm_rate = mean(alarm)` over the evaluation slice,
    4. `best_MSPC_TPR_at_TNR90 = max(T2_TPR_at_TNR90, Q_TPR_at_TNR90)` and `best_MSPC_source in {'T2','Q'}`.
       Tie-break: if `T2_TPR_at_TNR90 == Q_TPR_at_TNR90`, set `best_MSPC_source='T2'`.
    5. `empirical_ARL0` may be `NaN` only when fewer than two alarms occur in the evaluated sequence; if `NaN`, record reason in `reports/run_manifest.json`,
    6. allowed `eval_scope` values are fixed: `outer_fold` and `lockbox`,
    7. `fold_index` type is fixed: always write as a string.
       Outer-fold rows serialize the 1-based `outer_fold` as `"1"`, `"2"`, etc; the lockbox row writes `"LOCKBOX"`.
    8. outer-fold rows use `eval_scope='outer_fold'` and `fold_index` equal to the serialized 1-based `outer_fold`,
    9. lockbox row uses `eval_scope='lockbox'` and `fold_index='LOCKBOX'`.
13. `reports/operational_cost_curves.csv`:
   1. one row per `cost_ratio` value in the pre-registered set from Section 6,
   2. `cost_ratio`,
   3. expected cost columns for `primary_scientific`, `primary_operational`, `challenger_scientific`, `challenger_operational`,
   4. plus `all_pass_baseline`, `all_flag_baseline`.
   5. expected cost definition (lockbox evaluation slice):
      1. let `C_FP=1` and `C_FN=cost_ratio` (so `cost_ratio = C_FN / C_FP`),
      2. compute lockbox confusion-matrix counts `FP` and `FN` at the corresponding frozen threshold,
      3. write per-wafer expected cost as `(C_FP*FP + C_FN*FN) / lockbox_n` for each cost column.
   6. baseline cost definitions (lockbox evaluation slice):
      1. `all_pass_baseline = (cost_ratio * lockbox_fails) / lockbox_n` (flags none, so `FP=0` and `FN=lockbox_fails`),
      2. `all_flag_baseline = (lockbox_n - lockbox_fails) / lockbox_n` (flags all, so `FP=lockbox_passes` and `FN=0`).
14. `reports/feature_report.csv`:
   1. scope is the frozen primary supervised model only.
   2. one row per transformed feature selected in the frozen primary Phase-3 model (row count = frozen `k`).
   3. `feature_index`, `feature_type`, `feature_name_or_source_col`, `selection_frequency`, `conditional_effect_magnitude`, `expected_contribution`, `fold_jaccard_stability`, `cluster_id`.
   4. `selection_frequency` and `fold_jaccard_stability` are computed for this same selector (the frozen primary selector) using that selector's Stage B tuple set `(outer_fold, seed)` (derived from `reports/feature_stability_by_seed.csv`).
   5. `feature_type in {'value','missing_indicator'}` must be consistent with Section 8 naming contract: value features `X{j}` and missing indicators `M{j}` (raw feature index `j` is 0-based).
   6. `cluster_id` must follow the fixed clustering rule from Section 8 (`|corr| >= 0.95` connected components on imputed value features).
   7. `conditional_effect_magnitude` is taken from the Phase-3 frozen primary full-DEV refit (absolute coefficient on the scaled feature, i.e., post-scaler).
   8. `cluster_id` is defined for value features only; missing-indicator features must use `cluster_id=NA`.
15. `reports/drift_gate_summary.csv`:
   1. `model_scope`, `dev_fail_rate`, `lockbox_fail_rate`, `abs_prevalence_shift`, `ks_pvalue_scores`, `max_PSI`, `median_PSI`, `psi_feature_count`, `drift_gate_status`, `lockbox_claims_allowed`.
   2. allowed `model_scope` values: `primary_frozen`, `challenger_frozen`.
   3. one row per evaluated frozen supervised model:
      always `primary_frozen`, plus `challenger_frozen` only when challenger is available.
   4. `psi_feature_count` definition is fixed: number of value features in the PSI feature set for that model (`min(10, count_selected_value_features)`); 0 if none.
16. `reports/run_manifest.json`:
    1. seed sets by stage, split boundaries, fixed constants, search grid, checkpoint/restart metadata, and environment versions (Python, numpy, pandas, scikit-learn, scipy, skrebate),
    2. `lane_b_feasible` always present (`true` when feasible, `false` when infeasible),
    3. `lane_b_infeasible_reason` present only when `lane_b_feasible=false`,
    4. `challenger_available` always present and fallback reason when false,
    5. Phase-2 frozen config identity hash per available role (`config_hash`),
    6. config hash object form is fixed:
       1. always hash an object with exactly these keys: `selector`, `k`, `C`, `scaler`, `n_neighbors`,
       2. for non-ReliefF selectors, `n_neighbors` must be present and set to JSON `null` (not omitted).
    7. hash spec: `config_hash = SHA-256` over UTF-8 canonical JSON with sorted keys and compact separators `(",", ":")`,
    8. float normalization before hashing is fixed:
       1. the manifest must not contain `NaN`/`inf` floats; use JSON `null` for missing numeric fields,
       2. normalize every finite float `x` to `float(f\"{x:.6g}\")` (6 significant figures) before serialization,
       3. hash the exact bytes written to disk: `json.dumps(..., sort_keys=True, separators=(\",\", \":\"), ensure_ascii=True).encode('utf-8')`.
    9. Required keys (minimum; must be present even if value is null):
      1. `manifest_version` (string),
      2. `strategy_doc_path` (string) and `strategy_doc_sha256` (string),
      3. `git_commit` (string) and `git_dirty` (bool),
      4. `python_executable` (string),
      5. `library_versions` (object with at least: `python`, `numpy`, `pandas`, `sklearn`, `scipy`, `skrebate`),
      6. `seed_policy` (object with keys for Lane A, Stage B, Phase 2, Phase 3),
      7. `dev_lockbox_split` (object: `N_total_after_NaT_drop`, `N_dev`, `N_lockbox`, and the lockbox rule string),
      8. `outer_fold_plan_used` (string enum: `primary_3fold` or `fallback_3fold` or `fallback_2fold`) and `outer_fold_week_ranges` (list of per-fold `{outer_fold, train_weeks, test_weeks}`),
      9. `lane_b_feasible` (bool), and when false: `lane_b_infeasible_reason` (string),
      10. `challenger_available` (bool), and when false: `challenger_unavailable_reason` (string),
      11. `frozen_primary` (object with: `selector`, `k`, `C`, `scaler`, `n_neighbors` (nullable), `config_hash`),
      12. `frozen_challenger` (object or null; same keys as `frozen_primary`),
      13. `frozen_thresholds` (object keyed by role, then by `scientific` and `operational` threshold values),
      14. `drift_gate_results` (object keyed by role, including `drift_gate_status` and `lockbox_claims_allowed`),
      15. if any `empirical_ARL0` is `NaN`, include `empirical_ARL0_nan_reason` (string).

---

## 11) Manager-Facing Outputs (Final Report Requirements)

1. Weekly flagged wafers at scientific and operational thresholds.
2. Weekly fail capture and miss counts.
3. Review workload estimates and recommended alert policy.
4. Stable top features grouped into high-correlation clusters.
5. Supervised vs MSPC at matched `TNR=90%`.
6. Operational framing of workload:
   1. `weekly_rate = DEV_sample_count / DEV_week_count`,
   2. `DEV_week_count = count_unique(week_idx)` over DEV (weeks with at least one wafer in DEV),
   3. `predicted_flag_fraction` from full-DEV post-freeze predictions (report separately for scientific vs operational thresholds, for each available frozen model),
   4. Stage B mean flagged fraction shown as robustness diagnostic,
   5. lockbox flagged fraction shown as holdout observation.

Language rule:

1. Report feature relationships as prioritization associations, not causal proofs.

---

## 12) Final Report Outline

1. Executive summary.
2. Problem context and data realities (imbalance, missingness, temporal shift).
3. Lane A replication results (`strict` and `+MI`).
4. Lane B selection results (Stage A diagnostic + Stage B multi-seed ranking + Phase 2 freeze).
5. Lockbox results for available frozen models and thresholds.
6. Operational thresholding and workload/cost tradeoffs.
7. MSPC comparison and feature-actionability section.
8. Limitations and deployment caveats.

Mandatory limitations section content:

1. Single dataset, single process context, limited time window.
2. Temporal non-i.i.d. behavior and drift risk.
3. Anonymous features and non-causal interpretation.
4. Revalidation required under baseline process shift.
5. Historical-window workload estimates are not guaranteed bounds.

---

## 13) Claim Policy

Global precedence:

1. Drift gate status governs lockbox claim eligibility per frozen supervised model.
2. If a frozen supervised model's drift gate is `HIGH_SHIFT`, superiority claims are disallowed for that model.

Can claim:

1. Replication-lane improvement if CI supports it.
2. Time-aware selection is more deployment-realistic than random CV.
3. Thresholding was set without lockbox tuning.
4. Supervised advantage over MSPC only if, on the lockbox slice, the supervised model's `TPR_at_TNR90` (from `reports/final_lockbox_result.csv`) exceeds MSPC's `best_MSPC_TPR_at_TNR90` (from the `eval_scope='lockbox'` row in `reports/mspc_baseline.csv`) and that model's drift gate is not `HIGH_SHIFT`.
5. If Lane B BER is worse than Lane A BER, describe this as expected under stricter temporal validation.
6. If MSPC matches or exceeds supervised at `TNR=90%`, report this as a valid finding.
7. Benchmark-improvement claim versus `33.5%` BER is allowed only when the `Replication-Strict F-test` mean BER 95% CI upper bound is below `0.335` (CI from `reports/baseline_replication_summary.csv` using percentile bootstrap on Lane A fold BER; `n_boot=1000`, seed `42`).
8. Primary head-to-head benchmark claim versus `33.5%` must use `Replication-Strict F-test` BER (matched selector to McCann & Johnston 2008); other selectors are supportive evidence only.

Cannot claim:

1. Causality from feature importance.
2. Cross-fab or long-horizon generalization without new validation.
3. Lockbox superiority after any post-lockbox tuning.
4. Supervised superiority when MSPC is equal or better at matched `TNR=90%`.
5. Any superiority claim when drift gate is `HIGH_SHIFT`.

---

## 14) Pre-Registration Checklist

1. Lane B lockbox holdout rule (`last floor(0.15*N)` by sorted `(timestamp, raw_row_id)`).
2. Outer fold plan, fallback hierarchy, fail minimum, and inner-CV feasibility gate (`min(n_fail, n_pass) >= 5` for any `StratifiedKFold(n_splits=5, ...)` slice).
3. Seed policy by stage (`42`, `{42,11,23,37,59}`, `{42,11,23,37,59}`, `42`).
4. Stage A role and fixed settings.
5. Stage B selector set and Pearson/F-test de-dup behavior.
6. Stage B and Phase 2 explicit search grids.
7. Inner scoring and tie-break rule (`ROC-AUC` and BER computed per inner fold and averaged as an unweighted mean over inner folds; BER within `0.01`; deterministic config tie-break).
8. Inner BER threshold derivation rule (inner-train only).
9. Outer-test threshold derivation rule (outer-train only).
10. Stage B method ranking and deterministic tie-break chain.
11. Challenger eligibility and deterministic tie-break chain.
12. No-eligible-challenger fallback behavior and artifact rules.
13. Phase 2 freeze rule for each available role.
14. Phase 3 refit and threshold-derivation rule.
15. Operational threshold constants (`<=10%` mean weekly cap, `TNR=90%`, cost-ratio set).
16. Drift gate criteria and claim restrictions.
17. Required artifacts, row grains, and uniqueness constraints.
18. `run_manifest.json` deterministic hash policy and required keys.
19. ReliefF implementation determinism: use `skrebate.ReliefF` (no `random_state`); only CV split assignment depends on seeds.
20. Multi-seed feature-stability aggregation rule over `(outer_fold, seed)` tuple units.
21. Benchmark-claim lock rule: `33.5%` head-to-head claim uses `Replication-Strict F-test` with pre-registered CI criterion.
22. Deterministic threshold tie-break rules for scientific, operational-cap, and `TNR=90%` extraction points.
23. Label contract (pass/fail mapping, `y_bin` positive class, and `True+`/`True-` conventions).
24. Lane B transformed feature identity contract (value `X{j}` and missing indicator `M{j}` indexing, and missing-only indicator accounting rule).
25. PSI computation rule (quantile edges on DEV, duplicate-edge collapse, open-ended extreme bins, missing-bin handling, bin fractions over all samples, `eps`, and formula).
26. Supervised thresholding score/inequality and candidate rule (`p_hat=Pr(y_bin=1)`, predict fail iff `p_hat >= threshold`, unique-score candidates plus sentinels).
27. Lane A replication scope, pairing, and thresholding: full dataset (`DEV+LOCKBOX`) with paired folds across `Replication-Strict` and `Replication+MI`, and fold-train-only BER-optimal threshold derivation for strict KRR scoring.
28. Feature clustering rule: Pearson correlation using the Phase-3 frozen primary's imputer (fit on full DEV) applied to DEV value features only (exclude missing indicators and all lockbox rows).
29. Timestamp parsing contract (day-first format `DD/MM/YYYY HH:MM:SS`, `errors='coerce'`, UTC-naive no-conversion).
30. Selector implementation contract (definitions for S2N/Welch-t/F-test/Pearson/ReliefF/Gram-Schmidt, undefined-score handling, and feature-level tie-break).
31. Unweighted tuple-mean convention for Stage B method ranking metrics (each `(outer_fold, seed)` counts equally).
32. Reporting metric computation pins (Section 8): `PR_AUC=sklearn.metrics.average_precision_score` (not trapezoid PR integral), plus `ROC_AUC=roc_auc_score`, `MCC=matthews_corrcoef`, and `F2=fbeta_score(beta=2, zero_division=0)`.
