# SECOM Final End-to-End Report Strategy

## Status

This is the canonical strategy document for the project.  
It supersedes prior draft strategy/critique documents for execution decisions.

---

## 1) End Goal

Deliver a report that is strong for a semiconductor hiring manager and defensible to technical reviewers:

1. Reproduce legacy-style SECOM baselines for comparability.
2. Select a deployment-realistic model using strict time-aware validation.
3. Report operational impact (fails caught, alarm burden, cost tradeoff), not just ML metrics.

---

## 2) Non-Negotiable Validation Rules

1. No leakage: all preprocessing/selection/model steps fit on training data only per split.
2. No lockbox tuning: lockbox touched once after full config freeze.
3. Drift is a required diagnostic gate before interpreting lockbox (defined in Section 8).
4. Claims must include uncertainty and scope limits (single dataset, single period, non-causal feature interpretation).
5. Random seed policy: all random operations use `random_state=42` unless explicitly overridden and logged in `reports/run_manifest.json`.

Config freeze definition:

1. Config freeze occurs immediately after applying Section 6 selection logic to Stage B outer-fold results.
2. After freeze: no method additions, no grid changes, no threshold-policy changes, no retuning.

---

## 3) Data Partition Contract

1. Parse timestamps and drop rows with unparseable timestamps (`NaT`) before any sorting/splitting.
2. Sort remaining samples by timestamp.
3. Reserve lockbox by sample count (not time duration): after sorting by timestamp, lockbox is the last `floor(0.15 * N)` samples.
4. Use first 85% as DEV for all method and threshold decisions.
5. Any previously viewed lockbox results are diagnostic-only and non-final.
6. `N` in the lockbox formula is the row count after dropping `NaT` timestamps.

---

## 4) Two-Lane Phase 1 Plan

## 4.1 Lane A: Replication (benchmark comparability)

Purpose: compare to published SECOM framing as cleanly as possible.

Fixed protocol:

1. Feature count: `k=40`.
2. Baseline selectors: `S2N`, `Welch-t`, `F-test`, `Pearson`, `ReliefF`, `Gram-Schmidt`.
3. Random 10-fold stratified CV (`shuffle=True`, `random_state=42`) for literature-style comparability.
4. Scaler: `StandardScaler`.
5. Imputer strategy: `SimpleImputer(strategy='median')`.
6. Literature-faithful classifier: Kernel Ridge (`kernel='rbf'`, `alpha=1.0`, `gamma='scale'`), decision threshold at score `0.0`.
7. No threshold tuning in Lane A (fixed classifier decision boundary only).
8. Metrics: BER, True+, True- (plus CI on BER).
9. Lane A KernelRidge label encoding: fit with `y_krr in {-1, +1}` using `y_krr = 2*y_bin - 1`; the `0.0` threshold is defined on this encoding.

Required ablation in replication lane:

1. `Replication-Strict`: median imputation with `add_indicator=False`.
2. `Replication+MI`: median imputation with `add_indicator=True`.
3. Keep everything else identical (splits, seeds, classifier, scaling, `k`, thresholds).
4. Report paired deltas and CI for `+MI` vs strict.
5. CI method for paired deltas: percentile bootstrap on fold-level paired BER deltas (`n_boot=1000`, RNG seed `42`).

Interpretation:

1. `Replication-Strict` is the closest benchmark comparison.
2. `Replication+MI` shows incremental value from missingness information.

## 4.2 Lane B: Model selection (deployment-realistic)

Purpose: choose final model under temporal drift and operational constraints.

Outer time-aware fold design (used in both Stage A and Stage B):

1. Build weekly bins on DEV from sorted timestamps using dataset-anchored 7-day bins:
   - `week_idx = floor((timestamp - t_min_dev) / 7 days)`
   - bins are anchored to `t_min_dev`, not ISO/calendar-week boundaries.
   - fold descriptions below use 1-based week labels where `week N` maps to `week_idx = N-1`.
2. Primary split plan (3 anchored expanding windows):
   - Fold 1: train weeks 1-5, test weeks 6-last_DEV_week
   - Fold 2: train weeks 1-7, test weeks 8-last_DEV_week
   - Fold 3: train weeks 1-9, test weeks 10-last_DEV_week
   - test windows intentionally overlap; this is anchored expanding-window backtesting (not disjoint i.i.d. CV) to prioritize deployment realism and recency.
3. Minimum acceptable fail count per outer test window: `20`.
4. Deterministic fallback (used only if any primary fold has `<20` fails):
   - Fold 1: train weeks 1-4, test weeks 5-last_DEV_week
   - Fold 2: train weeks 1-6, test weeks 7-last_DEV_week
   - Fold 3: train weeks 1-8, test weeks 9-last_DEV_week
5. If fallback still violates fail minimum, switch to 2-fold deterministic fallback:
   - Fold A: train weeks 1-6, test weeks 7-last_DEV_week
   - Fold B: train weeks 1-8, test weeks 9-last_DEV_week
6. If 2-fold fallback still violates fail minimum, declare Lane B infeasible and report Lane A only (no deployment-realistic model-selection claim).
7. Per-fold fail counts are mandatory in all fold-level result artifacts.

Stage A (method screening):

1. Purpose: eliminate weak selectors under fixed settings.
2. Uses outer folds only.
3. No inner CV, no hyperparameter tuning.

Stage B (nested tuning):

1. Purpose: tune finalists and freeze primary/challenger configs.
2. Outer folds: same 3 anchored expanding windows as Stage A.
3. Inner CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
4. Inner scoring metric: ROC-AUC (threshold-independent inner objective for stable ranking under low fail counts).
5. Outer-fold evaluation threshold policy is fixed: for each `(config, outer_fold)`, derive BER-optimal threshold on outer-train only (`best_threshold_by_ber`), then apply unchanged to outer-test for BER/True+/True-.
6. Finalist promotion rule from Stage A:
   - promote top 2 distinct methods by Stage A mean BER (after Pearson/F-test de-dup),
   - promote a 3rd only if its mean BER is within `0.02` of the 2nd method's mean BER.

This satisfies breadth (all selectors) and depth (strong finalists) without exploding runtime.

---

## 5) Model-Selection Search Space

## 5.1 Selectors in scope (Lane B)

1. `S2N`
2. `Welch-t`
3. `F-test`
4. `Pearson`
5. `ReliefF`
6. `Gram-Schmidt`

Note:

1. Keep all selectors in Stage A computation sweep (including `Pearson`) for completeness.
2. Pearson `|r|` ranking and ANOVA F-test ranking are mathematically equivalent for binary targets; collapse to one narrative row in reporting.
3. Promote only F-test to Stage B (Pearson excluded from Stage B grid to avoid duplicate tuning).
4. Embedded challengers are not used in Stage A method screening.
5. `L1`/`Elastic Net` are excluded from formal Lane B evaluation to avoid optional post-hoc model shopping.
6. Stage A fixed-settings screening can false-eliminate selectors whose best settings differ from defaults; this risk is accepted for auditability and runtime control.
7. If de-dup reduces finalist count below 2, backfill from the next-ranked distinct Stage A method.

## 5.2 Hyperparameters

Stage A (fixed, all selectors; no tuning):

1. `k=40` fixed.
2. `C=1.0` fixed.
3. Scaler fixed to `RobustScaler`.
4. Missing mode fixed to `values + indicators`.
5. Threshold policy fixed to BER-optimal on outer-train (Youden/`best_threshold_by_ber`).
6. Classifier fixed to `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`.
7. ReliefF fixed setting in Stage A: `n_neighbors=10`.

Stage B (nested tuning, top 2 selectors + optional 3rd within 0.02 BER):

1. `k in {10, 20, 40}`.
2. `C in {0.01, 0.1, 1.0, 10.0}`.
3. Scaler in `{StandardScaler, RobustScaler}`.
4. Missing mode fixed to `values + indicators` (not a Stage B grid dimension).
5. Classifier fixed to `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)`.

Selector-specific (where applicable):

1. ReliefF neighbor count sweep: `n_neighbors in {5, 10, 20}`.
2. Stage B expected workload (top 2 selectors, primary 3-fold outer plan): `2 x 3 x 4 x 2 x 5 x 3 = 720` inner fits.
3. If ReliefF is one of the promoted top-2 selectors, expected total is `1440` inner fits (`360` non-ReliefF branch + `1080` ReliefF-neighbor branch).
4. If ReliefF is promoted and a 3rd selector is also promoted, expected total is `1800` inner fits.
5. If 2-fold outer fallback is used, multiply inner-fit counts above by `2/3`.

---

## 6) Selection Logic

Use lexicographic decision rules to avoid metric shopping:

1. Primary: minimize mean outer BER (unweighted arithmetic mean across outer folds).
2. Define near-equal models pre-run as `abs(delta_BER) <= 0.02`.
3. Pre-registered business preference for model selection: fail-catch priority (higher True+).
4. Challenger eligibility floor: mean outer BER must be `<= 0.40`.
5. Challenger selection preference (reported, not primary): among eligible non-primary configs, choose highest True-.
6. Final tie-break: simpler model (smaller `k`, fewer moving parts).

Notes:

1. No post-hoc shadow guardrail is used; fold-wise outer results already provide temporal stability evidence.
2. No threshold-policy tuning is included in Stage B model ranking.
3. Business preference is pre-registered as fail-catch priority before Stage B execution.
4. The `0.02` near-equal margin is conservative relative to fold-level noise and mainly acts as a guard against overclaiming small differences.
5. Because outer test windows overlap by design, Stage B fold summaries are treated as time-backtest evidence, not independent i.i.d. CV estimates.

---

## 7) Threshold Policy (Post-Selection)

Threshold is finalized only after model config freeze, using full DEV.

Report two operating points:

1. Scientific operating point: BER-optimal threshold (Youden/BER-tuned on DEV).
2. Operational operating point: threshold selected from pre-registered review-capacity rule.

Never use lockbox to choose threshold.

Pre-registered operational constants:

1. Review-capacity cap: mean weekly flagged fraction on DEV must be `<= 10%` at the chosen operational threshold.
2. Secondary matched-comparison operating point: `TNR=90%`.
3. Cost-ratio sensitivity set: `FN:FP in {1, 2, 5, 10, 20}`.
4. Operational-threshold selection rule: choose the highest-TPR threshold whose mean weekly flagged fraction on DEV is `<=10%`.
5. Operational threshold is deterministic from the review-capacity cap; cost-ratio curves are sensitivity analyses, not additional tuning criteria.
6. Capacity-cap caveat: the cap is enforced on mean weekly flagged fraction; weekly peaks may exceed `10%` and must be reported as an operational risk diagnostic.
7. `TNR=90%` matched-point rule: use the lowest threshold such that `TNR >= 0.90` (equivalently highest-TPR threshold under the TNR constraint); if multiple thresholds satisfy this with same TNR, pick the one with higher TPR.
8. Mean weekly flagged fraction is computed as an unweighted mean of per-week flagged fractions:
   - for each DEV week `w`: `frac_w = flagged_w / wafers_w`,
   - `mean_weekly_flagged_fraction = mean_w(frac_w)`.
   - week definition for this computation must match Section 4.2 exactly:
     dataset-anchored 7-day bins with `week_idx = floor((timestamp - t_min_dev)/7 days)`.
9. Threshold-calibration transparency: both scientific and operational thresholds are derived from the frozen model's in-sample predictions on full DEV; lockbox remains the out-of-sample validation.

---

## 8) Lockbox Protocol

Scope:

1. This section applies only if Lane B is feasible under the fail-count rules in Section 4.2.

Protocol:

1. Train frozen primary and challenger on full DEV.
2. Apply frozen thresholds.
3. Evaluate once on lockbox.
4. No retuning after lockbox.
5. Drift gate (mandatory before lockbox interpretation):
   - compute on frozen primary model and frozen preprocessing:
   - prevalence shift: `abs(lockbox_fail_rate - DEV_fail_rate)`.
   - score-distribution shift: two-sample KS test on predicted probabilities (`DEV` vs `lockbox`), with `alpha=0.01`.
   - feature-distribution shift: PSI on frozen-primary top-10 selected value features (10 equal-frequency DEV bins per feature), then report `max_PSI` and `median_PSI`.
   - value-feature definition for PSI: original sensor measurement columns only (exclude missing-indicator-derived columns).
   - top-10 ranking rule for PSI: among frozen primary selected value features, rank by absolute standardized logistic coefficient from the full-DEV frozen fit; use top 10, or all if fewer than 10 are available.
   - drift gate status:
     - `PASS`: prevalence shift `< 0.02`, KS `p >= 0.01`, and `max_PSI < 0.30`.
     - `CAUTION`: any one criterion violated.
     - `HIGH_SHIFT`: two or more criteria violated.
   - if status is `HIGH_SHIFT`, do not make superiority claims from lockbox; report descriptive performance only and explicitly attribute uncertainty to distribution shift.
6. `final_lockbox_result.csv` reports one row per `(config, threshold_policy)` pair: primary+scientific, primary+operational, challenger+scientific, challenger+operational.
7. If lockbox degrades vs DEV means, compare first to the most temporally similar outer fold before concluding general failure.
8. If lockbox BER is better than DEV mean BER, first attribute possible uplift to the documented prevalence/drift shift before claiming stronger generalization.
9. If Lane B is infeasible, do not publish deployment-model lockbox comparisons; report Lane A outputs only and mark lockbox model-selection fields as not applicable in `run_manifest.json`.
10. Lockbox MSPC companion evaluation:
   - fit MSPC on full DEV pass wafers only (autoscaler, PCA, and UCLs from DEV pass wafers),
   - score lockbox samples with that frozen MSPC model,
   - compute `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, and `best_MSPC_TPR_at_TNR90` on lockbox for the claim comparison in Section 13.

---

## 9) Metrics Policy

Lane scope:

1. Lane A replication reports only BER, True+, True- (plus BER CI).
2. Lane B and lockbox reporting use the full Section 9.2 and 9.3 metric suite.
3. Lane B uncertainty reporting: fold-wise values + mean +/- std (do not report formal 95% CI from 3 outer folds as inferential evidence).
4. Lane B outer folds are overlapping expanding-window backtests; interpret summary statistics as temporal robustness diagnostics rather than independent-fold inference.

## 9.1 Primary selection metrics

1. BER (decision metric)

## 9.2 Primary reporting metrics

1. BER, True+, True-
2. ROC-AUC, PR-AUC
3. MCC
4. F2
5. Lane A: report mean, std, and 95% CI across folds.
6. Lane B: report fold-wise values and mean +/- std.
7. Lockbox: report point estimates (single holdout window; no fold-based CI).

## 9.3 Secondary diagnostics

1. Brier score and calibration diagnostics
2. Weekly alarm burden and fails caught/missed
3. Cost curves and break-even region
4. MSPC baseline comparison (`Hotelling T2`, `Q-SPE`)
5. Feature stability and redundancy analysis
6. MSPC fold discipline: for each outer fold, fit autoscaler + PCA + UCLs on that fold's outer-train pass wafers only, then score that fold's outer-test set; define alarm stream by union rule `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)` for `alarm_rate` and `empirical_ARL0`.
7. MSPC implementation spec reference: frozen formulas per `docs/improvement_plan.md` (Section 5.5): `T2` UCL (Tracy-Young-Mason), `Q-SPE` UCL (Jackson-Mudholkar), contribution plots (Kourti-MacGregor).
8. Calibration caveat: with low fail counts, calibration-bin CI will be wide; report 95% CI per bin and avoid over-interpreting point estimates.
9. Brier dual baselines: report both all-pass baseline (`p`) and prevalence-constant baseline (`p*(1-p)`) using the evaluation-slice prevalence `p` for each fold/lockbox slice.
10. Feature stability minimum contents:
   - selection frequency across outer folds,
   - conditional effect magnitude on selected folds (defined as mean absolute standardized logistic coefficient, i.e., `mean(|coef_j|)` across folds where feature `j` is selected),
   - expected contribution (frequency x conditional magnitude),
   - `fold_jaccard_stability` defined as per-feature pairwise selection agreement:
     fraction of outer-fold pairs in which feature `j` is selected in both folds,
   - cluster assignment.
11. Feature-cluster correlation matrix scope: compute on full DEV (post-freeze), excluding lockbox rows, using imputed original feature values only (exclude missing-indicator columns).

---

## 10) Phase 2: Semiconductor-Manager-Facing Outputs

These are required in the final report:

1. Weekly flagged wafers at each operating point.
2. Weekly fail capture and miss counts.
3. Review workload estimate and recommended alert policy.
4. Top stable features grouped into correlated clusters for engineer triage.
5. Supervised vs MSPC baseline comparison at matched operating conditions.
6. Matched operating condition is pre-defined as `TNR=90%`.
7. Feature clusters are defined as connected components of the feature-correlation graph with edge rule `|corr| >= 0.95`.
8. Recommended alert policy must report:
   - scientific threshold value,
   - operational threshold value,
   - expected weekly flagged wafers at each threshold.
9. Weekly production rate definition for operations framing:
   - `weekly_rate = DEV_sample_count / DEV_week_count`,
   - `DEV_week_count = max(week_idx) + 1`,
   - `predicted_flag_fraction` source for expected workload = full-DEV post-freeze flagged fraction from the frozen selected `(config, threshold_policy)` at its frozen threshold,
   - report Stage B mean outer-test flagged fraction as a robustness diagnostic and report lockbox flagged fraction separately as observed holdout behavior.
10. MSPC false-alarm behavior summary using `alarm_rate` and `empirical_ARL0` (outer-fold and lockbox scopes).

Language requirement:

1. Associations for prioritization, not causal proof.

---

## 11) Required Artifacts

1. `reports/baseline_replication_strict.csv`
2. `reports/baseline_replication_with_missing_indicators.csv`
3. `reports/baseline_missing_indicator_ablation.csv`
4. `reports/timeaware_selector_screening.csv`
5. `reports/timeaware_model_selection.csv`
6. `reports/splitwise_timeaware_results.csv`
7. `reports/final_lockbox_result.csv`
8. `reports/mspc_baseline.csv`
9. `reports/operational_cost_curves.csv`
10. `reports/feature_report.csv`
11. `reports/run_manifest.json` (seeds, split boundaries, fixed constants, tuned grid, final frozen config hash).
12. `reports/drift_gate_summary.csv`

Minimum artifact column requirements:

1. `reports/splitwise_timeaware_results.csv`:
   - `config_id`, `method`, `k`, `C`, `scaler`, `fold_index`, `train_window`, `test_window`, `test_fails`, `BER`, `True+`, `True-`, `threshold_policy`, `threshold_value`.
2. `reports/mspc_baseline.csv`:
   - `fold_index`, `eval_scope` (`outer_fold` or `lockbox`), `T2_AUC`, `Q_AUC`, `alarm_rate`, `empirical_ARL0`, `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, `best_MSPC_TPR_at_TNR90`, `best_MSPC_source`.
   - `alarm_rate` and `empirical_ARL0` must be computed from the union alarm stream: `alarm = (T2 > UCL_T2) OR (Q > UCL_Q)`.
   - lockbox sentinel and completeness rules:
     - for lockbox MSPC companion evaluation, set `eval_scope='lockbox'` and `fold_index='LOCKBOX'`,
     - required non-null fields for lockbox row: `eval_scope`, `fold_index`, `T2_AUC`, `Q_AUC`, `alarm_rate`, `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, `best_MSPC_TPR_at_TNR90`, `best_MSPC_source`,
     - `empirical_ARL0` may be `NaN` only when fewer than two alarms occur in the evaluated sequence; if `NaN`, include reason in `run_manifest.json`.
3. `reports/operational_cost_curves.csv`:
   - `cost_ratio`, expected cost columns for `primary_scientific`, `primary_operational`, `challenger_scientific`, `challenger_operational`, plus `all_pass_baseline`, `all_flag_baseline`.
4. `reports/feature_report.csv`:
   - `feature_index`, `feature_name_or_source_col`, `selection_frequency`, `conditional_effect_magnitude`, `expected_contribution`, `fold_jaccard_stability`, `cluster_id`.
5. `reports/run_manifest.json`:
   - include `config_hash` as `SHA-256` of UTF-8 JSON with sorted keys and rounded floats (6 significant figures);
   - hashed config must include at least: `method`, `k`, `C`, `scaler`, `missing_mode`, `threshold_policy`, and split-definition identifiers.
6. Lane A replication artifacts:
   - `reports/baseline_replication_strict.csv` and `reports/baseline_replication_with_missing_indicators.csv` columns: `selector`, `fold`, `BER`, `True+`, `True-`, `n_train`, `n_test`, `n_test_fails`.
   - `reports/baseline_missing_indicator_ablation.csv` columns: `selector`, `BER_strict`, `BER_MI`, `delta_BER`, `CI_lower`, `CI_upper`, `n_boot`.
7. `reports/timeaware_selector_screening.csv` (Stage A summary; one row per method):
   - `method`, `n_splits`, `mean_BER`, `std_BER`, `mean_True+`, `mean_True-`, `min_test_fails`, `promoted_to_stage_b`.
8. `reports/timeaware_model_selection.csv` (Stage B summary; one row per config):
   - `config_id`, `method`, `k`, `C`, `scaler`, `n_splits`, `mean_BER`, `std_BER`, `mean_True+`, `std_True+`, `mean_True-`, `std_True-`, `is_primary`, `is_challenger`.
9. `reports/final_lockbox_result.csv`:
    - `config_id`, `method`, `threshold_policy`, `threshold_value`, `BER`, `True+`, `True-`, `ROC_AUC`, `PR_AUC`, `MCC`, `F2`, `lockbox_n`, `lockbox_fails`.
10. `best_MSPC_TPR_at_TNR90` definition: `max(T2_TPR_at_TNR90, Q_TPR_at_TNR90)`; `best_MSPC_source in {'T2','Q'}` records the source metric.
11. `reports/drift_gate_summary.csv` (one row for frozen-primary lockbox assessment):
    - `model_scope` (`primary_frozen`), `dev_fail_rate`, `lockbox_fail_rate`, `abs_prevalence_shift`, `ks_pvalue_scores`, `max_PSI`, `median_PSI`, `drift_gate_status` (`PASS`/`CAUTION`/`HIGH_SHIFT`), `lockbox_claims_allowed` (boolean).

---

## 12) Final Report Outline (Hiring-Manager Ready)

1. Executive summary (what model, what impact, what limits).
2. Problem context and dataset realities (imbalance, drift, missingness).
3. Replication results (strict and +MI).
4. Time-aware model-selection results (Stage A and Stage B).
5. Lockbox one-time results (frozen configs only).
6. Operational deployment framing (alarm burden, cost, recommended threshold).
7. MSPC comparison and feature-actionability section.
8. Limitations and what is needed before production deployment.

Minimum required limitations content (Section 12.8):

1. Single dataset / single fab / single ~14-week window; no demonstrated cross-fab generalization.
2. Temporal non-i.i.d. behavior (autocorrelation + drift) within the observed period.
3. Anonymous feature IDs; analysis is association-based and not causal-intervention ready.
4. Recalibration/revalidation required when process baseline shifts.
5. Thresholds and review-capacity estimates are historical-window estimates, not guaranteed weekly bounds.

---

## 13) Claim Policy

Global precedence rule:

1. Section 8 drift gate controls lockbox claim eligibility.
2. If drift gate status is `HIGH_SHIFT`, lockbox superiority claims are disallowed regardless of metric deltas.

Can claim:

1. Improvement under your replicated framing when CI supports it.
2. Time-aware validation is more deployment-realistic than random CV.
3. Operational thresholding was chosen without test-set leakage.
4. Supervised model advantage over MSPC only if supervised `TPR_at_TNR90` exceeds MSPC at the same `TNR` and Section 8 drift gate status is not `HIGH_SHIFT`.
5. If Lane B BER is worse than Lane A BER, interpret this as expected under stricter time-aware validation; Lane B is the deployment estimate, Lane A is comparability.
6. If Section 8 drift gate status is not `HIGH_SHIFT` and MSPC matches or beats supervised at `TNR=90%`, report this as a valid positive finding: labeled supervision did not add discrimination advantage in this window.

Cannot claim:

1. Causality from feature importance alone.
2. Generalization beyond this fab/window without new data.
3. Lockbox superiority if any post-lockbox tuning occurred.
4. Supervised superiority over MSPC when MSPC is equal or better at matched `TNR=90%`.
5. Any lockbox superiority claim when Section 8 drift gate status is `HIGH_SHIFT`.

Claim uncertainty methods:

1. Lane A paired ablation CI: percentile bootstrap on fold deltas (`n_boot=1000`, seed `42`).
2. Benchmark-improvement CI: percentile bootstrap on Lane A fold BER (`n_boot=1000`, seed `42`); only claim improvement over `33.5%` when the model BER 95% CI upper bound is below `0.335`.
3. Benchmark comparison selector lock: primary head-to-head claim versus `33.5%` uses Replication-Strict `F-test` BER (matched selector to McCann & Johnston 2008). Other selectors are supportive evidence only.

