# Runbook 02: Lane A Replication

Canonical source: `docs/final_end_to_end_report_strategy_merged.md` (Sections 4.1, 8, 10, 13, 14).

## Objective

Produce benchmark-comparable replication results with and without missing indicators, using paired folds and fixed settings.

## Inputs

1. Full dataset (`DEV+LOCKBOX`) after Section 3 preprocessing/sorting.
2. Selector set: `S2N`, `Welch-t`, `F-test`, `Pearson`, `ReliefF`, `Gram-Schmidt`.

## Outputs

1. `reports/baseline_replication_strict.csv`
2. `reports/baseline_replication_with_missing_indicators.csv`
3. `reports/baseline_missing_indicator_ablation.csv`
4. `reports/baseline_replication_summary.csv`

## Fixed Protocol

1. 10-fold `StratifiedKFold(shuffle=True, random_state=42)`.
2. Same folds for both runs (paired ablation).
3. `k=40`.
4. Imputer:
   1. strict: `SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=False)`
   2. +MI: `SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)`
5. Scaler: `StandardScaler(with_mean=True, with_std=True)`.
6. Classifier: `KernelRidge(kernel='rbf', alpha=1.0, gamma=None)`.
7. Label transform for KRR target: `y_krr = 2*y_bin - 1`.
8. Decision threshold (strict KRR): derive BER-optimal threshold on the fold-train predictions only; apply unchanged to that fold test split.
9. Lane A thresholding must be train-only within each fold (never use fold-test labels to pick threshold).
10. ReliefF parameter is fixed: `n_neighbors=10`.
11. Selector behavior (formulas, eps constants, undefined-score handling, deterministic tie-breaks) must follow canonical Section 4.3.

## Procedure

1. Run strict mode for all 6 selectors across 10 folds.
2. Run +MI mode for all 6 selectors across the exact same 10 folds.
3. Compute fold-level `BER`, `True+`, `True-`, plus `n_train`, `n_test`, `n_test_fails`.
4. Compute paired ablation:
   1. per-fold `delta_BER = BER_strict - BER_MI`
   2. 95% paired bootstrap CI for mean delta (`n_boot=1000`, seed 42)
5. Compute summary table per `(selector, replication_mode)`:
   1. mean/std (`std` uses `ddof=1`)
   2. 95% CI for mean metrics via fold bootstrap (`n_boot=1000`, seed 42).

## Exit Criteria

1. Both modes exist for every selector.
2. Fold IDs are identical between strict and +MI.
3. CI methodology is exactly the pre-registered bootstrap method.
4. For strict KRR, threshold is derived on fold-train only and then applied unchanged to fold-test.

## Claim-Linked Checks

1. Benchmark claim uses only `Replication-Strict F-test`.
2. 33.5% claim gate uses 95% CI upper bound from `baseline_replication_summary.csv`.
