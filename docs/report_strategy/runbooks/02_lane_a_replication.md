# Runbook 02: Lane A Replication

Canonical sources:

1. `docs/spec/04-two-lane-plan/04.1-lane-a-replication.md`
2. `docs/spec/06-threshold-policy.md`
3. `docs/spec/08-metrics-and-feature-identity/08.1-metric-definitions.md`
4. `docs/spec/10-required-artifacts-and-schemas/10.2-lane-a-artifacts.md`
5. `docs/spec/13-claim-policy.md`
6. `docs/spec/14-pre-registration-checklist.md`

## Objective

Produce benchmark-comparable replication results with and without missing indicators using a global OOF configuration-and-threshold protocol.

## Inputs

1. Full dataset (`DEV+LOCKBOX`) after Section 3 preprocessing/sorting.
2. Selector set: `S2N`, `Welch-t`, `F-test`, `Pearson`, `ReliefF`, `Gram-Schmidt`.

## Outputs

1. `reports/lane_a_global_sweep.csv`
2. `reports/lane_a_global_best_config.csv`
3. `reports/lane_a_global_fold_metrics.csv`
4. `reports/lane_a_global_summary.csv`
5. `reports/lane_a_global_ablation.csv`
6. `reports/lane_a_global_full_fit_summary.csv`

## Fixed Protocol

1. 10-fold `StratifiedKFold(shuffle=True, random_state=42)`.
2. Same folds for strict and +MI runs (paired ablation).
3. `k=40`.
4. Imputer:
   1. strict: `SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=False)`
   2. +MI: `SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)`
5. Scaler: `StandardScaler(with_mean=True, with_std=True)`.
6. Official classifiers: `krr` (tuned Kernel Ridge) and `logreg` (tuned Logistic Regression).
7. Optional benchmark-only classifier: `krr_strict` (`KernelRidge(kernel='rbf', alpha=1.0, gamma=None)`).
8. Label transform for KRR target: `y_krr = 2*y_bin - 1`.
9. Lane A tuning/thresholding protocol:
   1. evaluate each candidate config by pooled OOF BER,
   2. derive BER-optimal `threshold_oof_global` from pooled OOF predictions/labels for that config,
   3. pick one best config per `(selector, classifier, replication_mode)` via min OOF BER,
   4. tie-break deterministically by ascending parameter tuple (for KRR, `gamma=None` sorts before numeric gamma),
   5. compute fold metrics at the selected config's `threshold_oof_global`.
10. ReliefF parameter is fixed: `n_neighbors=10`.
11. Full-data threshold in `lane_a_global_full_fit_summary.csv` is diagnostic-only and not used for Lane A tuning claims.
12. Selector behavior (formulas, eps constants, undefined-score handling, deterministic tie-breaks) must follow canonical Section 4.3.

## Procedure

1. For each classifier and replication mode, enumerate deterministic config grids over selector and classifier parameters.
2. For each config, run all 10 folds, stack OOF scores, calibrate `threshold_oof_global`, and compute pooled OOF BER.
3. Select the best config per `(selector, classifier, replication_mode)` using objective+tie-break rules.
4. Write per-fold metrics for each selected best config to `lane_a_global_fold_metrics.csv`.
5. Build summary per `(selector, classifier, replication_mode)` from selected-config fold metrics:
   1. mean/std (`std` uses `ddof=1`)
   2. 95% CI for mean metrics via fold bootstrap (`n_boot=1000`, seed `42`).
6. Build strict-vs-MI ablation per `(selector, classifier)`:
   1. per-fold `delta_BER = BER_strict - BER_MI`
   2. 95% paired bootstrap CI for mean delta (`n_boot=1000`, seed 42).
7. Refit selected best configs on full Lane A data and write diagnostic full-data thresholds/metrics to `lane_a_global_full_fit_summary.csv`.

## Exit Criteria

1. Exactly one best config row exists per `(selector, classifier, replication_mode)`.
2. Exactly 10 fold rows exist per selected triplet in `lane_a_global_fold_metrics.csv`.
3. CI methodology is exactly the pre-registered bootstrap method.
4. Full-data threshold rows are marked diagnostic-only.

## Claim-Linked Checks

1. Benchmark claim uses only `classifier='krr', selector='F-test', replication_mode='strict'`.
2. 33.5% claim gate uses the 95% CI upper bound from `lane_a_global_summary.csv` for that anchored row.
