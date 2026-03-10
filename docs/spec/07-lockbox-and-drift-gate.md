---
spec_id: "07"
title: "Lockbox Protocol and Drift Gate"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §7"
applies_to:
  - "Lane B"
used_by_runbooks:
  - "../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
  - "../report_strategy/runbooks/05_artifacts_and_claim_checks.md"
keywords:
  - lockbox
  - drift gate
  - psi
  - ks test
  - mspc
---
# 07 Lockbox Protocol and Drift Gate

## Scope

This section applies only if Lane B is feasible.

## Lockbox Procedure

1. Score lockbox once using frozen primary (and frozen challenger if available).
2. Use frozen scientific and operational thresholds.
3. No retuning after lockbox.

## Mandatory Drift Gate Before Interpreting Lockbox Superiority

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

## Lockbox MSPC Companion Evaluation

1. Fit MSPC on full-DEV pass wafers only.
2. Freeze autoscaler, PCA, and UCLs from DEV pass wafers.
3. Score lockbox once and report:
   1. `T2_TPR_at_TNR90`,
   2. `Q_TPR_at_TNR90`,
   3. `best_MSPC_TPR_at_TNR90`.
4. MSPC implementation spec (autoscaling, PCA component selection, UCL formulas, and contributions) is pinned to `docs/report_strategy/improvement_plan.md`.

## See Also

- [06 Threshold Policy](06-threshold-policy.md)
- [08.1 Metric Definitions](08-metrics-and-feature-identity/08.1-metric-definitions.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10-required-artifacts-and-schemas/10.4-freeze-lockbox-and-mspc-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10-required-artifacts-and-schemas/10.5-feature-drift-and-run-manifest.md)
