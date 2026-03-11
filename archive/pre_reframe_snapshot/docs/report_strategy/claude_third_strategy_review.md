# Claude Third Strategy Review

## Purpose

This document reviews `final_end_to_end_report_strategy.md` after the seven-issue
round in `claude_second_strategy_review.md` was addressed. All seven prior items
are correctly resolved. This round identifies what remains.

---

## Critical Error

### §4.1: KernelRidge Label Encoding Is Not Specified

The canonical document specifies:

> "Literature-faithful classifier: Kernel Ridge (`kernel='rbf'`, `alpha=1.0`,
> `gamma='scale'`), decision threshold at score `0.0`."

The decision at score `0.0` is only correct if KernelRidge is trained on labels
in `{-1, +1}`. With `{-1, +1}` encoding, the two classes straddle the zero axis
symmetrically and the threshold at `0.0` is the natural binary separating point.

The project's data loading converts the original labels from `{-1, +1}` raw values
to `{0, 1}` binary (via `y_bin = (y_raw == 1).astype(int)`). With `{0, 1}` labels:

- KernelRidge is an unconstrained regression estimator. It estimates
  approximately `E[y | x]`, producing predictions clustered near `0` for passes
  and near `1` for fails, with the global mean at roughly `0.066` (prevalence).
- Almost all predictions are positive (above `0`).
- Threshold at `0.0` therefore flags virtually every sample: TPR ≈ 1.0, TNR ≈ 0.0,
  BER ≈ 50%.
- This is degenerate and produces no useful discrimination in Lane A.

McCann & Johnston (2008) used the standard kernel ridge regression convention of
`{-1, +1}` binary encoding, which is why their threshold sits at `0.0`. The
strategy document correctly specifies the threshold but omits the label encoding
that makes it valid.

**Required addition to §4.1:** After item 5 (classifier spec), add:

> "Lane A label encoding: for KernelRidge fitting only, convert binary labels to
> `{-1, +1}` as `y_krr = 2 * y_bin - 1`. All other pipeline steps (imputer, scaler,
> selector) use the standard `{0, 1}` binary labels. The `{-1, +1}` encoding is
> Lane A-specific and does not affect Lane B."

Without this, a correct and literal reading of the document produces a Lane A
experiment where every sample is classified as fail, all BER values are 50%, and
the comparison to the 2008 benchmark is entirely invalid.

---

## Moderate Issues

### Artifacts 4–5 Have No Minimum Column Specifications

`§11` now specifies minimum columns for six artifact types. Two required artifacts
remain without column definitions:

- `reports/timeaware_selector_screening.csv` (artifact 4)
- `reports/timeaware_model_selection.csv` (artifact 5)

These are the primary outputs from Stage A and Stage B respectively. They appear
in the required artifact list and will be referenced in the hiring-manager report.
Without column specifications, implementations will vary and the artifacts cannot
be validated against protocol.

`splitwise_timeaware_results.csv` (artifact 6) provides the fold-level detail.
Artifacts 4 and 5 are the fold-aggregated summaries. Suggested minimum columns:

**`timeaware_selector_screening.csv`** (Stage A summary — one row per method):
`method, mean_BER, std_BER, mean_True+, mean_True-, fold_1_BER, fold_2_BER,
fold_3_BER, fold_1_fails, fold_2_fails, fold_3_fails, promoted_to_stage_b`

**`timeaware_model_selection.csv`** (Stage B summary — one row per config):
`config_id, method, k, C, scaler, mean_BER, std_BER, mean_True+, std_True+,
mean_True-, std_True-, is_primary, is_challenger`

---

### §13.2 Benchmark Claim Does Not Specify Which Selector

The benchmark-improvement claim reads:

> "Bootstrap on Lane A fold BER (n_boot=1000, seed 42); only claim improvement
> over 33.5% when the model BER 95% CI upper bound is below 0.335."

Lane A produces results for six selectors. The 2008 benchmark of 33.5% is the
`F-test` result from McCann & Johnston. Claiming improvement over that benchmark
is meaningful when comparing our `F-test` (Replication-Strict) against theirs —
the protocol is otherwise matched. Claiming improvement using a different selector
(e.g., our `S2N`) against the 2008 `F-test` result compares different methods and
is not a head-to-head improvement claim.

The claim policy should specify which selector's fold BER drives the benchmark
comparison. Without this, the document permits claiming improvement by selecting
the best-performing selector after seeing results, which is the post-hoc model
shopping the protocol is designed to prevent.

**Required addition to §13.2:** "The benchmark comparison uses Replication-Strict
`F-test` BER as the matched selector (same selector as McCann & Johnston 2008).
If other Replication-Strict selectors also beat `33.5%` with CI support, report
them as supportive evidence, not as the primary benchmark comparison claim."

---

## Minor Issues

### §9.3.11: Feature Cluster Correlation Should Specify Imputed Features

The rule is "compute on full DEV (post-freeze), excluding lockbox rows." It does
not specify whether correlation is computed on raw features (which have ~5.9%
missing values) or imputed features.

Correlation computed on raw features with NaN requires pairwise complete-case
analysis, which means different pairs use different sample subsets. This can
produce a non-positive-semidefinite correlation matrix, which breaks connected
component computation (some pairs with few overlapping non-NaN samples will have
unreliable correlation estimates).

Correlation on imputed features (median-imputed, add_indicator mode) is stable and
uses the full DEV sample for every feature pair. This is the correct approach.

**Required addition to §9.3.11:** "Use full DEV features after applying the fitted
imputer from the frozen primary model (i.e., imputed feature values, not raw).
Correlation is computed on original features only (590 columns), not on the missing
indicator columns added by `add_indicator=True`."

The second clause is also worth clarifying: the 0/1 binary indicators are perfectly
anti-correlated with their source feature's missingness pattern, not with the
features themselves. Including them in the |corr| ≥ 0.95 clustering would
produce spurious cluster links between unrelated features that happen to share
the same missing-data pattern.

---

### §10: Weekly Production Rate Is Not Defined

`§10.1` and `§10.8` require "weekly flagged wafers" and "expected weekly flagged
wafers at each threshold." Computing these requires a weekly production rate that
is not stated in the document.

The SECOM dataset provides this directly: DEV samples / DEV weeks. With
approximately 1331 DEV samples over roughly 12 DEV weeks, the estimated rate is
~111 wafers/week. But this should be stated explicitly rather than derived
implicitly, because it affects the operational output framing.

**Required addition to §10:** "Estimate weekly production rate as:
`rate = DEV_sample_count / DEV_week_count`, where `DEV_week_count` is the
highest `week_idx` in DEV plus one. Apply this rate to the predicted flag
fraction to compute expected weekly flagged wafers."

---

### §12.8: Limitations Section Has No Pre-Specified Content

`§12.8` is "Limitations and what is needed before production deployment." This
section has no pre-specified minimum content in the document.

An open-ended limitations section is a vector for motivated reasoning: after seeing
results, the author can choose which limitations to emphasize or downplay. More
critically, a hiring manager expects to see specific production-readiness thinking,
not a generic disclaimer.

This section has been flagged in two prior review rounds and remains unaddressed.
The minimum content should be pre-registered before execution. Suggested
minimum requirements:

1. Single dataset, single fab, single 14-week window — no demonstrated
   cross-fab generalization.
2. Non-i.i.d. data (temporal autocorrelation, documented fail-rate drift) — the
   model was trained and validated within one distribution shift.
3. No fab-process metadata — feature IDs are anonymous sensor indices; causal
   intervention is not supported by this analysis.
4. Requires recalibration if process baseline changes.
5. Threshold was set on a historical training window; review-capacity estimates
   are averaged over the DEV period, not per-week guaranteed bounds.

---

### §3: Missing Timestamp Handling Not Specified

`§3.1` says "Sort all samples by timestamp." The SECOM dataset has some samples
where the timestamp cannot be parsed (NaT). How these samples are handled affects
N (the total count in §3.2's `floor(0.15 * N)` formula).

If NaT samples are sorted to the beginning (pandas default for `sort_values`):
they enter the earliest DEV training folds and could contaminate the temporal
ordering claim.

If NaT samples are dropped before sorting: N is reduced, and the partition
boundaries shift.

**Minimal required addition to §3:** "Samples with unparseable timestamps are
excluded before sorting and partitioning. N in `floor(0.15 * N)` is the count
after exclusion."

---

## Summary

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | §4.1 KernelRidge label encoding missing — threshold at 0.0 requires {-1, +1} | Correctness error | Critical |
| 2 | Artifacts 4–5 (Stage A and Stage B summary CSVs) have no column specs | Specification gap | Moderate |
| 3 | §13.2 benchmark claim selector unspecified — allows post-hoc selector cherry-picking | Protocol gap | Moderate |
| 4 | §9.3.11 feature cluster should use imputed features; indicators should be excluded | Correctness / clarity | Minor |
| 5 | §10 weekly production rate not defined | Specification gap | Minor |
| 6 | §12.8 limitations section content not pre-specified | Protocol gap | Minor |
| 7 | §3 missing timestamp handling not stated | Specification gap | Minor |

---

## What Was Correctly Resolved From Prior Round

All seven issues in `claude_second_strategy_review.md` are correctly addressed:

1. §7.6 TNR matched-point direction fixed (lowest threshold ✓)
2. Week index 0-based vs 1-based ambiguity resolved (explicit N-1 mapping ✓)
3. "Last 15% by time" — now by sample count after sorting ✓
4. `TPR_at_TNR90` disambiguated (T2, Q, best columns added ✓)
5. "Conditional effect magnitude" now defined (mean |coef_j|) ✓
6. Stage B workload note for ReliefF promotion added ✓
7. Lane A replication artifact column specs added ✓
