# Claude Second Strategy Review

## Purpose

This document reviews `final_end_to_end_report_strategy.md` after the 14-issue
round in `claude_final_strategy_review.md` was addressed. All 14 prior items
have been correctly resolved. This round identifies remaining issues in the
updated document.

---

## Critical Error

### §7.6: TNR Matched-Point Rule Direction Is Reversed

The updated rule reads:

> "TNR=90% matched-point rule: use the highest threshold such that TNR >= 0.90;
> if multiple thresholds satisfy this with same TNR, pick the one with higher TPR."

This is logically inverted. Here is why:

As the classification threshold increases:
- Fewer samples are predicted positive.
- TNR increases (more true negatives, fewer false positives).
- TPR decreases (more missed fails).

The set of thresholds satisfying `TNR >= 0.90` is therefore `{T : T >= T*}` where
`T*` is the lowest threshold at which TNR reaches 90%. This set includes every
threshold from `T*` up to the maximum. "The **highest** threshold with TNR >= 0.90"
selects the maximum threshold — where TNR approaches 1.0 and TPR approaches 0.

That is not the TNR=90% operating point. It is the maximum-specificity point.

Concrete example:

| Threshold | TNR   | TPR  |
|-----------|-------|------|
| 0.30      | 0.85  | 0.90 |
| 0.45      | 0.90  | 0.70 |
| 0.60      | 0.94  | 0.45 |
| 0.80      | 0.99  | 0.15 |

Thresholds with TNR >= 0.90: {0.45, 0.60, 0.80}.
- "Highest threshold": 0.80 → TNR=0.99, TPR=0.15. Comparing to MSPC here means
  comparing at TNR=99%, not TNR=90%. The comparison is invalid.
- "Lowest threshold": 0.45 → TNR=0.90, TPR=0.70. This is the operating point at
  TNR=90%, maximum sensitivity at that specificity. This is correct.

The `improvement_plan.md §5.5` (which the canonical doc defers to for metric
definitions) confirms the intent: "TPR at TNR=90% — Read from ROC curve:
sensitivity at 90% specificity." Sensitivity at 90% specificity is TPR at the
point where TNR=90%, which is the **lowest** threshold with TNR >= 0.90.

The §7.4 operational threshold rule is correctly stated for comparison:
"highest-TPR threshold with flagged rate <= 10%." The TNR=90% rule should follow
the same logical form.

**Required fix to §7.6:** Replace "use the highest threshold such that TNR >= 0.90"
with "use the lowest threshold such that TNR >= 0.90 (equivalently: highest-TPR
threshold with TNR >= 0.90)."

The tie-break clause ("if multiple thresholds satisfy this with same TNR, pick the
one with higher TPR") is correct as written.

---

## Moderate Issues

### Off-by-One Between the Week Formula and Fold Descriptions

§4.2 now specifies:

> `week_idx = floor((timestamp - t_min_dev) / 7 days)`

This formula produces 0-based indices: the first week is `week_idx = 0`, the
second is `week_idx = 1`, and so on.

The fold descriptions immediately below use 1-based language:

> Fold 1: train weeks 1-5, test weeks 6-last_DEV_week

If an implementer reads "weeks 1-5" as `week_idx` values 1, 2, 3, 4, 5, all
samples in `week_idx = 0` (the first week of DEV) are excluded from every training
fold. For a dataset with ~112 samples per week, this silently drops ~112 samples
from all training folds and is never caught by the fail-count gate.

If the intent is that "weeks 1-5" means the first five weeks (i.e., `week_idx`
0 through 4), the document needs to state that mapping explicitly. Without it,
the formula and the fold descriptions are inconsistent in their indexing base.

**Required addition to §4.2:** "Fold descriptions use 1-based week numbering.
'Week N' corresponds to `week_idx = N-1`. Fold 1 train set is all samples with
`week_idx` in {0, 1, 2, 3, 4}; test set is all samples with `week_idx >= 5`."

---

### §3: "Last 15% by Time" Is Ambiguous

"Reserve last 15% by time" is read two ways:

1. **Time-duration**: last 15% of the calendar span (if data spans 91 days,
   the last 13.65 days). Samples are not uniformly distributed, so this fraction
   of the time window contains an unpredictable number of samples.
2. **Sample-count after sorting**: last 15% of samples when sorted by timestamp
   (floor(0.15 × 1567) = 235 samples). This is what prior documents show (~236
   samples, Oct 5–Oct 17).

These can differ. SECOM data may have production rate variation by week.
Interpretation 1 could place 15% of time in a period with 10% of samples, or 20%.

The lockbox fail-count is approximately fixed under interpretation 2 but variable
under interpretation 1. For reproducibility and for the drift diagnostic (DEV vs
lockbox fail rates), the sample-count interpretation should be stated explicitly.

**Required addition to §3:** "The 15% cutoff is by sample count: sort all samples
by timestamp, then the last `floor(0.15 × N)` samples are the lockbox."

---

## Minor Issues

### `TPR_at_TNR90` in mspc_baseline.csv Is Ambiguous

`§11` specifies `mspc_baseline.csv` with the column `TPR_at_TNR90`. MSPC produces
two separate scores: T² and Q-SPE. The comparison at TNR=90% (§10.5–6) requires
specifying which score is used.

Options:
- T² at TNR=90%
- Q-SPE at TNR=90%
- The higher of the two (best MSPC discriminator)
- A combined "either alarm" rule's TPR at TNR=90%

The claim policy (§13.4) says "supervised model advantage over MSPC only if
supervised TPR_at_TNR90 exceeds MSPC at the same TNR." Without knowing which MSPC
score drives `TPR_at_TNR90`, this claim condition is not precisely executable.

The artifact spec should add separate columns (`T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`)
or specify that `TPR_at_TNR90` uses the better of T² and Q independently scored.

---

### "Conditional Effect Magnitude" Is Not Defined

`§9.3 bullet 10` requires "conditional effect magnitude on selected folds" as part
of the feature stability report. No metric definition is given.

Candidates: logistic regression coefficient magnitude (absolute value of the
weight after scaling), Cohen's d between fail/pass distributions for that feature,
or the S2N score. These are not equivalent and produce different rankings.

For the feature report to be reproducible and interpretable by an engineer, the
metric should be specified. Suggested: "coefficient magnitude from the fitted
LogisticRegression on the outer training fold, after the pipeline scaler and
selector have been applied, averaged across folds where the feature was selected."

---

### Stage B Workload Calculation Underestimates When ReliefF Is Promoted

§5.2 states:

> "Stage B expected workload (top 2 selectors): 2 x 3 x 4 x 2 x 5 x 3 = 720 inner fits."

The header says "top 2-3 selectors." If ReliefF is one of the top 2-3 and is
promoted to Stage B, it adds an additional grid dimension (`n_neighbors ∈ {5, 10, 20}`).
The ReliefF-specific workload is:
3 n_neighbors × 3 k × 4 C × 2 scalers × 5 inner folds × 3 outer folds = 1080.

Total if ReliefF is promoted with two other selectors: 720 + 1080 = 1800 inner fits.
This is not a blocker, but the runtime estimate in the document will be wrong if
ReliefF is promoted, which affects scheduling. The note should include the
ReliefF conditional: "If ReliefF is promoted, add 1080 inner fits (3 additional
n_neighbors values); total becomes 1800."

---

### Lane A Replication Artifacts Have No Column Specifications

`§11` specifies minimum columns for four artifacts: `splitwise_timeaware_results.csv`,
`mspc_baseline.csv`, `operational_cost_curves.csv`, and `feature_report.csv`.

Artifacts 1–3 in §11 — the Lane A replication CSVs — have no column specifications:

1. `baseline_replication_strict.csv`
2. `baseline_replication_with_missing_indicators.csv`
3. `baseline_missing_indicator_ablation.csv`

Without column specs, each could be implemented differently and could not be
reliably compared or validated. Minimum suggested columns:

For artifacts 1–2:
`selector, fold, BER, True+, True-, n_train, n_test, n_test_fails`

For artifact 3 (the ablation summary):
`selector, BER_strict, BER_MI, delta_BER, CI_lower, CI_upper, n_boot`

---

## What Was Correctly Resolved From Prior Round

All 14 issues in `claude_final_strategy_review.md` are correctly addressed:

1. MSPC per-fold training scope now explicit in §9.3.6 ✓
2. Weekly bin construction now deterministic (dataset-anchored formula) ✓
3. Brier baseline prevalence now uses evaluation-slice p ✓
4. Lane B CI correctly limited to mean ± std ✓
5. Challenger BER eligibility floor added (≤ 0.40) ✓
6. TNR=90% interpolation rule added (§7.6 — direction needs fix, see above) ✓ (rule present, direction wrong)
7. feature_report.csv column spec added ✓
8. run_manifest.json hash algorithm specified ✓
9. Pearson ≡ F-test stated as always-true ✓
10. Lane B > Lane A BER pre-registered interpretation ✓
11. MSPC competitive case pre-registered interpretation ✓
12. MSPC spec cross-reference qualified as frozen ✓
13. Feature cluster correlation matrix scope specified ✓
14. Lockbox better-than-DEV case added ✓

---

## Summary

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | §7.6 TNR matched-point direction is reversed | Correctness error | Critical |
| 2 | Off-by-one: 0-based formula vs 1-based fold descriptions | Ambiguity | Moderate |
| 3 | "Last 15% by time" — sample-count vs time-duration | Ambiguity | Moderate |
| 4 | `TPR_at_TNR90` ambiguous (T² vs Q vs combined) | Specification gap | Minor |
| 5 | "Conditional effect magnitude" metric undefined | Specification gap | Minor |
| 6 | Stage B workload underestimates if ReliefF promoted | Documentation | Minor |
| 7 | Lane A artifact column specs missing | Specification gap | Minor |
