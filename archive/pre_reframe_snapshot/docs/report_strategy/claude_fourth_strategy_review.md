# Claude Fourth Strategy Review

## Purpose

This document reviews `final_end_to_end_report_strategy.md` after the nine-item
round in Codex's self-review was fully addressed. All prior review items across
three rounds are correctly resolved. This round identifies what remains.

The document is now in good shape. The issues below are fewer and lower-severity
than previous rounds.

---

## Moderate Issues

### No Lockbox MSPC Evaluation Protocol or Artifact

The claim policy (§13.4) says:

> "Supervised model advantage over MSPC only if supervised TPR_at_TNR90 exceeds
> MSPC at the same TNR."

The lockbox protocol (§8) specifies how the supervised model is evaluated on the
lockbox. The final_lockbox_result.csv contains supervised model results.

MSPC is evaluated in §9.3 as a secondary diagnostic per outer fold on DEV.
The `mspc_baseline.csv` artifact contains fold-level MSPC results.

For the §13.4 claim to be executable, there must be a corresponding MSPC
lockbox evaluation: MSPC fitted on full DEV pass-wafers, then scored on the
lockbox. Without this, the claim comparison is between the supervised model
on the lockbox and MSPC on DEV folds — two different time periods with
documented different fail rates (~7.1% DEV, ~3.8% lockbox). Comparing a
supervised lockbox result to a DEV MSPC result is not a matched-condition
comparison.

**Required addition to §8 or §9.3:** "In addition to the per-fold evaluation,
fit MSPC on full DEV pass-wafers (analogous to the supervised model's full-DEV
training) and evaluate T² and Q on the lockbox. Report T2_TPR_at_TNR90 and
Q_TPR_at_TNR90 for this lockbox MSPC evaluation. Use this result for the §13.4
claim comparison against supervised lockbox results."

The `mspc_baseline.csv` or `final_lockbox_result.csv` should include a row or
separate columns for the lockbox MSPC evaluation. If added to mspc_baseline.csv,
use `fold_index = 'lockbox'` as a sentinel row.

---

### Stage B "Top 2-3 Finalists" Count Is Not Pinned

§5.1 note 7 says:

> "Stage B promotion rule: select top 2-3 distinct methods by Stage A mean BER
> (after de-dup); if de-dup reduces count below target, backfill with next-ranked
> distinct method."

"Top 2-3" is not a fixed pre-registered count. The decision of whether to promote
2 or 3 methods is left open. This means the count can be set after seeing Stage A
results, which is a post-hoc decision. A protocol that allows "between 2 and 3"
finalists without a decision rule provides one degree of freedom to choose
the finalist pool that produces the desired primary model.

The §5.2 Stage B workload note implies "top 2" as the base case (720 inner fits
for 2 selectors). The "2-3" wording in §5.1 is inconsistent with this.

**Required fix:** Pin the finalist count. Suggested rule: "Promote top 2 distinct
methods. Add a 3rd if the 3rd-ranked distinct method has Stage A mean BER within
0.02 of the 2nd-ranked (using the same near-equal margin defined in §6.2). If
Pearson de-dup reduces the pool below 2, backfill from the next-ranked distinct
method." This is fully pre-registerable and uses the ±0.02 margin already in §6.

---

## Minor Issues

### `fold_jaccard_stability` Column Is Ambiguous

`feature_report.csv` (§11.4) includes a column `fold_jaccard_stability`. This
appears alongside `selection_frequency` in the same per-feature row.

"Fold-set Jaccard stability" as defined in §9.3.10 is a property of the overall
selection set, not of individual features. The standard set Jaccard between two
fold selections S_i and S_j is |S_i ∩ S_j| / |S_i ∪ S_j|, averaged across all
fold pairs. This is a single scalar for the entire selector run, identical for
every row in the table.

If `fold_jaccard_stability` is meant to be the global set Jaccard reported
redundantly in every row, it wastes a column and could mislead. If it is meant to
be a per-feature metric (e.g., the proportion of fold pairs where feature j was
in both fold selections), it is a distinct metric from global Jaccard and should
be named accordingly.

**Required clarification in §9.3.10 and §11.4:** Specify whether `fold_jaccard_stability`
is (a) the global fold-pair Jaccard for the entire selection, reported as a constant
in every row, or (b) a per-feature metric such as "pairwise selection agreement
rate for feature j" defined as the fraction of fold pairs (i, j_pair) in which
feature j was selected in both. Option (b) is more informative and more natural
as a per-feature column.

---

### "Mean Weekly Flagged Fraction" Computation Not Pinned

§7.1 and §7.4 both use "mean weekly flagged fraction on DEV." This is a
non-trivial computation when weekly production rates are uneven:

- Interpretation A (unweighted): mean of per-week flagged fractions. If week 3
  has 50 wafers and week 10 has 150 wafers, both weeks get equal weight.
- Interpretation B (weighted / equivalent to overall): total predicted positives
  / total DEV samples. This is the overall flagged fraction.

For a hiring manager, "10% of wafers flagged per week" most naturally means
interpretation A (each week gets equal say). But these two interpretations can
give different operational threshold values when production is uneven.

**Required clarification in §7.4:** "Mean weekly flagged fraction is computed as
the unweighted mean of per-week flagged fractions: for each week, compute
(n_flagged_that_week / n_wafers_that_week), then average across all DEV weeks.
This interpretation treats each week equally regardless of production volume."

---

### Operational Threshold Uses In-Sample Predictions Without Explicit Statement

§7 says "Threshold is finalized only after model config freeze, using full DEV."
The frozen model is trained on full DEV and the Youden threshold is derived from
the model's predictions on that same DEV data (in-sample). The operational threshold
(10% flagged cap) is also derived from the model's in-sample DEV predictions.

Using in-sample predictions for threshold calibration is standard practice and
acceptable at this dataset size, but it means the threshold is derived from the
very data the model was trained on. A technical reviewer may raise this. The
document should acknowledge it:

**Suggested addition to §7:** "Both thresholds are derived from the frozen model's
in-sample predictions on full DEV (the same data used for training). This is
standard calibration practice. The lockbox serves as the out-of-sample validation
that the thresholds generalise; threshold leakage into the lockbox is prevented
by the no-post-lockbox-tuning rule."

---

## What Was Correctly Resolved From the Previous Round

All items from Codex's self-review are correctly addressed:

1. Stage A ReliefF n_neighbors=10 fixed ✓
2. Lane A imputer strategy pinned (median) + ablation add_indicator logic explicit ✓
3. Stage B finalist promotion clarified with backfill rule ✓
4. final_lockbox_result.csv column schema added ✓
5. best_MSPC_TPR_at_TNR90 defined as max(T2, Q) with source column ✓
6. predicted_flag_fraction source defined (Stage B outer-test mean) ✓
7. Capacity cap clarified as mean weekly + peak-week risk note ✓
8. Fallback path made deterministic (3-fold → 2-fold → Lane B infeasible) ✓
9. Benchmark CI method pinned as percentile bootstrap ✓

And all items from the third review round are correctly addressed:

1. KernelRidge label encoding specified ({-1, +1}) ✓
2. timeaware_selector_screening.csv and timeaware_model_selection.csv schemas added ✓
3. Benchmark claim selector locked to Replication-Strict F-test ✓
4. Feature cluster uses imputed features, excludes indicator columns ✓
5. Weekly production rate formula defined ✓
6. Limitations section minimum content pre-specified ✓
7. Missing timestamp handling added to §3 ✓

---

## Summary

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | No lockbox MSPC evaluation protocol — §13.4 claim comparison is across datasets without it | Protocol gap | Moderate |
| 2 | Stage B "top 2-3" finalist count not pinned to a decision rule | Protocol gap | Moderate |
| 3 | `fold_jaccard_stability` column: per-feature vs global metric undefined | Specification gap | Minor |
| 4 | "Mean weekly flagged fraction" computation not specified (unweighted vs weighted) | Specification gap | Minor |
| 5 | In-sample threshold derivation not acknowledged for technical defensibility | Transparency | Minor |

---

## Overall Assessment

The document is implementation-ready for nearly all components. Issues 3–5 are
refinements that will not prevent correct implementation but could cause
reviewer questions. Issues 1 and 2 require protocol decisions before coding
begins.

Issue 1 (lockbox MSPC) is the most consequential: without it, the final
report cannot make a clean supervised-vs-MSPC comparison on the held-out data.
All other MSPC vs supervised comparisons in the report would then be on DEV data
only, which technically makes the comparison part of the model-selection process
rather than a final evaluation.