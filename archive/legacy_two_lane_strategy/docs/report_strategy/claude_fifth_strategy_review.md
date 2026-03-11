# Claude Fifth Strategy Review

## Purpose

This document performs a top-to-bottom coherence audit of
`final_end_to_end_report_strategy.md` after all five round-4 issues were
resolved. The focus shifts from specification gaps to logical consistency:
does each section make sense in isolation, do sections coordinate correctly
with each other, and does every element serve the document's stated end goal?

All five round-4 issues are confirmed resolved before this review begins.

---

## Round 4 Resolutions — Confirmed

1. Lockbox MSPC companion evaluation added to §8.10; `mspc_baseline.csv` gains
   `eval_scope` column. ✓
2. Stage B finalist count pinned in §4.2 Stage B item 6 (top 2 + optional 3rd
   within 0.02 BER); §5.2 header updated. ✓
3. `fold_jaccard_stability` defined as per-feature pairwise selection agreement
   in §9.3.10. ✓
4. Mean weekly flagged fraction defined as unweighted per-week average in
   §7.8. ✓
5. In-sample threshold calibration acknowledged in §7.9. ✓

---

## Moderate Issues

### §8.5 Drift Gate and §13 Claim Policy Are Not Coordinated

§8.5 states:

> "if status is `HIGH_SHIFT`, do not make superiority claims from lockbox;
> report descriptive performance only."

§13 "Can claim" item 4 states:

> "Supervised model advantage over MSPC only if supervised `TPR_at_TNR90`
> exceeds MSPC at the same `TNR`."

§13 "Cannot claim" item 4 states:

> "Supervised superiority over MSPC when MSPC is equal or better at matched
> `TNR=90%`."

The §13 claim rules are framed purely in terms of metric comparison and say
nothing about drift gate status. An implementer following §13 alone could
legitimately read the lockbox MSPC and supervised TPR_at_TNR90 values, find
supervised > MSPC, and make the claim — even if §8.5 declared `HIGH_SHIFT`.
The §8.5 `HIGH_SHIFT` override is not visible from §13.

This creates a real execution gap: §8 and §13 give inconsistent instructions
for the same decision, and a reader who processes the document linearly will
encounter §13 claim rules without knowing they are conditionally suppressed
by §8.5.

**Required fix:** Add to §13 "Can claim" item 4 a qualifying condition:
"...only if supervised `TPR_at_TNR90` exceeds MSPC at the same `TNR` **and
the lockbox drift gate status is not `HIGH_SHIFT`** (see §8.5)." Also add a
corresponding note to §8.5: "Under `HIGH_SHIFT`, the supervised-vs-MSPC
comparison in §13.4 is descriptive only; do not apply the superiority-claim
rule."

---

### §8.5 PSI Spec: "Top-10 Selected Value Features" Is Undefined on Two Dimensions

§8.5 specifies:

> "PSI on frozen-primary top-10 selected value features (10 equal-frequency
> DEV bins per feature), then report `max_PSI` and `median_PSI`."

Two terms are undefined:

**"value features":** This term does not appear anywhere else in the document.
The implied meaning is "original sensor features (columns 0–589), not the
missing-indicator columns added by `add_indicator=True`." This is a reasonable
intent — computing PSI on binary 0/1 indicator columns is meaningless — but
the term needs to be defined. Suggested replacement: "original feature values
(excluding missing-indicator columns)."

**"top-10" by what ranking criterion:** If the frozen primary was selected
with k=40, there are 40 selected features. The document needs to specify which
10 to use for PSI. Options include top-10 by selection frequency, by
`expected_contribution`, or by `conditional_effect_magnitude` — three
different rankings that `feature_report.csv` contains. These can produce
different PSI values because the feature distributions vary. If k=10, "top-10"
equals all selected features (no ambiguity). If k=20 or k=40, the choice
matters.

Additionally, the PSI threshold for `HIGH_SHIFT` uses `max_PSI < 0.30`. The
industry standard for "significant shift" is typically `PSI > 0.20`. The
document uses a more lenient cutoff (0.30). This is a design choice but should
be stated as intentional: it is calibrated for small-N semiconductor data
where PSI can be elevated by random variation.

**Required fix:** Replace "top-10 selected value features" with:
"top-10 original-value selected features (excluding missing-indicator columns)
by `expected_contribution` rank from `feature_report.csv`, computed on the
frozen primary model's selected set."

---

### §7 Weekly Flagged Fraction Does Not Cross-Reference the Week Bins From §4.2

§7.8 defines:

> "for each DEV week `w`: `frac_w = flagged_w / wafers_w`"

"DEV week w" is not anchored to any binning rule. The operational threshold is
chosen so that `mean_w(frac_w) <= 10%`. The selected threshold value directly
depends on how DEV weeks are defined.

§4.2 defines the week bins as:

> "`week_idx = floor((timestamp - t_min_dev) / 7 days)`"

This is dataset-anchored, not calendar-week. If an implementer uses ISO
calendar weeks for the §7 computation and dataset-anchored bins for fold
construction, the boundaries are different and the operational threshold value
will differ from what fold-construction assumptions imply.

**Required fix:** Add to §7.8: "DEV weeks use the same dataset-anchored 7-day
bins as §4.2: `week_idx = floor((timestamp - t_min_dev) / 7 days)`. Each
week label in §7 maps to the corresponding `week_idx` bin."

---

## Minor Issues

### `empirical_ARL0` Is Computed but Never Used

`mspc_baseline.csv` (§11.2) requires the column `empirical_ARL0`. Average
Run Length to false alarm (ARL0) is a standard MSPC diagnostic measuring how
many consecutive pass-wafers are processed before a false alarm fires.

Nowhere in the document is `empirical_ARL0` consumed:

- §10 (manager outputs) does not include it.
- §12.7 (MSPC comparison section) has no pre-specified content, so it may or
  may not appear.
- §13 (claim policy) does not reference it.

This means `empirical_ARL0` is computed, stored, and then not connected to
any report element or claim. Two options:

- (a) Remove it from `mspc_baseline.csv` if it is not used.
- (b) Add a sentence to §12.7 or §10 specifying where it appears: "Report
  empirical ARL0 per fold as an MSPC operational diagnostic — the average
  number of consecutive pass-wafer test samples between false alarms at the
  UCL threshold."

Option (b) is preferred if the MSPC section will appear in the hiring-manager
report, since ARL0 is the practitioner metric semiconductor engineers
understand (vs. TPR/TNR, which is ML framing).

Note: if ARL0 is retained, §9.3.6 should specify whether it is computed on
outer-train pass wafers (in-sample, biased low) or outer-test pass wafers
(out-of-sample, preferred). Currently §9.3.6 only specifies fitting and scoring
for T²/Q; ARL0 computation scope is not stated.

---

### `mspc_baseline.csv` Lockbox Row: `fold_index` Sentinel and Column Scope Undefined

§11.2 lists `fold_index` as the first column of `mspc_baseline.csv`. For
outer fold rows, `fold_index ∈ {1, 2, 3}` (or `{A, B}` for the 2-fold
fallback). The lockbox row added by §8.10 will need a `fold_index` value.
No sentinel is specified.

Additionally, `mspc_baseline.csv` has columns `T2_AUC`, `Q_AUC`,
`alarm_rate`, and `empirical_ARL0` which are meaningful for DEV fold
evaluations but have different semantics (and potentially lower reliability)
for the lockbox window (~236 samples). The document does not specify which
columns are required vs optional for the lockbox row.

**Required minimum additions:**

1. State the `fold_index` sentinel for the lockbox row — suggested:
   `fold_index = 'lockbox'`.
2. Specify which columns are required for the lockbox row. At minimum:
   `T2_TPR_at_TNR90`, `Q_TPR_at_TNR90`, `best_MSPC_TPR_at_TNR90`,
   `best_MSPC_source`, and `eval_scope = 'lockbox'`. Columns like
   `empirical_ARL0` are optional for the lockbox row and should be marked
   `NaN` if not computed.

---

## What Is Coherent and Solid

The following were verified to be internally consistent and purposeful:

**Section-to-end-goal mapping:** All three §1 end goals are fully served.
Goal 1 (replication) → Lane A, §4.1, artifacts 1–3, §12.3.
Goal 2 (deployment-realistic model) → Lane B, §4.2–§6, artifacts 4–6, §12.4–5.
Goal 3 (operational impact) → §7, §10, §12.6, artifact 9.

**Pipeline logic:** The Stage A → Stage B → freeze → threshold → lockbox
sequence is coherent. Each step depends only on outputs of prior steps. No
circular dependencies.

**Overlapping test windows now correctly framed:** §4.2 explains overlapping
windows as intentional ("deployment realism and recency"). §6 note 5 pre-
registers that fold summaries are "time-backtest evidence, not independent
i.i.d. CV estimates." §9.4 repeats this for metrics reporting. These three
statements are consistent and complete the interpretive chain. ✓

**Finalist promotion rule consistency:** §4.2 Stage B item 6 defines the
promotion logic (top 2, + optional 3rd within 0.02). §5.1 note 7 covers the
backfill edge case. §5.2 header says "top 2 selectors + optional 3rd within
0.02 BER." All three are consistent and non-overlapping. ✓

**0.02 near-equal margin used in two places:** The same 0.02 margin governs
both the Stage B finalist promotion (§4.2 item 6) and the primary/challenger
selection tie-break (§6.2). This is intentional and coherent — the same
judgment of "too close to call" applies in both contexts. No inconsistency. ✓

**MSPC lockbox companion creates a proper matched comparison:** §8.10 fits
MSPC on full DEV pass wafers and scores lockbox. The supervised model (§8.1)
is also trained on full DEV and scored on lockbox. Same training data scope,
same evaluation period. The §13.4 supervised-vs-MSPC claim is now executable
as a matched comparison. ✓

**`best_MSPC_TPR_at_TNR90` definition consistent:** §11.10 defines it as
`max(T2_TPR_at_TNR90, Q_TPR_at_TNR90)`. §8.10 uses the same definition.
This is optimistic for MSPC (gives it its best statistic), making the
supervised-advantage claim harder to make — which is correctly conservative. ✓

**Drift gate thresholds:** All three criteria (prevalence shift < 0.02,
KS p ≥ 0.01, max_PSI < 0.30) are quantitative and executable as written.
The three-level gate (PASS / CAUTION / HIGH_SHIFT) has a pre-registered
consequence for HIGH_SHIFT. ✓

**Lane A and Lane B use compatible metric definitions:** Both use BER, True+,
True-. KernelRidge in Lane A outputs a binary decision (threshold at 0.0 on
{-1,+1} labels), and LogisticRegression in Lane B outputs a continuous score
thresholded at Youden. BER is defined the same way for both. ✓

**Benchmark comparison is locked and narrowly defined:** §13.3 specifies
Replication-Strict F-test BER as the matched selector, which is correct
(same selector as McCann & Johnston 2008). CI method is pinned
(percentile bootstrap, n_boot=1000). Claim condition is strict (CI upper
bound < 0.335). This is as conservative as it should be. ✓

---

## Summary

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | §8.5 HIGH_SHIFT drift status not referenced in §13.4 claim policy — sections give conflicting guidance | Logic gap | Moderate |
| 2 | §8.5 PSI: "top-10 selected value features" — ranking criterion and term "value features" undefined | Specification gap | Moderate |
| 3 | §7.8 "DEV week w" does not cross-reference §4.2 week binning | Cross-reference gap | Moderate |
| 4 | `empirical_ARL0` computed and stored but not connected to any report output or claim | Orphaned metric | Minor |
| 5 | `mspc_baseline.csv` lockbox row: `fold_index` sentinel undefined; required columns not specified | Specification gap | Minor |

---

## Overall Assessment

The document is coherent. The two-lane architecture, the Stage A → B pipeline,
the lexicographic selection rule, the threshold policy, and the lockbox
protocol all form a logically sound and self-consistent system. Every major
section can be traced to the §1 end goal.

The three moderate issues above are the only logical gaps that would cause
confusion during implementation. Issue 1 (drift gate / claim policy
coordination) is the most consequential: without it, an implementer could
make a supervised-vs-MSPC superiority claim on a HIGH_SHIFT lockbox, which
§8.5 prohibits. Issues 2 and 3 affect determinism of the PSI computation
and operational threshold computation respectively. All three are one-line
or two-line fixes.

The document is otherwise ready for implementation.
