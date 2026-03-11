# Claude Final Strategy Review

## Purpose

This document reviews `final_end_to_end_report_strategy.md` after all prior
challenge rounds have concluded. It does not re-litigate settled decisions
(inner CV, Stage A structure, two thresholds, BER-primary, lockbox hygiene).
It raises new issues that were not addressed in prior rounds and that would
surface during implementation.

Issues are grouped by severity: blockers (wrong results without resolution),
clarity gaps (ambiguous enough to produce different implementations), and
narrative gaps (results that have no pre-registered interpretation).

---

## Executive Assessment

The canonical document is functionally strong. The two-lane structure, nested
CV design, threshold policy, and claim discipline are all correct. What remains
is not structural — it is specification completeness. Three issues would cause
incorrect results if not resolved before running. Six would produce differing
implementations depending on the implementer. Two scenarios have no
pre-registered narrative, which means the report framing will be determined
post-hoc.

---

## Blockers

### 1. MSPC Training Scope Per Fold Is Not Stated

`§9.3` lists MSPC as a secondary diagnostic and `§11` specifies `mspc_baseline.csv`
with a `fold_index` column. The presence of `fold_index` implies MSPC is evaluated
per outer fold. But the canonical document never states that MSPC must be **fit
separately on each fold's outer-train pass-wafers**.

`improvement_plan.md §5.5` does say "Train PCA on autoscaled pass-wafers from the
outer training fold" — but that document is not the execution standard. The canonical
strategy document is. If an implementer follows the canonical doc alone and fits MSPC
once on full DEV, they introduce leakage: the PCA mean, variance, loadings, and UCLs
are computed using test observations that MSPC then evaluates.

This is not a minor note. MSPC trained on full DEV and evaluated on a test fold is
fundamentally different from MSPC trained on the training fold only. The T² and Q
scores will be lower for the test set (the PCA space has already absorbed test-set
variance), and UCLs will be tighter. Both effects flatter MSPC's apparent performance.

**Required addition to §9.3 or §11:** "MSPC is fit on pass-wafers within each outer
training window only. The autoscaler mean/variance, PCA loadings, component count,
and UCLs are computed on training pass-wafers for that fold. These fitted objects
are applied to the corresponding outer test set. This training discipline matches
the supervised model and ensures the comparison is valid."

---

### 2. Weekly Bin Construction Is Undefined

`§4.2` says "Build weekly bins on DEV from sorted timestamps." It does not define
what constitutes a week.

Two reasonable interpretations produce different fold assignments:

1. **Rolling 7-day bins from dataset start**: Week 1 = samples within the first 7
   days, Week 2 = days 8–14, etc. Week number = floor((day - day_0) / 7).

2. **ISO calendar weeks**: Week 1 = first ISO calendar week that contains the
   dataset start, etc. Calendar weeks can start on Monday and span month
   boundaries.

For the SECOM data (Jul 19 – ~Oct 17), the dataset spans approximately 14 weeks.
Rolling bins from day 0 and ISO calendar weeks will not produce the same week
assignments at the boundaries. This directly affects which samples land in which
fold, which affects fail counts per fold, and potentially whether the fallback
split is triggered.

The fold design is precise (train weeks 1–5, test weeks 6+, etc.) and the minimum
fail count is a hard gate. A 1-day shift in bin boundaries could move a test fold
from 21 fails to 19 fails and trigger the fallback when the primary should have
been used.

**Required addition to §4.2:** Specify the bin construction rule. The simplest
correct definition: "Week index = floor((t - t_min).days / 7) using the minimum
timestamp in DEV as the reference. This produces 0-based week indices (week 0,
week 1, ...) aligned to the dataset start, not calendar boundaries."

---

### 3. Brier Baseline Prevalence Uses Wrong Scope

`§9.3 bullet 8` specifies:

> "report both all-pass baseline (p) and prevalence-constant baseline (p*(1-p))
> using the evaluation-slice prevalence p (full-dataset values are roughly
> 0.0664 and 0.062)"

The phrase "using the evaluation-slice prevalence" is correct. The hardcoded
parenthetical "full-dataset values are roughly 0.0664 and 0.062" is wrong for
anything other than full-dataset evaluation.

The DEV fail rate is approximately 7.1% (not 6.64%). The lockbox fail rate is
approximately 3.8%. Each outer fold test window has its own fail rate — the later
folds may have lower fail rates if the drift documented in the strategy is real.

If the full-dataset value 0.0664 is used as the Brier all-pass baseline for the
lockbox slice (which has ~3.8% fails), the baseline is inflated relative to the
actual lockbox prevalence. The supervised model's Brier score is evaluated against
a stiffer baseline than it should face. This makes the model look better relative
to the baseline than it actually is.

**Required fix:** Remove the hardcoded parenthetical values or qualify them
explicitly as the full-dataset reference only. State: "compute p from the
evaluation slice itself for each Brier baseline report." For per-fold Lane B
diagnostics, use each fold's test-set prevalence. For the lockbox report, use
the lockbox prevalence.

---

## Clarity Gaps

### 4. Lane B 95% CI Is Not Computable From 3 Outer Folds

`§9.2` specifies "Mean, std, and 95% CI where applicable." For Lane B, the outer
metric distribution has 3 observations (one per fold). A t-distribution 95% CI
with df=2 has a critical value of approximately 4.30. The resulting intervals are
not informative — they will typically span most of the [0, 0.5] BER range.

This is not a reason to avoid reporting uncertainty, but it requires an honest
statement of what the numbers mean. The document should specify:

1. For Lane B metrics: report fold-by-fold values plus mean ± std. Do not label
   this a "95% CI."
2. If a CI is required: use percentile bootstrap treating the 3 fold values as
   observations (n_boot=1000, seed=42). Note that this CI is not asymptotically
   valid with 3 observations and should be interpreted as a spread measure, not
   a frequentist coverage guarantee.
3. For comparison to the 2008 benchmark: the CI method is already specified in
   `§13.2` (bootstrap on Lane A fold BERs). The Lane A CI from 10 folds is
   meaningful. Treat Lane B fold spread as qualitative temporal stability evidence,
   not a formal CI.

---

### 5. Challenger Selection Rule Has No BER Floor

`§6.4` says the challenger is selected with "false-alarm control preference (higher
True-)." This implies: among all non-primary configs, choose the one with highest
True-. But this is not stated explicitly, and it raises a gap:

What if the config with the highest True- has a BER of 40%? Is that a valid
challenger for a report that demonstrates improvement over the 33.5% benchmark?
A challenger with BER well above the benchmark undermines the operational framing.

The document should pre-register a challenger eligibility rule. Suggested
addition to §6: "Challenger eligibility: mean outer BER within 0.10 of the
dataset-wide fail-rate complement (i.e., BER ≤ 0.40 given ~6.64% fails, which
represents above-chance performance). Among eligible configs, select the
non-primary config with the highest mean outer True-."

Without this floor, the worst-case outcome is reporting a challenger that catches
fewer fails than a naive predict-all-flag baseline, which is not defensible.

---

### 6. TNR=90% Operating Point Has No Interpolation Rule

`§7` and `§10.6` both reference TNR=90% as the matched comparison point for MSPC.
The supervised model produces a continuous score; the threshold grid is discrete.
No threshold will produce exactly TNR=90%.

Without a rule, two valid implementations exist:
- Nearest TNR to 90%
- Highest threshold where TNR ≥ 90% (i.e., at least 90% specificity, conservative)
- Interpolation between adjacent thresholds

These give different TPR values, which affects whether the supervised model
"beats" MSPC at matched conditions per `§13.4`.

**Required addition to §7 or §10:** "The TNR=90% operating point is defined as
the highest threshold for which TNR ≥ 0.90 (conservative, ensures specificity
is not below target). If multiple thresholds give identical TNR, use the one with
higher TPR."

---

### 7. feature_report.csv Has No Artifact Column Specification

`§11` specifies minimum columns for three artifacts:
- `splitwise_timeaware_results.csv` — 14 columns defined
- `mspc_baseline.csv` — 5 columns defined
- `operational_cost_curves.csv` — cost ratio + model config columns defined

Artifact 10, `feature_report.csv`, has no column specification. Its content is
described in `§9.3 bullet 9` with five required items. The artifact section
should add the minimum column requirements matching `§9.3.9`:

```
feature_index, feature_name (or original column index), selection_frequency,
conditional_effect_magnitude, expected_contribution, fold_jaccard_stability,
cluster_id
```

Without this, implementations will produce heterogeneous feature reports that
cannot be compared if the analysis is re-run.

---

### 8. run_manifest.json Config Hash Algorithm Is Undefined

`§11 bullet 11` lists "final frozen config hash" as a required field in
`run_manifest.json`. No hash algorithm or serialization format is specified.

This matters for auditability — the hash exists precisely to enable future
verification that no post-freeze changes occurred. A hash of an unstable string
representation (e.g., Python's default dict ordering pre-3.7, float precision
differences) would fail to reproduce.

**Required addition to §11:** "Config hash: SHA-256 of the UTF-8-encoded
JSON serialization of the frozen config dict with keys sorted alphabetically
and float values rounded to 6 significant figures. Include selector name,
k, C, scaler, missing_mode, threshold_policy in the hashed dict."

---

### 9. Pearson ≡ F-test Is Always True, Not Conditional

`§5.1 notes 2–3` say:

> "If Pearson and F-test are empirically equivalent, collapse them."
> "If Pearson and F-test are equivalent in Stage A, promote only one into Stage B."

The word "if" implies this equivalence might not hold. It will always hold. For
binary targets and feature-ranking purposes, Pearson |r|² is a monotone
transformation of the ANOVA F-statistic. The two selectors rank all features in
identical order for any dataset. Jaccard similarity between their selected sets
is 1.0 by construction.

The "if" language could cause an implementer to check empirical equivalence,
find slight numerical differences due to floating-point arithmetic, and conclude
the selectors are not equivalent — then keep both in Stage B, doubling the grid
without adding information.

**Correction:** Replace "If Pearson and F-test are empirically equivalent" with
"Pearson |r| ranking and F-test ranking are mathematically equivalent for binary
targets (r² is monotone with F). They will always produce identical selected
feature sets. In Stage A, run both for record completeness; collapse to one row
in reporting. Promote only F-test to Stage B; Pearson is excluded from the Stage B
grid."

---

## Narrative Gaps

### 10. Lane B BER > Lane A BER Has No Pre-Registered Interpretation

The document presents Lane A and Lane B as separate lanes but does not address the
expected comparison outcome in the report.

Lane B (time-aware outer folds, smaller effective training sets due to anchored
expanding windows) will typically produce higher BER than Lane A (10-fold
stratified random CV, more training data per fold, optimistic validation). This
is the whole point of using time-aware validation — it reveals the optimism in
random CV.

If the report shows Lane B BER = 35% and Lane A BER = 31%, a reader without
pre-registered framing might interpret this as "the deployment-realistic model is
worse." Without a pre-registered statement, the report author will explain this
post-hoc, which a technical reviewer will correctly identify as motivated
interpretation.

**Required pre-registration addition to §4 or §13:** "If Lane B mean BER exceeds
Lane A mean BER, this is expected. Random CV overestimates performance by allowing
test samples from future periods to inform training. The Lane B estimate is the
deployment-realistic one. The Lane A result is reported for benchmark comparability,
not as a target to match."

---

### 11. MSPC Competitive or Superior at TNR=90% Has No Pre-Registered Narrative

`§13.4` says "Supervised model advantage over MSPC only if supervised TPR_at_TNR90
exceeds MSPC at the same TNR." The claim policy correctly blocks overclaiming. But
it does not pre-register what the report says when MSPC is equal or better.

If MSPC wins, an implementer might frame this as a failure of the project.
It is not. A finding that a properly implemented control chart matches a labeled
supervised model is a substantive and credible result for a semiconductor hiring
manager — it answers the practitioner's actual question ("does adding labels help
beyond what I already run?").

**Required pre-registration addition to §13:** "If MSPC TPR_at_TNR90 is equal to
or higher than the supervised model, the pre-registered conclusion is: labeled
outcome data does not provide discrimination advantage over multivariate statistical
process control at this operating condition and time window. The MSPC approach
is the primary recommendation; the supervised model is retained as a
complementary view. This finding should be reported honestly and is not a
negative result — it is a quantitative answer to a question practitioners care
about."

---

## Minor Observations

These do not require changes before execution but should be noted.

### 12. MSPC Spec Cross-Reference Is Fragile

`§9.3 bullet 6` references `docs/improvement_plan.md (Section 5.5)` for the
MSPC implementation spec. The canonical strategy document is the execution
standard, but the formula authority lives in a separate document that could be
edited independently.

For auditability, the canonical document should either reproduce the key UCL
formulas inline or note them as frozen: "UCL formulas as specified in
improvement_plan.md §5.5 dated [version]. The relevant formulas are:
T² UCL (Tracy-Young-Mason), Q UCL (Jackson-Mudholkar). Implementers should
verify formula versions before running."

---

### 13. Feature Cluster Correlation Matrix Scope Is Unspecified

`§10.7` defines feature clusters as connected components of the correlation graph
with edge rule `|corr| >= 0.95` but does not specify which data the correlation
matrix is computed from.

Options: raw DEV data, full DEV training set of the final model, or full dataset
including lockbox. Since feature clustering is for report interpretation (not model
selection), using full DEV after config freeze is appropriate and does not introduce
leakage. But this should be stated: "Correlation matrix for cluster definition is
computed on the full DEV feature set after imputation, using the fitted imputer
from the final frozen model. Lockbox samples are excluded."

---

### 14. Lockbox Better-Than-DEV Case Has No Guidance

`§8.6` addresses "if lockbox degrades vs DEV means." It does not address the
inverse: lockbox BER lower (better) than DEV mean.

This is possible and would likely indicate the lockbox period was a lower-stress
process window (consistent with the documented fail-rate drop from ~7.1% to ~3.8%).
A better lockbox result should not be treated as confirmation of generalization —
it may simply reflect that the lockbox is easier. The drift diagnostic already
covers this, but `§8` should note: "If lockbox BER is below DEV mean BER,
attribute this first to the documented prevalence drop before concluding the
model generalizes well. Report the lockbox result at its actual prevalence and
do not use it to revise upward the DEV-derived performance estimates."

---

## Summary Table

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | MSPC per-fold training not stated in canonical doc | Blocker | High |
| 2 | Weekly bin construction undefined (rolling vs calendar) | Blocker | High |
| 3 | Brier baseline prevalence hardcoded to full-dataset value | Blocker | Medium |
| 4 | Lane B 95% CI uncomputable from 3 folds | Clarity gap | Medium |
| 5 | Challenger BER floor not specified | Clarity gap | Medium |
| 6 | TNR=90% interpolation rule missing | Clarity gap | Medium |
| 7 | feature_report.csv has no column spec | Clarity gap | Low |
| 8 | run_manifest.json hash algorithm undefined | Clarity gap | Low |
| 9 | Pearson ≡ F-test is always true, not conditional | Clarity gap | Low |
| 10 | Lane B > Lane A BER — no pre-registered narrative | Narrative gap | Medium |
| 11 | MSPC wins — no pre-registered narrative | Narrative gap | Medium |
| 12 | MSPC spec cross-reference fragility | Minor | Low |
| 13 | Feature cluster correlation matrix scope | Minor | Low |
| 14 | Lockbox better-than-DEV — no guidance | Minor | Low |

---

## What Is Solid

These do not need to change:

1. Two-lane structure (Lane A replication, Lane B deployment-realistic) is
   elegant and defensible.
2. Nested CV design (Stage A fixed-settings screen, Stage B nested tuning,
   stratified inner, time-aware outer) is correct and well-specified.
3. Threshold policy (Youden + operational cap, no lockbox tuning) is rigorous.
4. Lexicographic selection rule with pre-declared business preference is clean
   and audit-proof.
5. Lockbox protocol (once only, both configs, both thresholds) is correct.
6. Claim policy (CI-gated benchmark claims, MSPC branching, scope limits) is
   appropriately conservative.
7. Two-threshold reporting (scientific + operational) gives the report practical
   utility beyond pure ML metrics.
8. Missing indicator ablation (Strict vs +MI) is a well-designed and
   underappreciated contribution that the hiring manager audience will value.
