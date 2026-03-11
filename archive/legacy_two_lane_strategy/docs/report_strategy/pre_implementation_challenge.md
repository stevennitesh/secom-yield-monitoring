# Pre-Implementation Challenge: final_end_to_end_report_strategy.md

*This is the final challenge before implementation begins. Every section is reviewed
as an implementer would read it — looking for ambiguities, missing specifications,
structural inconsistencies, and decisions that will be made ad hoc at coding time
because the document does not resolve them.*

*Issues are classified: [BLOCKER] will cause incorrect implementation without
resolution. [AMBIGUOUS] will cause inconsistent implementation across runs or
implementers. [MISSING] an important detail not present. [MINOR] worth noting but
workable.*

---

## Section 2 — Non-Negotiable Validation Rules

**[AMBIGUOUS] "Config freeze" is not defined (Rule 2)**

Rule 2 says "lockbox touched once after full config freeze." But the document never
defines when exactly config freeze occurs. The implied answer is "after Stage B
completes and the selection rule in §6 is applied" — but this is not stated. An
implementer who re-runs Stage B with different grid values after seeing preliminary
outer fold results has not violated the letter of Rule 2, because "freeze" was never
formally defined. Should read: "Config freeze occurs immediately after the §6
selection rule is applied to Stage B outer fold results. No grid changes, method
additions, or parameter adjustments after that point."

---

## Section 4.1 — Lane A Replication

**[BLOCKER] Classifier is not specified**

Lane A says "keep everything else identical (splits, seeds, classifier family, scaling,
k, thresholds)" but never states what the classifier family is. The 2008 paper used
kernel ridge regression. The project uses logistic regression with balanced class
weights. These are different classifiers with different inductive biases. An implementer
who reads only this document will not know which to use. The Lane A description must
explicitly state: "Classifier: `LogisticRegression(class_weight='balanced',
solver='lbfgs', max_iter=3000)`. This differs from the 2008 kernel ridge classifier —
this deviation must be disclosed in the replication section of the final report."

**[BLOCKER] Scaler for Lane A is not specified**

Section 5.2 specifies `RobustScaler` for Stage A (Lane B) and `{StandardScaler,
RobustScaler}` for Stage B. Lane A has no scaler specification at all. The 2008 kernel
ridge classifier is not scale-invariant when used with linear features, but the
original paper's scaling choice is unknown. Two options: (a) use `StandardScaler` in
Lane A to be closer to the most common 2008-era choice, (b) use `RobustScaler`
consistent with the rest of the project. Either choice is defensible, but it must be
stated, and any deviation from the 2008 protocol must be disclosed. Without this
specification, two implementers will make different choices.

**[BLOCKER] Lane A threshold policy is not specified**

Lane A ablation says "keep everything else identical (splits, seeds, classifier family,
scaling, k, thresholds)" but the threshold policy is not stated. The 2008 paper most
likely used the default 0.5 threshold or the classifier's native decision boundary —
it did not implement `best_threshold_by_ber`. If Lane A uses Youden's J threshold
optimisation, the BER will be lower than the 2008 benchmark through a mechanism the
2008 paper did not use. If Lane A uses fixed 0.5, the comparison is closer to the
original protocol. This must be specified explicitly. The Replication-Strict variant
should use the same threshold policy as the 2008 paper (most likely fixed 0.5 or
no threshold tuning). The Replication+MI variant can use either policy, but must be
consistent with Strict for a valid paired comparison.

**[AMBIGUOUS] Random seed for Lane A 10-fold CV is not specified**

10-fold random stratified CV produces different fold assignments under different random
seeds. The document says "keep everything else identical (splits, seeds, ...)" but
never specifies what seed to use. Without a fixed seed, Lane A results are not
reproducible across machines, runs, or implementers. Specify: `random_state=42` (or
any fixed value) for `StratifiedKFold` in Lane A.

**[AMBIGUOUS] CI computation method for Lane A ablation is not specified**

"Report paired deltas and CI for +MI vs strict." — CI computed how? Options:
bootstrap on fold-level BER differences, paired t-test on 10 fold-level BER values,
Wilcoxon signed-rank test. Each gives a different CI. Paired t-test on 10 values has
low power and assumes normality. Bootstrap (n=1000 resamples of fold-level deltas)
is more appropriate. Must be specified before implementation.

---

## Section 4.2 — Lane B Stage Structure

**[BLOCKER] Outer fold structure is not specified**

The document says "test windows large enough for minority-class stability" but never
specifies: (a) the number of outer folds, (b) the fold boundary rule, or (c) the
minimum fail count per outer test window. This is the most critical unresolved
implementation detail in the document. Every metric in Stage A and Stage B depends
on this.

Without specification, an implementer will choose fold boundaries arbitrarily. Two
valid implementations could produce entirely different outer BER estimates and reach
different method selection conclusions — without either being wrong per the document.

Required specification:
```
Outer folds: 3 anchored expanding windows
Fold 1: train weeks 1–5,  test weeks 6–12  (target ~35-40 fails in test)
Fold 2: train weeks 1–7,  test weeks 8–12  (target ~25-30 fails in test)
Fold 3: train weeks 1–9,  test weeks 10–12 (target ~20-22 fails in test)
Minimum acceptable fails per test window: 20
```
The exact week boundaries must be computed from the actual timestamp distribution,
but the fold count (3) and the expanding-window structure must be fixed in the
document.

**[BLOCKER] Inner tuning description precedes Stage A/B distinction — creates scope ambiguity**

The document structure in §4.2 is:

```
Outer validation: [applies to both stages]
Inner tuning:
  1. Stratified 5-fold CV
  2. ROC-AUC scoring
Stage structure:
  1. Stage A — no inner CV
  2. Stage B — full nested CV
```

An implementer reading linearly encounters "stratified 5-fold CV" before reaching "Stage
A has no inner CV." The inner tuning description must be scoped to Stage B explicitly,
or moved to the Stage B subsection. Current layout risks a Stage A implementation that
applies inner CV because the inner tuning description appears before the Stage A
clarification.

Recommended restructure:
```
Stage A:
  - Outer time-aware evaluation only
  - Fixed settings (see §5.2)
  - No inner CV

Stage B:
  - Outer: anchored expanding-window folds (same structure as Stage A)
  - Inner: stratified 5-fold CV, scoring = roc_auc
  - Grid: see §5.2
```

---

## Section 5.1 — Selectors in Scope

**[BLOCKER] L1/Elastic Net "optional" creates a post-hoc inclusion decision**

"Optional embedded challenger: L1 or Elastic Net." Optional under what conditions?
Decided by whom? When? The notebooks showed L1 underperformed substantially (BER
~0.418 vs F-test ~0.314 under random CV). Including it "optionally" creates a decision
point that can become a leakage vector: if Stage B results are disappointing, add
L1/EN; if good, don't bother. This is post-hoc model shopping.

Decision required now, before seeing any new results:
- Include: commit to running it through Stage A alongside the other methods. It will
  likely be eliminated. The audit trail shows it was tested and found inferior.
- Exclude: state explicitly "L1 and Elastic Net were tested in exploratory notebooks
  and found substantially inferior to filter methods under random CV. They are excluded
  from formal evaluation."

Either is defensible. "Optional" is not.

**[AMBIGUOUS] Stage A false-elimination risk not disclosed**

Stage A evaluates all methods at C=1.0 fixed. ReliefF and Gram-Schmidt at C=1.0
may underperform their best operating point. A method eliminated at C=1.0 in Stage A
could have survived at C=0.1 in Stage B. This is an inherent two-stage screening
limitation and should be explicitly disclosed: "Stage A screening at fixed C=1.0 may
false-eliminate methods whose optimal regularization differs from C=1.0. This risk is
accepted in exchange for auditability and reduced runtime."

---

## Section 5.2 — Hyperparameters

**[BLOCKER] ReliefF neighbor count sweep values not specified**

"ReliefF neighbor count sweep" — what values? `n_neighbors ∈ {10, 20, 50}`?
`{5, 10, 20}`? Without specific values, ReliefF tuning is implemented with an
arbitrary range. Must specify.

**[MINOR] Stage B grid size should be stated for implementer awareness**

With 2 promoted methods × 3 k × 4 C × 2 scalers = 48 configs per inner fold, 5 inner
folds, 3 outer folds: 720 total inner fits for Stage B. Manageable, but worth stating
so the implementer can estimate runtime before starting.

---

## Section 6 — Selection Logic

**[BLOCKER] Business preference not pre-declared in the document**

Note 3 says "business preference must be declared before Stage B execution" but does not
declare it. This is the document where the pre-declaration must live. Leaving it to
be decided at execution time allows the business preference to be chosen after seeing
partial Stage B results, which makes it a post-hoc selection criterion.

The project's stated purpose is process/yield engineering, where missing a fail is more
costly than a false alarm. The pre-declared preference should be stated here:

**Pre-declared business preference: fail-catch priority (higher True+).**

This means: if two configs are within ±0.02 BER of each other, the one with higher
mean outer True+ is preferred. This is recorded now, before any Stage B results are
seen.

**[AMBIGUOUS] ±0.02 BER threshold is within the noise floor for most fold comparisons**

With outer test windows of 20–40 fails, the per-fold BER standard error is
approximately 0.05–0.08. A mean BER difference of 0.02 across 3 folds is smaller than
the standard error on any individual fold. The ±0.02 threshold is conservative —
meaning the tie-break will rarely trigger because real differences between competitive
methods (F-test vs Welch-t) will typically be either below noise or above 0.02 in
absolute value, not precisely at 0.02. This is not wrong but should be acknowledged:
"The ±0.02 threshold is conservative relative to fold-level noise. It will rarely
trigger; in practice the primary BER rule will almost always determine the winner."

---

## Section 7 — Threshold Policy

**[AMBIGUOUS] Operational threshold derivation at what cost ratio?**

"Operational operating point: threshold selected from review-capacity/cost curve."
The cost curve plots expected cost vs cost ratio r ∈ [1, 20]. The operational threshold
is the threshold that minimises expected cost at a specific r. But what r?

Two valid approaches:
1. Pre-specify a representative r (e.g., r=10, meaning a missed fail costs 10× a false
   alarm, which is reasonable for semiconductor backend-of-line processes). Report the
   operational threshold at r=10 and show the cost curve so engineers can read off
   their specific r.
2. Report the cost curve only and let the engineer choose their threshold from it.

Either is valid but must be specified. Without this, "operational threshold" is not
a deterministic output — it requires a human decision at report time.

**[MISSING] Lockbox degradation interpretation guidance**

If lockbox BER is substantially higher than mean outer BER, how should this be
interpreted? Given the DEV/lockbox fail rate shift (7.1% → 3.8%), some degradation
is expected. The report must contextualise this: "The lockbox period (Oct 5–17) has
a lower fail rate (3.8%) than the DEV period (7.1%). Performance degradation on the
lockbox may reflect temporal drift in the process, not model failure. Compare lockbox
results against Outer Fold 3 (the fold with the most similar temporal position to the
lockbox) rather than the overall DEV mean."

This guidance should be in the document so it's not crafted after seeing the lockbox
results.

---

## Section 8 — Lockbox Protocol

**[AMBIGUOUS] Both thresholds applied to lockbox, but not stated explicitly**

§7 establishes two thresholds (scientific and operational). §8 says "apply frozen
thresholds" (plural). It's implied that both are applied and both lockbox results are
reported. But which row of `final_lockbox_result.csv` corresponds to which threshold?
The artifact structure should state: one row per (config, threshold_policy) combination.
Primary + scientific threshold, primary + operational threshold, challenger + scientific
threshold, challenger + operational threshold = 4 rows minimum.

---

## Section 9 — Metrics Policy

**[BLOCKER] Metrics policy does not distinguish Lane A from Lane B**

Section 9 lists metrics without specifying which apply to Lane A (replication) and
which to Lane B (time-aware selection). Lane A should only report BER, True+, True-
(to match the 2008 protocol). Reporting PR-AUC or MCC for Lane A replication and
comparing them to the 2008 paper (which only reported BER/True+/True-) would be
misleading. An implementer who applies the full §9 metric suite to Lane A will
produce numbers that have no 2008 comparison point and confuse the replication
narrative.

Required: explicit statement that §9.2 and §9.3 metrics apply to Lane B outer fold
results and the lockbox. Lane A reports only BER, True+, True- (plus CI on BER for
the improvement claim).

**[MISSING] Calibration feasibility caveat**

§9.3 item 1: "Brier score and calibration diagnostics." With ~35 fails in the largest
outer test window, a 10-bin reliability diagram has on average 3–4 fail events per bin.
Calibration curve confidence intervals will be very wide at this sample size. This must
be noted at the point of specification, not discovered during implementation: "Calibration
curves must include 95% CI on each bin. Wide intervals at this sample size are expected
and must be disclosed — do not interpret point estimates without CI context."

**[MISSING] Brier score dual baseline not specified**

§9.3 item 1 says "Brier score" without specifying the baseline. Two different baselines
exist: (a) all-pass classifier: Brier = prevalence ≈ 0.0664; (b) constant-prevalence
classifier (always predict 0.066): Brier = p(1−p) ≈ 0.062. Both should be reported
alongside the model Brier score. Without stating this, the implementer will compute
the model Brier score with no reference point, making the number uninterpretable.

**[MISSING] Feature stability and redundancy analysis content undefined**

§9.3 item 5: "Feature stability and redundancy analysis." This is three words describing
something that has a complex implementation. Required content:
- Selection frequency per feature across 3 outer folds (0, 1, 2, or 3)
- Mean |coeff × feature_std| conditional on selection (not averaged over folds where
  the feature was never selected — this would dilute the signal)
- Expected contribution = selection frequency × conditional |coeff × std|
- Jaccard similarity between fold 1 and fold 3 selected feature sets (temporal
  stability diagnostic)
- Feature cluster assignments from EDA correlation matrix (|corr| ≥ 0.95 threshold)

Without this specification, `feature_report.csv` will be implemented as a simple
importance ranking table, missing the stability and cluster analysis that makes it
useful for a process engineer.

**[MISSING] MSPC artifact content undefined**

§9.3 item 4 references MSPC metrics. `mspc_baseline.csv` (§11 item 8) has no content
definition. Required columns:
- Per-outer-fold: T²-AUC, Q-AUC at UCL threshold
- Empirical ARL₀ on training pass-wafers (not the theoretical 1/α = 100)
- Supervised model TPR and MSPC TPR at matched TNR (e.g., TNR=90%)
- Per-fold alarm rates at UCL

**[MISSING] Operational cost curve artifact content undefined**

`operational_cost_curves.csv` (§11 item 9) has no content definition. Required:
- Rows: r ∈ {1, 2, 3, 5, 7, 10, 15, 20} (or finer grid)
- Columns: expected cost at each r for primary+Youden, primary+operational,
  challenger+Youden, all-pass baseline (r × prevalence), all-flag baseline (1−prevalence)

---

## Section 10 — Phase 2 Outputs

**[BLOCKER] "Matched operating conditions" for MSPC comparison not defined**

"Supervised vs MSPC baseline comparison at matched operating conditions." Matched on
what? Options: same TNR (specificity), same FP count per week, same alarm rate, same
threshold value. These give different comparisons and different conclusions. Must
specify: "Match on TNR. At TNR=90% (10% false alarm rate), report TPR of supervised
model and MSPC T²/Q separately. This is the standard matched-specificity comparison
used in the process monitoring literature."

**[AMBIGUOUS] Feature cluster definition not specified**

"Top stable features grouped into correlated clusters for engineer triage." How are
clusters defined? By the EDA pre-computed correlation pairs at |corr|≥0.95? By
hierarchical clustering with a distance threshold? By a graph-connected-components
algorithm on the correlation network? The EDA already computed 316 highly correlated
pairs at |corr|≥0.95. The simplest defensible approach: use connected components of
the |corr|≥0.95 graph. Each connected component is one cluster. State this explicitly.

**[MISSING] "Recommended alert policy" is undefined**

§10 item 3: "Review workload estimate and recommended alert policy." Alert policy in
what form? Options: a threshold recommendation (flag when predicted probability ≥ X),
a ranked priority list (flag top N wafers per week), a SPC-style decision rule. Without
a definition, this becomes a vague narrative. Recommended form: "State the Youden
threshold in probability terms, the expected weekly review count at that threshold, and
note that the operational threshold from the cost curve is used when review capacity is
constrained."

---

## Section 11 — Required Artifacts

**[AMBIGUOUS] `splitwise_timeaware_results.csv` content not defined**

This artifact presumably contains per-fold outer BER for each method evaluated in
Stage B (and possibly Stage A). But the exact columns are not stated. Does it include
Stage A per-fold results, Stage B per-fold results, or both? For each fold: which
metrics (BER only? BER + True+ + True-?)? Required definition: "Per-fold outer test
results for Stage B finalist configs. Columns: config_id, method, k, C, scaler,
fold_index, fold_train_weeks, fold_test_weeks, fold_n_fails_test, BER, True+, True-,
threshold_used."

---

## Section 13 — Claim Policy

**[MISSING] CI computation method not specified for benchmark improvement claim**

"Improvement under your replicated framing when CI supports it." — CI computed how?
Must specify: "Bootstrap CI on Lane A fold-level BER (n=1000 resamples of the 10
fold-level BER values). Improvement is claimed only when the 95% CI upper bound for
our mean BER falls below 33.5% (the McCann & Johnston F-test benchmark)."

**[MISSING] MSPC comparison claim branch is absent**

The claim policy covers the supervised model against the 2008 benchmark but says
nothing about the MSPC comparison result. The honest policy is:
- If supervised model TPR > MSPC TPR at TNR=90%: claim "the supervised model improves
  on the MSPC baseline in fail detection rate at matched specificity."
- If MSPC TPR ≥ supervised TPR: do not claim supervised superiority. Report honestly:
  "MSPC achieves comparable or better fail detection. The supervised model provides a
  probability score and labeled-outcome alignment that MSPC cannot, but does not
  improve upon MSPC's detection rate at this specificity level."

This branching logic must be pre-specified. It cannot be written after seeing results.

---

## No Random Seed Policy Anywhere in the Document

**[BLOCKER] No random state specification for any random operation**

The document specifies no random seeds for any of the following operations:
- Lane A 10-fold stratified CV
- Stage B 5-fold stratified inner CV
- Bootstrap CI computation
- Any future random-k feature ablation

Without fixed seeds, results are not reproducible across machines or implementers. The
document must specify: "All random operations use `random_state=42` unless stated
otherwise. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for inner CV.
`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` for Lane A."

---

## Summary: Issues by Priority

### Blockers — resolve before any implementation starts

| # | Issue | Location |
|---|---|---|
| 1 | Outer fold count and boundaries not specified | §4.2 |
| 2 | Inner tuning description scope ambiguous (Stage A vs B) | §4.2 |
| 3 | Lane A classifier not specified | §4.1 |
| 4 | Lane A scaler not specified | §4.1 |
| 5 | Lane A threshold policy not specified | §4.1 |
| 6 | L1/EN "optional" must be decided now | §5.1 |
| 7 | ReliefF neighbor count values not specified | §5.2 |
| 8 | Business preference not pre-declared in document | §6 |
| 9 | Metrics policy not scoped to Lane A vs Lane B | §9 |
| 10 | MSPC matched operating condition not defined | §10 |
| 11 | No random seed policy | throughout |

### Ambiguous — resolve to ensure consistent implementation

| # | Issue | Location |
|---|---|---|
| 12 | Config freeze timing not defined | §2 |
| 13 | Lane A CI method for ablation paired delta | §4.1 |
| 14 | Stage A false-elimination risk not disclosed | §5.1 |
| 15 | ±0.02 threshold conservative relative to noise floor | §6 |
| 16 | Operational threshold at what cost ratio r | §7 |
| 17 | Lockbox degradation interpretation guidance | §8 |
| 18 | Lockbox artifact rows (which config × threshold combinations) | §8 |
| 19 | Feature cluster definition method | §10 |
| 20 | Alert policy definition | §10 |
| 21 | `splitwise_timeaware_results.csv` column definition | §11 |
| 22 | CI computation method for benchmark improvement claim | §13 |

### Missing — add before final document is accepted as canonical

| # | Issue | Location |
|---|---|---|
| 23 | Calibration feasibility caveat (3-4 fails per bin) | §9.3 |
| 24 | Brier score dual baseline values specified | §9.3 |
| 25 | Feature stability analysis content fully defined | §9.3 |
| 26 | `mspc_baseline.csv` column definition | §11 |
| 27 | `operational_cost_curves.csv` column definition | §11 |
| 28 | MSPC comparison claim branching logic | §13 |

---

## What Is Genuinely Ready for Implementation

The following sections are unambiguous and correct. Implementation can proceed on
these without further specification:

- §2 Rules 1, 3, 4 (leakage, drift gate, claims)
- §3 Data partition (sort by timestamp, 85/15 split, lockbox discipline)
- §4.1 Lane A ablation structure (Strict vs +MI variants, paired delta — method TBD)
- §4.2 Stage A/B distinction (fixed settings vs nested, once blocker 2 is resolved)
- §5.2 Stage A fixed settings (k=40, C=1.0, RobustScaler, values+indicators, Youden)
- §5.2 Stage B grid (k, C, scaler values — once ReliefF neighbor values are added)
- §6 Lexicographic selection structure (once business preference is pre-declared)
- §7 Two-threshold framework (once cost ratio for operational threshold is specified)
- §8 Lockbox one-time evaluation
- §9.1 BER as primary selection metric
- §9.2 Primary reporting metrics list
- §9.3 item 6 MSPC spec reference to improvement_plan.md
- §10 Weekly wafer counts, feature clustering direction (once cluster method defined)
- §12 Report outline structure
- §13 Cannot-claim items (all three are clear)
