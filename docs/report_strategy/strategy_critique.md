# Strategy Critique: Challenging the Codex End-to-End Report Strategy

*This document reviews `end_to_end_report_strategy.md` against `improvement_plan.md`,
identifies genuine improvements Codex adds, challenges specific decisions where the
reasoning is weak or wrong, and proposes a reconciled position for each disagreement.*

---

## 1) Where Codex Adds Genuine Value

These are points in Codex's document that are better than my improvement plan and
should be adopted.

### 1.1 Two-stage screening structure (Stage A → Stage B)

Codex proposes a formal two-stage process: Stage A is a broad screen across all feature
selection methods at fixed settings, Stage B is focused nested tuning on the top 2–3
survivors. My document skips straight to nested tuning on F-test and Welch-t, justified
by notebook observations.

The Codex approach is more defensible in a hiring context because it is auditable. If a
hiring manager asks "why didn't you test ReliefF in your final setup?", the answer
"Stage A showed it wasn't competitive on time-aware splits" is stronger than "the
notebook suggested it". Stage A is the formal paper trail for method elimination.

**Adopt**: The two-stage structure. Clarify the execution of each stage (see Section 3).

### 1.2 Scaler as a tuning dimension in Stage B

Codex includes `{StandardScaler, RobustScaler}` in the Stage B search grid. My document
simply recommends switching to RobustScaler based on reasoning about outlier sensitivity,
without empirically testing the assumption.

The Codex approach is more rigorous. RobustScaler is better in theory for sensor data
with equipment excursions — but theory is not the same as evidence on this specific
dataset. If StandardScaler produces lower outer BER on these 14 weeks of data, the
rationale for RobustScaler needs to be revised. Including both in the grid answers the
question empirically rather than by assumption.

**Adopt**: `{StandardScaler, RobustScaler}` in the Stage B grid.

### 1.3 Cleaner report artifact naming

Codex separates:
- `timeaware_selector_screening.csv` (Stage A)
- `timeaware_model_selection.csv` (Stage B)
- `splitwise_timeaware_results.csv`
- `mspc_baseline.csv`
- `operational_cost_curves.csv`

My document collapses Stage A and B results into `model_selection_summary.csv`. Codex's
separation is cleaner because a reviewer of the artifacts can immediately see which
results come from screening (Stage A) and which from rigorous selection (Stage B).

**Adopt**: Codex's artifact naming and separation.

### 1.4 Final report section structure (Section 8)

Codex's 7-section narrative is more coherent than my Section 6:

1. Problem and business context
2. Dataset characteristics and observed drift
3. Baseline replication (literature comparability)
4. Time-aware validation (deployment realism)
5. Final operating points (primary + challenger)
6. Phase 2 operations (MSPC, alarm burden, cost)
7. Limitations and future work

This ordering answers increasingly operational questions and ends with honest
limitations. The drift section appearing early (position 2) is correct — drift is
context that every subsequent result must be interpreted through.

**Adopt**: This report structure.

---

## 2) Where Codex Is Wrong

These are specific decisions in Codex's document where the reasoning is flawed or the
conclusion contradicts evidence already established in the notebooks.

### 2.1 Inner CV: "Inner time-aware splits" is the wrong choice for this dataset

**Codex says** (Section 3.2, Step 4): "Inner time-aware splits for tuning (for
shortlisted methods)."

**This is wrong for a practical reason, not a theoretical one.**

With approximately 60–80 fails in each outer training set, a time-aware inner split
gives fewer fails per inner test window than 5-fold stratified CV. The inner loop's job
is to reliably distinguish a good configuration from a bad one. If the inner test
windows contain 8–12 fails each, the ROC-AUC signal is so noisy that the grid search
selects configurations near-randomly. You have then gone through the computational cost
of a nested CV scheme and gotten less configuration-selection information than a
stratified split would have provided.

The theoretical concern about temporal leakage in the inner loop is real — adjacent
wafers are correlated. But the inner loop does not produce the reported metrics. It only
selects k, C, and scaler. Mild optimism in inner AUC scores does not propagate to the
outer test set results, which are what you report and which are computed on held-out
time-future data.

The fail-count-per-fold problem cannot be solved by making the inner splits larger
without reducing their number, which further reduces signal. The tradeoff resolves
clearly in favour of stratified inner CV for this dataset. Codex's recommendation
prioritises methodological purity over practical utility, which is the wrong priority
when you have 104 total fails.

**Decision**: Inner CV uses 5-fold stratified k-fold. This is not a compromise — it is
the correct choice given the data.

### 2.2 Stage A scope is redundant with existing notebook results

**Codex says** (Section 3.2, Step 5): Stage A screens S2N, Welch-t, F-test, Pearson,
ReliefF, Gram-Schmidt at k=40, fixed threshold, across the same time-aware outer splits
as Stage B.

**Three problems:**

**Problem 1: This is already done.** The time-aware notebook already ran all these
methods on rolling time-aware splits. Stage A as described is repeating that analysis
with marginally larger outer windows. The result will be the same — F-test and Welch-t
competitive, ReliefF and Gram-Schmidt weaker — because we are using the same data. If
the goal of Stage A is to formally document the screening decision, referencing and
extending the existing notebook results is more efficient than a full rerun.

**Problem 2: F-test and Pearson are identical.** The notebooks confirmed these select
exactly the same 40 features (Jaccard=1.0) because they are mathematically equivalent
for two-class problems. Running both in Stage A wastes computation and creates
confusingly identical results in the screening table.

**Problem 3: Fixed threshold at k=40 is a weak screen.** Stage A uses a fixed threshold
(presumably 0.5 or train-tuned) and fixed k=40. But methods differ substantially in
their sensitivity to k. ReliefF at k=40 performed worse than ReliefF at k=10 in some
notebook folds. Screening at one k value may eliminate methods that would be competitive
at a different k. A screening step that can produce false eliminations is not a reliable
filter.

**Better Stage A**: Use the existing notebook time-aware results as Stage A evidence,
supplemented by a clean rerun under the new (larger) outer fold structure. Eliminate
Pearson (duplicate of F-test). Carry forward F-test, Welch-t, and S2N as Stage B
candidates, with ReliefF as an optional challenger if runtime permits.

### 2.3 Threshold policy in the Stage B grid introduces a circular dependency

**Codex says** (Section 3.2, Step 6): Stage B tunes threshold policy `{fixed 0.5,
train-tuned BER}`.

**This is wrong for two reasons.**

**Reason 1: We already know the answer.** The notebooks implemented `best_threshold_by_ber`
and demonstrated it outperforms fixed 0.5. Including fixed 0.5 in the Stage B grid is
testing a question that is already answered. It adds grid entries, more inner CV
computation, and a column in the results table that will consistently show "train-tuned
BER wins." This is wasted effort in exchange for the appearance of thoroughness.

**Reason 2: It creates a metric dependency loop.** If Stage B uses the inner ROC-AUC
to select configurations (which is the correct inner scoring metric), and then separately
tests threshold policies as a grid dimension, the best threshold policy for maximising
inner ROC-AUC is undefined — ROC-AUC is threshold-free. You would need a different
inner scoring metric to evaluate threshold policies, which then re-introduces the BER
dependency in the inner loop. The clean separation (inner loop optimises AUC for ranking,
outer train optimises threshold for deployment) breaks down as soon as you add threshold
policy to the inner grid.

**Decision**: Always use Youden's J threshold optimisation (`best_threshold_by_ber`) on
outer training data. Never include fixed 0.5 as a grid option.

### 2.4 Codex's Phase 2 metric suite is too vague to implement

**Codex says** (Section 4.2): "Report: wafers flagged per week, fails caught per week,
fails missed per week, review burden rate, detection lag to first alarm."

**What is missing:**

- No PR-AUC. For a 6.64% fail rate, ROC-AUC is misleading — a random classifier has
  PR-AUC ≈ 0.066, not 0.5. PR-AUC is the threshold-free metric that accurately
  represents minority-class discrimination. Its absence is a gap for the ML audience.

- No MCC. The Matthews Correlation Coefficient is the single most informative balanced
  metric for binary classification with severe imbalance. It ranges from −1 to +1 and
  is equivalent to the Pearson correlation between predictions and labels. Its exclusion
  leaves the metric suite without a defensible single-number summary.

- No F2 score. Recall-weighted F-score is standard for fault detection contexts where
  missing a fail costs more than a false alarm. F1 (equal weight) is less appropriate
  than F2 here.

- No Brier score or calibration assessment. A model that ranks fails correctly but
  assigns poorly calibrated probabilities is unsafe for cost-curve decisions, because
  the cost curve depends on the absolute probability values. Codex's document mentions
  cost curves but omits the calibration check that validates the probabilities feeding
  into those curves.

- No MSPC UCL formulas. Codex says "implement Hotelling T² and Q-SPE" without stating
  the UCL derivation. T² uses the Tracy-Young-Mason (1992) F-distribution approximation.
  Q-SPE uses the Jackson-Mudholkar (1979) chi-squared approximation. These are different
  formulas. Conflating them produces incorrect upper control limits. An implementer
  following Codex's document will need to look these up separately or will use the wrong
  formula for one of them.

- No contribution plot methodology. Codex says "contribution-style diagnostics for
  alarms and top features" without specifying the method. The Kourti-MacGregor (1996)
  squared-loading decomposition is the industry standard — naming it matters because
  there are multiple contribution methods that give different answers and the choice is
  not obvious from first principles.

- No ARL₀ empirical computation note. Codex mentions "detection lag to first alarm"
  but does not distinguish between ARL₀ (false alarm rate when in control) and
  detection lag (time to first alarm after a fault). These are separate metrics with
  separate computation requirements. ARL₁ cannot be computed reliably for sporadic
  failures — only detection lag can.

### 2.5 No bootstrap CI requirement on the benchmark comparison claim

**Codex says** (Section 5): "Can claim: Baseline replication quality and improvements
under comparable random CV framing."

The McCann & Johnston (2008) benchmark reports 33.5% ±2.2% BER. The current notebooks
achieve approximately 31.4% under random CV. The difference is ~2.1 percentage points
against a benchmark with ±2.2% standard error. This is not a statistically secure
improvement without a confidence interval on the current result.

If the bootstrap CI on the current mean BER overlaps with 33.5%, the improvement claim
cannot be made without qualification. Codex's document claims this improvement is
claimable without requiring the CI to be computed first. That is overclaiming.

**Decision**: The BER improvement claim is contingent on a bootstrap CI confirming the
difference is statistically meaningful. The claim cannot be made without this.

### 2.6 Drift is noted but not required before interpreting results

**Codex says** (Section 8, point 2): "Dataset characteristics and observed drift" is a
section in the final report.

The DEV fail rate is 7.1%. The lockbox fail rate is 3.8%. This is a ~47% relative drop
in fail rate between the two periods. This is not just a dataset characteristic to
describe — it is a critical diagnostic that must be resolved before the lockbox result
can be interpreted. If the lockbox performance appears weaker than DEV performance, the
reason could be: (a) temporal model degradation, (b) process improvement (fewer real
fails), or (c) statistical noise from small fail counts. These three explanations have
completely different implications for deployment.

Codex treats drift as contextual background. It should be a required analysis step that
gates interpretation of the lockbox result, not a narrative introduction.

---

## 3) Reconciled Strategy

The following resolves each disagreement and merges the best elements of both documents.

### 3.1 Validation structure

```
Full dataset: 1567 samples, 14 weeks
│
├── LOCKBOX: last 15% by time (~236 samples)
│   Gate: touch only after drift analysis, full config freeze, and DEV training.
│   Threshold: must be output of best_threshold_by_ber on full DEV.
│
└── DEV: first 85% by time (~1331 samples)
      │
      ├── Stage A: Broad screen [Codex structure, narrowed scope]
      │     Use existing notebook rolling-CV results + clean rerun on new outer folds.
      │     Methods: F-test, Welch-t, S2N (drop Pearson as F-test duplicate)
      │     ReliefF optional if runtime permits.
      │     Fixed k=40, train-tuned threshold.
      │     Output: screening table → select top 2 for Stage B.
      │
      └── Stage B: Focused nested tuning [my structure, with Codex's scaler addition]
            Outer: 3 anchored expanding-window folds (time-aware)
            Inner: 5-fold stratified k-fold, no purge gap [my recommendation]
            Inner scoring metric: roc_auc [my recommendation]
            Grid: method {F-test, Welch-t}
                  × k {10, 20, 40}
                  × C {0.01, 0.1, 1.0, 10.0}
                  × scaler {StandardScaler, RobustScaler} [Codex addition]
            Threshold: always Youden's J on outer train [not in grid]
            Output: primary config + challenger with fully specified settings.
```

### 3.2 Inner CV decision — final position

Inner CV uses stratified k-fold. Time-aware inner splits are methodologically cleaner
but are the wrong tradeoff for a dataset with 104 total fails. Fail-count stability in
the inner loop takes priority over avoiding mild temporal leakage, because:

1. The inner loop is not reported — leakage does not contaminate outer test results.
2. Stratified inner CV gives 12–16 fails per inner test fold vs fewer with time-aware
   inner splits, providing meaningfully more stable AUC signal for configuration selection.
3. The outer test folds are time-aware, which is what prevents the fundamental
   train-on-future / test-on-past leakage that would invalidate reported metrics.

### 3.3 Threshold — final position

Always train-tuned via Youden's J (best_threshold_by_ber). Never fixed 0.5. Not a grid
dimension. This is already implemented in both notebooks and has already been shown to
outperform fixed 0.5 on training data. There is no reason to re-test it.

### 3.4 Full metric suite — adoption

Codex's Phase 2 operational metrics are correct but incomplete. The full suite from
`improvement_plan.md` Section 5 stands, including:
- Group 2: ROC-AUC, PR-AUC, MCC
- Group 3: Precision, F1, F2, Youden's J
- Group 4: Brier score (with correct dual baseline), calibration curve with CI
- Group 5: T² (Tracy-Young-Mason UCL), Q-SPE (Jackson-Mudholkar UCL), T²-AUC,
  Q-AUC, ARL₀ empirical, detection lag, Kourti-MacGregor contribution plots
- Group 6: Wafer counts per week, review burden, cost curve, break-even ratio,
  TPR at TNR=90%/95%

### 3.5 Report artifacts — reconciled list

Adopt Codex's naming with additions from improvement_plan.md:

```
reports/
  baseline_replication.csv          ← Stage A, random CV, literature comparison
  timeaware_selector_screening.csv  ← Stage A, time-aware, method elimination
  timeaware_model_selection.csv     ← Stage B, nested CV, config selection
  splitwise_timeaware_results.csv   ← per-fold results for Stage B winning configs
  mspc_baseline.csv                 ← T², Q-SPE results on same outer folds
  operational_cost_curves.csv       ← cost curve at r ∈ [1, 20] for primary + challenger
  final_lockbox_result.csv          ← one-time lockbox evaluation
  feature_report.csv                ← selection freq, conditional contribution, Jaccard
```

### 3.6 Report narrative — adopt Codex's structure

1. Problem and business context (yield monitoring, fail cost asymmetry)
2. Dataset characteristics and observed drift [required diagnostic before section 4]
3. Baseline replication (literature comparability, bootstrap CI on improvement claim)
4. Time-aware validation (deployment realism, per-fold results, uncertainty)
5. Final chosen operating points (primary + challenger with cost-curve selection)
6. Phase 2 operations (MSPC comparison, contribution plots, alarm burden, cost tradeoff)
7. Limitations (104 fails, single fab, single 14-week window, calibration caveat)

### 3.7 What to claim — consolidated

| Claim | Condition |
|---|---|
| BER improvement over 2008 benchmark | Only after bootstrap CI confirms significance |
| Time-aware validation is more realistic than random CV | Always — no condition needed |
| Supervised model adds TPR over MSPC T²/Q | Only if the MSPC comparison results support it |
| Threshold was not tuned on test data | Always — this is guaranteed by the protocol |
| Selected features are associated with failures in this window | Always — with non-causal language |
| Results generalise beyond Jul–Oct 2008 | Never |
| Results generalise to other fabs | Never |

---

## 4) Summary of Changes to Each Document

### Changes to adopt from Codex into improvement_plan.md

1. Two-stage structure (Stage A screening → Stage B nested tuning)
2. Scaler `{StandardScaler, RobustScaler}` as a grid dimension in Stage B
3. Codex's report artifact naming and separation
4. Codex's 7-section final report narrative structure

### Changes rejected from Codex

1. Inner time-aware splits → replaced with stratified inner CV (Section 2.1)
2. Threshold policy in grid → rejected (Section 2.3)
3. Stage A running Pearson alongside F-test → rejected (duplicate)
4. Phase 2 metric vagueness → replaced with full metric suite from improvement_plan.md
5. BER improvement as an unconditional claim → gated on bootstrap CI (Section 2.5)

### Changes improvement_plan.md retains over Codex

- Full metric suite with formulas and feasibility caveats (Groups 1–6)
- MSPC implementation detail: autoscaling note, UCL formulas, contribution method
- Calibration curve feasibility caveat (35 fails per test window)
- Drift analysis as a required gate before lockbox interpretation
- Bootstrap CI requirement on the benchmark claim
- Detection lag vs ARL₁ distinction
- ARL₀ must be computed empirically, not taken as 1/α by definition
- Feature report: conditional contribution, Jaccard stability across folds
- Explicit non-causal language in claims
