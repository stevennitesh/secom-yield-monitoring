# SECOM Yield Monitoring: Validation Playbook

## 1) Problem We Are Solving

We are building a model to identify semiconductor production failures (`fail=1`) from process/sensor signals so engineers can:

1. Detect likely yield excursions early.
2. Rank important signals/features for investigation.
3. Balance fail detection against false alarms.

This is not only a classification problem. It is a **time-varying industrial monitoring** problem where model validation must reflect forward-in-time deployment.

## 2) Dataset Reality and Why Validation Is Hard

Observed facts from EDA:

1. Samples/features: `1567 x 590`.
2. Class imbalance: `104 fails` (~`6.64%` fail rate).
3. Missingness is substantial for a subset of features:
   - `32` features with >`40%` missing.
   - up to ~`91%` missing in worst columns.
4. Strong redundancy:
   - many highly correlated / near-duplicate feature pairs.
5. Time drift exists:
   - weekly fail rate changed substantially over time.

Implications:

1. Plain accuracy is misleading.
2. Random CV can look optimistic versus forward-time performance.
3. Small fail counts in test windows can make results noisy.
4. Threshold and `k` selection can become unstable if validation windows are too small.

## 3) What We Did So Far (Troubleshooting Path)

### 3.1 Data integrity + hygiene

1. Verified row alignment between `secom.data` and labels/timestamps.
2. Confirmed label mapping and fail prevalence.
3. Removed low-information features:
   - constants, near-constants.
4. Identified duplicate/redundant clusters.

Result: feature count reduced to a cleaner candidate set.

### 3.2 Missingness strategy checks

Tested:

1. Median-only imputation.
2. Median + missing indicators.

Finding:

1. `median + indicators` gave better BER / fail recall than median-only.
2. Missingness has signal, but values still matter.

### 3.3 Method comparisons under random CV

Compared S2N, Welch-t, F-test/Pearson, ReliefF, Gram-Schmidt, L1, Elastic Net.

Pattern:

1. ReliefF looked best in random CV.
2. Embedded methods (L1/EN) underperformed in this setup.

### 3.4 Time-aware evaluations

Ran forward/anchored splits and multi-cut checks.

Pattern:

1. Performance dropped versus random CV (expected under drift).
2. Method ranking became split-dependent.
3. F-test and Welch-t became more competitive/robust.

### 3.5 Uncertainty check

Compared split-by-split deltas with bootstrap/sign-test style reasoning:

1. BER differences between top methods were small and often non-significant.
2. Tradeoff was operational:
   - one method catches more fails,
   - another reduces false alarms.

## 4) What Went Wrong (and Why)

1. Early windows had too few fail events in outer tests.
   - This inflated metric variance.
2. Different split schemes gave different winners.
   - Not contradiction; expected under drift + class imbalance.
3. A "great lockbox run" after trying many configs is not valid evidence.
   - Once lockbox guides choices, it is no longer an untouched final test.

## 5) Correct Validation Logic Going Forward

## 5.1 Core principles

1. Never tune on outer test.
2. Tune `method`, `k`, and threshold using training-only data (inner time-aware splits).
3. Use outer time-aware windows for unbiased model-selection estimates.
4. Keep one final untouched lockbox for one-time final reporting.

## 5.2 Recommended split design

1. Reserve last `~15%` by time as lockbox.
2. Use the earlier `~85%` as dev.
3. On dev, use larger outer test windows (70/30-style anchored windows) to increase fail count stability.
4. Inside each outer-train, run inner time-aware tuning.

## 5.3 Candidate set to tune

Primary candidates:

1. `F-test` (simple, robust baseline).
2. `Welch-t` (imbalance-aware test behavior).

Optional challenger:

1. ReliefF (only if runtime budget allows after primary tuning is stable).

Tune:

1. `k ∈ {10, 20, 40, 60}` (optionally 80).
2. Threshold policy:
   - fixed `0.5`,
   - train-only tuned threshold (BER objective or business constraint objective).

## 5.4 Model selection rule

1. Primary criterion: mean/median outer BER on dev.
2. Tie-break 1: business preference
   - maximize fail catch (True+), or
   - maximize pass specificity (True-).
3. Tie-break 2: smaller `k` (simpler model).

## 5.5 Finalization rule

1. After selecting a single config on dev:
   - train on full dev,
   - evaluate once on lockbox.
2. Do not retune after seeing lockbox.

## 6) Why This Is the Right Next Step

This protocol directly addresses each failure mode we observed:

1. Drift: handled by forward-time outer/inner splits.
2. Instability from low fail counts: mitigated by larger outer windows.
3. Selection leakage: prevented by nested tuning.
4. Over-interpretation of one lucky split: avoided via repeated outer windows + uncertainty summaries.
5. Lockbox contamination: avoided by one-time final use.

## 7) Concrete Next Step Checklist

1. Freeze split definitions (dev + lockbox).
2. Run nested time-aware tuning on dev for `F-test` and `Welch-t`.
3. Compare BER/True+/True- across outer windows.
4. Select one primary + one challenger with explicit tradeoff rationale.
5. Run one-time lockbox evaluation for the selected primary.
6. Freeze protocol and begin implementation in `src/`.

## 8) Deliverables to Produce Before Coding

1. `model_selection_summary.csv`
   - method, k, threshold policy, outer BER mean/std, True+/True- mean/std.
2. `splitwise_results.csv`
   - each outer split result for selected candidates.
3. `final_lockbox_result.csv`
   - one-time primary model final metrics.
4. `feature_report.csv`
   - selected features, selection frequency, indicator/value type, direction notes.

---

This is the point where methodology becomes stable enough to implement without rework.
