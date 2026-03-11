# Codex Assessment: `docs/improvement_plan.md`

## Executive Summary

The improvement plan is strong and directionally correct. It identifies the real issue (validation instability under drift), calls out lockbox contamination, and proposes a more defensible protocol.

The main adjustments I recommend are:

1. Keep time-aware validation as the primary tuning/evaluation path (not random inner CV as default).
2. Reduce scope for phase 1 (freeze supervised pipeline first; make MSPC a phase-2 extension).
3. Treat preprocessing choices (for example `StandardScaler` vs `RobustScaler`) as tunable, not fixed assumptions.
4. Re-establish an untouched final holdout if the previous lockbox influenced decisions.

## 1) What The Plan Gets Right

## 1.1 Correct diagnosis

1. Class imbalance and temporal drift are the core sources of instability.
2. Small outer test windows produced too few fail events and noisy BER.
3. The lockbox cannot be used for iterative tuning.

These are the right root causes.

## 1.2 Correct hygiene principles

1. Fit all preprocessing inside training folds only.
2. Optimize threshold on training data only.
3. Separate model selection from final one-time lockbox evaluation.

These are non-negotiable and are correctly emphasized.

## 1.3 Correct reporting mindset

1. Report random-CV and time-aware results as different questions.
2. Avoid causal claims from observational feature importance.
3. Frame operational tradeoff (fail catch vs false alarms), not a single metric-only story.

This is exactly how hiring managers and process engineers will read the project.

## 2) Where I Would Adjust The Plan

## 2.1 Inner-loop CV choice

The plan argues for random stratified inner CV for stability. I disagree with making that the default.

Reasoning:

1. You observed real time drift.
2. If deployment is forward-in-time, inner tuning should preserve time direction to avoid selecting configurations that depend on future regimes.
3. Stratified random inner CV can be kept as a sensitivity check, but not the primary tuning method.

Recommendation:

1. Primary: anchored time-aware inner splits.
2. Sensitivity: stratified inner CV (report how much settings change).

## 2.2 Fixed preprocessing claims

The plan recommends switching to `RobustScaler` directly.

Reasoning:

1. This may help with excursions/outliers, but should be validated, not assumed.
2. Current results are sensitive enough that preprocessing should be part of tuning.

Recommendation:

1. Tune scaler as a small choice: `{StandardScaler, RobustScaler}`.

## 2.3 Scope control

The plan includes a large phase-1 scope (full metric suite, MSPC baseline, calibration, cost curves, contribution methods).

Reasoning:

1. This is valuable, but it can delay getting a frozen, leak-safe core model.
2. Resume projects benefit from a clear staged story: "robust core first, extensions second."

Recommendation:

1. Phase 1: finalize supervised validation protocol + lockbox result + feature report.
2. Phase 2: MSPC and extended operational framing.

## 2.4 Lockbox status

Because lockbox results already influenced discussion, the previous lockbox should be treated as diagnostic.

Recommendation:

1. If possible, carve a new untouched final window.
2. If not possible, clearly state that final numbers come from nested time-aware DEV evaluation and the existing lockbox is non-final.

## 3) Recommended Next-Step Protocol (What To Do Now)

## 3.1 Freeze data partitions

1. Sort by timestamp.
2. Reserve a final holdout window (last 10-15%) as untouched lockbox.
3. Use earlier data as DEV for all tuning.

## 3.2 Tune only two methods first

1. `F-test`
2. `Welch-t`

Search space:

1. `k in {10, 20, 40, 60}`
2. `C in {0.1, 1.0, 10.0}` for logistic
3. scaler in `{StandardScaler, RobustScaler}`
4. threshold policy in `{fixed 0.5, train-tuned by BER}`

## 3.3 Nested time-aware validation on DEV

1. Outer: anchored windows with larger test segments (70/30-style) to increase fail count stability.
2. Inner: anchored time-aware splits for tuning.
3. Selection criterion: mean outer BER.
4. Tie-break:
   - higher True+ if fail misses are more costly,
   - else higher True- if false alarms are more costly,
   - then smaller `k`.

## 3.4 Final one-time evaluation

1. Fit selected config on full DEV.
2. Apply selected threshold policy without change.
3. Evaluate once on untouched lockbox.
4. Stop tuning.

## 4) Deliverables To Produce Immediately After Freeze

1. `model_selection_summary.csv`
   - method, k, C, scaler, threshold policy, outer BER mean/std, True+/True- mean/std
2. `splitwise_results.csv`
   - each outer split result for top configurations
3. `final_lockbox_result.csv`
   - one-time final metrics
4. `feature_report.csv`
   - selected features, value vs missing-indicator flag, selection frequency

## 5) What To Tell Hiring Managers

Keep the story simple and defensible:

1. "We compared feature-selection methods under strict forward-time validation."
2. "Random CV looked better but was optimistic under drift."
3. "We selected the final model using nested time-aware tuning and one-time holdout evaluation."
4. "Missingness carried signal, but best performance came from combining values and missing indicators."
5. "Final model choice reflects an explicit fail-catch vs false-alarm tradeoff."

That is the most credible way to present this project.
