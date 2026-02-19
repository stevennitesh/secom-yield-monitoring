# SECOM End-to-End Strategy: From Current Analysis to Final Reports

## Purpose

This document consolidates:

1. Current analysis status and what it means.
2. Validation decisions and rationale.
3. Phase 1 plan (scientific replication + rigorous modeling).
4. Phase 2 plan (semiconductor-manager-facing methods).
5. Exact path from where we are now to report-ready outputs.

This is the working contract for finishing the project in a defensible, resume-ready way.

## 1) Where We Are Now

## 1.1 Completed findings (from notebooks)

1. Data integrity is validated (rows aligned, labels parsed, timestamps parsed).
2. Data is highly imbalanced (`~6.64%` fails).
3. Missingness is substantial for a subset of features and is partially informative.
4. Time drift is real (weekly fail-rate variation is large).
5. Redundancy is high (many highly correlated / near-duplicate features).

## 1.2 Modeling observations so far

1. Random CV and time-aware results do not agree on a single stable winner.
2. ReliefF looked strong in random CV but was less consistent in time-aware settings.
3. F-test and Welch-t are competitive in time-aware settings with operational tradeoff:
   - one can favor fail recall,
   - the other can favor specificity.
4. Differences are often small relative to split variance.

## 1.3 What is trustworthy vs diagnostic

1. Trustworthy for selection:
   - nested, time-aware DEV results only.
2. Diagnostic only:
   - any lockbox result that influenced tuning choices.

Implication: we must enforce a strict final holdout protocol before final claims.

## 2) Core Validation Logic (Final Stance)

1. Keep two separate evaluation lanes:
   - Lane A: baseline replication (apples-to-apples with literature).
   - Lane B: deployment-realistic validation (time-aware).
2. Use time-aware nested validation as primary for model choice.
3. Tune threshold on training data only.
4. Use one untouched final lockbox exactly once.
5. Report uncertainty (split-wise variation and confidence intervals), not only means.

## 3) Phase 1: Scientific + Rigorous Validation

Phase 1 has two lanes that must both be reported.

## 3.1 Lane A: Baseline replication (for comparability)

Goal:

1. Replicate classic SECOM feature-selection benchmark framing.

Protocol:

1. Methods: `S2N`, `t`, `F`, `Pearson`, `ReliefF`, `Gram-Schmidt`.
2. Feature count: `k=40`.
3. Evaluation: 10-fold random CV.
4. Metrics: BER, True+, True-.

Important:

1. Keep this lane as close as possible to the published setup.
2. Any deviations (classifier, preprocessing details) must be explicitly disclosed.

Output:

1. A clean comparison table versus literature baseline BER.

## 3.2 Lane B: Deployment-realistic validation (primary decision lane)

Goal:

1. Select model settings that survive time drift and realistic deployment constraints.

Protocol:

1. Sort by timestamp.
2. Reserve a final untouched lockbox (last 10-15% by time).
3. Use earlier data as DEV.
4. On DEV:
   - Outer anchored time-aware splits with larger test windows (70/30-style windows) to reduce fail-count noise.
   - Inner time-aware splits for tuning (for shortlisted methods).
5. Stage A broad screen (fixed settings, same outer splits for all selectors):
   - evaluate `S2N`, `Welch-t`, `F-test`, `Pearson` (optional if identical to F-test), `ReliefF`, `Gram-Schmidt`,
   - optional one embedded baseline (`L1` or `Elastic Net`),
   - use fixed `k` (for example `k=40`) and fixed threshold policy for fair first-pass ranking.
6. Stage B focused nested tuning on top 2-3 methods from Stage A:
   - `k in {10,20,40,60}`,
   - logistic `C in {0.1,1,10}`,
   - scaler `{StandardScaler, RobustScaler}`,
   - threshold policy `{fixed 0.5, train-tuned BER}`.
7. Select by:
   - primary: mean outer BER,
   - tie-break: operational preference (True+ vs True-),
   - tie-break: smaller `k`.

Output:

1. Frozen primary model and challenger with fully specified settings.

## 3.3 Final one-time lockbox run (after freeze)

Protocol:

1. Train chosen config on full DEV.
2. Apply frozen threshold policy.
3. Evaluate once on lockbox.
4. No further tuning after this run.

Output:

1. Final unbiased lockbox result table.

## 4) Phase 2: Semiconductor-Manager-Facing Methods

Phase 2 is for operational credibility and stakeholder appeal.

## 4.1 MSPC baseline (industry language)

Implement and compare:

1. PCA-based `Hotelling T²`.
2. PCA residual `Q-SPE`.

Using same outer time-aware discipline as supervised model.

## 4.2 Operational metrics managers care about

Report:

1. Wafers flagged per week.
2. Fails caught per week.
3. Fails missed per week.
4. Review burden rate.
5. Detection lag to first alarm.

## 4.3 Decision economics

1. Cost curve across FN/FP cost ratios.
2. Break-even operating region.
3. Head-to-head operating points:
   - supervised model,
   - MSPC baseline.

## 4.4 Diagnostics and actionability

1. Feature report (selected values + missing-indicator signals).
2. Contribution-style diagnostics (for alarms and top features).
3. Explicit non-causal language (association, not root-cause proof).

## 5) What We Should Claim (and Not Claim)

Can claim:

1. Baseline replication quality and improvements under comparable random CV framing.
2. More realistic deployment validation via time-aware protocol.
3. Explicit operational tradeoff and threshold policy selection discipline.

Should not claim:

1. Causality from feature importance alone.
2. Generalization beyond observed period/fab without additional validation.
3. Lockbox superiority if lockbox was used iteratively during tuning.

## 6) From Now to Reports: Concrete Execution Sequence

1. Freeze data split policy:
   - define DEV and untouched lockbox.
2. Run Phase 1 Lane A baseline replication table.
3. Run Phase 1 Lane B Stage A broad selector screen on DEV.
4. Run Phase 1 Lane B Stage B focused nested tuning on top 2-3 selectors.
5. Freeze primary + challenger configs.
6. Run one-time lockbox evaluation.
7. Produce Phase 1 report tables and plots.
8. Implement Phase 2 MSPC + operational metrics + cost framing.
9. Produce integrated final report for hiring-manager narrative.

## 7) Required Report Artifacts

1. `reports/baseline_replication.csv`
2. `reports/timeaware_selector_screening.csv` (Stage A)
3. `reports/timeaware_model_selection.csv` (Stage B)
4. `reports/splitwise_timeaware_results.csv`
5. `reports/final_lockbox_result.csv`
6. `reports/feature_report.csv`
7. `reports/mspc_baseline.csv` (Phase 2)
8. `reports/operational_cost_curves.csv` (Phase 2)

## 8) Suggested Final Report Structure

1. Problem and business context (yield monitoring and fail-risk).
2. Dataset characteristics and observed drift.
3. Baseline replication results (literature comparability).
4. Time-aware validation results (deployment realism).
5. Final chosen operating points (primary + challenger).
6. Phase 2 operations section (MSPC, alarm burden, cost tradeoffs).
7. Limitations and future work.

---

The project is now at the transition point from exploratory analysis to protocol freeze and report production.
