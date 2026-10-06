# 04 Temporal Robustness Study

## Scope and Claim Boundary

Temporal robustness is secondary evidence: a balanced-logistic-regression role stress study plus a bounded DEV-only KRR comparison, neither an operational transfer certification. The last chronological 15% remains a retrospective later evaluation block already exposed through full-dataset benchmarks and prior reports. Existing LOCKBOX filenames are supported, but never imply a fresh confirmatory lockbox.

## Chronological Roles

Stable-sort by timestamp and raw sample ID. DEV outer tests are disjoint calendar-week blocks: fixed boundaries at 50%, two-thirds, five-sixths, and the end of DEV's calendar-week span. Each training region is the earlier expanding prefix. Boundaries are independent of labels; test failure counts never redesign periods. Low/absent evaluation class counts accompany guarded rates and explicit availability fields. Insufficient training/calibration classes make the temporal layer unavailable with a warning, without invalidating benchmarks.

Reserve the final chronological 20% of each training region as calibration. Tune on the earlier FIT prefix using three deterministic chronological inner splits (at least two must be feasible): validation blocks follow expanding prefixes at 50%, two-thirds, five-sixths, and the end. Each inner training region itself has an earlier FIT and later calibration portion. Train/selector/scaler fits use FIT; thresholds use calibration only. Validation outcomes can select inner configs, but outer evaluation labels cannot choose periods, configs, fitted state, or thresholds. Remove shuffled tuning and random-seed pseudo-replications.

Select configurations by BER then deterministic simplicity/order. DEV joint outer selection chooses the family from inner metrics within each outer train. Family outer comparisons are exploratory. Final primary/challenger roles are chosen from chronological inner searches on the final FIT prefix, never a retrospective outer family minimum. Reuse prepared selector rankings across classifier C values and feature-budget prefixes.

## Retained Model and Diagnostics

Fit the retained role model on FIT, score calibration, freeze scientific BER and illustrative mean-weekly operating thresholds, and do not refit on calibration. Evaluate future samples once. Manager workload comes from held-out DEV calibration predictions and names that region. The policy constrains mean weekly flagged fraction, not a hard cap in every week; it is illustrative and calibrated using that same region.

KS compares held-out calibration and future scores from the same retained model. Raw-feature PSI may use FIT as a descriptive reference. Explicitly report selected-indicator missingness-rate shift and collection-regime association without causal attribution. Heuristic drift gates never authorize superiority: confirmatory claims are always unavailable for the retrospective later block.

MSPC fits PCA on pass-only FIT samples. Held-out calibration determines T2 versus Q source and operating thresholds before evaluation. Persist truly frozen-threshold confusion counts separately from retrospective evaluation-label ROC TNR90 diagnostics. Observed mean inter-alarm spacing over all samples is not in-control ARL0.

## Uncertainty and Evidence

Persist disjoint DEV joint held-out prediction IDs/configs/thresholds and recomputable confusion counts. Report later-block N, failure count, TP/FP/TN/FN, and exact binomial TPR/TNR intervals. These intervals condition on the fixed fitted model and assume independent Bernoulli trials; temporal dependence and model-selection uncertainty are excluded. Sparse class counts limit interpretation. Report outer fold mean/std/range descriptively; do not claim independent future replications or algorithm-performance CIs.

Temporal warnings and restrictions do not erase original or tuned benchmark evidence. No operational superiority, production readiness, causal process-driver, or intervention benefit claim follows from this dataset.

Inner validation periods with only one class cannot supply a BER objective; omit them without moving their boundaries. Require at least two useful fixed inner periods or report temporal evidence unavailable. Outer evaluation periods remain fixed regardless of their labels or failure counts.

## Bounded DEV KRR Comparator

Use the same fixed DEV test periods and raw evaluation IDs as logistic regression. KRR uses median imputation, StandardScaler, supported configured benchmark selector families, k10/20/40 and ReliefF neighbors5/10/20, and the shared tuned alpha/gamma-multiplier owner from spec 02. Reuse train-fitted ranking prefixes and byte-identical score fits. It adds no later15% KRR freeze/role/MSPC/manager suite.

Predeclare joint, values-only, and values-plus-indicators procedures for final 20% calibration (main) and final 30% calibration (sensitivity). Each path independently tunes inside its earlier 80% or 70% FIT, respectively. Inner training prefixes likewise reserve that path's fraction for threshold calibration before chronological validation. No 30% calibration row can participate in its model fitting, selector fitting or configuration selection. Joint input mode is inner-BER-selected; outer minima cannot choose a procedure, window or champion. Retain the FIT model and freeze its calibration threshold. Infeasible paths have explicit unavailability reasons and warnings; never move test periods or fail the primary benchmarks.

## Calibration Fragility

For selected LR DEV joint and final primary/challenger models, and each available KRR procedure, persist held-out calibration scores, raw IDs, exact FIT IDs, timestamps, counts, config identity and frozen threshold. Report N, failures, passes, BER change per one failed example=1/(2 failures) and per one pass=1/(2 passes), where defined. Fewer than 10 failures is a declared fragility warning, neither a feasibility minimum nor permission to redesign splits.

Fixed-model leave-one-failure-out recalibration removes each failed calibration example in turn, changes only the threshold, and reports threshold range plus flagged-fraction range on the same complete calibration score set. It is available only with at least two failures and one pass. Undefined diagnostics remain null. These are descriptive calibration-instability ranges, not CIs or future/algorithm uncertainty. Report per-period BER and ROC AUC to distinguish threshold weakness from ranking weakness. No evaluation labels select windows, thresholds, or models.

Inner row boundaries may share a timestamp when ordered by the stable `(timestamp, raw_row_id)` key. Keep those fixed boundaries and disjoint raw IDs; the inner receipt audit allows nondecreasing timestamps. Search receipts flag `inner_timestamp_tie`, selected comparator metrics count `n_inner_timestamp_ties`, and calibration receipts/diagnostics flag `fit_calibration_timestamp_tie`. Stable raw row ID order is deterministic bookkeeping; it establishes neither physical event order nor independence within tied timestamps. Reports disclose this limitation. Outer calendar test blocks remain strictly later.
