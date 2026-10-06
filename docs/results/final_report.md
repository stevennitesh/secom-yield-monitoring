# Can anonymous manufacturing measurements identify recorded failures?

## Executive Summary

This SECOM study asks whether recorded manufacturing measurements distinguish a failed test from a passed test. When failures are rare, predicting pass for everyone can look accurate while catching no failures. The useful comparison gives failures and passes equal weight, then examines the tradeoff between catching failures and raising false alerts.

- **Benchmark:** mean balanced error 31.43% → 30.86%; failures caught 70 → 66; false alerts 445 → 371. Rates average held-out folds; counts pool their predictions.
- **Chronological stress:** kernel ridge averages 47.47% balanced error and flags 88.5% of 750 later samples. This separate procedure tests transfer; the final later block is retrospective.

**Read:** [Scope](#dataset-and-study-scope) · [Reference](#original-replication-design) · [Tuning](#tuned-benchmark-design) · [Tradeoff](#original-vs-tuned-benchmark-comparison) · [Inputs](#feature-stability-and-interpretation) · [Later samples](#temporal-robustness-stress-test) · [Gaps](#industrialization-gaps) · [Conclusions](#conclusions-and-next-data-requirements) · [Technical tables](#technical-appendix) · [Provenance](#provenance-appendix)

## What I Built

A reproducible Python study that compares a literature-inspired reference benchmark with bounded tuning, checks the complete selection procedure on unseen test samples, and then tests transfer to later samples. Training-only transformations, saved predictions and independent artifact checks make the results traceable.

## Dataset and Study Scope

The files contain **1,567 samples, 590 anonymous measurement columns, 104 failures and 1,463 passes**. Predicting pass for everyone would give 93.36% accuracy and catch zero failures. 4.54% of measurement cells are missing.

**Failure recall** is the fraction of failures caught. **Pass specificity** is the fraction of passes correctly left unflagged. **Balanced error** is the average of the missed-failure rate and the false-alert rate on passes: ½ × [(1 − recall) + (1 − specificity)]. Lower is better; an always-pass rule has 50% balanced error.

A row represents a production entity whose physical unit is undocumented. Measurements are anonymous, and their availability before the outcome is not established. This study predicts recorded test labels; it does not demonstrate early warning or intervention benefit. Public metadata describe 591 features, while the reference SECOM measurement file contains 590.

## Original Replication Design

The reference is a literature-inspired fixed-feature-budget benchmark, not an exact replication of the published classifier and protocol. We compared 6 column-selection methods using Kernel ridge and Logistic regression. **Kernel ridge regression (KRR)** learns a nonlinear score using similarity between samples; **logistic regression (LR)** learns a weighted combination of inputs with a logistic link.

**Cross-validation** rotates which samples are held aside. Shuffled, class-stratified outer splits preserve class proportions and evaluate unseen samples. Both benchmarks used 10 held-out test folds. Inside each outer training portion, inner splits choose the selector, model, input mode, settings and alert threshold. Median imputation, scaling and feature selection see training data only. The chosen procedure is refitted on outer training data, then evaluated once on its untouched outer test samples. The test labels never choose the model or threshold.

**Calibration** means choosing the score threshold that triggers an alert. The benchmark chooses it from pooled inner held-out scores and freezes it before the outer refit. Scores can shift after refitting; that is a remaining calibration limitation. The evaluation covers the complete selection procedure, rather than a model picked retrospectively for its best test result.

## Original Replication Results

Rates below are fold means. Counts sum the once-held-out predictions. Standard deviation (SD) and ranges describe fold variation, not algorithm-performance confidence intervals.

| Procedure | Balanced error | Failure recall | Pass specificity | Fold spread (SD) | Fold range | Failures caught | False alerts on passes | Samples flagged |
|---|---|---|---|---|---|---|---|---|
| Complete selected procedure | 31.43% | 67.55% | 69.59% | 8.02 pp | 15.20%–42.81% | 70 | 445 | 32.9% |
| Measurements only | 32.73% | 73.36% | 61.18% | 6.91 pp | 22.79%–42.81% | 76 | 568 | 41.1% |
| Measurements + missing flags | 30.10% | 68.36% | 71.44% | 8.12 pp | 15.20%–42.12% | 71 | 418 | 31.2% |
| Missing flags only | 43.27% | 46.27% | 67.18% | 10.03 pp | 22.35%–58.36% | 48 | 480 | 33.7% |
| Always predict pass | 50.00% | 0.00% | 100.00% | 0.00 pp | 50.00%–50.00% | 0 | 0 | 0.0% |

## Tuned Benchmark Design

The paired benchmark protocol keeps test samples, folds and seeds fixed while comparing declared input budgets and model settings. The reference allows up to 40 selected inputs; tuning compares budgets of 10, 20 and 40 inputs. Full grids and candidate counts are in the technical appendix. For dimension-relative kernel-ridge candidates, kernel width scales with the actual selected input count. Stronger regularization wins exact balanced-error ties after fewer features.

This is bounded parameter coverage, not global optimization. Model settings, input mode and threshold remain selected entirely inside training data. Separately predefined measurements-only and measurements-plus-missing-flags procedures provide contrasts; their test results do not replace the complete selected procedure as the headline.

## Tuned Benchmark Results

| Procedure | Balanced error | Failure recall | Pass specificity | Fold spread (SD) | Fold range | Failures caught | False alerts on passes | Samples flagged |
|---|---|---|---|---|---|---|---|---|
| Complete selected procedure | 30.86% | 63.64% | 74.65% | 6.18 pp | 22.82%–41.64% | 66 | 371 | 27.9% |
| Measurements only | 29.99% | 65.64% | 74.37% | 4.99 pp | 22.82%–39.94% | 68 | 375 | 28.3% |
| Measurements + missing flags | 30.61% | 65.36% | 73.42% | 10.92 pp | 14.18%–43.90% | 68 | 389 | 29.2% |
| Missing flags only | 43.27% | 46.27% | 67.18% | 10.03 pp | 22.35%–58.36% | 48 | 480 | 33.7% |
| Always predict pass | 50.00% | 0.00% | 100.00% | 0.00 pp | 50.00%–50.00% | 0 | 0 | 0.0% |

## Original vs Tuned Benchmark Comparison

The complete selected procedure changes balanced error from **31.43% to 30.86%**, failure recall from **67.55% to 63.64%**, and pass specificity from **69.59% to 74.65%**. The pooled counts change from 70 failures caught / 445 false alerts to 66 caught / 371 false alerts. Tuning produces fewer failures caught and fewer false alerts. Read both rates and counts together; a balanced-error change alone does not establish uniform improvement.

On the paired test folds, tuning lowers balanced error in 5, raises it in 4, and ties in 1. Mean reduction: 0.57 percentage points. This is descriptive, not a significance test.

![Complete reference and tuned procedures: balanced error and pooled alert counts](figures/benchmark_comparison.png)

Both benchmarks used 10 held-out test folds. Complete procedures evaluated on unseen samples. Error bars show fold means and ranges; count panels pool held-out predictions. Fold spread is descriptive.

<details>
<summary>Supporting input-mode contrasts</summary>

![Balanced-error reduction after tuning](figures/tuned_vs_original_delta.png)

Positive values mean lower balanced error after tuning. The same folds and held-out samples support each contrast; no significance claim is made.

</details>

## Feature Stability and Interpretation

Most of the search selects existing columns. Imputation fills missing measurements and scaling puts inputs on comparable scales; the explicit feature-engineering contrast adds a flag recording whether each measurement was missing. Ratios, interactions, trends and process-informed features were not systematically explored. The engineering space is not exhausted.

The chart shows how often anonymous inputs were selected across overlapping training folds, for each study's exploratory lowest-error family. A family is a selector/model/input combination. This family was identified from test summaries for description and is not an independently validated champion. Column numbers are zero-based file positions, not named sensors. Missing flags can encode measurement-collection changes as well as process state.

Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof or validated process-driver identification. Stability across resamples matters more than a single full-fit ranking, and missing-indicator features are kept distinct from raw value features.

For example, missing flags for columns 109, 110, 111, 244, 245, 246, 382, 383, 384, 516, 517, 518 have identical missingness patterns. Their full-sample missing rate by month is 2008-07: 100.0%; 2008-08: 100.0%; 2008-09: 57.3%; 2008-10: 17.3%. These descriptive associations do not identify root causes or prove independent sensor effects.

![Anonymous-input selection frequency](figures/feature_stability.png)

Exploratory family stability and scaled-coefficient heuristics. Bars use selection frequency only; fitted coefficients remain in the technical appendix.

## Temporal Robustness Stress Test

The shuffled benchmark asks about other samples from this dataset. The chronological stress test asks whether earlier measurements and labels transfer to later calendar periods. It is secondary evidence and does not replace the benchmark result.

Fixed nonoverlapping calendar test blocks follow expanding earlier training regions. Model selection uses deterministic chronological splits inside the earlier fitting portion. A held-out portion of each training region supplies calibration scores for a retained model; the model is never refitted after its threshold is frozen. Ranking AUC is a supporting measure of how well scores order failures above passes; 0.5 is chance ordering, and AUC does not set the alert threshold.

### Later-period logistic-regression results

| Later test period | Procedure | Test samples / failures | Balanced error | Failure recall | Pass specificity | Ranking AUC | Failures caught | False alerts on passes | Samples flagged |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Complete selected logistic-regression procedure | 244 / 7 | 50.06% | 28.57% | 71.31% | 0.462 | 2 | 68 | 28.7% |
| 2 | Complete selected logistic-regression procedure | 299 / 9 | 50.69% | 0.00% | 98.62% | 0.579 | 0 | 4 | 1.3% |
| 3 | Complete selected logistic-regression procedure | 207 / 17 | 50.98% | 76.47% | 21.58% | 0.402 | 13 | 149 | 78.3% |

Mean per-period balanced error: 50.58%. Per-period results matter: pooling samples can conceal weak transfer when failure prevalence and alert rates differ between periods.

### Kernel-ridge comparison and calibration-window sensitivity

Kernel ridge uses the same fixed later test samples, a standard scaler and the shared bounded tuning grid. Each declared calibration-window sensitivity is independently selected using only its own earlier fitting portion. It changes both training size and calibration size; it is a compound sensitivity comparison, not an isolated threshold experiment. The two models also differ in grids and preprocessing, so this is no clean single-factor algorithm comparison.

| Procedure | Mean period balanced error | Period 1: error / AUC / flagged | Period 2: error / AUC / flagged | Period 3: error / AUC / flagged |
|---|---|---|---|---|
| Complete selected procedure (20% calibration) | 47.47% | 42.83% / 0.727 / 86.1% | 49.14% / 0.589 / 98.3% | 50.45% / 0.487 / 77.3% |
| Measurements only (20% calibration) | 46.48% | 42.83% / 0.727 / 86.1% | 43.39% / 0.610 / 53.8% | 53.24% / 0.469 / 29.5% |
| Measurements + missing flags (20% calibration) | 48.04% | 44.54% / 0.315 / 3.7% | 49.14% / 0.589 / 98.3% | 50.45% / 0.487 / 77.3% |
| Complete selected procedure (30% calibration) | 45.85% | 42.86% / 0.555 / 0.4% | 43.28% / 0.680 / 87.0% | 51.42% / 0.478 / 55.6% |
| Measurements only (30% calibration) | 47.33% | 42.86% / 0.555 / 0.4% | 47.72% / 0.492 / 6.7% | 51.42% / 0.478 / 55.6% |
| Measurements + missing flags (30% calibration) | 48.13% | 52.77% / 0.455 / 19.7% | 43.28% / 0.680 / 87.0% | 48.34% / 0.525 / 73.4% |

The main kernel-ridge procedure flags 88.5% of the 750 later samples. Its mean period balanced error is 47.47%; its differently weighted pooled balanced error is 50.34%. Flagged fraction measures alert volume alongside recall. Per-period AUC separates score-ordering weakness from threshold weakness. Compare declared window paths descriptively; no sensitivity arm is promoted as a winner.

### Calibration counts and threshold sensitivity

Removing one failed calibration example and recalibrating the same fixed scores checks threshold fragility. The resulting flagged-fraction range is evaluated on the full original calibration score set. It describes calibration instability only, not future uncertainty or a confidence interval. Fewer than ten calibration failures triggers a warning; it does not move a split boundary.

| Later test period | Calibration samples / failures / passes | Logistic regression flagged range | Kernel ridge (20%) flagged range |
|---|---|---|---|
| 1 | 117 / 4 / 113 | 0.9%–39.3% | 9.4%–57.3% |
| 2 | 166 / 3 / 163 | 1.2%–83.7% | 77.7%–94.6% |
| 3 | 225 / 6 / 219 | 20.0%–81.8% | 18.7%–84.4% |

The compact table shows selected logistic-regression and main-window kernel-ridge thresholds on matching later periods. [Full calibration diagnostics](#chronological-diagnostics) are in the **Temporal Robustness Stress Test** appendix section; expand it to see every predefined window/input procedure and final model. A single failure changes balanced error by 1/(2 × calibration failures). For illustration, three failures give a 16.7-percentage-point step. More calibration data also leaves less data for fitting. Calibration sensitivity and score ordering must be read together; a different threshold alone cannot establish stable transfer.

### The final later block: retrospective, frozen thresholds

The final 15% of samples was already exposed in full-dataset benchmarking and earlier reports. It is a retrospective check, not independent confirmation. Only logistic-regression roles and a multivariate statistical process control (MSPC) baseline are evaluated here. MSPC uses principal components fitted on earlier passing samples; its score and alert threshold are chosen on calibration data. No later-block kernel-ridge result is implied.

| Frozen rule | Failures caught | Failures missed | False alerts on passes | Passes left unflagged |
|---|---|---|---|---|
| Primary model: balanced-error threshold | 7 | 2 | 168 | 58 |
| Primary model: workload-limited threshold | 1 | 8 | 9 | 217 |
| Comparison model: balanced-error threshold | 2 | 7 | 17 | 209 |
| Comparison model: workload-limited threshold | 2 | 7 | 18 | 208 |

The later block contains 9 failures and 226 passes. Exact intervals in the appendix are conditional on a fixed model and independent Bernoulli trials; they exclude temporal dependence and model-selection uncertainty. Retrospective thresholds chosen from evaluation labels to reach 90% pass specificity remain a separate ranking diagnostic, not frozen operating performance.

![Later-block frozen alerts](figures/lockbox_vs_mspc.png)

The later block contains 9 failures and 226 passes. Existing frozen rules; class counts determine rate precision. No superiority or production claim follows.

### Measurement and score distribution checks

Raw-measurement distribution comparisons use the earlier model-fitting samples as their reference. Score comparisons use held-out calibration scores from the same retained model. Missing-rate and prevalence changes are descriptive. These references answer different questions; their heuristic warnings cannot establish causes or authorize superiority.

![Later measurement and score shifts](figures/temporal_drift.png)

Primary logistic-regression model: raw-feature stability index versus earlier fitting measurements; score-distribution test versus held-out calibration. Secondary descriptive evidence.

### Hypothetical workload and cost

The workload-limited threshold constrains the unweighted mean weekly flagged fraction on calibration data to 10%. It does not cap each week, or establish future workload. Cost ratios compare the assumed cost of missing a failure with the assumed cost of a false alert on a pass. Costs and capacity are hypothetical, not measured production outcomes.

![Calibration workload and hypothetical costs](figures/workload_cost_framing.png)

Calibration-only summaries used to choose operating thresholds. The mean-weekly policy is not an individual-week hard cap; hypothetical costs do not validate deployment value.

## Industrialization Gaps

- No stable device/tool/chamber identifier for unseen-device validation.
- No intervention or maintenance history.
- No explicit regime-change metadata.
- No downstream decision or action outcome data.
- Anonymous features limit process interpretation.
- Single-dataset evidence only.
- Operational framing in this report is illustrative, not production-validated.

### Next Data Requirements

- Next data collection should add device- or tool-level identifiers, intervention logs, and longer-horizon cross-context validation.
- A production-grade study would also require deployment decision objectives and cost accounting.
- Named measurements, verified pre-outcome timing and intervention outcomes are needed to test causal or process explanations.

## Conclusions and Next Data Requirements

Bounded tuning lowers mean balanced error by 0.57 percentage points (31.43% to 30.86%), with fewer failures caught (70 to 66) and fewer false alerts (445 to 371). These results describe the complete selection procedure under shuffled sampling; they do not establish uniform improvement. Later-period logistic-regression balanced error averages 50.58%. The main kernel-ridge stress test averages 47.47% balanced error. It flags most later samples (88.5%), which describes the alert burden alongside failure recall. The available later-period error means exceed the shuffled tuned benchmark: these stress tests do not reproduce its class-balanced performance. Their different training and calibration procedures limit direct algorithm comparisons. The selected main-window procedures have 3–6 failed calibration examples per period. Sparse calibration failures trigger threshold-fragility warnings; the recalibration ranges describe instability, not future uncertainty.

Named measurements, pre-outcome timing, device/tool context, intervention records and independent later data are needed before claims about stable operational benefit. There is no production-readiness, causal, early-warning or fresh confirmatory superiority claim.

## Technical Appendix

Machine field names and search tables below support reproducibility. They do not define additional headline results.

<details>
<summary>Original Replication Design</summary>

### Appendix: Original Replication Design

Recorded outer fold counts: reference 10; tuned 10. The protocol uses shuffled stratified outer folds and stratified inner cross-validation. Recorded inner folds: 3. Recorded seed: 42. Imputation, scaling, and selection are fitted within each inner training split. Pooled inner out-of-fold scores jointly select parameters and a BER threshold; ties use deterministic simplicity/order. The threshold is frozen before the chosen pipeline is refitted on outer training data and evaluated once on outer test data. The inner-OOF to outer-refit score-distribution difference remains a calibration limitation; no third nesting is claimed.

Recorded feature budgets: 40. Recorded ReliefF neighbor counts: 10. Recorded KRR regularization strengths: 0.1/1.0/10.0; kernel-width settings: automatic/0.01/0.1/1.0. Declared KRR configurations per selector budget: 12. Recorded logistic-regression regularization settings: 0.01/0.1/1.0/10.0. BER is the primary inner objective; AUC is supporting. Exact ties prefer fewer features, then stronger regularization (larger KRR alpha, smaller logistic C), followed by deterministic remaining order.


</details>

<details>
<summary>Original Replication Search Summary</summary>

### Appendix: Original Replication Search Summary

#### Original Search Space

| selector | classifier | mode | evaluated_configs | k_values | c_values | alpha_values | gamma_values | gamma_multiplier_values | n_neighbors_values |
|---|---|---|---|---|---|---|---|---|---|
| S2N | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| S2N | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| S2N | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| S2N | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| Ttest | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Ttest | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| Ttest | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Ttest | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| F-test | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| F-test | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| F-test | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| F-test | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| ReliefF | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 1 |
| ReliefF | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 1 |
| ReliefF | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 1 |
| ReliefF | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 1 |
| Gram-Schmidt | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Gram-Schmidt | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| Gram-Schmidt | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Gram-Schmidt | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| Pearson | krr | strict | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Pearson | logreg | strict | 4 | 1 | 4 | 0 | 0 | 0 | 0 |
| Pearson | krr | with_missing_indicators | 12 | 1 | 0 | 3 | 3 | 0 | 0 |
| Pearson | logreg | with_missing_indicators | 4 | 1 | 4 | 0 | 0 | 0 | 0 |

#### Original Selected Configurations

Modal configurations describe inner selections across folds and full-data interpretation fits, not a new performance estimate.

| selector | classifier | mode | k | C | alpha | gamma | gamma_multiplier | n_neighbors | selected_count | mean_inner_ROC_AUC | mean_inner_BER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ReliefF | krr | strict | 40 | n/a | 1.000 | 0.010 | n/a | 10.000 | 5 | 0.743 | 0.296 |
| F-test | krr | strict | 40 | n/a | 10.000 | 0.100 | n/a | n/a | 6 | 0.703 | 0.319 |
| Pearson | krr | strict | 40 | n/a | 10.000 | 0.100 | n/a | n/a | 6 | 0.703 | 0.319 |
| Ttest | krr | strict | 40 | n/a | 10.000 | 0.100 | n/a | n/a | 6 | 0.703 | 0.319 |
| F-test | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | n/a | 5 | 0.700 | 0.315 |
| Pearson | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | n/a | 5 | 0.700 | 0.316 |
| Ttest | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | n/a | 5 | 0.699 | 0.316 |
| ReliefF | logreg | with_missing_indicators | 40 | 0.100 | n/a | n/a | n/a | 10.000 | 6 | 0.715 | 0.306 |
| ReliefF | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | 10.000 | 4 | 0.729 | 0.291 |
| ReliefF | logreg | strict | 40 | 0.100 | n/a | n/a | n/a | 10.000 | 7 | 0.725 | 0.318 |
| F-test | logreg | strict | 40 | 0.010 | n/a | n/a | n/a | n/a | 7 | 0.689 | 0.330 |
| Pearson | logreg | strict | 40 | 0.010 | n/a | n/a | n/a | n/a | 7 | 0.689 | 0.330 |
| Ttest | logreg | strict | 40 | 0.010 | n/a | n/a | n/a | n/a | 7 | 0.689 | 0.330 |
| Gram-Schmidt | logreg | strict | 40 | 0.010 | n/a | n/a | n/a | n/a | 7 | 0.658 | 0.363 |
| S2N | logreg | with_missing_indicators | 40 | 0.010 | n/a | n/a | n/a | n/a | 4 | 0.685 | 0.341 |
| Ttest | logreg | with_missing_indicators | 40 | 0.010 | n/a | n/a | n/a | n/a | 4 | 0.701 | 0.322 |
| F-test | logreg | with_missing_indicators | 40 | 0.010 | n/a | n/a | n/a | n/a | 4 | 0.701 | 0.324 |
| Pearson | logreg | with_missing_indicators | 40 | 0.010 | n/a | n/a | n/a | n/a | 4 | 0.701 | 0.325 |
| S2N | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | n/a | 5 | 0.696 | 0.323 |
| Gram-Schmidt | logreg | with_missing_indicators | 40 | 0.100 | n/a | n/a | n/a | n/a | 4 | 0.637 | 0.376 |
| Gram-Schmidt | krr | strict | 40 | n/a | 10.000 | 0.010 | n/a | n/a | 4 | 0.675 | 0.344 |
| Gram-Schmidt | krr | with_missing_indicators | 40 | n/a | 10.000 | n/a | n/a | n/a | 4 | 0.666 | 0.351 |
| S2N | logreg | strict | 40 | 0.010 | n/a | n/a | n/a | n/a | 4 | 0.676 | 0.352 |
| S2N | krr | strict | 40 | n/a | 10.000 | n/a | n/a | n/a | 4 | 0.716 | 0.304 |


</details>

<details>
<summary>Original Replication Results</summary>

### Appendix: Original Replication Results

#### Joint Held-out Procedure and Baselines

Headline source: `benchmark_procedure_summary.csv`, recomputed from `benchmark_predictions.csv`. Fold mean, standard deviation, and range are descriptive because training samples overlap. They are not algorithm-performance confidence intervals.

| procedure | mean_BER | std_BER | min_BER | max_BER | mean_True+ | mean_True- | pooled_TP | pooled_FP | pooled_TN | pooled_FN |
|---|---|---|---|---|---|---|---|---|---|---|
| joint | 0.314 | 0.080 | 0.152 | 0.428 | 0.675 | 0.696 | 70 | 445 | 1018 | 34 |
| values_only | 0.327 | 0.069 | 0.228 | 0.428 | 0.734 | 0.612 | 76 | 568 | 895 | 28 |
| values_and_indicators | 0.301 | 0.081 | 0.152 | 0.421 | 0.684 | 0.714 | 71 | 418 | 1045 | 33 |
| missingness_only | 0.433 | 0.100 | 0.223 | 0.584 | 0.463 | 0.672 | 48 | 480 | 983 | 56 |
| all_pass | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 | 1.000 | 0 | 0 | 1463 | 104 |

#### Exploratory Nested Family Comparisons

| selector | classifier | mode | mean_BER | fold_min | fold_max | mean_TPR | mean_TNR |
|---|---|---|---|---|---|---|---|
| ReliefF | krr | with_missing_indicators | 0.306 | 0.152 | 0.427 | 0.655 | 0.734 |
| F-test | krr | strict | 0.308 | 0.156 | 0.428 | 0.751 | 0.632 |
| Pearson | krr | strict | 0.308 | 0.156 | 0.428 | 0.751 | 0.632 |
| Ttest | krr | strict | 0.308 | 0.156 | 0.428 | 0.751 | 0.632 |
| ReliefF | krr | strict | 0.309 | 0.203 | 0.418 | 0.722 | 0.660 |
| ReliefF | logreg | with_missing_indicators | 0.315 | 0.116 | 0.476 | 0.639 | 0.731 |
| F-test | krr | with_missing_indicators | 0.316 | 0.136 | 0.421 | 0.713 | 0.656 |
| Pearson | krr | with_missing_indicators | 0.316 | 0.136 | 0.425 | 0.713 | 0.656 |
| Ttest | krr | with_missing_indicators | 0.316 | 0.136 | 0.425 | 0.713 | 0.656 |
| Ttest | logreg | with_missing_indicators | 0.321 | 0.201 | 0.478 | 0.674 | 0.685 |
| F-test | logreg | with_missing_indicators | 0.321 | 0.201 | 0.478 | 0.674 | 0.684 |
| Pearson | logreg | with_missing_indicators | 0.321 | 0.201 | 0.478 | 0.674 | 0.684 |
| F-test | logreg | strict | 0.339 | 0.190 | 0.499 | 0.693 | 0.630 |
| Pearson | logreg | strict | 0.339 | 0.190 | 0.499 | 0.693 | 0.630 |
| Ttest | logreg | strict | 0.339 | 0.190 | 0.499 | 0.693 | 0.630 |
| S2N | krr | with_missing_indicators | 0.343 | 0.248 | 0.471 | 0.683 | 0.631 |
| ReliefF | logreg | strict | 0.348 | 0.228 | 0.465 | 0.636 | 0.669 |
| Gram-Schmidt | krr | with_missing_indicators | 0.349 | 0.254 | 0.537 | 0.673 | 0.630 |
| Gram-Schmidt | logreg | strict | 0.357 | 0.217 | 0.540 | 0.645 | 0.642 |
| S2N | logreg | with_missing_indicators | 0.358 | 0.270 | 0.435 | 0.740 | 0.544 |
| Gram-Schmidt | krr | strict | 0.359 | 0.276 | 0.538 | 0.606 | 0.677 |
| S2N | krr | strict | 0.364 | 0.247 | 0.504 | 0.674 | 0.597 |
| Gram-Schmidt | logreg | with_missing_indicators | 0.378 | 0.277 | 0.523 | 0.607 | 0.638 |
| S2N | logreg | strict | 0.418 | 0.307 | 0.566 | 0.624 | 0.540 |

#### Supporting Benchmark Metrics

| selector | classifier | mode | mean_ROC_AUC | mean_PR_AUC | mean_MCC | mean_F2 |
|---|---|---|---|---|---|---|
| ReliefF | krr | with_missing_indicators | 0.741 | 0.231 | 0.217 | 0.390 |
| F-test | krr | strict | 0.733 | 0.219 | 0.197 | 0.378 |
| Pearson | krr | strict | 0.733 | 0.219 | 0.197 | 0.378 |
| Ttest | krr | strict | 0.733 | 0.219 | 0.197 | 0.378 |
| ReliefF | krr | strict | 0.748 | 0.227 | 0.202 | 0.384 |
| ReliefF | logreg | with_missing_indicators | 0.752 | 0.255 | 0.203 | 0.377 |
| F-test | krr | with_missing_indicators | 0.738 | 0.240 | 0.193 | 0.374 |
| Pearson | krr | with_missing_indicators | 0.738 | 0.240 | 0.193 | 0.374 |
| Ttest | krr | with_missing_indicators | 0.738 | 0.240 | 0.193 | 0.374 |
| Ttest | logreg | with_missing_indicators | 0.726 | 0.198 | 0.192 | 0.370 |
| F-test | logreg | with_missing_indicators | 0.726 | 0.198 | 0.191 | 0.369 |
| Pearson | logreg | with_missing_indicators | 0.726 | 0.198 | 0.191 | 0.369 |
| F-test | logreg | strict | 0.723 | 0.188 | 0.166 | 0.350 |
| Pearson | logreg | strict | 0.723 | 0.188 | 0.166 | 0.350 |
| Ttest | logreg | strict | 0.723 | 0.188 | 0.166 | 0.350 |
| S2N | krr | with_missing_indicators | 0.717 | 0.185 | 0.163 | 0.349 |
| ReliefF | logreg | strict | 0.731 | 0.214 | 0.160 | 0.340 |
| Gram-Schmidt | krr | with_missing_indicators | 0.713 | 0.216 | 0.153 | 0.335 |
| Gram-Schmidt | logreg | strict | 0.701 | 0.190 | 0.151 | 0.338 |
| S2N | logreg | with_missing_indicators | 0.691 | 0.175 | 0.144 | 0.333 |
| Gram-Schmidt | krr | strict | 0.703 | 0.227 | 0.153 | 0.334 |
| S2N | krr | strict | 0.709 | 0.180 | 0.138 | 0.326 |
| Gram-Schmidt | logreg | with_missing_indicators | 0.684 | 0.182 | 0.133 | 0.314 |
| S2N | logreg | strict | 0.675 | 0.178 | 0.081 | 0.274 |

#### Paired Missing-indicator Ablation

Positive delta_BER means values-only BER minus values-plus-indicators BER. Paired fold deltas are descriptive.

| selector | classifier | BER_reference | BER_missing_indicator | delta_BER | std_delta_BER | min_delta_BER | max_delta_BER |
|---|---|---|---|---|---|---|---|
| S2N | krr | 0.364 | 0.343 | 0.021 | 0.040 | -0.051 | 0.090 |
| S2N | logreg | 0.418 | 0.358 | 0.060 | 0.068 | -0.022 | 0.181 |
| Ttest | krr | 0.308 | 0.316 | -0.007 | 0.046 | -0.073 | 0.062 |
| Ttest | logreg | 0.339 | 0.321 | 0.018 | 0.035 | -0.036 | 0.084 |
| F-test | krr | 0.308 | 0.316 | -0.007 | 0.046 | -0.073 | 0.062 |
| F-test | logreg | 0.339 | 0.321 | 0.017 | 0.034 | -0.036 | 0.084 |
| ReliefF | krr | 0.309 | 0.306 | 0.003 | 0.073 | -0.155 | 0.094 |
| ReliefF | logreg | 0.348 | 0.315 | 0.033 | 0.059 | -0.046 | 0.125 |
| Gram-Schmidt | krr | 0.359 | 0.349 | 0.010 | 0.062 | -0.100 | 0.126 |
| Gram-Schmidt | logreg | 0.357 | 0.378 | -0.021 | 0.050 | -0.096 | 0.053 |
| Pearson | krr | 0.308 | 0.316 | -0.007 | 0.046 | -0.073 | 0.062 |
| Pearson | logreg | 0.339 | 0.321 | 0.017 | 0.034 | -0.036 | 0.084 |

#### UCI Original Benchmark Reference

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) describes KRR; [McCann and Johnston (2010), Table 2](https://proceedings.mlr.press/v6/mccann10a/mccann10a.pdf) labels its baseline Naive Bayes. These are reference context, not an exact classifier/protocol replication claim.

| UCI method | local selector | UCI BER % | UCI True+ % | UCI True- % | local BER % | local True+ % | local True- % |
|---|---|---|---|---|---|---|---|
| S2N | S2N | 34.5 +/- 2.6 | 57.8 +/- 5.3 | 73.1 +/- 2.1 | 36.4 | 67.4 | 59.7 |
| Ttest | Ttest | 33.7 +/- 2.1 | 59.6 +/- 4.7 | 73.0 +/- 1.8 | 30.8 | 75.1 | 63.2 |
| Relief | ReliefF | 40.1 +/- 2.8 | 48.3 +/- 5.9 | 71.6 +/- 3.2 | 30.9 | 72.2 | 66.0 |
| Pearson | Pearson | 34.1 +/- 2.0 | 57.4 +/- 4.3 | 74.4 +/- 4.9 | 30.8 | 75.1 | 63.2 |
| Ftest | F-test | 33.5 +/- 2.2 | 59.1 +/- 4.8 | 73.8 +/- 1.8 | 30.8 | 75.1 | 63.2 |
| Gram Schmidt | Gram-Schmidt | 35.6 +/- 2.4 | 51.2 +/- 11.8 | 77.5 +/- 2.3 | 35.9 | 60.6 | 67.7 |

Interpretation note: the local Ttest row uses a pooled two-sample t statistic to align with the UCI selector label; Welch-t remains available only as an explicit local selector. Binary-label ANOVA F-test ranking and absolute Pearson correlation ranking are mathematically monotonic for non-constant features, so they can select the same 40-feature set and produce identical local rows. The UCI reference table reports separate Ftest and Pearson rows, which should be read as that benchmark's implementation/protocol definitions rather than a guarantee that the two selectors are distinct under this replication.


</details>

<details>
<summary>Tuned Benchmark Design</summary>

### Appendix: Tuned Benchmark Design

Recorded outer fold counts: reference 10; tuned 10. The protocol uses shuffled stratified outer folds and stratified inner cross-validation. Recorded inner folds: 3. Recorded seed: 42. Imputation, scaling, and selection are fitted within each inner training split. Pooled inner out-of-fold scores jointly select parameters and a BER threshold; ties use deterministic simplicity/order. The threshold is frozen before the chosen pipeline is refitted on outer training data and evaluated once on outer test data. The inner-OOF to outer-refit score-distribution difference remains a calibration limitation; no third nesting is claimed.

Recorded feature budgets: 10/20/40. Recorded ReliefF neighbor counts: 5/10/20. Recorded KRR regularization strengths: 0.1/1.0/10.0/100.0; dimension-relative kernel-width multipliers: 0.1/0.2/1.0/2.0. Declared KRR configurations per selector budget: 16. Recorded logistic-regression regularization settings: 0.01/0.1/1.0/10.0. BER is the primary inner objective; AUC is supporting. Exact ties prefer fewer features, then stronger regularization (larger KRR alpha, smaller logistic C), followed by deterministic remaining order.


</details>

<details>
<summary>Tuned Benchmark Search Summary</summary>

### Appendix: Tuned Benchmark Search Summary

#### Tuned Search Space

| selector | classifier | mode | evaluated_configs | k_values | c_values | alpha_values | gamma_values | gamma_multiplier_values | n_neighbors_values |
|---|---|---|---|---|---|---|---|---|---|
| S2N | krr | strict | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| S2N | logreg | strict | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| S2N | krr | with_missing_indicators | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| S2N | logreg | with_missing_indicators | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| Ttest | krr | strict | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Ttest | logreg | strict | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| Ttest | krr | with_missing_indicators | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Ttest | logreg | with_missing_indicators | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| F-test | krr | strict | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| F-test | logreg | strict | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| F-test | krr | with_missing_indicators | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| F-test | logreg | with_missing_indicators | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| ReliefF | krr | strict | 1440 | 3 | 0 | 4 | 8 | 4 | 3 |
| ReliefF | logreg | strict | 360 | 3 | 4 | 0 | 0 | 0 | 3 |
| ReliefF | krr | with_missing_indicators | 1440 | 3 | 0 | 4 | 8 | 4 | 3 |
| ReliefF | logreg | with_missing_indicators | 360 | 3 | 4 | 0 | 0 | 0 | 3 |
| Gram-Schmidt | krr | strict | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Gram-Schmidt | logreg | strict | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| Gram-Schmidt | krr | with_missing_indicators | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Gram-Schmidt | logreg | with_missing_indicators | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| Pearson | krr | strict | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Pearson | logreg | strict | 120 | 3 | 4 | 0 | 0 | 0 | 0 |
| Pearson | krr | with_missing_indicators | 480 | 3 | 0 | 4 | 8 | 4 | 0 |
| Pearson | logreg | with_missing_indicators | 120 | 3 | 4 | 0 | 0 | 0 | 0 |

#### Modal Selected Configurations

Modal configurations describe inner selections across folds and full-data interpretation fits, not a new performance estimate.

| selector | classifier | mode | k | C | alpha | gamma | gamma_multiplier | n_neighbors | selected_count | mean_inner_ROC_AUC | mean_inner_BER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F-test | krr | with_missing_indicators | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 2 | 0.699 | 0.306 |
| Pearson | krr | with_missing_indicators | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 2 | 0.699 | 0.306 |
| Ttest | krr | with_missing_indicators | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 2 | 0.699 | 0.306 |
| F-test | krr | strict | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 3 | 0.699 | 0.305 |
| Pearson | krr | strict | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 3 | 0.699 | 0.305 |
| Ttest | krr | strict | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 3 | 0.699 | 0.305 |
| Gram-Schmidt | krr | strict | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 3 | 0.702 | 0.310 |
| ReliefF | krr | strict | 20 | n/a | 10.000 | 0.050 | 1.000 | 10.000 | 2 | 0.760 | 0.262 |
| S2N | krr | strict | 10 | n/a | 100.000 | 0.200 | 2.000 | n/a | 2 | 0.695 | 0.308 |
| F-test | logreg | strict | 10 | 1.000 | n/a | n/a | n/a | n/a | 3 | 0.690 | 0.314 |
| Pearson | logreg | strict | 10 | 1.000 | n/a | n/a | n/a | n/a | 3 | 0.690 | 0.314 |
| Ttest | logreg | strict | 10 | 1.000 | n/a | n/a | n/a | n/a | 3 | 0.690 | 0.314 |
| ReliefF | logreg | with_missing_indicators | 20 | 10.000 | n/a | n/a | n/a | 5.000 | 2 | 0.720 | 0.282 |
| Gram-Schmidt | logreg | with_missing_indicators | 20 | 0.010 | n/a | n/a | n/a | n/a | 2 | 0.663 | 0.346 |
| S2N | logreg | with_missing_indicators | 20 | 0.010 | n/a | n/a | n/a | n/a | 3 | 0.684 | 0.333 |
| S2N | krr | with_missing_indicators | 20 | n/a | 10.000 | 0.100 | 2.000 | n/a | 3 | 0.712 | 0.300 |
| Gram-Schmidt | logreg | strict | 10 | 0.010 | n/a | n/a | n/a | n/a | 3 | 0.679 | 0.337 |
| ReliefF | krr | with_missing_indicators | 40 | n/a | 10.000 | 0.050 | 2.000 | 10.000 | 3 | 0.737 | 0.282 |
| ReliefF | logreg | strict | 20 | 10.000 | n/a | n/a | n/a | 5.000 | 2 | 0.744 | 0.280 |
| F-test | logreg | with_missing_indicators | 10 | 0.100 | n/a | n/a | n/a | n/a | 2 | 0.703 | 0.303 |
| Pearson | logreg | with_missing_indicators | 10 | 0.100 | n/a | n/a | n/a | n/a | 2 | 0.703 | 0.303 |
| Ttest | logreg | with_missing_indicators | 10 | 0.100 | n/a | n/a | n/a | n/a | 2 | 0.703 | 0.303 |
| Gram-Schmidt | krr | with_missing_indicators | 40 | n/a | 0.100 | 0.003 | 0.100 | n/a | 2 | 0.701 | 0.317 |
| S2N | logreg | strict | 40 | 1.000 | n/a | n/a | n/a | n/a | 2 | 0.693 | 0.321 |


</details>

<details>
<summary>Tuned Benchmark Results</summary>

### Appendix: Tuned Benchmark Results

#### Joint Held-out Procedure and Baselines

Headline source: `benchmark_tuned_procedure_summary.csv`, recomputed from `benchmark_tuned_predictions.csv`. Fold mean, standard deviation, and range are descriptive because training samples overlap. They are not algorithm-performance confidence intervals.

| procedure | mean_BER | std_BER | min_BER | max_BER | mean_True+ | mean_True- | pooled_TP | pooled_FP | pooled_TN | pooled_FN |
|---|---|---|---|---|---|---|---|---|---|---|
| joint | 0.309 | 0.062 | 0.228 | 0.416 | 0.636 | 0.746 | 66 | 371 | 1092 | 38 |
| values_only | 0.300 | 0.050 | 0.228 | 0.399 | 0.656 | 0.744 | 68 | 375 | 1088 | 36 |
| values_and_indicators | 0.306 | 0.109 | 0.142 | 0.439 | 0.654 | 0.734 | 68 | 389 | 1074 | 36 |
| missingness_only | 0.433 | 0.100 | 0.223 | 0.584 | 0.463 | 0.672 | 48 | 480 | 983 | 56 |
| all_pass | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 | 1.000 | 0 | 0 | 1463 | 104 |

#### Exploratory Nested Family Comparisons

| selector | classifier | mode | mean_BER | fold_min | fold_max | mean_TPR | mean_TNR |
|---|---|---|---|---|---|---|---|
| F-test | krr | with_missing_indicators | 0.301 | 0.186 | 0.421 | 0.705 | 0.693 |
| Pearson | krr | with_missing_indicators | 0.301 | 0.186 | 0.425 | 0.705 | 0.692 |
| Ttest | krr | with_missing_indicators | 0.301 | 0.186 | 0.425 | 0.705 | 0.692 |
| F-test | krr | strict | 0.307 | 0.186 | 0.428 | 0.705 | 0.682 |
| Pearson | krr | strict | 0.307 | 0.186 | 0.428 | 0.705 | 0.682 |
| Ttest | krr | strict | 0.307 | 0.186 | 0.428 | 0.705 | 0.682 |
| ReliefF | krr | strict | 0.309 | 0.228 | 0.399 | 0.636 | 0.745 |
| F-test | logreg | with_missing_indicators | 0.310 | 0.193 | 0.478 | 0.665 | 0.716 |
| Pearson | logreg | with_missing_indicators | 0.310 | 0.193 | 0.478 | 0.665 | 0.716 |
| Ttest | logreg | with_missing_indicators | 0.310 | 0.193 | 0.478 | 0.665 | 0.716 |
| F-test | logreg | strict | 0.314 | 0.216 | 0.428 | 0.705 | 0.666 |
| Pearson | logreg | strict | 0.314 | 0.216 | 0.428 | 0.705 | 0.666 |
| Ttest | logreg | strict | 0.314 | 0.216 | 0.428 | 0.705 | 0.666 |
| ReliefF | krr | with_missing_indicators | 0.318 | 0.162 | 0.423 | 0.634 | 0.729 |
| Gram-Schmidt | krr | strict | 0.323 | 0.222 | 0.458 | 0.657 | 0.696 |
| ReliefF | logreg | with_missing_indicators | 0.325 | 0.142 | 0.486 | 0.587 | 0.763 |
| S2N | krr | with_missing_indicators | 0.329 | 0.216 | 0.464 | 0.665 | 0.678 |
| ReliefF | logreg | strict | 0.335 | 0.161 | 0.459 | 0.585 | 0.745 |
| S2N | krr | strict | 0.337 | 0.185 | 0.508 | 0.664 | 0.662 |
| Gram-Schmidt | krr | with_missing_indicators | 0.341 | 0.216 | 0.430 | 0.626 | 0.692 |
| S2N | logreg | with_missing_indicators | 0.343 | 0.213 | 0.439 | 0.674 | 0.640 |
| Gram-Schmidt | logreg | with_missing_indicators | 0.349 | 0.285 | 0.428 | 0.596 | 0.706 |
| Gram-Schmidt | logreg | strict | 0.350 | 0.217 | 0.453 | 0.578 | 0.721 |
| S2N | logreg | strict | 0.381 | 0.206 | 0.566 | 0.635 | 0.603 |

#### Supporting Benchmark Metrics

| selector | classifier | mode | mean_ROC_AUC | mean_PR_AUC | mean_MCC | mean_F2 |
|---|---|---|---|---|---|---|
| F-test | krr | with_missing_indicators | 0.730 | 0.204 | 0.213 | 0.389 |
| Pearson | krr | with_missing_indicators | 0.730 | 0.204 | 0.213 | 0.389 |
| Ttest | krr | with_missing_indicators | 0.730 | 0.204 | 0.213 | 0.389 |
| F-test | krr | strict | 0.725 | 0.197 | 0.205 | 0.385 |
| Pearson | krr | strict | 0.725 | 0.197 | 0.205 | 0.385 |
| Ttest | krr | strict | 0.725 | 0.197 | 0.205 | 0.385 |
| ReliefF | krr | strict | 0.740 | 0.235 | 0.218 | 0.391 |
| F-test | logreg | with_missing_indicators | 0.737 | 0.191 | 0.209 | 0.387 |
| Pearson | logreg | with_missing_indicators | 0.737 | 0.191 | 0.209 | 0.387 |
| Ttest | logreg | with_missing_indicators | 0.737 | 0.191 | 0.209 | 0.387 |
| F-test | logreg | strict | 0.737 | 0.200 | 0.199 | 0.378 |
| Pearson | logreg | strict | 0.737 | 0.200 | 0.199 | 0.378 |
| Ttest | logreg | strict | 0.737 | 0.200 | 0.199 | 0.378 |
| ReliefF | krr | with_missing_indicators | 0.744 | 0.241 | 0.204 | 0.379 |
| Gram-Schmidt | krr | strict | 0.712 | 0.202 | 0.191 | 0.372 |
| ReliefF | logreg | with_missing_indicators | 0.728 | 0.223 | 0.204 | 0.375 |
| S2N | krr | with_missing_indicators | 0.708 | 0.172 | 0.181 | 0.363 |
| ReliefF | logreg | strict | 0.724 | 0.217 | 0.189 | 0.363 |
| S2N | krr | strict | 0.704 | 0.189 | 0.171 | 0.352 |
| Gram-Schmidt | krr | with_missing_indicators | 0.707 | 0.192 | 0.171 | 0.350 |
| S2N | logreg | with_missing_indicators | 0.723 | 0.192 | 0.163 | 0.349 |
| Gram-Schmidt | logreg | with_missing_indicators | 0.710 | 0.201 | 0.168 | 0.341 |
| Gram-Schmidt | logreg | strict | 0.711 | 0.198 | 0.167 | 0.340 |
| S2N | logreg | strict | 0.692 | 0.178 | 0.120 | 0.308 |

#### Paired Missing-indicator Ablation

Positive delta_BER means values-only BER minus values-plus-indicators BER. Paired fold deltas are descriptive.

| selector | classifier | BER_reference | BER_missing_indicator | delta_BER | std_delta_BER | min_delta_BER | max_delta_BER |
|---|---|---|---|---|---|---|---|
| S2N | krr | 0.337 | 0.329 | 0.008 | 0.045 | -0.083 | 0.067 |
| S2N | logreg | 0.381 | 0.343 | 0.038 | 0.085 | -0.067 | 0.181 |
| Ttest | krr | 0.307 | 0.301 | 0.006 | 0.018 | -0.018 | 0.041 |
| Ttest | logreg | 0.314 | 0.310 | 0.005 | 0.037 | -0.088 | 0.043 |
| F-test | krr | 0.307 | 0.301 | 0.006 | 0.018 | -0.018 | 0.041 |
| F-test | logreg | 0.314 | 0.310 | 0.005 | 0.037 | -0.088 | 0.043 |
| ReliefF | krr | 0.309 | 0.318 | -0.009 | 0.071 | -0.136 | 0.084 |
| ReliefF | logreg | 0.335 | 0.325 | 0.010 | 0.049 | -0.070 | 0.112 |
| Gram-Schmidt | krr | 0.323 | 0.341 | -0.017 | 0.040 | -0.103 | 0.029 |
| Gram-Schmidt | logreg | 0.350 | 0.349 | 0.002 | 0.033 | -0.068 | 0.053 |
| Pearson | krr | 0.307 | 0.301 | 0.006 | 0.018 | -0.018 | 0.041 |
| Pearson | logreg | 0.314 | 0.310 | 0.005 | 0.037 | -0.088 | 0.043 |

</details>

<details>
<summary>Original vs Tuned Benchmark Comparison</summary>

### Appendix: Original vs Tuned Benchmark Comparison

The procedures use identical held-out sample IDs/folds. Positive paired delta below is original BER minus tuned BER; the spread is descriptive, not a significance test.

Joint paired delta_BER mean=0.006, std=0.047, range=-0.076 to 0.077.

</details>

<details>
<summary>Feature Stability and Interpretation</summary>

### Appendix: Feature Stability and Interpretation

Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof or validated process-driver identification. Stability across resamples matters more than a single full-fit ranking, and missing-indicator features are kept distinct from raw value features.

Selection frequency across overlapping outer training folds is descriptive. Full-data coefficients are in-sample associations. absolute_scaled_coefficient is the absolute full-fit scaled logistic coefficient; stability_weighted_coefficient is frequency times that coefficient. Neither is an expected economic contribution or causal effect. KRR coefficient fields remain unavailable.

#### Original Exploratory Family Feature Interpretation

- Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof or validated process-driver identification. Stability across resamples matters more than a single full-fit ranking, and missing-indicator features are kept distinct from raw value features.
- Scaled coefficient magnitudes are unavailable for this exploratory family; the table shows stability.

| feature | type | selection_frequency | cluster_id |
|---|---|---:|---:|
| M112 | missing_indicator | 1.000 | n/a |
| M247 | missing_indicator | 1.000 | n/a |
| M345 | missing_indicator | 1.000 | n/a |
| M346 | missing_indicator | 1.000 | n/a |
| M385 | missing_indicator | 1.000 | n/a |
| M519 | missing_indicator | 1.000 | n/a |
| M578 | missing_indicator | 1.000 | n/a |
| M579 | missing_indicator | 1.000 | n/a |
| M580 | missing_indicator | 1.000 | n/a |
| M581 | missing_indicator | 1.000 | n/a |
#### Tuned Exploratory Family Feature Interpretation

- Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof or validated process-driver identification. Stability across resamples matters more than a single full-fit ranking, and missing-indicator features are kept distinct from raw value features.
- Scaled coefficient magnitudes are unavailable for this exploratory family; the table shows stability.

| feature | type | selection_frequency | cluster_id |
|---|---|---:|---:|
| X103 | value | 1.000 | 100.000 |
| X21 | value | 1.000 | 21.000 |
| X348 | value | 1.000 | 268.000 |
| X434 | value | 1.000 | 304.000 |
| X510 | value | 1.000 | 350.000 |
| X59 | value | 1.000 | 57.000 |
| X129 | value | 0.900 | 122.000 |
| X430 | value | 0.900 | 304.000 |
| X431 | value | 0.900 | 305.000 |
| X435 | value | 0.900 | 304.000 |
#### Missingness Context

These full-sample diagnostics explain selected missing indicators; they are not additional model validation or causal evidence.

- `M112`, `M247`, `M385`, `M519` have identical missingness masks. Missing rate by month: 2008-07: 31.7%; 2008-08: 17.5%; 2008-09: 46.9%; 2008-10: 89.4%. Overall pass/fail missing rates: 46.6% / 31.7%.
- `M72`, `M73`, `M345`, `M346` have identical missingness masks. Missing rate by month: 2008-07: 15.9%; 2008-08: 32.3%; 2008-09: 62.0%; 2008-10: 66.6%. Overall pass/fail missing rates: 51.8% / 34.6%.
- `M578`, `M579`, `M580`, `M581` have identical missingness masks. Missing rate by month: 2008-07: 23.8%; 2008-08: 62.2%; 2008-09: 68.6%; 2008-10: 51.3%. Overall pass/fail missing rates: 60.8% / 56.7%.

Co-occurring missingness can encode acquisition regime or time as well as process state. Repeated selection does not identify independent sensor effects, root causes, or a stable deployment signal.

</details>

### Chronological diagnostics

<details>
<summary>Temporal Robustness Stress Test</summary>

### Appendix: Temporal Robustness Stress Test

Temporal robustness status: `warning`

#### Temporal Robustness Design

The retained logistic-regression role study and a bounded DEV-only KRR comparator are separate secondary stress studies. The last chronological 15% is a retrospective later evaluation block already exposed through the full-dataset benchmark and earlier reports; it is not a fresh confirmatory lockbox. DEV uses fixed nonoverlapping calendar test blocks and expanding training prefixes. Inner tuning uses deterministic chronological splits. The last chronological 20% of each training region is held-out calibration. Tuning/model fitting use the earlier fit prefix; thresholds use calibration scores from that retained model, with no refit on calibration after freezing.

#### DEV-only KRR and Calibration Sensitivity

KRR uses the shared tuned alpha grid and dimension-relative gamma multipliers with StandardScaler. Joint input mode and configuration are chosen by earlier FIT chronological inner BER. Values-only and combined procedures use the same fixed evaluation periods. The predeclared 30% calibration sensitivity tunes independently within its earlier 70% FIT. It is descriptive and cannot promote an outer-period winner or a later-block KRR champion.

| fold | procedure | available | unavailable_reason | BER | ROC_AUC | n_inner_timestamp_ties | True+ | True- | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | krr_cal20_joint | True | n/a | 0.428 | 0.727 | 0 | 1.000 | 0.143 | 7 | 34 | 203 | 0 |
| 1 | krr_cal20_values_only | True | n/a | 0.428 | 0.727 | 0 | 1.000 | 0.143 | 7 | 34 | 203 | 0 |
| 1 | krr_cal20_values_and_indicators | True | n/a | 0.445 | 0.315 | 0 | 0.143 | 0.966 | 1 | 229 | 8 | 6 |
| 2 | krr_cal20_joint | True | n/a | 0.491 | 0.589 | 0 | 1.000 | 0.017 | 9 | 5 | 285 | 0 |
| 2 | krr_cal20_values_only | True | n/a | 0.434 | 0.610 | 0 | 0.667 | 0.466 | 6 | 135 | 155 | 3 |
| 2 | krr_cal20_values_and_indicators | True | n/a | 0.491 | 0.589 | 0 | 1.000 | 0.017 | 9 | 5 | 285 | 0 |
| 3 | krr_cal20_joint | True | n/a | 0.504 | 0.487 | 0 | 0.765 | 0.226 | 13 | 43 | 147 | 4 |
| 3 | krr_cal20_values_only | True | n/a | 0.532 | 0.469 | 0 | 0.235 | 0.700 | 4 | 133 | 57 | 13 |
| 3 | krr_cal20_values_and_indicators | True | n/a | 0.504 | 0.487 | 0 | 0.765 | 0.226 | 13 | 43 | 147 | 4 |
| 1 | krr_cal30_joint | True | n/a | 0.429 | 0.555 | 0 | 0.143 | 1.000 | 1 | 237 | 0 | 6 |
| 1 | krr_cal30_values_only | True | n/a | 0.429 | 0.555 | 0 | 0.143 | 1.000 | 1 | 237 | 0 | 6 |
| 1 | krr_cal30_values_and_indicators | True | n/a | 0.528 | 0.455 | 0 | 0.143 | 0.802 | 1 | 190 | 47 | 6 |
| 2 | krr_cal30_joint | True | n/a | 0.433 | 0.680 | 0 | 1.000 | 0.134 | 9 | 39 | 251 | 0 |
| 2 | krr_cal30_values_only | True | n/a | 0.477 | 0.492 | 0 | 0.111 | 0.934 | 1 | 271 | 19 | 8 |
| 2 | krr_cal30_values_and_indicators | True | n/a | 0.433 | 0.680 | 0 | 1.000 | 0.134 | 9 | 39 | 251 | 0 |
| 3 | krr_cal30_joint | True | n/a | 0.514 | 0.478 | 1 | 0.529 | 0.442 | 9 | 84 | 106 | 8 |
| 3 | krr_cal30_values_only | True | n/a | 0.514 | 0.478 | 1 | 0.529 | 0.442 | 9 | 84 | 106 | 8 |
| 3 | krr_cal30_values_and_indicators | True | n/a | 0.483 | 0.525 | 1 | 0.765 | 0.268 | 13 | 51 | 139 | 4 |

#### Calibration Counts and Threshold Fragility

Fewer than ten calibration failures is a fragility warning, not a reason to move a chronological boundary. BER steps are 1/(2 failures) and 1/(2 passes). Leave-one-failure-out recalibration keeps fitted scores fixed; flagged-fraction ranges use the same full calibration score set. LOFO ranges are undefined with fewer than two failures or no passes; class-specific BER steps are undefined when that class is absent. These ranges describe calibration instability only, not confidence intervals, future uncertainty or algorithm uncertainty. Per-period AUC distinguishes ranking weakness from threshold weakness. Inner rows follow stable (timestamp, raw_row_id) order; equal-timestamp boundaries are flagged in search/calibration receipts. Disjoint IDs and nondecreasing timestamps preserve the fixed splits. Raw row ID does not establish physical event order or independence within tied timestamps. Outer calendar tests remain strictly later.

| fold | procedure | calibration_n | calibration_fails | calibration_passes | BER_step_failure | BER_step_pass | threshold | fragile_calibration | fit_calibration_timestamp_tie | lofo_available | lofo_threshold_min | lofo_threshold_max | lofo_flagged_fraction_min | lofo_flagged_fraction_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | temporal_joint | 117 | 4 | 113 | 0.125 | 0.004 | 0.503 | True | False | True | 0.503 | 0.885 | 0.009 | 0.393 |
| 2 | temporal_joint | 166 | 3 | 163 | 0.167 | 0.003 | 0.683 | True | False | True | 0.133 | 0.683 | 0.012 | 0.837 |
| 3 | temporal_joint | 225 | 6 | 219 | 0.083 | 0.002 | 0.087 | True | False | True | 0.087 | 0.472 | 0.200 | 0.818 |
| 0 | logreg_final_primary | 267 | 17 | 250 | 0.029 | 0.002 | 0.185 | False | False | True | 0.185 | 0.188 | 0.566 | 0.577 |
| 0 | logreg_final_challenger | 267 | 17 | 250 | 0.029 | 0.002 | 0.508 | False | False | True | -inf | 0.559 | 0.041 | 1.000 |
| 1 | krr_cal20_joint | 117 | 4 | 113 | 0.125 | 0.004 | -0.783 | True | False | True | -0.783 | -0.258 | 0.094 | 0.573 |
| 1 | krr_cal20_values_only | 117 | 4 | 113 | 0.125 | 0.004 | -0.783 | True | False | True | -0.783 | -0.258 | 0.094 | 0.573 |
| 1 | krr_cal20_values_and_indicators | 117 | 4 | 113 | 0.125 | 0.004 | 0.004 | True | False | True | -0.130 | 0.004 | 0.009 | 0.991 |
| 2 | krr_cal20_joint | 166 | 3 | 163 | 0.167 | 0.003 | -1.199 | True | False | True | -1.199 | -0.907 | 0.777 | 0.946 |
| 2 | krr_cal20_values_only | 166 | 3 | 163 | 0.167 | 0.003 | -0.400 | True | False | True | -0.400 | -0.393 | 0.380 | 0.398 |
| 2 | krr_cal20_values_and_indicators | 166 | 3 | 163 | 0.167 | 0.003 | -1.199 | True | False | True | -1.199 | -0.907 | 0.777 | 0.946 |
| 3 | krr_cal20_joint | 225 | 6 | 219 | 0.083 | 0.002 | -0.534 | True | False | True | -0.534 | -0.126 | 0.187 | 0.844 |
| 3 | krr_cal20_values_only | 225 | 6 | 219 | 0.083 | 0.002 | -0.428 | True | False | True | -0.428 | -0.323 | 0.173 | 0.236 |
| 3 | krr_cal20_values_and_indicators | 225 | 6 | 219 | 0.083 | 0.002 | -0.534 | True | False | True | -0.534 | -0.126 | 0.187 | 0.844 |
| 1 | krr_cal30_joint | 175 | 7 | 168 | 0.071 | 0.003 | 0.145 | True | False | True | -0.280 | 0.145 | 0.023 | 0.771 |
| 1 | krr_cal30_values_only | 175 | 7 | 168 | 0.071 | 0.003 | 0.145 | True | False | True | -0.280 | 0.145 | 0.023 | 0.771 |
| 1 | krr_cal30_values_and_indicators | 175 | 7 | 168 | 0.071 | 0.003 | -0.002 | True | False | True | -0.002 | 1.48e-06 | 0.131 | 0.149 |
| 2 | krr_cal30_joint | 248 | 7 | 241 | 0.071 | 0.002 | -0.915 | True | False | True | -0.915 | 0.356 | 0.218 | 0.915 |
| 2 | krr_cal30_values_only | 248 | 7 | 241 | 0.071 | 0.002 | -0.008 | True | False | True | -0.248 | -1.98e-05 | 0.165 | 0.851 |
| 2 | krr_cal30_values_and_indicators | 248 | 7 | 241 | 0.071 | 0.002 | -0.915 | True | False | True | -0.915 | 0.356 | 0.218 | 0.915 |
| 3 | krr_cal30_joint | 338 | 11 | 327 | 0.045 | 0.002 | -0.798 | False | False | True | -0.798 | -0.715 | 0.482 | 0.589 |
| 3 | krr_cal30_values_only | 338 | 11 | 327 | 0.045 | 0.002 | -0.798 | False | False | True | -0.798 | -0.715 | 0.482 | 0.589 |
| 3 | krr_cal30_values_and_indicators | 338 | 11 | 327 | 0.045 | 0.002 | -0.161 | False | False | True | -0.161 | -0.077 | 0.172 | 0.589 |

#### Temporal Joint Held-out Procedure

| fold | procedure | ROC_AUC | BER | True+ | True- | TP | TN | FP | FN | n_test | n_test_fails | TPR_available | TNR_available | BER_available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | temporal_joint | 0.462 | 0.501 | 0.286 | 0.713 | 2 | 169 | 68 | 5 | 244 | 7 | True | True | True |
| 2 | temporal_joint | 0.579 | 0.507 | 0.000 | 0.986 | 0 | 286 | 4 | 9 | 299 | 9 | True | True | True |
| 3 | temporal_joint | 0.402 | 0.510 | 0.765 | 0.216 | 13 | 41 | 149 | 4 | 207 | 17 | True | True | True |

#### Temporal Model Selection Summary

Roles are chosen from chronological inner selection on the final fit prefix. Outer family ranking remains exploratory.

##### Selector Ranking and Modal Configurations

| selector | status | mean_BER | mean_True+ | mean_True- | modal_k | modal_C | modal_scaler | modal_n_neighbors |
|---|---|---|---|---|---|---|---|---|
| S2N | primary | 0.531 | 0.455 | 0.482 | 10 | 0.010 | RobustScaler | n/a |
| ReliefF | challenger | 0.535 | 0.252 | 0.677 | 10 | 0.010 | RobustScaler | 10.000 |
| Gram-Schmidt | supporting | 0.497 | 0.694 | 0.312 | 10 | 0.100 | RobustScaler | n/a |
| F-test | supporting | 0.515 | 0.330 | 0.639 | 20 | 0.010 | RobustScaler | n/a |
| Ttest | supporting | 0.515 | 0.330 | 0.639 | 20 | 0.010 | RobustScaler | n/a |

#### Lockbox Metrics

Frozen-threshold confusion counts accompany rates and exact binomial TPR/TNR intervals where available. When a class is absent, guarded numerical rate placeholders are marked unavailable; they are not evidence of that class recall. Intervals are conditional on a fixed model and independent Bernoulli trials; temporal dependence and model-selection uncertainty are excluded. TNR90 thresholds selected from evaluation labels remain retrospective ROC diagnostics.

| role | selector | threshold_policy | threshold_value | BER | True+ | True- | ROC_AUC | PR_AUC | MCC | F2 | lockbox_n | lockbox_fails | TPR_available | TNR_available | BER_available | low_failure_count_warning | threshold_at_TNR90 | TNR_at_TNR90 | TPR_at_TNR90 | TP | TN | FP | FN | TPR_exact_lower | TPR_exact_upper | TNR_exact_lower | TNR_exact_upper | interval_semantics | evaluation_semantics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | S2N | scientific | 0.185 | 0.483 | 0.778 | 0.257 | 0.618 | 0.102 | 0.015 | 0.166 | 235 | 9 | True | True | True | True | 0.397 | 0.903 | 0.333 | 7 | 58 | 168 | 2 | 0.400 | 0.972 | 0.201 | 0.319 | conditional_fixed_model_independent_trials_only | retrospective_later_block |
| primary | S2N | operational | 0.449 | 0.464 | 0.111 | 0.960 | 0.618 | 0.102 | 0.068 | 0.109 | 235 | 9 | True | True | True | True | 0.397 | 0.903 | 0.333 | 1 | 217 | 9 | 8 | 0.003 | 0.482 | 0.926 | 0.982 | conditional_fixed_model_independent_trials_only | retrospective_later_block |
| challenger | ReliefF | scientific | 0.508 | 0.426 | 0.222 | 0.925 | 0.640 | 0.237 | 0.103 | 0.182 | 235 | 9 | True | True | True | True | 0.466 | 0.903 | 0.333 | 2 | 209 | 17 | 7 | 0.028 | 0.600 | 0.882 | 0.956 | conditional_fixed_model_independent_trials_only | retrospective_later_block |
| challenger | ReliefF | operational | 0.504 | 0.429 | 0.222 | 0.920 | 0.640 | 0.237 | 0.098 | 0.179 | 235 | 9 | True | True | True | True | 0.466 | 0.903 | 0.333 | 2 | 208 | 18 | 7 | 0.028 | 0.600 | 0.877 | 0.952 | conditional_fixed_model_independent_trials_only | retrospective_later_block |

#### Drift and Claim Restrictions

KS compares held-out calibration scores from the same retained model with future scores. Raw-feature PSI uses the fit reference descriptively. Missingness-rate changes indicate collection-regime association, not causes. Heuristic gates cannot authorize superiority or confirmatory claims.

| model_scope | dev_fail_rate | lockbox_fail_rate | abs_prevalence_shift | ks_pvalue_scores | max_PSI | median_PSI | psi_feature_count | drift_gate_status | confirmatory_claims_allowed | score_reference | selected_indicator_missingness_rates | missingness_reference | max_missingness_rate_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | 0.071 | 0.038 | 0.033 | 0.000425 | 4.052 | 2.447 | 6 | HIGH_SHIFT | False | held_out_calibration_same_retained_model | [{"feature": "M72", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M73", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M345", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M346", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}] | FIT_original_column_masks | 0.338 |
| challenger | 0.071 | 0.038 | 0.033 | 1.21e-09 | 4.224 | 1.350 | 10 | HIGH_SHIFT | False | held_out_calibration_same_retained_model | [{"feature": "M72", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M73", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M345", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M346", "fit_missing_rate": 0.43661971830985913, "later_missing_rate": 0.774468085106383}, {"feature": "M112", "fit_missing_rate": 0.2892018779342723, "later_missing_rate": 0.948936170212766}, {"feature": "M247", "fit_missing_rate": 0.2892018779342723, "later_missing_rate": 0.948936170212766}, {"feature": "M385", "fit_missing_rate": 0.2892018779342723, "later_missing_rate": 0.948936170212766}, {"feature": "M519", "fit_missing_rate": 0.2892018779342723, "later_missing_rate": 0.948936170212766}, {"feature": "M562", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M563", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M564", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M565", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M566", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M567", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M568", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M569", "fit_missing_rate": 0.25258215962441316, "later_missing_rate": 0.00425531914893617}, {"feature": "M578", "fit_missing_rate": 0.6366197183098592, "later_missing_rate": 0.4425531914893617}, {"feature": "M579", "fit_missing_rate": 0.6366197183098592, "later_missing_rate": 0.4425531914893617}, {"feature": "M580", "fit_missing_rate": 0.6366197183098592, "later_missing_rate": 0.4425531914893617}, {"feature": "M581", "fit_missing_rate": 0.6366197183098592, "later_missing_rate": 0.4425531914893617}] | FIT_original_column_masks | 0.660 |
- retrospective_later_block_not_fresh_confirmatory_lockbox
- no_production_readiness_or_superiority_claim
- primary_high_shift_blocks_lockbox_superiority_claim
- challenger_high_shift_blocks_lockbox_superiority_claim

#### Supervised vs MSPC

MSPC fits PCA on pass-only fit samples. Calibration freezes T2/Q source and thresholds before evaluation. Frozen confusion counts and retrospective TNR90 diagnostics are separate. Observed mean inter-alarm spacing across all samples is not in-control ARL0.

| eval_scope | fold_index | T2_AUC | Q_AUC | alarm_rate | observed_mean_inter_alarm_spacing | T2_TPR_at_TNR90 | Q_TPR_at_TNR90 | calibration_selected_MSPC_TPR_at_TNR90 | calibration_selected_MSPC_source | frozen_threshold | T2_frozen_threshold | Q_frozen_threshold | frozen_BER | frozen_TPR | frozen_TNR | TP | TN | FP | FN | TNR90_semantics | source_selection_region | T2_calibration_TPR_at_TNR90 | Q_calibration_TPR_at_TNR90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| outer_fold | 1 | 0.651 | 0.530 | 0.098 | 8.261 | 0.429 | 0.143 | 0.429 | T2 | 54.475 | 54.475 | 4413.270 | 0.431 | 0.143 | 0.996 | 1 | 236 | 1 | 6 | retrospective_evaluation_ROC_diagnostic | held_out_calibration | 0.250 | 0.250 |
| outer_fold | 2 | 0.378 | 0.584 | 0.174 | 5.765 | 0.000 | 0.111 | 0.000 | T2 | 25.144 | 25.144 | 66058.636 | 0.507 | 0.000 | 0.986 | 0 | 286 | 4 | 9 | retrospective_evaluation_ROC_diagnostic | held_out_calibration | 0.333 | 0.000 |
| outer_fold | 3 | 0.380 | 0.555 | 0.039 | 26.143 | 0.059 | 0.059 | 0.059 | T2 | 30.703 | 30.703 | 2699.252 | 0.516 | 0.000 | 0.968 | 0 | 184 | 6 | 17 | retrospective_evaluation_ROC_diagnostic | held_out_calibration | 0.000 | 0.000 |
| lockbox | LOCKBOX | 0.538 | 0.401 | 0.013 | 50.500 | 0.111 | 0.111 | 0.111 | T2 | 26327.453 | 26327.453 | 3181164.124 | 0.507 | 0.000 | 0.987 | 0 | 223 | 3 | 9 | retrospective_evaluation_ROC_diagnostic | held_out_calibration | 0.059 | 0.059 |

#### Illustrative Operational Framing

Workload comes from held-out DEV calibration predictions. The operational policy constrains unweighted mean weekly flagged fraction to 10%, not every week's hard cap. Calibration workload is used to select that policy; future operating cost and production capacity remain unvalidated.

| evaluation_region | workload_semantics | role | selector | threshold_policy | predicted_flag_fraction | mean_weekly_flag_fraction | mean_weekly_flagged_samples | mean_weekly_fail_captures | mean_weekly_fail_misses |
|---|---|---|---|---|---|---|---|---|---|
| held_out_DEV_calibration | illustrative_mean_weekly_policy_not_per_week_cap | primary | S2N | scientific | 0.577 | 0.578 | 51.333 | 5.000 | 0.667 |
| held_out_DEV_calibration | illustrative_mean_weekly_policy_not_per_week_cap | primary | S2N | operational | 0.139 | 0.099 | 12.333 | 1.333 | 4.333 |
| held_out_DEV_calibration | illustrative_mean_weekly_policy_not_per_week_cap | challenger | ReliefF | scientific | 0.090 | 0.080 | 8.000 | 0.667 | 5.000 |
| held_out_DEV_calibration | illustrative_mean_weekly_policy_not_per_week_cap | challenger | ReliefF | operational | 0.101 | 0.099 | 9.000 | 0.667 | 5.000 |

##### Cost Curves

| cost_ratio | primary_scientific | primary_operational | challenger_scientific | challenger_operational | all_pass_baseline | all_flag_baseline |
|---|---|---|---|---|---|---|
| 1 | 0.723 | 0.072 | 0.102 | 0.106 | 0.038 | 0.962 |
| 2 | 0.732 | 0.106 | 0.132 | 0.136 | 0.077 | 0.962 |
| 5 | 0.757 | 0.209 | 0.221 | 0.226 | 0.191 | 0.962 |
| 10 | 0.800 | 0.379 | 0.370 | 0.374 | 0.383 | 0.962 |
| 20 | 0.885 | 0.719 | 0.668 | 0.672 | 0.766 | 0.962 |

</details>


## Provenance Appendix

Executed modeling source: `faf5b9e1f35bee43daba8cf0b5a7bb863cb4df55f7f9fa2fc159fcfe914bd40e`. Executed study-spec identity: `0e4d008b2ad68142a38628e9808443f7b7693fecbe7f5673df9c64808f8d4722`. Git dirty state: `True`.

The execution manifest describes model training. A presentation-only export records its current rendering source separately in the publication audit record; it does not relabel the executed source. Tables and figures read unchanged audited CSVs. No model fitting occurs during rendering.

Method references: [selection bias](https://jmlr.org/papers/v11/cawley10a.html), [threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html), [cross-validation variance limits](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf).
