# SECOM Yield Monitoring: Improvement Plan
*Reviewed and revised. Review critique available on request.*

---

## 1) Current State Assessment

### 1.1 What the notebooks already do correctly

**Class imbalance handling**
`class_weight="balanced"` is used in every `LogisticRegression` call across both
notebooks. This rescales the loss function so each class contributes proportionally
regardless of sample count. With a 6.64% fail rate this is non-negotiable and is
already done.

**Threshold optimisation**
A custom `best_threshold_by_ber` function is implemented in both notebooks. It searches
a grid over [0.05, 0.95], evaluates BER at each candidate, and returns the threshold
that minimises BER on training data only, then applies it to the test set. This is
mathematically equivalent to maximising Youden's J statistic (TPR + TNR − 1). The
threshold is never tuned on the test set. This is correct.

**Missing value strategy**
`SimpleImputer(strategy="median", add_indicator=True)` is used throughout. The
`add_indicator=True` flag appends a binary column per imputed feature, preserving the
information that a value was absent. The EDA confirmed that median+indicators outperforms
median-only, meaning missingness itself carries signal. This is correctly implemented.

**Pipeline structure**
Preprocessing (imputation → scaling → feature selection → classifier) is built as an
sklearn `Pipeline`, which prevents data leakage during cross-validation by re-fitting
each step on training data only.

### 1.2 Implication

The model implementation is sound. The instability observed under time-aware evaluation
is not caused by the model — it is caused by the validation scheme. That is the root of
all remaining problems.

---

## 2) What Is Wrong With the Current Validation

### 2.1 Outer test windows are too small

The rolling window CV used 8 folds with approximately 235 samples per test window. At a
6.64% fail rate that is roughly 15 fails per outer test fold. A BER estimate on 15 events
has a standard error large enough to make method comparisons meaningless. The ±0.08–0.09
BER standard deviation seen across folds is a direct consequence of this.

The fix is not more folds — it is larger folds. Three anchored expanding-window folds,
each with a test window covering 30–40% of the DEV set, gives 20–35 fails per fold.
Fewer, larger folds are better than many small ones when the minority class is rare.
Note: even the largest fold will be noisy given only 104 total fails; this limitation
cannot be engineered away and must be disclosed in the final writeup.

### 2.2 The lockbox was touched multiple times

The notebook evaluates the lockbox while still making configuration decisions (k=10 vs
k=40, threshold choice). Once a lockbox result influences any decision it is no longer a
lockbox. The reported BER of 0.377 cannot be cited as an unbiased final estimate. It
must be treated as contaminated.

The fix is strict sequencing: freeze all configuration choices using only the DEV set,
retrain on full DEV, then evaluate the lockbox exactly once. Critically, the threshold
applied to the lockbox must be the output of `best_threshold_by_ber` fitted on full DEV
data — not a round number chosen by inspection. Any manually chosen threshold (e.g.
0.5) invalidates the claim that threshold was not tuned on the test set.

### 2.3 Inner CV should use stratified k-fold without a purge gap

Random stratified k-fold is the correct choice for the inner loop. The reason is
practical: with approximately 60–80 fails in each outer training set, 5-fold stratified
inner CV produces roughly 12–16 fails per inner test fold — enough signal to distinguish
good configurations from bad ones. Time-aware inner splits on the same data would give
fewer fails per inner window, making configuration selection noisier, not better.

The inner loop is not reported. It only needs to select k and C reliably. Mild temporal
leakage in the inner scores does not contaminate the outer test results, which are what
you report. The tradeoff favours stability of configuration selection over methodological
purity in the inner loop.

One previous draft added a "purge gap" to the stratified inner CV. That was internally
inconsistent: a purge gap is only meaningful when fold boundaries are at fixed temporal
positions, which stratified CV does not have. The fix is to drop the purge gap, not to
replace stratified CV with time-aware splits.

### 2.4 Inner CV scoring metric is threshold-dependent

The inner loop currently uses BER to select k and method. BER depends on a threshold
that has not been chosen yet at that stage. The inner loop should instead use ROC-AUC,
which is threshold-free and measures the classifier's ability to rank fails above passes
across all thresholds. The threshold is then separately optimised on the outer training
fold using `best_threshold_by_ber`, as is already done.

---

## 3) Phase 1 — Improving BER Against the Original Benchmark

### 3.1 The official benchmark

McCann & Johnston (2008) reported results using a kernel ridge classifier, 10-fold
random cross-validation, and 40 features selected by filter methods. No threshold
optimisation was applied in their protocol. Best result:

| Feature Selection Method | BER % | True+ % | True− % |
|---|---|---|---|
| F-test | **33.5 ±2.2** | 59.1 ±4.8 | 73.8 ±1.8 |
| T-test | 33.7 ±2.1 | 59.6 ±4.7 | 73.0 ±1.8 |
| Pearson | 34.1 ±2.0 | 57.4 ±4.3 | 74.4 ±4.9 |
| S2N | 34.5 ±2.6 | 57.8 ±5.3 | 73.1 ±2.1 |
| Gram-Schmidt | 35.6 ±2.4 | 51.2 ±11.8 | 77.5 ±2.3 |
| ReliefF | 40.1 ±2.8 | 48.3 ±5.9 | 71.6 ±3.2 |

The current project's notebooks achieve approximately 31.4% BER under random CV with
logistic regression and median+indicator imputation — already below the original.

**Important caveats on this comparison**: The original used a kernel ridge classifier
without threshold optimisation. The current implementation uses logistic regression with
Youden's J threshold tuning on training data, and adds missing-value indicators not
present in the original. These are not identical protocols. To claim a valid improvement,
bootstrap confidence intervals on fold-level BER must confirm the current mean BER is
statistically below 33.5% given the benchmark's ±2.2% standard error. Without this, the
improvement claim is not statistically supported.

**Two BER numbers will always be reported**:
1. Random CV BER — for direct literature comparison only, matching the 2008 protocol
2. Time-aware BER — for honest deployment estimation, expected to be higher

These measure different things and must never be compared to each other directly.

### 3.2 Pipeline improvements

The model architecture (logistic regression, class-balanced weights, filter feature
selection, median+indicator imputation) is appropriate and does not need to change. The
following targeted adjustments improve the bias-variance operating point:

**Replace StandardScaler with RobustScaler**
Semiconductor sensor data regularly produces outliers from equipment excursions, tool PM
events, and recipe changes. StandardScaler is sensitive to these events. RobustScaler
centres on the median and scales on the IQR, reducing outlier influence on the linear
classifier without requiring explicit outlier removal.

**Tune regularisation strength C jointly with k**
The current implementation uses C=1.0 (default), which is arbitrary. Regularisation
strength and number of selected features interact: with more features, stronger
regularisation may be needed. A grid of C ∈ {0.01, 0.1, 1.0, 10.0} × k ∈ {10, 20, 40}
gives 12 configurations per inner fold — small enough not to overfit the inner loop.

**Use ROC-AUC as inner CV scoring metric**
Replace BER with roc_auc for configuration selection. This separates the ranking problem
(inner loop) from the threshold decision (outer loop), and is more stable under class
imbalance.

**Keep existing threshold optimisation**
The `best_threshold_by_ber` function applied on the outer training fold is correct.

### 3.3 Validation scheme

```
Full dataset: 1567 samples, Jul 19 – Oct 17, 2008 (14 weeks)
│
├── LOCKBOX: last 15% by time (~236 samples, Oct 5 – Oct 17)
│   Fail rate: ~3.8%  [NOTE: substantially lower than DEV — see drift note below]
│   Touch exactly once. Threshold must come from DEV best_threshold_by_ber, not manual.
│
└── DEV: first 85% by time (~1331 samples, Jul 19 – Oct 5, ~12 weeks)
      Fail rate: ~7.1% (~93 fails)
      │
      ├── Outer fold 1: train weeks 1–5,  test weeks 6–12  (~40 fails in test)
      ├── Outer fold 2: train weeks 1–7,  test weeks 8–12  (~30 fails in test)
      ├── Outer fold 3: train weeks 1–9,  test weeks 10–12 (~22 fails in test)
      │
      └── Inside each outer train:
            Inner 5-fold stratified k-fold (random, no purge gap)
            Scoring metric: roc_auc
            Grid: method {F-test, Welch-t} × k {10, 20, 40} × C {0.01, 0.1, 1.0, 10.0}
```

Note: Outer fold 3 will still have approximately 22 fails in the test window. This is
above the ~15-fail problem threshold but remains noisy. Per-fold results must be
reported individually — do not report only the mean, as the variance across folds is
real signal about temporal instability.

### 3.4 Drift analysis (required before interpreting results)

The DEV fail rate is 7.1% and the lockbox fail rate is 3.8% — nearly half. This is a
substantial downward shift in marginal fail rate over time. Before reporting any model
result, a weekly fail rate chart (already computable from `weekly["fail_rate"]` in the
notebooks) must be included and discussed. This shift implies one of: (a) genuine
process improvement during the period, (b) a time-varying production campaign, or (c)
statistical noise given small weekly fail counts. Without diagnosing this, the lockbox
performance degradation has no interpretable cause. A process engineer will ask this
question immediately.

### 3.5 Method selection

F-test and Pearson select identical features (Jaccard=1.0) for the same k on this
dataset — they are mathematically equivalent for two-class problems. No reason to
evaluate both.

Primary candidates:
- **F-test**: Tended toward higher TNR in time-aware evaluation (passes more good
  wafers through without review)
- **Welch-t**: Tended toward higher TPR (catches more fails at the cost of more false
  alarms). Accounts for unequal class variance.

This tradeoff is operationally meaningful and should be preserved as two documented
operating points, not collapsed into a single winner by a metric. The cost curve
(Section 5.6) is what resolves the choice given a specific fab's cost assumptions.

ReliefF is deprioritised: competitive under random CV but not under time-aware
validation, and substantially slower to fit. The `relief.py` stub in `src/` should be
implemented as a lower-priority optional module, not a required deliverable.

---

## 4) Baselines

Baselines establish what the model must beat and why it beats it.

| Baseline | Purpose | Notes |
|---|---|---|
| **Predict-all-pass** | Trivial floor. Cost of having no model at all. | BER=50%, TPR=0%, TNR=100% |
| **Predict-all-fail** | Trivial ceiling. Useless operationally. | BER=50%, TPR=100%, TNR=0% |
| **McCann & Johnston 2008** (F-test, kernel ridge, random CV) | Direct literature comparison. Best: 33.5% ±2.2%. | Requires CI on our result to claim improvement |
| **LR, no feature selection** (post-imputation feature space) | Ablation: does feature selection add value beyond regularisation alone? | "Post-imputation" means all 456 value columns + all missing indicator columns produced by add_indicator=True — approximately 590 total |
| **LR, random k features** averaged over 50 seeds | Ablation: is ranking signal better than random selection? | Must average over ≥50 random seeds; a single draw is not a valid ablation |
| **MSPC T²** (pass-wafers only, no labels) | Industry standard. What process engineers already run. | Trained on outer-train pass-wafers only |
| **MSPC Q-SPE** (pass-wafers only, no labels) | Complementary industry standard. Catches novel fault modes. | Same training discipline as T² |

---

## 5) Full Metrics Suite

### 5.1 Group 1 — Original Benchmark Metrics

Computed under random CV to match the 2008 protocol for literature comparison, and
separately under time-aware CV for deployment estimation.

| Metric | Definition |
|---|---|
| **BER** | 1 − 0.5(TPR + TNR) |
| **TPR** (Fail Detection Rate) | TP / (TP + FN) |
| **TNR** (Pass-through Rate) | TN / (TN + FP) |

### 5.2 Group 2 — Threshold-Free Ranking Metrics

| Metric | Definition | Why include |
|---|---|---|
| **ROC-AUC** | Area under TPR vs FPR curve | Standard, comparable across papers |
| **PR-AUC** (Average Precision) | Area under Precision vs Recall curve | More honest than ROC-AUC at 6.64% fail rate. A random classifier has PR-AUC ≈ 0.066, not 0.5. ROC-AUC can appear high even when the minority class is largely missed. |
| **MCC** (Matthews Correlation Coefficient) | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Balanced single metric from −1 to +1. Equivalent to Pearson correlation between predicted and actual labels. Handles imbalance correctly by construction and is harder to inflate than F1 or accuracy. |

### 5.3 Group 3 — Threshold-Dependent Classification Metrics

All computed at the Youden's J optimal threshold derived from outer training data only.

| Metric | Definition | Why include |
|---|---|---|
| **Precision** (PPV) | TP / (TP + FP) | Of wafers flagged, what fraction are actually failing |
| **F1** | 2 · (P · R) / (P + R) | Equal-weight harmonic mean |
| **F2** | 5 · (P · R) / (4 · P + R) | β=2, recall-weighted. Reflects that missing a fail costs more than a false alarm. Standard for fault detection. |
| **Youden's J** | TPR + TNR − 1 | Distance from random (J=0) to perfect (J=1). The quantity directly maximised by threshold optimisation. |

### 5.4 Group 4 — Probability Calibration

**Feasibility caveat**: With approximately 35 fails in the largest outer test window,
a 10-bin reliability diagram has on average 3–4 fail events per bin. Confidence
intervals on observed fail rates in each bin will be very wide. Calibration assessment
is included because it is methodologically important, but conclusions must be hedged.
Do not apply Platt scaling or isotonic regression — there are too few positive samples
for these post-processors to be reliable.

| Metric | Definition | Notes |
|---|---|---|
| **Brier Score** | Mean squared error of predicted probabilities: E[(p̂ − y)²] | Two baselines to compare against: (1) all-pass classifier: Brier = prevalence = 0.0664; (2) constant-prevalence classifier (always predict 0.066): Brier = p(1−p) ≈ 0.062. These are different and both should be stated. |
| **Calibration curve** | Observed fail rate vs predicted probability in binned groups | Plot with 95% confidence intervals on each bin. At this sample size the intervals will be wide — this is informative, not a bug. |

### 5.5 Group 5 — MSPC Metrics (Industry Standard Baseline)

MSPC is the process engineering standard for yield monitoring. It is trained on
pass-wafers only (no labels needed) and monitors whether new observations are
statistically consistent with the in-control process.

**MSPC preprocessing is separate from supervised preprocessing**
For PCA-based MSPC, all sensors must be autoscaled: subtract the column mean and divide
by the column standard deviation, both computed on the training pass-wafer set only.
PCA is sensitive to scale differences; if features are not autoscaled, high-variance
sensors dominate loadings regardless of process relevance. This is distinct from
RobustScaler used in the supervised pipeline — do not reuse the supervised scaler for
MSPC.

**PCA training**
Train PCA on autoscaled pass-wafers from the outer training fold. Retain enough
components to explain 95% of variance as a starting point, then verify empirically: at
the selected number of components A, compute the T² and Q alarm rates on training
pass-wafers. The empirical false alarm rate should be close to the nominal 1% (for a
99% UCL). If it deviates substantially, adjust A. This verification step is standard
MSPC practice.

**Why both T² and Q are needed**
T² and Q are complementary, not redundant. T² measures deviation along the principal
directions of normal process variation (within the PCA subspace). Q measures deviation
orthogonal to those directions. A process shift that amplifies a known source of
variation raises T². A novel fault mode with no historical precedent raises Q but leaves
T² unaffected. A complete MSPC system monitors both simultaneously.

| Metric | Definition | Notes |
|---|---|---|
| **Hotelling's T²** | T² = (x − x̄)ᵀ P Λ⁻¹ Pᵀ (x − x̄), where x̄ is the training pass-wafer column mean, P are the retained PC loadings, Λ is the diagonal matrix of retained eigenvalues | x̄ must be computed on training pass-wafers and applied to test observations. Centering on the training mean is required — RobustScaler median-centres on the median which is not the same thing. |
| **T² UCL** | F-distribution approximation (Tracy-Young-Mason, 1992) at 99% confidence | UCL_T² = A(n−1)(n+1) / (n(n−A)) · F(α, A, n−A), where n is training sample count and A is number of retained PCs |
| **Q-SPE** | Q = ‖(I − PPᵀ)(x − x̄)‖², the residual variance not explained by the PCA model | Complementary to T². Sensitive to fault modes outside the normal variation subspace. |
| **Q UCL** | Jackson-Mudholkar approximation (1979) based on first three moments of the residual distribution | UCL_Q ≈ θ₁[c_α √(2θ₂h₀²/θ₁) + 1 + θ₂h₀(h₀−1)/θ₁²]^(1/h₀), where θᵢ are functions of eigenvalues of the residual covariance. This is a different formula from T² UCL and must be implemented separately. |
| **ARL₀** (In-control ARL) | Mean wafers between false alarms when process is in control | Must be computed empirically on training pass-wafers (count consecutive non-alarm runs), not taken as 1/α = 100 by definition. i.i.d. assumption required for 1/α to hold; semiconductor data is autocorrelated and will deviate. |
| **Detection lag** | Median number of wafers from first fail event in a cluster to first T² or Q alarm | Replaces ARL₁ for this dataset. ARL₁ requires sustained process shifts; SECOM fails appear sporadic. Compute only if contiguous fail windows of 3+ consecutive fails exist in the test set. Otherwise report per-event detection rate at threshold. |
| **T²-ROC-AUC** | Use T² score as a binary classifier; compute AUC vs fail labels | Quantifies how well T² separates fails from passes without committing to a threshold |
| **Q-ROC-AUC** | Same for Q-SPE | Tests whether novel fault modes (high Q) are more discriminative than centroid distance (high T²) |
| **Contribution plot** | Per-sensor decomposition of T² exceedance using the Kourti-MacGregor (1996) method: contribution of sensor j = p²ⱼ · (xⱼ − x̄ⱼ)² / λⱼ, summed over retained PCs | The primary diagnostic for process engineers: which sensors drove the out-of-control signal and in which direction. Implement the simple squared-loading decomposition (Kourti-MacGregor) not the complete decomposition — it is interpretable and is the industry standard. |

### 5.6 Group 6 — Yield Engineering Framing

These translate model performance into fab operational language. They are reframings of
existing metrics, not new computations.

**Dataset context for calculations**
- Time range: 14 weeks
- Average weekly throughput: ~112 wafers/week (1567 / 14)
- Average weekly fails: ~7.4 wafers/week (104 / 14)
- Note: these averages mask substantial week-to-week variation in fail rate

| Metric | Definition |
|---|---|
| **Wafers flagged per week** | (TP + FP) / weeks in test window |
| **Fails caught per week** | TP / weeks |
| **Fails missed per week** | FN / weeks |
| **Review burden %** | (TP + FP) / total wafers in window |
| **TPR at TNR=90%** | Read from ROC curve: sensitivity at 90% specificity |
| **TPR at TNR=95%** | Sensitivity at 95% specificity — tighter false alarm constraint |
| **Cost curve** | Expected cost per wafer = (r × FN + FP) / N, for r ∈ [1, 20]. Plot vs both baselines: all-pass cost = r × (fail rate); all-flag cost = (1 − fail rate). Shows the cost-ratio range over which the model is profitable. **Assumptions**: uniform cost per FP, cost per FN = r × cost per FP, no downstream recovery or rework credit. These simplifications must be stated explicitly. |
| **Break-even cost ratio** | The minimum r at which model expected cost < min(all-pass cost, all-flag cost). Below this ratio, the model does not pay for itself. |

**Operating point recommendation**
The cost curve resolves the F-test vs Welch-t choice. Plot both operating points on the
same cost curve. The operating point with lower expected cost at the fab's estimated
cost ratio is the recommendation. Present both; let the cost assumption drive the
selection rather than an arbitrary metric preference. This is the same answer as
Question 6 in Section 6 — these two questions are not separate.

---

## 6) How the Metrics Fit Together in the Final Report

The metrics above are structured into a narrative answering the questions a hiring
manager or process engineer would ask in sequence.

```
Section A: Can this beat the 2008 paper?
  → Group 1 metrics under random CV vs McCann & Johnston benchmark
  → With bootstrap CI to confirm statistical significance of improvement

Section B: What does deployment actually look like?
  → Group 1 + Group 2 + Group 3 under time-aware CV, per-fold results shown
  → Acknowledge: time-aware BER will be higher than random CV. This is expected.
  → Drift analysis: weekly fail rate chart, explain DEV/lockbox fail rate difference

Section C: Are the probabilities usable?
  → Group 4 (Brier, calibration curve with CI)
  → Caveat: ~35 test fails mean calibration assessment is indicative, not definitive

Section D: Why not just use the control chart you already run?
  → Group 5: MSPC T²/Q as unsupervised baseline vs supervised model
  → Honest framing: test whether the supervised model adds TPR beyond MSPC, not assert it
  → If MSPC wins: position it as primary, supervised as label-based screener
  → If supervised wins: explain what labeled outcome data adds over process deviation

Section E: What does this mean for the fab floor and which config should we use?
  → Group 6: yield impact tables, cost curve for both F-test and Welch-t operating points
  → Contribution plots: which sensors drive T² alarms (MSPC) and which features matter most (supervised)
  → The cost curve is also the operating point selector — these are the same question
```

---

## 7) Implementation Sequence

### Step 1: Implement src/ modules

The `src/secom/` package contains nine empty stub files. Priority order:

1. `io.py` — load raw data, align rows, return DataFrame with timestamp index
2. `preprocess.py` — sklearn Pipeline with RobustScaler, SimpleImputer(add_indicator=True), feature selector, LR(class_weight="balanced")
3. `metrics.py` — BER, TPR, TNR, ROC-AUC, PR-AUC, MCC, F1, F2, Youden's J, Brier score (with correct baselines: all-pass=0.0664, prevalence=0.062), cost_at_ratio, yield_impact_table, constrained_tpr_at_tnr
4. `cv.py` — anchored_splits (already prototyped in notebook — replicate faithfully), nested_cv_run
5. `feature_select/univariate.py` — F-test and Welch-t as sklearn-compatible transformers with a consistent fit/transform interface
6. `mspc.py` — fit_pca_on_passes (with autoscaling separate from supervised scaler), compute_t2, compute_q, ucl_t2_tracy, ucl_q_jackson_mudholkar, contribution_kourti_macgregor, arl0_empirical

**ReliefF stub**: `relief.py` is a lower-priority optional module. Implement only after
all above are complete and tested.

### Step 2: Notebook 04 — Clean validation and BER improvement

Using src/ modules: implement nested CV scheme from Section 3.3. Run all baselines
including LR-no-selection and LR-random-k (50-seed average). Compute bootstrap CI on
mean BER vs McCann benchmark. Output: `model_selection_summary.csv`, `splitwise_results.csv`.

### Step 3: Notebook 05 — MSPC baseline

Implement PCA + T² + Q-SPE on the same outer splits using the same train/test
discipline. Compute T²-AUC, Q-AUC, empirical ARL₀, contribution plots, detect whether
contiguous fail clusters exist (determines ARL₁ feasibility). Output: `mspc_baseline_results.csv`.

### Step 4: Notebook 06 — Full metric suite and yield framing

Using the winning config from Step 2 (without touching the lockbox): compute all Group
2–6 metrics, calibration curves with CI, cost curves for both F-test and Welch-t
operating points, feature stability table (Jaccard similarity of selected features
across outer folds), and feature report.

### Step 5: Lockbox evaluation

Retrain winning config on full DEV. Apply `best_threshold_by_ber` to full DEV to
obtain the deployment threshold (never a manually chosen value). Evaluate once on
lockbox. Output: `final_lockbox_result.csv`.

### Step 6: Feature report

Compute: selection frequency per feature across outer folds, mean |coeff × feature_std|
conditional on selection (SHAP is equivalent to this for linear models — use the
direct computation rather than the SHAP library), expected contribution = selection
frequency × conditional |coeff × std| (prevents averaging across folds where the
feature was never selected), Jaccard similarity table between fold 1 and fold 3 selected
sets (temporal stability diagnostic). Output: `feature_report.csv`.

---

## 8) What to Claim and What Not to Claim

**Claim**: Under a comparable random CV protocol, the model achieves lower BER than
McCann & Johnston (2008), with improvement attributed to class-balanced logistic
regression, Youden's J threshold optimisation on training data, and missing-value
indicator features. A bootstrap CI must confirm this difference is statistically
meaningful given the benchmark's ±2.2% standard error before this claim is made.

**Do not claim**: The time-aware BER improves on the 2008 benchmark. The 2008 paper
used random CV. Time-aware BER is higher and reflects a different and more honest
evaluation. Comparing them directly would mislead.

**Test, do not pre-assert**: Whether the supervised model improves on the MSPC T²/Q
baseline in fail detection rate. This is a hypothesis to be tested. If MSPC wins,
report that honestly and position MSPC as the primary recommendation with the supervised
model as a complementary screener.

**Claim**: The operating threshold was selected by maximising Youden's J on training
data only, using `best_threshold_by_ber`. The threshold was not tuned on the test or
lockbox set.

**Do not claim**: A specific threshold is optimal for deployment. The cost curve shows
which threshold minimises expected cost under a range of cost-ratio assumptions. The
actual cost ratio must be estimated from fab operations data not present in this dataset.

**Claim**: Selected features and contribution plots identify sensors statistically
associated with failure events during the Jul–Oct 2008 window at this specific fab.

**Do not claim**: These sensors are root causes of failure. Association is not
causation. Process engineering investigation and controlled experiments are required to
establish causality and define corrective action.

**Do not claim**: The model generalises beyond the Jul–Oct 2008 window or to other
fabs. The dataset covers 14 weeks of one production line. Temporal drift is already
evident within the window (DEV fail rate 7.1%, lockbox fail rate 3.8%). Any deployment
would require periodic retraining as the process evolves.
