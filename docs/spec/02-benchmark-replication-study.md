# 02 Benchmark Replication Study

## Scope and Estimand

Both studies estimate the complete selection procedure under shuffled sampling from SECOM. The original is a literature-inspired fixed-budget reproduction, not exact published-paper replication. The tuned extension changes selector budgets and the bounded KRR parameter grid; it does not change evaluation samples or the scientific objective.

## Shared Nested Design

Use all timestamp-validated rows. Both studies use identical ten shuffled stratified outer folds, seed 42, and three stratified inner folds (reduce only for training-class feasibility). Every imputer, scaler, and selector is fitted on each inner training split. Candidate validation scores are pooled inner out-of-fold scores. Jointly choose parameters and a BER-optimal threshold using only those scores/labels, then freeze the threshold, refit on the outer training set, and evaluate once on outer test data.

BER is the primary inner objective. Numerical ties use deterministic simplicity and stable family/input-mode order, never outer performance or an AUC band. ROC AUC remains a supporting diagnostic. Inner OOF models and the outer refit can have different score distributions; this calibration limitation is disclosed. No third nesting is required or claimed.

Within each outer train, joint selection chooses selector, classifier, input mode, selector parameters, classifier parameters, and threshold. Its held-out metrics are the headline. The values-only and values-plus-indicators joint procedures provide interpretable input-mode comparisons. Individually nested family metrics and paired family ablations remain exploratory comparisons; their retrospective minimum is not an independently validated champion.

## Fixed and Tuned Search Spaces

Both use S2N, pooled Ttest, F-test, ReliefF, Gram-Schmidt, and Pearson by default; focused subsets remain supported. The original uses k=40 and ReliefF neighbors=10. The tuned study uses k=10/20/40 and neighbors=5/10/20. The original retains alpha=[0.1,1,10] and gamma=[None,0.01,0.1,1]. Tuned KRR uses alpha=[0.1,1,10,100] and gamma multipliers=[0.1,0.2,1,2], each divided by the actual selected feature count for that fit. This grows KRR from 12 to 16 classifier configurations per selector budget; it does not promise the old numerical results. Balanced logistic regression retains C=[0.01,0.1,1,10]. Both modes use train-fitted median imputation and StandardScaler; the indicator mode includes train-established missing indicators.

## Baselines and Prediction Evidence

On exactly the same outer folds, report an all-pass baseline and missingness-only balanced logistic regression with fixed C=1 and no parameter/selector grid. Missingness masks refer to original columns; retain only training-varying masks and fit scaling on train. Inner-OOF threshold calibration fits these transforms within inner train. If no mask varies, use an uninformative constant 0.5 score with its declared BER threshold tie rule.

Persist compact outer predictions for joint, values-only, values-plus-indicators, missingness-only, and all-pass procedures: raw sample ID, fold, true label, score, frozen threshold, prediction, selected family/configuration, and procedure identity. Never persist fitted matrices or model binaries. Pooled confusion counts must recompute from these per-example predictions. Compare original and tuned joint procedures using the identical held-out samples/folds and descriptive paired fold deltas.

Fold mean/std/range and paired ablation delta spread are descriptive. Overlapping training folds do not yield algorithm-performance confidence intervals; remove fold-bootstrap CI claims. Full-data fitted scores and modal configurations support descriptive interpretation only.

## Reference Limits

[UCI](https://archive.ics.uci.edu/dataset/179/secom) describes KRR, while [McCann and Johnston (2010), Table 2](https://proceedings.mlr.press/v6/mccann10a/mccann10a.pdf) labels its baseline Naive Bayes. Keep this discrepancy and the local 590 measurement columns versus metadata 591 explicit. Physical units and pre-outcome availability are unestablished. No Naive Bayes suite is required.

Method context: [selection bias](https://jmlr.org/papers/v11/cawley10a.html), [threshold calibration](https://scikit-learn.org/stable/modules/classification_threshold.html), and [CV variance limits](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf).

## Reuse and Runtime Constraints

ReliefF uses the pinned skrebate 0.8.4 distance and feature-typing implementation. The default NumPy adapter preserves its binary, imputed-data neighbor ordering, mixed-feature ramp, normalization and summation order. Exact score/rank equivalence against the upstream implementation is required for runtime optimizations; runtime-only optimizations preserve folds, seeds, feature budgets and the currently declared search grids. The bounded tuned-grid revision above is an intentional study change. Import or fitting failures are errors, never permission to substitute a different algorithm. Rankings may be reused only for byte-identical training measurements and labels, shape, neighbor count and backend identity; the bounded cache returns copies and never stores validation labels. The adapter runs in one process. An explicit `SECOM_RELIEF_BACKEND=reference` run uses upstream scoring with at most four workers by default, adjustable through positive `SECOM_RELIEF_N_JOBS`. The backend and actual worker count are recorded in the full-run manifest. The full evidence run includes KRR and a balanced logistic-regression comparator; focused KRR-only entrypoints remain supported.

Benchmark score vectors may be reused only when the classifier configuration, training measurements and labels, evaluation measurements, shapes/dtypes/strides and fitting implementation match exactly. The in-memory cache is bounded to 4,096 entries, returns independent copies, stores no raw arrays or validation labels, and starts cold for a complete timed study. Every family still performs its specified parameter selection and evaluation; reuse eliminates only identical fits. The full-run manifest records cache hits, misses and retained score bytes.

## Simplicity and Gamma Receipts

Exact numerical BER ties prefer fewer selected-budget features, then stronger regularization (larger KRR alpha, smaller logistic C), then deterministic gamma multiplier/reference gamma, ReliefF neighbors, family and input-mode order. Family and joint selectors share this owner; no near-best tolerance or AUC promotion is permitted. The tolerance is only 1e-12 numerical equality.

`gamma_multiplier` is the stable tuned candidate identity. Effective gamma is resolved separately in each inner/outer/full selected matrix; `inner_selected_widths` records the inner dimensions and outer/full receipts retain numeric gamma. Modal identity uses the multiplier, so different train-established indicator widths do not create different candidates. Original None-gamma receipts remain intact. The model-score cache normalizes only None to1/actual width, preserving complete input/layout/factory keys, copy isolation and bounded eviction.
