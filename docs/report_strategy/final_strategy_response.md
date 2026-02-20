# Final Strategy Response: Challenging the Codex Challenge

*This document responds to `codex_challenge_to_strategy_critique.md` point by point.
It concedes where Codex is right, pushes back where the reasoning is weak, and
produces a final resolved position on each disputed decision.*

---

## Executive Response

Codex's challenge is more disciplined than its original document. Several points are
conceded. But the central criticism — that my recommendations are "too rigid" — is
largely a rhetorical move that substitutes procedural flexibility for methodological
clarity. "Too absolute" is only a valid critique when the absolute position is wrong.
In most cases here it is not.

The specific additions worth adopting: the two-threshold reporting approach and the
"diagnostic gate" language on drift. The specific additions to reject: the "time-aware
sensitivity check guardrail" (undefined), fixed 0.5 as a baseline reference (noise),
and PR-AUC as a primary selection metric (multi-criteria problem without a weighting
function).

---

## Point 1: Inner CV — "Too Absolute"

**Codex's position**: Stratified inner CV is directionally right but framed too
absolutely. Proposes a hybrid: stratified 5-fold as primary tuner, plus a "mandatory
time-aware sensitivity check" before promoting finalists.

**What Codex gets right**: The observation that inner CV design is a tradeoff between
variance reduction (stratified) and deployment alignment (time-aware) is technically
correct. The abstract tradeoff exists.

**The problem**: The "time-aware sensitivity check" is an undefined procedure.

Codex says: "If a config wins stratified inner but consistently degrades on time-aware
checks, do not promote it." This requires answering:

- How many time-aware inner splits constitute the check?
- What metric is used — inner ROC-AUC, or something else?
- What magnitude of degradation is "consistent"? 0.01 AUC drop? 0.05?
- When a config fails the check, what happens? Pick the second-best stratified winner
  and re-check? Run Stage B again with time-aware inner splits?

Codex's "hybrid" is not a protocol — it is a description of a decision that will be
made arbitrarily at implementation time. A validation strategy that defers key decisions
to the implementer's judgment is not more flexible; it is less reproducible. The whole
reason for specifying a protocol in writing is to prevent ad hoc decisions.

The per-fold outer test BER already performs the time-aware sensitivity check Codex
wants. If a config selected by stratified inner CV fails consistently on time-aware outer
test folds, that shows up in the outer fold variance — which I already require to be
reported fold-by-fold. The check is there. It just operates on reported outer metrics,
not on an undefined inner diagnostic.

**Final position: stratified inner CV, no undefined guardrail.** The outer fold
variance reports provide the time-aware sensitivity information Codex wants, without
adding a non-reproducible intermediate step.

---

## Point 2: Stage A — Concession

**Codex's position**: Stage A is not redundant. Exploratory notebooks are not a formal
selection record. The split design changed. Lockbox contamination means all prior
screening results are under a different (and evolving) protocol. A clean, frozen,
reproducible Stage A is needed for auditability.

**This is correct and I concede it.**

My critique framed Stage A as "repeating notebook work," which conflated the purpose.
The notebook runs were exploratory. Stage A under a frozen protocol is a formal record.
These are different things even if the data and code overlap substantially.

**What I maintain from my critique**: The scope of Stage A should be narrow. Drop
Pearson (mathematically identical to F-test on this dataset — including it produces
identical results and creates a confusing table entry). Include: F-test, Welch-t, S2N.
ReliefF optional if runtime permits. Gram-Schmidt included only if you want it formally
on record — its notebook results were weak under time-aware splits, but Codex's point
about reproducible records applies here too.

**Revised Stage A scope**: F-test, Welch-t, S2N, Gram-Schmidt (formal record), ReliefF
(optional). Pearson excluded.

---

## Point 3: Threshold Policy — Partial Concession

**Codex's position**: "Always Youden" is too rigid. BER-optimal threshold is not always
operationally optimal. Proposes two thresholds: BER-optimal (Youden) and
operations-constrained (capacity/cost-constrained). Also says fixed 0.5 can be retained
as a "baseline reference."

**What Codex gets right**: The two-threshold approach is genuinely useful.

In manufacturing monitoring, the threshold is ultimately set by operational constraints
— how many wafers an engineering team can review per week, what false alarm rate the fab
floor will tolerate before ignoring the system, or an explicit cost ratio from finance.
A model handed to a fab with only a Youden-optimal threshold has a threshold that
optimises a mathematical criterion, not an operational one. Reporting both gives the
engineer something to act on.

**Adopt**: Two thresholds — BER-optimal (Youden's J on full DEV) and
operations-constrained (minimum TNR at which TPR is maximised, parameterised by review
capacity). The second threshold is derived from the cost curve, not from an additional
grid search. The cost curve already produces this as a read-off at the fab's estimated
cost ratio.

**What Codex gets wrong**: Fixed 0.5 as a "baseline reference."

Codex says "fixed 0.5 can be retained as a baseline reference, not as winner-selection
criterion." This is a soft way of including 0.5 in the results. The effect is a column
in the results table that either:

(a) Consistently shows Youden beats 0.5 — in which case it adds nothing except length,
or (b) Shows 0.5 is competitive in some folds — in which case someone will interpret
this as evidence that 0.5 is a valid operating point, which it is not at a 6.64% fail
rate.

A "baseline reference" in a results table is indistinguishable from a candidate. If it
is not a candidate, it should not be in the table. The predict-all-pass and
predict-all-fail baselines already establish what "no threshold optimisation" looks like.
Fixed 0.5 is not meaningfully different from those as a reference point.

**Final position**: Two thresholds (Youden + operations-constrained). No fixed 0.5 in
results tables.

---

## Point 4: Metric Suite — Partial Concession on Tiering, Rejection on PR-AUC as Primary

**Codex's position**: Adding all metrics as first-class targets risks metric bloat.
Proposes tiered approach: primary selection metrics (BER, True+, True-, PR-AUC),
secondary diagnostics (MCC, Brier, calibration, alarm burden, cost curves).

**What Codex gets right**: The tiered framing is correct and useful.

I did not explicitly tier the metrics in `improvement_plan.md` or `strategy_critique.md`.
The distinction between selection metrics (what drives the config choice) and reporting
metrics (what characterises the chosen config) is important and should be stated. A
hiring manager reading a results table with 15 equally-weighted metrics will not know
which numbers matter.

**Adopt**: Tiered metric policy.

- Primary selection: outer mean BER (with True+/True- tie-break by business preference)
- Primary reporting: ROC-AUC, PR-AUC, MCC, F2, Youden's J
- Secondary diagnostics: Brier score, calibration curve, cost curve, yield impact table,
  MSPC comparison metrics

**What Codex gets wrong**: PR-AUC should not be a primary selection metric.

Codex puts PR-AUC in the primary tier alongside BER. This introduces a multi-criteria
selection problem: what happens when one config has lower BER but another has higher
PR-AUC? You need a weighting function to resolve this, which is exactly the same
problem you were trying to avoid by not putting threshold policy in the grid.

The selection criterion should be one primary metric. BER is the right choice because:

1. It is the original benchmark metric — selection on BER makes the comparison to
   McCann & Johnston (2008) clean.
2. It is already optimised by the threshold procedure (Youden's J minimises BER) —
   the entire pipeline is coherent around BER.
3. PR-AUC is threshold-free, which makes it incomparable to a BER measured at a
   specific threshold. You cannot meaningfully combine a threshold-free ranking metric
   with a threshold-dependent selection decision.

PR-AUC belongs in primary reporting (it characterises the model's discrimination
ability), not in primary selection (it does not have a well-defined relationship to the
threshold-dependent operating point being chosen).

**Final position**: Primary selection = outer mean BER. PR-AUC = primary reporting
metric, not selection criterion.

---

## Point 5: Benchmark Claim — Agreement with Minor Addition

**Codex's position**: Agrees CI is required. Adds that claims should be phrased as
"improves under our replicated framing" not "strictly beats the original benchmark."

**Agreement**: This phrasing is better and should be adopted in all writeup language.
The protocol differences (logistic regression vs kernel ridge, threshold tuning vs
none, missing indicators added) mean the comparison is "under a comparable framing,"
not an identical one. Claiming an identical comparison would be technically false.

**No challenge needed here.** This is a useful refinement, not a disagreement.

---

## Point 6: Drift Gate — Adopt Codex's Phrasing, Maintain Substance

**Codex's position**: My term "resolved" is wrong. Drift often cannot be fully resolved
at this dataset size. Better framing: drift is a "diagnostic gate" (must be quantified
and caveated) not a "resolution gate" (must be explained definitively).

**Codex is right on the language.** I used "resolved" loosely when I meant "diagnosed
and documented." With 14 weeks of data and small weekly fail counts, the cause of the
7.1% → 3.8% fail rate drop cannot be definitively determined from this dataset alone.

**Adopt**: "Diagnostic gate" language. The lockbox can be reported once the drift is
quantified (weekly fail rate chart produced, magnitude acknowledged, possible
explanations listed), even without a definitive causal explanation.

**The substantive requirement is unchanged**: The drift characterisation must appear in
the report before the lockbox results are presented, and the lockbox result must be
interpreted in light of it. A 3.8% fail rate in the lockbox vs 7.1% in DEV means the
lockbox is testing the model under a lower-prevalence regime. A process engineer needs
to know this before interpreting the lockbox TPR.

---

## Final Resolved Protocol

This is the single executable specification that survives both rounds of challenge.

### Data partitioning

```
LOCKBOX: last 15% by time (~236 samples, Oct 5–Oct 17)
  - Touch once, after config freeze and full DEV training.
  - Threshold from best_threshold_by_ber on full DEV only.

DEV: first 85% (~1331 samples, Jul 19–Oct 5)
```

### Required prerequisite: drift diagnostic

Before any DEV evaluation results are presented, produce the weekly fail rate chart
from `weekly["fail_rate"]`. Document the DEV vs lockbox fail rate difference (7.1% vs
3.8%). List possible explanations. This is a diagnostic gate — proceed once documented,
not only once explained.

### Stage A: Formal method screening on DEV

```
Methods:  F-test, Welch-t, S2N, Gram-Schmidt
          (ReliefF optional if runtime permits)
          Pearson excluded — mathematically identical to F-test on this dataset
Fixed:    k=40, train-tuned Youden threshold, one classifier (LR balanced)
Outer:    3 anchored expanding-window time-aware folds (same as Stage B)
Inner:    None (Stage A uses fixed settings — no tuning in Stage A)
Output:   timeaware_selector_screening.csv
          Top 2 methods promoted to Stage B
```

Note: Stage A has no inner CV because it uses fixed settings. The purpose is elimination,
not optimisation. Running inner CV in Stage A would make it Stage B.

### Stage B: Focused nested tuning on DEV

```
Methods:  Top 2 from Stage A (expected: F-test, Welch-t)
Outer:    3 anchored expanding-window time-aware folds
Inner:    5-fold stratified k-fold
          Scoring metric: roc_auc (threshold-free)
          No time-aware guardrail — per-fold outer results serve this purpose
Grid:     k ∈ {10, 20, 40}
          C ∈ {0.01, 0.1, 1.0, 10.0}
          scaler ∈ {StandardScaler, RobustScaler}
Selection: primary = mean outer BER
           tie-break = True+/True- by business preference
           tie-break = smaller k
Output:   timeaware_model_selection.csv
          Frozen primary + challenger configs
```

### Threshold finalisation

After config freeze, two thresholds derived from full DEV:

1. **BER-optimal (Youden's J)**: from `best_threshold_by_ber` on full DEV predictions.
   Use for scientific comparability and lockbox reporting.
2. **Operations-constrained**: read from cost curve at the fab's estimated cost ratio.
   Use for yield engineering framing and the operational operating point.

Fixed 0.5 is not reported as a candidate or baseline. Predict-all-pass establishes the
no-threshold-optimisation floor.

### Lockbox evaluation

Retrain frozen config on full DEV. Apply both frozen thresholds. Evaluate once.
Output: `final_lockbox_result.csv`.

### Metrics tiering

**Primary selection metric** (drives config choice in Stage B):
- Outer mean BER

**Primary reporting metrics** (characterise the chosen config):
- ROC-AUC, PR-AUC, MCC, F2, Youden's J, TPR, TNR

**Secondary diagnostics** (context, reported with uncertainty caveats):
- Brier score with dual baseline (all-pass=0.0664, prevalence=0.062)
- Calibration curve with 95% CI on each bin (feasibility caveat: ~35 test fails)
- Cost curve at r ∈ [1, 20], both thresholds plotted
- Yield impact table (wafers/week, review burden)
- MSPC T² and Q-SPE with T²-AUC and Q-AUC
- Feature report with selection frequency and conditional contribution

---

## Concession Summary

| Point | Concession to Codex |
|---|---|
| Stage A redundancy | Conceded — Stage A is needed as an auditable record, not for discovery |
| Two thresholds | Adopted — Youden + operations-constrained from cost curve |
| Drift language | Adopted — "diagnostic gate" not "resolution gate" |
| Metric tiering | Adopted — explicit primary/secondary tier distinction |

## Rejections Summary

| Point | Rejection |
|---|---|
| Time-aware sensitivity check guardrail | Undefined procedure; per-fold outer BER already provides this |
| Fixed 0.5 as baseline reference | Adds noise to results tables; predict-all-pass serves this function |
| PR-AUC as primary selection metric | Creates unresolvable multi-criteria conflict with BER; belongs in reporting |
| "Stratified is too absolute" label | Not a critique of the conclusion, only of the phrasing; the conclusion is correct |

---

## What Is Now Settled

These decisions are closed. Further challenges should present new evidence or a
specific failure mode of the protocol, not restate the flexibility-vs-rigidity framing.

1. Inner CV: 5-fold stratified, no guardrail.
2. Outer CV: 3 anchored expanding-window time-aware folds.
3. Threshold: Youden's J + operations-constrained. No fixed 0.5.
4. Selection metric: outer mean BER. PR-AUC is a reporting metric.
5. Stage A: required for auditability. Narrow scope, no inner CV, no Pearson.
6. Drift: diagnostic gate before lockbox interpretation. Quantify and caveat.
7. Lockbox: touch once, threshold from full DEV best_threshold_by_ber.
