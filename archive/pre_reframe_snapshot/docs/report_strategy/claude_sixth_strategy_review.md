# Claude Sixth Strategy Review

## Purpose

Top-to-bottom review of `final_end_to_end_report_strategy.md` after all
round-5 issues and Codex's three self-review items were resolved. This round
confirms those resolutions and identifies any remaining implementation-blocking
issues.

---

## Round 5 + Codex Self-Review Resolutions — Confirmed

Round 5 (Claude fifth review):

1. §8.5 drift gate HIGH_SHIFT now explicitly overrides §13.4 claim policy via
   global precedence rule in §13. ✓
2. PSI "value-feature definition" (original sensor columns only) and "top-10
   ranking rule" (by absolute standardized logistic coefficient from full-DEV
   frozen fit) both added to §8.5. ✓
3. §7.8 week definition for flagged fraction now cross-references §4.2 exactly
   (`week_idx = floor((timestamp - t_min_dev)/7 days)`). ✓
4. `empirical_ARL0` is no longer orphaned: §10.10 consumes it in manager-facing
   output. ✓
5. `mspc_baseline.csv` lockbox row now has sentinel (`fold_index='LOCKBOX'`,
   `eval_scope='lockbox'`) and completeness rules (required non-null fields;
   `empirical_ARL0` may be NaN with reason logged). ✓

Codex self-review:

6. `drift_gate_summary.csv` added as artifact 12 with full schema. ✓
7. §13 global precedence rule added; "Can claim" item 4 and item 6 now
   condition on drift gate status not being `HIGH_SHIFT`; "Cannot claim"
   item 5 added. ✓
8. §5.2 ReliefF top-2 case (1440 inner fits) made explicit alongside the
   3-selector case (1800). ✓

---

## Remaining Issues

### §5.2 Workload Annotation "(720 + 720 ReliefF-neighbor branch)" Is Arithmetically Wrong

The document states:

> "If ReliefF is one of the promoted top-2 selectors, expected total is `1440`
> inner fits (`720 + 720` ReliefF-neighbor branch)."

The total (1440) is correct. The breakdown annotation (720 + 720) is not.

When ReliefF replaces one of the two selectors in the top-2 pool, the
components are:

- Non-ReliefF selector: `1 x 3 x 4 x 2 x 5 x 3 = 360` inner fits.
- ReliefF selector: `3 x 3 x 4 x 2 x 5 x 3 = 1080` inner fits
  (ReliefF adds the `n_neighbors ∈ {5, 10, 20}` dimension).
- Total: `360 + 1080 = 1440`.

The "(720 + 720)" notation implies two equal halves. A developer reading this
to implement the grid loop will construct the wrong iteration structure:
two loops of 720 each rather than a 360-iteration loop for the non-ReliefF
selector and a 1080-iteration loop for ReliefF. The correct annotation is
"(360 non-ReliefF + 1080 ReliefF)".

For comparison, the 3-selector case in item 4 — "If ReliefF is promoted and
a 3rd selector is also promoted, expected total is `1800` inner fits" — is
correct (720 for the 2 non-ReliefF selectors + 1080 for ReliefF = 1800), and
has no annotation to mislead.

**Required fix:** Change "(720 + 720 ReliefF-neighbor branch)" to
"(360 non-ReliefF + 1080 ReliefF)".

---

### `alarm_rate` and `empirical_ARL0` in `mspc_baseline.csv` — Alarm Source Statistic Undefined

`mspc_baseline.csv` (§11.2) has single columns `alarm_rate` and
`empirical_ARL0`. MSPC produces two separate alarm streams: one from T² (vs
T² UCL) and one from Q-SPE (vs Q UCL). An alarm can be defined as:

- T² alarm only: `T2 > UCL_T2`.
- Q alarm only: `Q > UCL_Q`.
- "Either" alarm (union rule): `T2 > UCL_T2 OR Q > UCL_Q`.

These produce different `alarm_rate` values and different `empirical_ARL0`
values (the union rule has a higher false-alarm rate than either individual
rule). The correct operationally-used interpretation is the union rule: if
either chart alarms, the wafer is flagged. But without a stated rule,
implementations will differ.

This matters for §10.10 ("MSPC false-alarm behavior summary using
`alarm_rate` and `empirical_ARL0`"), which will be presented to a hiring
manager as a key MSPC characteristic. The alarm rate and ARL0 values depend
directly on this definition.

**Required addition to §9.3.6 or §11.2:** "For `alarm_rate` and
`empirical_ARL0`, define an alarm as the union rule: sample is flagged if
`T2 > UCL_T2` OR `Q > UCL_Q`. `alarm_rate` = fraction of test samples
triggering this union alarm. `empirical_ARL0` = mean count of consecutive
pass-wafer test samples between union alarms."

---

## What Was Verified Clean This Pass

**§1 → §12 traceability:** All three §1 end goals are fully covered by the
protocol. Every section is connected forward and backward with no orphaned
elements.

**§5.2 workload totals:** All three totals (720 base, 1440 with ReliefF in
top 2, 1800 with 3 selectors including ReliefF) are arithmetically correct.
Only the annotation for 1440 is wrong.

**§13 claim policy drift gate integration:** The global precedence rule,
"Can claim" items 4 and 6, and "Cannot claim" item 5 are now consistent with
§8.5. An implementer reading §13 alone will encounter the drift gate condition
before the claim rules. ✓

**§8.5 drift gate operability:** All three criteria are quantitative and
executable. The PSI ranking rule (absolute standardized logistic coefficient,
full-DEV frozen fit) is deterministic. The `drift_gate_summary.csv` records
all inputs and outputs needed to audit a gate decision. ✓

**`drift_gate_summary.csv` completeness:** The schema contains the raw inputs
to each gate criterion (`dev_fail_rate`, `lockbox_fail_rate`,
`abs_prevalence_shift`, `ks_pvalue_scores`, `max_PSI`, `median_PSI`) plus the
derived outputs (`drift_gate_status`, `lockbox_claims_allowed`). This is
sufficient for a full audit trail. ✓

**MSPC lockbox companion vs supervised lockbox — matched comparison:** Both
are trained on full DEV, both scored on lockbox samples. The §13.4 claim is
now a proper matched-condition comparison. ✓

**Artifact schema completeness:** All 12 artifacts have minimum column
specifications. No orphaned artifacts, no under-specified schemas. ✓

---

## Summary

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | §5.2 workload annotation "(720 + 720)" is wrong — correct breakdown is "360 + 1080" | Arithmetic annotation | Minor |
| 2 | `alarm_rate` and `empirical_ARL0` in `mspc_baseline.csv` — union vs T²-only vs Q-only undefined | Specification gap | Minor |

---

## Overall Assessment

The document is implementation-ready. The two remaining issues are minor:
issue 1 affects only a comment (the total is right), and issue 2 affects the
MSPC alarm-rate reporting (consequential for the §10.10 manager-facing output,
but not for the primary supervised model selection or claim policy).

After fixing these two items, no further review rounds are needed. The
protocol logic is sound, all major ambiguities have been closed over six
review rounds, and the full pipeline from data partition through claim
adjudication is deterministic and auditable.
