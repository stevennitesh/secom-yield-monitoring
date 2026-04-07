# Final Report Generator Design

**Date:** 2026-03-15

## Goal

Add a separate reporting step that turns audited study artifacts into a polished `final_report.md` with supporting figures, while keeping PDF export optional.

## Decisions

- `final_report.md` becomes the canonical generated report artifact.
- Report generation stays separate from `run_full_study`; it consumes an existing output directory.
- PDF export is optional and must not block markdown generation.
- Figures are static PNG files generated into `reports/figures/`.
- Report prose is derived from existing artifacts and manifest metadata; no manual editing step is required for the default output.

## Recommended Approach

Extend the existing reporting module rather than creating a second reporting system.

This keeps one artifact-driven reporting path, reuses current report-generation logic where it still helps, and minimizes maintenance overhead. The report generator should load the current CSV and manifest outputs, compute compact narrative summaries, render figures, and then write a finished markdown report.

## Output Surface

- Add `scripts/run_final_report.py` as the dedicated report-generation entry point.
- Keep the existing reporting module as the core implementation surface.
- Generate:
  - `runs/<study>/reports/final_report.md`
  - `runs/<study>/reports/figures/benchmark_comparison.png`
  - `runs/<study>/reports/figures/tuned_vs_original_delta.png`
  - `runs/<study>/reports/figures/feature_stability.png`
  - `runs/<study>/reports/figures/temporal_drift.png`
  - `runs/<study>/reports/figures/lockbox_vs_mspc.png`
  - `runs/<study>/reports/figures/workload_cost_framing.png`
- Support an optional `--export-pdf` flag that attempts PDF export without making markdown generation depend on it.

## Report Structure

The generated report should read like a finished case study rather than a scaffold.

Required sections:

1. Executive summary
2. What I built
3. Dataset and study scope
4. Benchmark replication
5. Tuned benchmark
6. Original vs tuned comparison
7. Feature stability and interpretation
8. Temporal robustness stress test
9. Industrialization gaps
10. Conclusions and next data requirements
11. Provenance appendix

Narrative rules:

- Replace placeholder lines with artifact-derived prose.
- Keep benchmark evidence primary and temporal evidence secondary.
- Surface temporal claim restrictions exactly when present.
- Write for hiring managers first and ML practitioners second.
- Include a short, explicit “What I built” section covering:
  - reproducible benchmark and tuned-study pipelines
  - temporal robustness workflow with drift and claim gating
  - audit and report generation from versioned artifacts

## Figure Set

Generate six figures:

1. `benchmark_comparison.png`
   - Ranked benchmark leaders with BER emphasis and uncertainty where available.
2. `tuned_vs_original_delta.png`
   - Per-configuration BER deltas between tuned and original benchmark outputs.
3. `feature_stability.png`
   - Top benchmark and tuned features, distinguishing value features from missing indicators.
4. `temporal_drift.png`
   - Drift-gate summary for the primary temporal model, including PSI and prevalence/score-shift context where available.
5. `lockbox_vs_mspc.png`
   - Supervised lockbox `TPR_at_TNR90` versus best MSPC comparator, annotated as descriptive-only if claims are restricted.
6. `workload_cost_framing.png`
   - Operational/scientific workload framing plus illustrative cost curves.

Each figure should be referenced directly from the markdown report with a short interpretation paragraph.

## Provenance Appendix

The report appendix should expose manifest-backed provenance:

- git commit
- dirty state
- Python executable
- library versions
- study spec path
- study spec hash
- study statuses
- temporal claim restrictions

## Error Handling

- Missing required report data should remain a hard failure.
- Optional PDF export failures should not block markdown generation.
- If a figure cannot be produced because its required artifact is missing, the report should say so explicitly instead of silently omitting the section.

## Validation

- Extend reporting tests to verify `final_report.md` generation and the new narrative sections.
- Add figure-generation assertions for the expected PNG outputs using synthetic artifacts.
- Add provenance assertions sourced from `run_manifest.json`.
- Keep the existing report/audit validation behavior aligned with the new canonical report artifact.

## Notes

- Do not commit these planning changes unless the user explicitly requests a commit.
