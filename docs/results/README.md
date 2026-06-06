# Results Evidence Snapshot

This directory is the public evidence surface for the current SECOM study results. It contains a curated snapshot copied from the generated full-study output under `runs/full_study/reports/`.

The full `runs/` directory stays gitignored because it contains large generated search tables and rerunnable experiment output. This folder keeps the files a public reader needs to verify the headline claims without checking in the entire run directory.

## What To Read First

| File | Purpose |
| --- | --- |
| `final_report.md` | Canonical generated report with the benchmark, tuned benchmark, temporal robustness, and industrialization-gap narrative. |
| `figures/*.png` | Report figures used by `final_report.md`. |
| `evidence/run_manifest.json` | Provenance record for the snapshot, including git commit, dirty status, Python version, library versions, study spec hash, and study-layer statuses. |
| `evidence/benchmark_summary.csv` | Original benchmark replication summary behind the headline BER result. |
| `evidence/benchmark_tuned_summary.csv` | Tuned benchmark summary behind the stricter nested-CV result. |
| `evidence/temporal_drift_summary.csv` | Drift evidence behind the temporal robustness warning. |
| `evidence/temporal_lockbox.csv` | Lockbox stress-test metrics used as descriptive temporal evidence. |

## Snapshot Status

- Source run: `runs/full_study`
- Git commit: `094315c4ea9f0f57ae5915009aa2f61e9a55ad3a`
- Git dirty at manifest refresh: `false`
- Primary benchmark study: `passed`
- Temporal robustness: `warning`
- Active claim restriction: `primary_high_shift_blocks_lockbox_superiority_claim`

## Regenerate

Place the UCI SECOM raw files under `data/raw/`, then run:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --strict
```

The generated report and full artifact set will be written under `runs/full_study/reports/`. This curated directory should be refreshed only when the public-facing evidence snapshot needs to change.
