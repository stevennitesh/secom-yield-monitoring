# Results Snapshot

Read [final_report.md](final_report.md) first. This small public snapshot was exported from `runs\finalization_20261003_optimized_v3` after the complete artifact audit passed. Git tracks the report, its figures, the unchanged execution manifest and the audit receipt. Detailed CSVs and the full source/artifact ZIP remain in ignored run storage.

The source execution records base Git commit `3a016025b702e4c3166b4d931020987dae9ed6e5`, dirty status `True` and exact source content hash `9c660d2bb051504fe86203e90797e1b60e3f14e4d8bca538bec4ea47a83f3814`. A dirty run includes uncommitted source changes; it does not pretend the base commit contains them. The unchanged [run manifest](evidence/run_manifest.json) records input hashes, resolved dependencies, study settings, timing and artifact hashes. The publication revision is the repository commit that eventually contains this snapshot.

- Primary benchmark status: `passed`.
- Temporal stress-test status: `warning`.
- Temporal claim restrictions: `primary_high_shift_blocks_lockbox_superiority_claim`.
- [Audit receipt](evidence/audit_receipt.json): errors, warnings, restrictions and exported-file hashes.
- Complete CSVs stay in `runs\finalization_20261003_optimized_v3/reports/`.
- The full artifact/source archive is `runs\finalization_20261003_optimized_v3\evidence\study_artifacts.zip`: all CSVs, manifest, report, figures and exact study source, without raw data. It is a local file, not a GitHub download; the public receipt records its hash.

## Reproduce and Audit

Use Python 3.11 or 3.12. See the root README for portable installation. Fetch verified data with `python scripts/fetch_secom.py`, then run into a fresh directory:

```bash
python scripts/run_full_study.py --input-dir data/raw --output-dir runs/reproduction --classifiers krr,logreg --progress --strict
python scripts/run_audit.py --output-dir runs/reproduction --strict
python scripts/export_results.py --output-dir runs/reproduction
```

The full study performs many repeated fits; this snapshot's manifest records the measured modeling duration and thread settings. Export saves the full archive under `runs/reproduction/evidence/study_artifacts.zip`. To audit that local archive without training, unzip it and run its `source/scripts/run_audit.py --output-dir <extracted-directory> --strict`. To reproduce the exact dirty source, install from the archive's `source/` directory and run its scripts. Original non-nested results, tuned family-wise estimates and chronological logistic-regression diagnostics retain separate claim scopes.
