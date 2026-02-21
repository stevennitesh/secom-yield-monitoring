# SECOM Yield Monitoring

Implementation of the merged report strategy under:

- `docs/final_end_to_end_report_strategy_merged.md`
- `docs/report_strategy/runbooks/*.md`

## CLI Sequence

1. `python scripts/run_01_split.py`
2. `python scripts/run_02_lane_a.py`
3. `python scripts/run_03_lane_b_stage_ab.py`
4. `python scripts/run_04_freeze_lockbox.py`
5. `python scripts/run_05_audit_claims.py`
