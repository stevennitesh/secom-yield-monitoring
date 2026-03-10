# AGENTS.md

Project-level operating contract for coding agents.

Status: active.

## 1) Scope

This file governs agent behavior, review discipline, and validation expectations.
Project logic and scientific contracts remain canonical in code, tests, and runbooks.

## 2) Execution Gate

- Default mode: analysis first.
- Do not edit files unless the latest user message contains `APPROVE_IMPLEMENT`.
- Without that token, read-only inspection, repo search, and non-mutating test/CLI runs are allowed.
- If the request is ambiguous and editing would be risky, ask one concise clarifying question.

## 3) Response Rules

- Be precise, direct, and concise.
- State assumptions, intended behavior changes, and decision criteria.
- For analysis-only requests, provide:
  - Findings
  - Tradeoffs
  - Recommendation
  - Minimal Plan
  - Approval Request
- For implementation requests, provide:
  - Solution
  - Behavior Delta (Before -> After)
  - Files Changed
  - Validation Run Log (command, exit code, runtime, result)
  - Risks / Limitations
  - Next Options
- When explaining code behavior, include file references with line numbers.

## 4) Implementation Rules

- Avoid hidden behavior changes.
- Reuse existing code paths where practical; do not duplicate logic without reason.
- If a function, method, or contract changes, check affected callers, imports, tests, and artifacts.
- Do not commit unless the user explicitly asks.

## 5) Validation Rules

After changes, report:

- Exact commands run
- Exit code for each command
- Runtime for each command
- What passed or failed
- Residual risks
- What was not tested and why

Validation minimums:

- Small change: targeted tests
- Medium change: targeted plus adjacent suite
- Large change: full suite plus CLI help smoke checks

Preferred test profiles:

- Fast:
  - `python -m pytest tests/test_metrics_threshold_optimization.py tests/test_split_contract.py tests/test_stage_b_selection.py -q`
- Lane A:
  - `python -m pytest tests/test_lane_a_replication.py tests/test_artifact_schema_and_claim_gates.py -q`
- Lane B:
  - `python -m pytest tests/test_stage_b_selection.py tests/test_freeze_lockbox_drift_mspc.py -q`
- Full:
  - `python -m pytest tests -q`

CLI smoke:

- `python scripts/run_01_split.py --help`
- `python scripts/run_02_lane_a.py --help`
- `python scripts/run_03_lane_b_stage_ab.py --help`
- `python scripts/run_04_freeze_lockbox.py --help`
- `python scripts/run_05_audit_claims.py --help`

## 6) Canonical References

If a requested change may affect contracts, check these first:

- Artifact names and required artifact sets: `src/secom/config.py`
- Artifact/schema validation: `src/secom/artifacts.py`
- QA and claim gates: `src/secom/qa.py`
- Split policy and feasibility rules: `src/secom/cv.py`
- Split workflow wiring: `src/secom/workflows/split_contract.py`
- Audit workflow: `src/secom/workflows/audit.py`
- Contract enforcement tests: `tests/test_split_contract.py`, `tests/test_artifact_schema_and_claim_gates.py`, `tests/test_lane_a_replication.py`, `tests/test_stage_b_selection.py`, `tests/test_freeze_lockbox_drift_mspc.py`

If `AGENTS.md` conflicts with code, tests, or runbooks, stop and ask before changing behavior.

## 7) Safety and Overrides

- Never use destructive commands unless explicitly requested.
- If unrelated changes appear, stop and ask how to proceed.
- If a requested change affects artifact names, schema columns, claim gates, validation rules, or split policy, ask for explicit confirmation before editing.

One-turn override format:

- `OVERRIDE_POLICY: <rule and scope>`

Example:

- `OVERRIDE_POLICY: skip approval token for this turn; only modify AGENTS.md wording`