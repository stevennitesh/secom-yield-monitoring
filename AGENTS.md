# AGENTS.md

Project-level operating contract for coding agents.

Status: active.

## 1) Project Objective

- Primary goals:
  1. Scientific goal:
     - Produce a faithful, reproducible, and auditable replication path for SECOM benchmark-style results (Lane A).
  2. Hireability goal:
     - Produce final artifacts and claim checks that are interviewer-defensible, clear, and technically rigorous.
  3. Engineering goal:
     - Maintain a stable time-aware model-selection/freeze pipeline (Lane B/lockbox path) with explicit QA gates.

- Non-goals:
  - One-off analyses that are not reproducible from scripts and artifacts.
  - Metric/claim changes that bypass artifact validation and audit checks.
  - Silent behavior changes without test coverage and documented rationale.

## 2) Execution Gate (Required)

- Default mode must be analysis-first.
- Agent must **not** edit files until the approval token appears in the latest user message.
- Required approval token before implementation: `APPROVE_IMPLEMENT`.
- If user asks for code directly without token:
  - return analysis + minimal plan + explicit approval request.
- Allowed without token:
  - read-only inspection,
  - repository search,
  - test/CLI runs that do not edit source files.
- If request is ambiguous:
  - ask exactly one clarifying question before editing.

## 3) Communication Style

- Responses must be precise, direct, and concise.
- Responses must include concrete numbers, assumptions, and decision criteria when applicable.
- Responses must use actionable bullets over long prose unless narrative detail is explicitly requested.
- When explaining code behavior, responses must include file references with line numbers.
- Responses must not omit implementation-relevant details unless user asks for a short summary.

## 4) Spec Completeness Contract

When writing plans/specs, include all of:

1. Scope
2. Non-goals
3. Files/functions to change
4. Interface/contract impacts
5. Data/schema impacts
6. Edge cases/failure modes
7. Test plan (unit/integration/smoke)
8. Acceptance criteria
9. Risks and rollback plan

If unknown, mark `UNKNOWN` and state what is needed to resolve.

- Include a dependency impact map:
  - for each changed file/function, list affected callers/importers and affected artifacts/contracts.
- Include decision rationale:
  - for each major design choice, state why it was chosen and why primary alternatives were not chosen.

## 5) Freshness / External Data

- Use up-to-date information by default when facts can drift.
- Use web search when user asks for latest/current/today, or when correctness may have changed.
- Provide source links and source dates for external claims.
- If browsing is unavailable, state that explicitly and continue best-effort.
- For web-backed answers, include a short source-audit block:
  - source (title + link),
  - publish/update date (if available),
  - access date,
  - confidence level and key uncertainty.

## 6) Coding Constraints

- Avoid introducing new bugs.
- Be memory efficient.
- Do not duplicate existing logic unnecessarily.
- Consolidate into existing code paths instead of adding helpers unless clarity/testability requires a new helper.
- If a method/helper is changed, check all imports/call-sites affected.
- No hidden behavior changes:
  - explicitly state intended behavior deltas before implementation,
  - after implementation, report actual before/after behavior in concrete terms.
- Do not commit unless user explicitly requests it.

## 7) Validation Requirements

After changes, report:

- Exactly what commands were run
- Exit code for each command
- Runtime for each command (or best available timing)
- What passed/failed
- Residual risks
- What was not tested and why

Minimum validation levels:

- Small change: targeted tests
- Medium change: targeted + adjacent suite
- Large/refactor: full suite + CLI/help smoke checks

## 8) Definition of Done

A task is done only if all apply:

- [ ] Requested behavior implemented
- [ ] No contract/schema regressions
- [ ] Required tests pass
- [ ] Outputs/artifacts validated
- [ ] Risks and follow-ups documented
- [ ] User explicitly accepts the reported outcome

## 9) Output Format Contract

For implementation tasks, use this fixed heading order (no skips):

1. `Solution`
2. `Behavior Delta (Before -> After)`
3. `Files Changed`
4. `Validation Run Log (command, exit code, runtime, result)`
5. `Risks / Limitations`
6. `Next Options`

For analysis-only tasks, use this fixed heading order (no skips):

1. `Findings`
2. `Tradeoffs`
3. `Recommendation`
4. `Minimal Plan`
5. `Approval Request`

- If a section is not applicable, include it and state `N/A` with one-line reason.

## 10) Test Command Profiles

- Fast:
  - `python -m pytest tests/test_metrics_threshold_optimization.py tests/test_split_contract.py tests/test_stage_b_selection.py -q`
- Lane A:
  - `python -m pytest tests/test_lane_a_replication.py tests/test_artifact_schema_and_claim_gates.py -q`
- Lane B:
  - `python -m pytest tests/test_stage_b_selection.py tests/test_freeze_lockbox_drift_mspc.py -q`
- Full:
  - `python -m pytest tests -q`
  - CLI smoke:
    - `python scripts/run_01_split.py --help`
    - `python scripts/run_02_lane_a.py --help`
    - `python scripts/run_03_lane_b_stage_ab.py --help`
    - `python scripts/run_04_freeze_lockbox.py --help`
    - `python scripts/run_05_audit_claims.py --help`

## 11) Critical Contracts

- Artifact schemas:
  - Canonical artifact names and required artifact sets:
    - `src/secom/config.py` (`ArtifactName`, `REQUIRED_ARTIFACTS_LANE_A_ONLY`, `REQUIRED_ARTIFACTS_LANE_B`).
  - Generic schema/enum/logic validation:
    - `src/secom/artifacts.py` -> `validate_schema_and_logic(...)`.
  - Lane A global artifact contract validation (shape/uniqueness/delta checks/anchor presence):
    - `src/secom/qa.py` -> `validate_lane_a_global_artifacts(...)`.
- Claim gates:
  - Lane A gate source:
    - `reports/lane_a_global_summary.csv`.
  - Lane A anchor requirements (enforced in `src/secom/workflows/audit.py` and `src/secom/qa.py`):
    - exactly one row for `(classifier='krr', selector='F-test', replication_mode='strict')`,
    - exactly one row for `(classifier='krr', selector='F-test', replication_mode='with_missing_indicators')`.
  - Lane A ablation consistency:
    - `delta_BER = BER_strict - BER_MI` must hold per `(selector, classifier)`.
  - Lane B claim checks (when `lane_b_feasible=True`):
    - lockbox MSPC row must exist in `reports/mspc_baseline.csv` (`eval_scope='lockbox'`),
    - if drift status is `HIGH_SHIFT`, scientific lockbox `TPR_at_TNR90` may not exceed lockbox MSPC TPR for that role.
  - Note:
    - no numeric BER benchmark threshold (e.g., `CI_upper_BER < X`) is currently enforced in code-level audit.
- Data split rules:
  - Deterministic sort and split:
    - lockbox is last `floor(0.15 * N)` rows after stable sort by `(timestamp, raw_row_id)`.
  - Dev weekly binning:
    - `week_label = floor((timestamp - min_timestamp)/7 days) + 1`.
  - Outer fold planning:
    - choose first feasible plan among primary/fallback windows such that each fold has `test_fails >= 20`.
  - Lane B feasibility:
    - requires feasible outer plan and minimum class count for inner CV (`min_class_count >= 5`) in each outer-train and full dev.
  - Canonical implementation:
    - `src/secom/cv.py`, `src/secom/workflows/split_contract.py`.

## 12) Risk Policy

- Never use destructive commands unless explicitly requested.
- If unexpected unrelated changes appear during work:
  - stop and ask user how to proceed.
- If a requested change conflicts with contract rules:
  - state conflict and ask for override decision.
- For high-impact contract changes:
  - stop and ask for explicit confirmation before changing docs/contracts/schemas,
  - examples: artifact names, schema columns, claim gates, validation rules, split policy.

Override protocol:

- The user may override a specific rule for a single turn by writing:
  - `OVERRIDE_POLICY: <rule and scope>`.
- Example:
  - `OVERRIDE_POLICY: skip approval token for this turn; only modify AGENTS.md wording`.
- The agent must restate the override, affected scope, and one-turn duration before applying it.
- Overrides do not persist unless the user explicitly requests updating this `AGENTS.md`.

## 13) Decision Log (Optional)

Append stable decisions to avoid repeating context.

Example format:

- `YYYY-MM-DD`: Decision, rationale, impact.

Current decisions:

- `2026-02-22`: Execution approval token is required before edits (`APPROVE_IMPLEMENT`).
  - Rationale: prevent accidental implementation when analysis/spec is requested.
  - Impact: analysis-first default; edits only after explicit approval.

- `2026-02-22`: Ambiguity handling requires one clarifying question before implementation.
  - Rationale: reduce interpretation errors.
  - Impact: fewer silent assumption-driven edits.

- `2026-02-22`: Response format is fixed-section for both analysis and implementation.
  - Rationale: improve consistency and reviewability.
  - Impact: predictable outputs with explicit validation and risk reporting.

- `2026-02-22`: Web-backed answers require a source-audit block.
  - Rationale: improve freshness and traceability of external facts.
  - Impact: each web-backed answer includes source/date/confidence metadata.

- `2026-02-22`: Code behavior explanations must include file references with line numbers.
  - Rationale: improve precision and auditability.
  - Impact: easier verification of claims against source.

- `2026-02-22`: High-impact contract changes require explicit confirmation before editing.
  - Rationale: reduce accidental contract drift.
  - Impact: schema/gate/split-policy changes always gated by user confirmation.

- `2026-02-22`: Added one-turn override protocol (`OVERRIDE_POLICY: <rule and scope>`).
  - Rationale: allow controlled exceptions without permanently weakening policy.
  - Impact: temporary, explicit, scoped overrides are supported and auditable.
