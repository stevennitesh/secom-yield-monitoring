# AGENTS.md

## Mission

Refactor this repo toward the best study and reporting design for the actual goal, even if that requires replacing existing specs, contracts, workflows, artifacts, tests, and CLI structure.

Backward compatibility is not required unless the user explicitly asks for it.

## Priorities

Optimize for this order:

1. correct study design
2. clear and defensible claims
3. clean architecture
4. coherent artifacts and audits
5. performance and implementation quality

Preserving outdated structure is not a goal.

## Execution

- The agent may inspect, edit, refactor, rename, remove, and replace files across the repo without repeated approval.
- A single user approval for the refactor thread is sufficient unless scope changes materially.
- The agent should proactively steer the project toward the right design, not just preserve the current one.
- The agent should not stop just because a change affects specs, artifacts, claim logic, validation rules, or workflow structure.
- Do not commit unless the user explicitly asks.

## Refactor Rules

- Prefer replacing bad structure over layering compatibility on top of it.
- Do not add compatibility shims unless they provide clear ongoing value.
- If the corrected design requires breaking old contracts, break them cleanly and update all affected code, tests, specs, and docs.
- Remove obsolete code, docs, tests, and runbooks when they no longer match the new design.
- Keep naming, file layout, and workflow boundaries aligned with the new study structure, not the old one.

## Validation

After changes, always report:

- commands run
- exit code
- runtime
- what passed or failed
- residual risks
- what was not tested

Validation should match the new design, not preserve the old one.

Minimums:

- small change: targeted tests
- medium change: targeted plus adjacent suite
- large refactor: full relevant suite plus any still-supported entrypoint smoke checks

If old tests or CLI entrypoints encode obsolete behavior, update or remove them rather than preserving them by default.

## Canonical Sources

During refactor, the canonical source of truth is:

1. the corrected study objective
2. the updated spec
3. the updated tests
4. the updated code

If old specs, runbooks, tests, or code conflict with the corrected design, replace them.

## Safety

- Never use destructive commands outside the repo scope unless explicitly requested.
- If unrelated changes materially interfere with the refactor, stop and ask.
- Do not commit unless the user explicitly asks.
