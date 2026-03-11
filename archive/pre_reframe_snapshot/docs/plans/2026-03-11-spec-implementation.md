# SECOM Spec Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring the codebase into parity with the canonical modular spec in `docs/spec/`, starting from the highest-risk contract surfaces and verifying each tranche before moving on.

**Architecture:** Implement in contract-first tranches: split/feasibility, Lane A, Lane B selection/freeze, then artifact/audit/report outputs. Each tranche updates code and tests together so every spec-backed behavior is defended by a narrow verification loop before the next tranche.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest, Serena symbol tools

---

### Task 1: Split And Feasibility Parity

**Files:**
- Modify: `src/secom/cv.py`
- Modify: `src/secom/workflows/split_contract.py`
- Test: `tests/test_split_contract.py`

**Step 1: Write or extend failing split-contract tests**

Add assertions for:
- DEV/LOCKBOX split size and ordering
- `week_label` / `week_idx` semantics
- fold-plan fallback selection and infeasibility signaling in manifest-relevant outputs

**Step 2: Run targeted split tests**

Run: `python -m pytest tests/test_split_contract.py -q`
Expected: failures that expose current contract gaps

**Step 3: Implement the minimal split/feasibility fixes**

Update `src/secom/cv.py` and `src/secom/workflows/split_contract.py` to match:
- Section 03
- Section 04.2.1
- manifest signaling required by Section 10.5

**Step 4: Re-run split tests**

Run: `python -m pytest tests/test_split_contract.py -q`
Expected: PASS

### Task 2: Lane A Replication Parity

**Files:**
- Modify: `src/secom/workflows/lane_a.py`
- Modify: `src/secom/qa.py`
- Modify: `src/secom/artifacts.py`
- Test: `tests/test_lane_a_replication.py`
- Test: `tests/test_artifact_schema_and_claim_gates.py`

**Step 1: Write or extend failing Lane A parity tests**

Cover:
- config search/tie-break behavior
- threshold rules (`threshold_oof_global`, `threshold_full_dataset`)
- Lane A artifact schema/shape parity

**Step 2: Run targeted Lane A tests**

Run: `python -m pytest tests/test_lane_a_replication.py tests/test_artifact_schema_and_claim_gates.py -q`
Expected: failures pinpointing Lane A/spec mismatches

**Step 3: Implement the minimal Lane A fixes**

Align code to:
- Section 04.1
- Section 06
- Section 10.2
- Section 13 benchmark-claim anchors

**Step 4: Re-run Lane A tests**

Run: `python -m pytest tests/test_lane_a_replication.py tests/test_artifact_schema_and_claim_gates.py -q`
Expected: PASS

### Task 3: Lane B Selection Parity

**Files:**
- Modify: `src/secom/workflows/lane_b.py`
- Modify: `src/secom/common/thresholds.py`
- Test: `tests/test_stage_b_selection.py`

**Step 1: Write or extend failing Stage B tests**

Cover:
- selector scope and de-dup
- inner selection/tie-break chain
- tuple-level reporting contract including flagged-fraction diagnostics

**Step 2: Run Stage B tests**

Run: `python -m pytest tests/test_stage_b_selection.py -q`
Expected: failures on remaining Stage B/spec gaps

**Step 3: Implement the minimal Stage B fixes**

Align code to:
- Section 04.2.3
- Section 05
- Section 09
- Section 10.3

**Step 4: Re-run Stage B tests**

Run: `python -m pytest tests/test_stage_b_selection.py -q`
Expected: PASS

### Task 4: Freeze, Lockbox, Drift, MSPC, And Manager Outputs

**Files:**
- Modify: `src/secom/workflows/freeze_lockbox.py`
- Modify: `src/secom/config.py`
- Modify: `src/secom/artifacts.py`
- Test: `tests/test_freeze_lockbox_drift_mspc.py`
- Test: `tests/test_artifact_schema_and_claim_gates.py`

**Step 1: Write or extend failing freeze/lockbox tests**

Cover:
- frozen thresholds and lockbox scoring
- drift gate outputs
- MSPC lockbox/outer-fold coverage
- manager-facing artifact outputs

**Step 2: Run targeted freeze/lockbox tests**

Run: `python -m pytest tests/test_freeze_lockbox_drift_mspc.py tests/test_artifact_schema_and_claim_gates.py -q`
Expected: failures on remaining freeze/spec mismatches

**Step 3: Implement the minimal freeze/lockbox fixes**

Align code to:
- Sections 06 and 07
- Sections 10.4 and 10.5
- Sections 11 through 14 where they affect emitted outputs

**Step 4: Re-run freeze/lockbox tests**

Run: `python -m pytest tests/test_freeze_lockbox_drift_mspc.py tests/test_artifact_schema_and_claim_gates.py -q`
Expected: PASS

### Task 5: Full Verification

**Files:**
- Verify: `tests/*.py`
- Verify: `scripts/run_01_split.py`
- Verify: `scripts/run_02_lane_a.py`
- Verify: `scripts/run_03_lane_b_stage_ab.py`
- Verify: `scripts/run_04_freeze_lockbox.py`
- Verify: `scripts/run_05_audit_claims.py`

**Step 1: Run full test suite**

Run: `python -m pytest tests -q`
Expected: PASS

**Step 2: Run CLI smoke checks**

Run:
- `python scripts/run_01_split.py --help`
- `python scripts/run_02_lane_a.py --help`
- `python scripts/run_03_lane_b_stage_ab.py --help`
- `python scripts/run_04_freeze_lockbox.py --help`
- `python scripts/run_05_audit_claims.py --help`

Expected: all exit `0`

**Step 3: Review artifact and claim pathways**

Re-check:
- `src/secom/config.py`
- `src/secom/artifacts.py`
- `src/secom/qa.py`
- `src/secom/workflows/audit.py`

Expected: no remaining spec/code drift for emitted artifacts and claim gates
