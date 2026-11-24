# ViMax Pipeline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline command that evaluates generated Track A/B shot clips with ViCLIP and reports protocol-aware IC/CC/GC results.

**Architecture:** Keep discovery, completeness checks, encoding/cache, pair metrics, aggregation, and report serialization in a standalone evaluation module. Add a thin CLI wrapper that reads the existing benchmark index and pipeline artifacts; do not change online generation. Unit tests use temporary trees, actual tiny video fixtures, and fake encoders.

**Tech Stack:** Python 3.12, NumPy, requests, OpenCV, pytest.

---

### Task 1: Add report metric primitives and coverage tests

**Files:**
- Create: `scripts/vimax_pipeline_eval.py`
- Create: `tests/test_vimax_pipeline_eval.py`

- [ ] **Step 1: Write failing tests** for cosine metrics, Track A sorted mapping, and positional scene mapping.
- [ ] **Step 2: Run those tests** and confirm import/behavior failures.
- [ ] **Step 3: Implement** metric summaries and Track A inventory/mapping primitives.
- [ ] **Step 4: Run tests** and confirm the cases pass.

### Task 2: Implement Track B inventory and sample completeness

**Files:**
- Modify: `scripts/vimax_pipeline_eval.py`
- Modify: `tests/test_vimax_pipeline_eval.py`

- [ ] **Step 1: Add failing tests** for nested `(scene_idx, shot_idx)` discovery, script/storyboard inventory mismatches, missing and zero-byte clips, and optional generated-scene diagnostics separation.
- [ ] **Step 2: Run tests** and confirm expected failures.
- [ ] **Step 3: Implement** artifact discovery, expected-vs-found checks, completeness records, and diagnostics.
- [ ] **Step 4: Run tests** and confirm all inventory cases pass.

### Task 3: Implement validated ViCLIP HTTP encoder, cache, and evaluation flow

**Files:**
- Modify: `scripts/vimax_pipeline_eval.py`
- Modify: `tests/test_vimax_pipeline_eval.py`

- [ ] **Step 1: Add failing tests** for HTTP contract validation, cache key invalidation/hits, corrupt-video context, and pair-weighted/story-macro aggregates.
- [ ] **Step 2: Run tests** and verify they fail for the missing behavior.
- [ ] **Step 3: Implement** OpenCV decode validation, `/embed` metadata/dimension/norm validation, fake encoder, JSON embedding cache, fail-fast sample/shot context, and per-track aggregation.
- [ ] **Step 4: Run tests** and verify evaluation behavior.

### Task 4: Add CLI and user documentation

**Files:**
- Create: `scripts/run_vimax_pipeline_eval.py`
- Create: `docs/vimax_pipeline_evaluation.md`
- Modify: `tests/test_vimax_pipeline_eval.py`

- [ ] **Step 1: Add failing CLI tests** for selectors, fake-encoder smoke mode, report marker, and non-zero exit when all selected samples are incomplete.
- [ ] **Step 2: Run tests** and confirm failures.
- [ ] **Step 3: Implement** CLI arguments and JSON report output; document commands/defaults and the custom-result boundary.
- [ ] **Step 4: Run focused tests** and inspect `--help` output.

### Task 5: Final verification

**Files:**
- Review all files above.

- [ ] Run `pytest -q tests/test_vimax_pipeline_eval.py`.
- [ ] Run relevant existing benchmark runner tests.
- [ ] Run a fake-encoder smoke evaluation against a temporary tree and inspect the report.
- [ ] Run static syntax compilation for both new scripts.
- [ ] Verify the diff includes only new evaluation code/docs/tests and does not alter generated videos or existing user changes.
