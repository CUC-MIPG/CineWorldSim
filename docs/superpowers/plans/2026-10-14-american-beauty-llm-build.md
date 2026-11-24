# American Beauty LLM Narrative Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable LLM-only L1-L4 CineWorldSim representation and Track A/B annotation pilot for `0001_American_Beauty` without modifying source MovieBench annotations.

**Architecture:** A single orchestration script reads source shot JSON files, calls the configured OpenAI-compatible Chat Completions API for each hierarchy level, validates coverage and JSON structure, and writes stage-specific artifacts under a new output directory. Each unit is written atomically with a status record so failed calls can be retried without recomputing successful units.

**Tech Stack:** Python 3.12, `openai`, `python-dotenv`, existing `agents.client`, JSON, pytest.

---

### Task 1: Add pure data and validation helpers

**Files:**
- Create: `narrative/moviebench_llm_builder.py`
- Test: `tests/test_moviebench_llm_builder.py`

- [ ] **Step 1: Write failing tests for L1/L2/L3 coverage validation and atomic JSON writes.**
- [ ] **Step 2: Run the focused tests and confirm they fail because the module is missing.**
- [ ] **Step 3: Implement only the pure helpers and schemas needed by the tests.**
- [ ] **Step 4: Run the focused tests and confirm they pass.**

### Task 2: Implement stage 0 and L1/L2 orchestration

**Files:**
- Modify: `narrative/moviebench_llm_builder.py`
- Create: `scripts/build_moviebench_cineworldsim_pilot.py`
- Test: `tests/test_moviebench_llm_builder.py`

- [ ] **Step 1: Add tests for deterministic source manifest creation and resume behavior.**
- [ ] **Step 2: Run tests and confirm the new cases fail.**
- [ ] **Step 3: Implement stage 0 source auditing, L1 prompt construction, L2 prompt construction, API calls, retries, and per-unit status files.**
- [ ] **Step 4: Run tests and confirm they pass.**

### Task 3: Implement L3/L4 aggregation and Track A/B outputs

**Files:**
- Modify: `narrative/moviebench_llm_builder.py`
- Modify: `scripts/build_moviebench_cineworldsim_pilot.py`
- Test: `tests/test_moviebench_llm_builder.py`

- [ ] **Step 1: Add tests for L3 complete assignment and Track B leakage checks.**
- [ ] **Step 2: Run tests and confirm the new cases fail.**
- [ ] **Step 3: Implement L3 aggregation, L4 movie-bible and story-arc calls, and deterministic Track A/B packaging from validated L4 results.**
- [ ] **Step 4: Run tests and confirm they pass.**

### Task 4: Execute the American Beauty pilot and audit outputs

**Files:**
- Create: `outputs/moviebench_cineworldsim/0001_American_Beauty/`
- Modify: `docs/moviebench_cineworldsim_processing_and_evaluation_plan.md`

- [ ] **Step 1: Run a no-network prompt/schema smoke test.**
- [ ] **Step 2: Run the real LLM pipeline using `.env`, with retries and unbuffered logs.**
- [ ] **Step 3: Validate stage counts, source coverage, failure records, and Track A/B leakage fields.**
- [ ] **Step 4: Record actual results and any API failures without claiming incomplete stages are complete.**
