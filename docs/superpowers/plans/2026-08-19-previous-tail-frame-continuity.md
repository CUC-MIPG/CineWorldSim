# Previous Tail-Frame Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, failure-safe LLM continuity judge that decides whether a previous event video's tail frame should be used as the next event's reference image.

**Architecture:** Keep current reference-image collection unchanged. Add a small `VideoAgent` decision method that compares previous/current narrative nodes through the existing VIDEO LLM client, parses a strict JSON result, and returns false on any uncertainty. Invoke it only when `--use-previous-tail-frame` is enabled, then persist the decision under `continuity_reference`.

**Tech Stack:** Python, existing OpenAI-compatible client, pytest, JSON task metadata.

---

### Task 1: Add failing continuity decision tests

**Files:**
- Modify: `tests/test_orchestrated_event_video_generation.py`
- Test: `tests/test_orchestrated_event_video_generation.py`

- [ ] **Step 1: Add tests for default-off, affirmative, negative, and failure-safe behavior.**
- [ ] **Step 2: Run the targeted tests and verify they fail because no continuity judge exists.**

### Task 2: Implement the continuity judge

**Files:**
- Modify: `agents/video_agent.py`

- [ ] **Step 1: Add a system prompt requiring comparison of event continuity, characters, locations, and objects.**
- [ ] **Step 2: Add a structured JSON parser with exact-boolean approval and false fallback.**
- [ ] **Step 3: Add a method that calls the existing VIDEO client and returns decision metadata.**

### Task 3: Integrate the judge into event generation

**Files:**
- Modify: `agents/video_agent.py`

- [ ] **Step 1: Track the previous narrative node together with the previous tail/audio paths.**
- [ ] **Step 2: Invoke the judge only when the tail-frame option is enabled and a previous tail frame exists.**
- [ ] **Step 3: Prepend the tail frame only for an affirmative decision.**
- [ ] **Step 4: Persist decision, score, reason, and final usage in `continuity_reference`.**

### Task 4: Verify regression safety

**Files:**
- No new files.

- [ ] **Step 1: Run the targeted continuity tests.**
- [ ] **Step 2: Run the existing video, skill, and pipeline regression suites.**
- [ ] **Step 3: Inspect the generated task metadata from a dry-run fixture.**
