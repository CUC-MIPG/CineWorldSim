# ViMax Track B Character Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover all relevant ViMax case characters for Track B by default, while preserving explicit CLI overrides.

**Architecture:** Add a Track B-only LLM extraction helper in the ViMax runner. Build its evidence from the story overview and textual shot descriptions, parse and validate returned names, and pass the resulting names to the existing role initialization pipeline. Track B simulation continues to receive only `story_overview`; the manifest records both the names and their source.

**Tech Stack:** Python, existing `agents.llm_tool.LLMJsonTool`, pytest.

---

### Task 1: Test character extraction and explicit selection

**Files:**
- Modify: `tests/test_vimax_track_ab_pipeline.py`
- Modify: `scripts/run_vimax_track_ab_pipeline.py`

- [ ] Test that extraction prompt contains `story_overview`, `first_frame`, and `video_prompt` evidence.
- [ ] Test extracted-name normalization/deduplication and clear failure on empty names using a fake JSON tool.
- [ ] Test explicit CLI names take precedence and are not sent through extraction.
- [ ] Run the focused test module and verify new tests fail before implementation.

### Task 2: Integrate discovery into Track B

**Files:**
- Modify: `scripts/run_vimax_track_ab_pipeline.py`
- Modify: `tests/test_vimax_track_ab_pipeline.py`

- [ ] Make `--track-b-characters` default to unset, distinguishing omitted from explicitly supplied values.
- [ ] Discover characters only when Track B is selected and no explicit names were supplied.
- [ ] Keep only `story_overview` as the `run_track_b()` simulation story input.
- [ ] Record the resolved names and source (`llm_extracted` or `cli_override`) in the output manifest.
- [ ] Run focused and adjacent CLI tests.

### Task 3: Verify behavior

**Files:**
- No additional files.

- [ ] Run `pytest -q tests/test_vimax_track_ab_pipeline.py tests/test_dual_backend_pipeline_cli.py`.
- [ ] Inspect the final diff and confirm no image/video generation or Track A behavior changed.

Git metadata is unavailable in the current workspace, so no commit step can be performed here.
