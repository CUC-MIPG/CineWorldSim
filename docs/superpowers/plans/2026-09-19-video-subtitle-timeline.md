# Video Subtitle Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one timestamped JSON timeline beside every generated event or auxiliary video so dialogue and explicitly supplied narration can be converted into subtitles.

**Architecture:** Add a pure `agents.video_timeline` module that converts task shot intervals and dialogue/narration fields into normalized segment records. Integrate it into `VideoAgent.process_orchestrated_events` after each task is fully planned; `story_description` is never treated as narration by default.

**Tech Stack:** Python, JSON, pytest, existing `VideoAgent` task and shot-plan structures.

---

### Task 1: Timeline Data Builder

**Files:**
- Create: `agents/video_timeline.py`
- Test: `tests/test_video_timeline.py`

- [x] Write tests for shot-interval timing, speaker parsing, explicit narration, story exclusion, and deterministic filename output.
- [x] Run the tests and confirm collection fails because the module does not exist.
- [ ] Implement normalization, timecode formatting, segment extraction, and JSON writing.
- [ ] Run the focused tests and confirm they pass.

### Task 2: VideoAgent Integration

**Files:**
- Modify: `agents/video_agent.py`
- Modify: `tests/test_orchestrated_event_video_generation.py`

- [ ] Add timeline metadata to event and auxiliary tasks after their shot plans and dialogue backfill are available.
- [ ] Write timelines for dry-run, resumed, successful, failed, and auxiliary tasks without changing video request behavior.
- [ ] Include `timeline_path` in task and generated summary rows.
- [ ] Add an integration assertion that the timeline is emitted and excludes `story_description` when no dialogue/narration exists.

### Task 3: Verification

**Files:**
- No additional production files.

- [ ] Run focused timeline and orchestration tests.
- [ ] Run the full relevant video-agent test module.
- [ ] Verify the generated JSON schema and filename mapping on an existing output directory without modifying existing videos.
