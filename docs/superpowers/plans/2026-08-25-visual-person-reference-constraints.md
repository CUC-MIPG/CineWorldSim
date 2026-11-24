# Visual Person Reference Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain image and video prompt generation so single-subject events stay free of unrelated people, while multi-person references are only enabled when the event truly requires them.

**Architecture:** Add one shared visual-prompt constraint helper and reuse it in character portrait, action keyframe, world asset, event keyframe, and H3 prompt builders. Add event-level reference-image policy metadata so video reference selection can cap character references to one by default and only expand for explicit multi-person events.

**Tech Stack:** Python, pytest, existing CineWorldSim prompt builders and narrative pipeline.

---

### Task 1: Add shared visual-person constraint helpers

**Files:**
- Create: `narrative/visual_prompt_constraints.py`
- Test: `tests/test_visual_prompt_constraints.py`

- [ ] **Step 1: Write the failing test**

```python
from narrative.visual_prompt_constraints import (
    build_people_control_clause,
    event_needs_multi_person_references,
)


def test_single_subject_clause_blocks_unrelated_people():
    clause = build_people_control_clause(["Hero"], allow_multi_people=False)
    assert "pedestrians" in clause.lower()
    assert "unrelated people" in clause.lower()
    assert "hero" in clause.lower()


def test_multi_person_detection_requires_interaction_cues():
    node = {"characters": ["Hero", "Guard"], "event_summary": "Hero confronts the guard at the door."}
    assert event_needs_multi_person_references(node) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_prompt_constraints.py -v`
Expected: FAIL because the module and functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# create helper functions that emit the clause and detect multi-person events from node text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_prompt_constraints.py -v`
Expected: PASS.

### Task 2: Thread the constraint clause through image prompt builders

**Files:**
- Modify: `narrative/simulation_scaffold.py`
- Modify: `narrative/event_keyframe_renderer.py`
- Modify: `scripts/run_dual_backend_narrative_world_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from types import SimpleNamespace

from narrative.event_keyframe_renderer import EventKeyframeRenderer
from narrative.simulation_scaffold import StateAnalysisMachine


def test_keyframe_prompt_mentions_no_extra_people():
    prompt = EventKeyframeRenderer.build_prompt(
        step=1,
        role_log={"role_name": "Hero", "next_action": {"action": "opens the door", "location": "Hall"}},
    )
    assert "pedestrians" in prompt.lower()
    assert "unrelated people" in prompt.lower()


def test_character_portrait_prompt_mentions_single_subject_only(monkeypatch, tmp_path):
    captured = []

    def fake_generate_image_nano_banana(**kwargs):
        captured.append(kwargs["prompt"])
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"image")
        return str(kwargs["output_path"])

    monkeypatch.setattr("agents.generation_backends.generate_image_nano_banana", fake_generate_image_nano_banana)
    monkeypatch.setattr(
        "agents.generation_backends.nano_banana_settings_from_env",
        lambda: {"api_key": "x", "base_url": "http://example.com", "model_name": "model", "timeout_sec": 1},
    )

    machine = StateAnalysisMachine()
    character = SimpleNamespace(
        state=SimpleNamespace(
            name="Hero",
            role_profile="A lone investigator.",
            final_goal="Find the truth.",
            known_relations=["Guard"],
        ),
        portrait_path="",
    )

    machine._generate_character_portrait(character=character, output_dir=tmp_path / "portraits", storyline="A test story.")

    assert captured
    assert "pedestrians" in captured[0].lower()
    assert "unrelated people" in captured[0].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_keyframe_renderer.py tests/test_narrative_visual_prompt_generation.py -v`
Expected: prompt text does not yet include the new constraint.

- [ ] **Step 3: Write minimal implementation**

```python
# import the helper and append the clause to portrait, action keyframe, and world asset prompts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_keyframe_renderer.py tests/test_narrative_visual_prompt_generation.py -v`
Expected: PASS.

### Task 3: Gate multi-person reference selection in video generation

**Files:**
- Modify: `agents/narrative_node_selector.py`
- Modify: `agents/video_agent.py`
- Test: `tests/test_orchestrated_event_video_generation.py`

- [ ] **Step 1: Write the failing test**

```python
from agents.video_agent import VideoAgent


def test_single_person_event_keeps_one_character_reference(tmp_path):
    refs = VideoAgent._node_reference_images(
        {"visual_assets": {"character_keyframes": ["a.png", "b.png"], "multi_person_reference_allowed": False}},
        include_character_keyframes=True,
        include_world_images=False,
        max_reference_images=4,
    )
    assert refs == ["a.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k reference_images -v`
Expected: current helper can still return multiple character refs.

- [ ] **Step 3: Write minimal implementation**

```python
# add event metadata and cap character references to one unless the event requires multi-person content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k reference_images -v`
Expected: PASS.

### Task 4: Extend H3 prompt text to encode the same rule

**Files:**
- Modify: `agents/h3_prompt_adapter.py`
- Test: `tests/test_orchestrated_event_video_generation.py`

- [ ] **Step 1: Write the failing test**

```python
prompt = VideoAgent._event_video_prompt(
    {
        "story_description": "Hero enters the room and confronts the guard.",
        "event_summary": "Hero enters the room and confronts the guard.",
        "video_description": "Hero opens the door and faces the guard inside.",
        "event_type": "conflict",
        "story_phase": "development",
        "characters": ["Hero", "Guard"],
        "locations": ["Room"],
        "items": ["Key"],
    },
    storyline="A test story.",
)
assert "unrelated people" in prompt.lower()
assert "pedestrians" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k h3_official_ref_style -v`
Expected: prompt lacks the new constraint.

- [ ] **Step 3: Write minimal implementation**

```python
# add the people-control clause to the subject_definitions / retention / detailed_description text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k h3_official_ref_style -v`
Expected: PASS.

### Task 5: Verify end-to-end prompt and reference behavior

**Files:**
- Test: `tests/test_orchestrated_event_video_generation.py`
- Test: `tests/test_event_keyframe_renderer.py`

- [ ] **Step 1: Run the targeted test suite**

Run: `pytest tests/test_event_keyframe_renderer.py tests/test_orchestrated_event_video_generation.py tests/test_skill_adapter.py -v`
Expected: all tests pass.

- [ ] **Step 2: Inspect generated prompt snippets if a failure appears**

Run: `pytest tests/test_event_keyframe_renderer.py tests/test_orchestrated_event_video_generation.py tests/test_narrative_visual_prompt_generation.py -vv`
Expected: single-person prompts forbid extras; multi-person prompts remain explicit and bounded to the listed characters.

- [ ] **Step 3: Commit**

```bash
git add narrative/visual_prompt_constraints.py narrative/simulation_scaffold.py narrative/event_keyframe_renderer.py scripts/run_dual_backend_narrative_world_pipeline.py agents/narrative_node_selector.py agents/video_agent.py agents/h3_prompt_adapter.py tests/test_visual_prompt_constraints.py tests/test_event_keyframe_renderer.py tests/test_orchestrated_event_video_generation.py docs/superpowers/plans/2026-08-25-visual-person-reference-constraints.md
git commit -m "feat: constrain visual prompts to relevant people"
```
