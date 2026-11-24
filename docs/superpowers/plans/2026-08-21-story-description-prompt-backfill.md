# Story Description Prompt Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-generated `story_description` for prompt-only use while preserving the existing `video_description` field unchanged in task JSON and downstream artifacts.

**Architecture:** Keep `video_description` as the canonical source field in event data, but generate a separate `story_description` from the event node plus pre-skill context before prompt construction. `VideoAgent` will write both narrative artifacts into task metadata, then the compact prompt path and `h3-prompt-writing` will prefer `story_description` when present and fall back to `video_description` only when needed.

**Tech Stack:** Python 3.12, existing `LLMJsonTool`, `pytest`, `VideoAgent`, `SkillAdapter`, `build_h3_ref_prompt`.

---

### Task 1: Add an LLM-backed story description tool

**Files:**
- Create: `agents/event_story_tool.py`
- Create: `tests/test_event_story_tool.py`
- Modify: `agents/llm_tool.py` only if shared JSON parsing helpers need to be reused

- [ ] **Step 1: Write the failing test**

```python
def test_event_story_tool_generates_story_description_and_artifacts(tmp_path):
    tool = EventStoryTool(
        llm_tool=FakeLLMJsonTool(
            {"story_description": "Character1 confronts Character2 in the corridor."}
        )
    )
    result = tool.generate(
        node={
            "event_summary": "A character reaches a corridor and speaks to another character.",
            "video_description": "An event-level summary of the scene logic.",
            "characters": ["Hero", "Guide"],
            "locations": ["Corridor"],
        },
        storyline="A test story.",
        output_dir=tmp_path / "story_artifacts",
        rank=1,
        node_id="NN_001",
    )
    assert result["status"] == "done"
    assert result["story_description"] == "Character1 confronts Character2 in the corridor."
    assert Path(result["artifact_path"]).exists()
    assert Path(result["meta_path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_story_tool.py -v`

Expected: fail because `agents.event_story_tool` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a small tool that:

```python
class EventStoryTool:
    def __init__(self, *, llm_tool: LLMJsonTool | None = None, retries: int = 3) -> None: ...
    def generate(
        self,
        *,
        node: dict[str, Any],
        storyline: str,
        output_dir: Path,
        rank: int,
        node_id: str,
        base_prompt_context: str = "",
    ) -> dict[str, Any]: ...
```

Implementation rules:

- Use `LLMJsonTool.run_json(...)`
- Ask for a JSON object with one key: `story_description`
- Keep the output narrative and chronological, not reasoning-style
- Fall back to `video_description` or `event_summary` only if the LLM fails
- Write `event_story_XXX_NODE.json` and `event_story_XXX_NODE.meta.json`
- Store the raw `story_description` string in the artifact, not a rephrased prompt block

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_story_tool.py -v`

Expected: pass.

- [ ] **Step 5: Add a regression test for fallback behavior**

```python
def test_event_story_tool_falls_back_to_video_description_when_llm_fails(tmp_path):
    tool = EventStoryTool(llm_tool=FailingLLMJsonTool())
    result = tool.generate(
        node={"video_description": "A calm establishing beat.", "event_summary": "Fallback event."},
        storyline="Fallback story.",
        output_dir=tmp_path / "story_artifacts",
        rank=1,
        node_id="NN_001",
    )
    assert result["story_description"] == "A calm establishing beat."
    assert result["status"] == "fallback"
```

---

### Task 2: Thread `story_description` through `VideoAgent` without changing `video_description`

**Files:**
- Modify: `agents/video_agent.py`
- Modify: `tests/test_orchestrated_event_video_generation.py`
- Create: `tests/test_video_agent_story_description.py` if the existing test file gets too dense

- [ ] **Step 1: Write the failing test**

```python
def test_process_orchestrated_events_persists_story_description_without_overwriting_video_description(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    # Build a one-node event and stub the story tool to return a narrative sentence.
    # Assert event_video_tasks.json contains both keys:
    # - video_description == original event description
    # - story_description == generated narrative text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k story_description -v`

Expected: fail because `VideoAgent` does not yet backfill `story_description`.

- [ ] **Step 3: Write minimal implementation**

Add a story backfill step before prompt resolution:

```python
prompt_node = dict(node)
story_result = self._backfill_event_story(
    node,
    storyline=storyline,
    output_dir=output_dir,
    rank=rank,
    node_id=node_id,
    enabled=story_backfill,
)
prompt_node["story_description"] = story_result["story_description"]
prompt_node["_story_backfill"] = story_result
```

Keep the original field untouched in task JSON:

```python
task["video_description"] = node.get("video_description", node.get("event_summary", ""))
task["story_description"] = prompt_node.get("story_description", "")
```

The new helper should mirror the dialogue backfill shape:

```python
def _backfill_event_story(...) -> tuple[dict[str, Any], dict[str, Any]]:
    # disabled -> keep node unchanged and return a disabled result
    # success -> write prompt-only story_description into prompt_node
    # failure -> fall back to video_description or event_summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrated_event_video_generation.py -k story_description -v`

Expected: pass.

- [ ] **Step 5: Add a regression assertion that `video_description` is not rewritten**

```python
assert tasks[0]["video_description"] == "An event-level summary of the scene logic."
assert tasks[0]["story_description"] == "Character1 confronts Character2 in the corridor."
```

---

### Task 3: Make prompt builders prefer `story_description`

**Files:**
- Modify: `agents/skills/adapter.py`
- Modify: `agents/h3_prompt_adapter.py`
- Modify: `tests/test_skill_adapter.py`
- Modify: `tests/test_orchestrated_event_video_generation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_skill_adapter_prefers_story_description_for_compact_prompt():
    prompt = SkillAdapter().build(
        "VideoAgent",
        compact_manifest,
        {
            "node": {
                "story_description": "Character1 confronts Character2 after the reveal.",
                "video_description": "Reasoning-style event summary that should not drive the prompt.",
                "characters": ["Hero", "Guide"],
                "locations": ["Corridor"],
                "items": ["Lantern"],
            },
            "storyline": "A test story.",
        },
    )["prompt"]
    assert "Character1 confronts Character2 after the reveal." in prompt
    assert "Reasoning-style event summary" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_adapter.py -v`

Expected: fail because the adapter still reads only `video_description`.

- [ ] **Step 3: Write minimal implementation**

Add a small helper in both prompt paths:

```python
def _prompt_story_description(node: dict[str, Any]) -> str:
    return _clean(
        node.get("story_description", "")
        or node.get("video_description", "")
        or node.get("event_summary", ""),
        limit=360,
    )
```

Then use that helper in:

- `_build_compact_event_prompt(...)`
- `_build_multi_shot_segment_prompt(...)`
- `_build_multi_shot_event_prompt(...)`
- `build_h3_ref_prompt(...)` for the `detailed_description` narrative text

Keep the existing labels and structure stable if possible:

- `video_description:` can remain the section label
- the content under it should come from `story_description` first

This keeps downstream prompt templates stable while changing the semantics of the text fed into them.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_adapter.py tests/test_orchestrated_event_video_generation.py -v`

Expected: pass.

- [ ] **Step 5: Add a regression test for H3 prompt text**

```python
def test_h3_prompt_uses_story_description_in_detailed_description(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    prompt = VideoAgent._event_video_prompt(
        {
            "story_description": "Hero confronts the guard and demands the key.",
            "video_description": "This should not dominate the prompt.",
            "characters": ["Hero", "Guard"],
            "locations": ["Hallway"],
        },
        storyline="A test story.",
    )
    assert "Hero confronts the guard and demands the key." in prompt
    assert "This should not dominate the prompt." not in prompt
```

---

### Task 4: Verify end-to-end compatibility and preserve old fields

**Files:**
- Modify: `tests/test_orchestrated_event_video_generation.py`
- Modify: `tests/test_skill_adapter.py`

- [ ] **Step 1: Add a full-path dry-run regression**

```python
def test_story_description_flows_into_tasks_and_prompt_but_video_description_remains_unchanged(
    tmp_path, monkeypatch
):
    # Build one event, run dry_run=True, and inspect event_video_tasks.json.
    # Assert:
    # - task["video_description"] still matches the original event field
    # - task["story_description"] exists
    # - task["prompt"] contains the story text
```

- [ ] **Step 2: Run the targeted test file**

Run: `pytest tests/test_orchestrated_event_video_generation.py -v`

Expected: all event-generation tests pass, including story-description coverage.

- [ ] **Step 3: Run the focused adapter tests**

Run: `pytest tests/test_skill_adapter.py tests/test_llm_tool.py -v`

Expected: pass.

- [ ] **Step 4: Inspect the generated task JSON schema**

Confirm the output still contains the old fields plus the new prompt-only field:

```json
{
  "video_description": "...",
  "story_description": "...",
  "dialogue_backfill": {...},
  "prompt": "..."
}
```

This is the final compatibility check before broader pipeline use.

---

### Self-Review Checklist

- `story_description` is LLM-generated, not deterministic
- `video_description` remains unchanged in task JSON
- prompt builders prefer `story_description` and fall back cleanly
- dialogue behavior stays intact
- tests cover both the prompt text and persisted task fields
- no new CLI surface is required for this change
