# Event-to-Multi-Shot TMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current CineWorldSim narrative-time and cinematic-world pipeline into an event-centric multi-shot cinematic generation framework.

**Architecture:** The current system already produces narrative deduction logs, world states, world deltas, and visual assets. The next layer should extract atomic events from these artifacts, bind them to relevant characters / locations / items / keyframes, expand each atomic event into a group of cinematic shots, and then send the shot group to video generation and post-production. This changes the core generation unit from `event -> one clip` to `event -> multi-shot sequence -> video segment`.

**Tech Stack:** Python, JSON artifacts, LangChain/OpenAI-compatible LLM calls, nano_banana, local FLUX, MagRef video generation, OmniVoice, ACE-Step, OpenCV, ffmpeg, pytest.

---

## 1. Current Baseline

The existing architecture contains these useful layers:

- `OrchestratorAgent`: parses user story and creates character agents.
- `WorldModelAgent`: extracts locations and important items from the story.
- `StateAnalysisMachine`: runs multi-character narrative deduction.
- `WorldDeltaJudge`: determines whether world assets change after each step.
- `WorldStateTracker`: writes step-indexed world states.
- `EventKeyframeRenderer`: generates role-level event keyframes.
- `PathOrchestratorAgent`: orders candidate nodes or shots.
- `ShotPlannerAgent`: converts ordered narrative nodes into shot prompts.
- `VideoAgent`: generates videos from shot prompts and visual references.
- `PostProductionAgent`: assembles video, dubbing, narration, music, and final cuts.

The current experimental world pipeline is:

```text
storyline
  -> character configs
  -> world model
  -> world images
  -> step_logs
  -> world_deltas
  -> world_states
  -> event keyframes
  -> HTML visualization
```

The new TMM extension starts after `step_logs + world_states + world_deltas + event_keyframes` are available.

## 2. Target Concept

The main technical shift is:

```text
Current:
Atomic event -> one generated video clip

Target:
Atomic event -> multi-shot storyboard group -> multiple generated clips -> one edited event segment
```

An atomic event is not a shot. It is a narrative unit such as:

```text
The journalist discovers the cracked compass inside the abandoned lighthouse.
```

A cinematic system should express this with multiple shots:

```text
EVT_001_SH001: establishing shot of the lighthouse interior
EVT_001_SH002: tracking shot of the journalist approaching the table
EVT_001_SH003: close-up of the cracked compass
EVT_001_SH004: reaction shot of the journalist realizing its significance
```

Therefore, the framework needs a new middle layer:

```text
Narrative + World Artifacts
  -> Atomic Event Layer
  -> Event Composition Layer
  -> Multi-Shot Storyboard Layer
  -> Video Execution Layer
```

## 3. Proposed New Modules

### 3.1 AtomicEventExtractor

Create:

- `agents/atomic_event_extractor.py`

Responsibility:

- Read `step_logs`, `world_states`, `world_deltas`, and `event_keyframes`.
- Extract cinematic atomic events.
- Merge role-level actions in the same step when they form one shared event.
- Preserve source traceability back to step IDs and role names.

Primary input:

```json
{
  "storyline": "string",
  "step_logs": [],
  "world_states": [],
  "world_deltas": [],
  "event_keyframes": []
}
```

Primary output:

```json
{
  "event_id": "EVT_001",
  "source_steps": [1],
  "source_roles": ["Journalist", "Caretaker"],
  "event_summary": "The journalist examines the cracked compass while the caretaker tries to interrupt.",
  "event_goal": "Reveal the first important clue and introduce resistance.",
  "story_phase": "development",
  "characters": ["Journalist", "Caretaker"],
  "locations": ["Abandoned Lighthouse"],
  "items": ["Cracked Compass"],
  "source_keyframes": [],
  "world_state_refs": ["world_state_step1.json"],
  "world_delta_refs": ["world_delta_step1.json"],
  "estimated_importance": 0.75
}
```

### 3.2 EventAssetBinder

Create:

- `agents/event_asset_binder.py`

Responsibility:

- Attach visual assets to each atomic event.
- Bind character references, scene references, item references, world update images, and event keyframes.
- Validate that every referenced asset exists on disk.

Output field:

```json
{
  "visual_assets": {
    "character_refs": [],
    "location_refs": [],
    "item_refs": [],
    "world_update_refs": [],
    "event_keyframes": []
  }
}
```

### 3.3 AtomicEventComposer

Create:

- `agents/atomic_event_composer.py`

Responsibility:

- Convert extracted atomic events into director-ready event packages.
- Assign duration budget, emotional function, visual objective, and continuity constraints.
- Decide whether the event needs 2, 3, 4, or more shots.

Primary output:

```json
{
  "event_id": "EVT_001",
  "event_summary": "string",
  "event_goal": "string",
  "event_duration_budget_sec": 18.0,
  "shot_count_target": 4,
  "dramatic_function": "clue discovery",
  "visual_function": "reveal object significance through progressive detail",
  "continuity_constraints": {
    "location": "Abandoned Lighthouse",
    "time_of_day": "storm night",
    "lighting": "cold lightning flashes and dim interior practicals",
    "must_show_items": ["Cracked Compass"],
    "must_show_characters": ["Journalist"]
  }
}
```

### 3.4 MultiShotStoryboardPlanner

Create:

- `agents/multi_shot_storyboard_planner.py`

Responsibility:

- Expand one composed atomic event into several shot descriptions.
- Enforce shot count, duration budget, continuity constraints, and asset bindings.
- Generate shot-level video prompts.

Primary output:

```json
{
  "event_id": "EVT_001",
  "shots": [
    {
      "shot_id": "EVT_001_SH001",
      "event_id": "EVT_001",
      "sequence": 1,
      "shot_type": "establishing shot",
      "duration_sec": 4.0,
      "camera_language": "wide static shot",
      "screen_action": "The abandoned lighthouse interior is revealed under storm light.",
      "shot_prompt": "Cinematic 16:9 wide establishing shot inside an abandoned lighthouse...",
      "asset_bindings": ["Abandoned_Lighthouse", "Journalist"],
      "transition_to_next": "cut to closer tracking shot",
      "continuity_notes": "same room, same storm lighting"
    }
  ]
}
```

### 3.5 EventVideoAssembler

Create:

- `agents/event_video_assembler.py`

Responsibility:

- Group generated shot videos by `event_id`.
- Assemble shots into an event-level video segment.
- Preserve shot order inside each event.
- Produce event-level metadata for final post-production.

Output:

```json
{
  "event_id": "EVT_001",
  "ordered_shot_ids": ["EVT_001_SH001", "EVT_001_SH002"],
  "shot_video_paths": [],
  "event_video_path": "outputs/.../events/EVT_001/event_segment.mp4",
  "duration_sec": 18.0
}
```

## 4. Pipeline After Extension

The new full framework should become:

```text
User story
  -> OrchestratorAgent
  -> CharacterAgent creation
  -> WorldModelAgent
  -> StateAnalysisMachine
  -> WorldDeltaJudge / WorldStateTracker
  -> EventKeyframeRenderer
  -> AtomicEventExtractor
  -> EventAssetBinder
  -> AtomicEventComposer
  -> MultiShotStoryboardPlanner
  -> PathOrchestratorAgent
  -> ShotPlannerAgent
  -> VideoAgent
  -> EventVideoAssembler
  -> StorySupervisorAgent
  -> PostProductionAgent
  -> final video
```

Important distinction:

- `PathOrchestratorAgent` orders atomic events globally.
- `MultiShotStoryboardPlanner` expands one atomic event locally into shots.
- `ShotPlannerAgent` turns those shots into executable video generation prompts.

## 5. File-Level Implementation Plan

### Task 1: Add Atomic Event Schema

**Files:**

- Modify: `agents/schemas.py`
- Test: `tests/test_atomic_event_schema.py`

- [ ] Add dataclasses or typed structures for `AtomicEvent`, `EventAssetBinding`, `ComposedEvent`, `EventShot`, and `EventShotGroup`.
- [ ] Include `to_dict()` / `from_dict()` helpers consistent with the existing project style.
- [ ] Test JSON round-trip serialization.
- [ ] Test that required IDs are stable: `EVT_001`, `EVT_001_SH001`.

Expected test command:

```bash
pytest tests/test_atomic_event_schema.py -q
```

Expected output:

```text
passed
```

### Task 2: Implement AtomicEventExtractor

**Files:**

- Create: `agents/atomic_event_extractor.py`
- Test: `tests/test_atomic_event_extractor.py`

- [ ] Load `step_logs`, `world_states`, `world_deltas`, and `event_keyframes`.
- [ ] Build one candidate event per step as the minimal baseline.
- [ ] Merge multiple role actions inside a step into a single event summary.
- [ ] Preserve role-level evidence under `source_roles`.
- [ ] Preserve world evidence under `world_state_refs` and `world_delta_refs`.
- [ ] Save output as `atomic_events.json`.

Expected output:

```json
{
  "event_count": 3,
  "events": []
}
```

### Task 3: Implement EventAssetBinder

**Files:**

- Create: `agents/event_asset_binder.py`
- Test: `tests/test_event_asset_binder.py`

- [ ] Read `world_model`, `world_states`, and `event_keyframes`.
- [ ] Match event characters to character refs when available.
- [ ] Match event locations to world model location images.
- [ ] Match event items to world model item images.
- [ ] Attach event keyframes from matching source steps and roles.
- [ ] Validate that all bound paths exist and are non-empty.
- [ ] Save output as `atomic_events_with_assets.json`.

Expected behavior:

- Missing optional assets should be recorded as warnings.
- Missing required event records should fail the test.

### Task 4: Implement AtomicEventComposer

**Files:**

- Create: `agents/atomic_event_composer.py`
- Test: `tests/test_atomic_event_composer.py`

- [ ] Accept asset-bound atomic events.
- [ ] Assign event duration budget.
- [ ] Assign target shot count.
- [ ] Generate dramatic and visual function fields.
- [ ] Add continuity constraints.
- [ ] Save output as `composed_events.json`.

Default duration rule:

```text
event_duration_budget_sec = clamp(10 + estimated_importance * 12, min=8, max=28)
```

Default shot count rule:

```text
shot_count_target = 3 if duration <= 14
shot_count_target = 4 if 14 < duration <= 22
shot_count_target = 5 if duration > 22
```

### Task 5: Implement MultiShotStoryboardPlanner

**Files:**

- Create: `agents/multi_shot_storyboard_planner.py`
- Test: `tests/test_multi_shot_storyboard_planner.py`

- [ ] Expand each composed event into `shot_count_target` shots.
- [ ] Ensure shot durations sum to the event duration budget.
- [ ] Bind every shot to relevant event assets.
- [ ] Ensure shot IDs follow `EVT_001_SH001`.
- [ ] Save output as `event_shot_groups.json`.

Shot pattern baseline:

```text
shot 1: establish spatial context
shot 2: show character action
shot 3: reveal important object / reaction
shot 4: consequence or transition
shot 5: optional escalation or emotional emphasis
```

### Task 6: Add Event-to-Shot Pipeline Script

**Files:**

- Create: `scripts/run_event_to_multishot_pipeline.py`
- Test: `tests/test_event_to_multishot_pipeline_cli.py`

- [ ] Accept `--pipeline-result-json`.
- [ ] Accept `--output-dir`.
- [ ] Run extractor, binder, composer, and multi-shot planner.
- [ ] Write all intermediate JSON files.
- [ ] Print the final `event_shot_groups.json` path.

Expected command:

```bash
python scripts/run_event_to_multishot_pipeline.py \
  --pipeline-result-json outputs/<run>/pipeline_result.json \
  --output-dir outputs/<run>/event_multishot
```

Expected outputs:

```text
atomic_events.json
atomic_events_with_assets.json
composed_events.json
event_shot_groups.json
```

### Task 7: Connect Multi-Shot Output to VideoAgent

**Files:**

- Modify: `agents/video_agent.py`
- Create: `agents/event_video_assembler.py`
- Test: `tests/test_event_video_assembler.py`

- [ ] Add support for event-level shot groups.
- [ ] Preserve shot order inside each event.
- [ ] Generate one video file per shot.
- [ ] Assemble shot videos into one event segment.
- [ ] Preserve event-level metadata for post-production.

Expected metadata:

```json
{
  "event_id": "EVT_001",
  "event_video_path": "...",
  "shot_video_paths": []
}
```

### Task 8: Add HTML Visualization

**Files:**

- Modify: `scripts/generate_narrative_world_pipeline_html.py`
- Create: `scripts/generate_event_multishot_html.py`
- Test: `tests/test_generate_event_multishot_html.py`

- [ ] Add a new section named `Event-to-Multi-Shot`.
- [ ] Show each atomic event.
- [ ] Show bound visual assets.
- [ ] Show generated shot cards under each event.
- [ ] Show shot duration and asset bindings.

Expected HTML sections:

```text
Narrative Time
World Space
Atomic Events
Event-to-Multi-Shot
Backend Comparison
```

### Task 9: Full Integration Script

**Files:**

- Create: `scripts/run_tmm_event_multishot_full.py`
- Test: `tests/test_tmm_event_multishot_full_cli.py`

- [ ] Run the existing dual-backend narrative world pipeline.
- [ ] Run event extraction.
- [ ] Run asset binding.
- [ ] Run event composition.
- [ ] Run multi-shot planning.
- [ ] Optionally run video generation.
- [ ] Optionally run post-production.
- [ ] Generate final HTML.

Recommended CLI:

```bash
conda run -n mifage python scripts/run_tmm_event_multishot_full.py \
  --storyline "A journalist enters an abandoned lighthouse and discovers a cracked compass." \
  --characters Journalist,Caretaker,HiddenWitness \
  --backend-mode all_banana \
  --max-steps 3 \
  --output-dir outputs/tmm_event_multishot_demo \
  --dry-run-video \
  --log-level INFO
```

## 6. LLM Prompt Requirements

The new LLM prompts should explicitly enforce:

- An atomic event is not a shot.
- One atomic event must be expressed through multiple shots.
- Each shot must have a unique cinematic function.
- Shot durations must sum to the event budget.
- Asset bindings must use only available assets.
- Continuity must be maintained across shots in the same event.

Recommended system instruction:

```text
You are a film director and storyboard planner.
Do not treat one narrative event as one continuous shot.
For each atomic event, design a multi-shot cinematic sequence.
Each shot must have a different visual function, camera language, duration, and continuity purpose.
Return JSON only.
```

## 7. Validation Criteria

A run is valid when:

- Every atomic event has at least 2 shots.
- Every generated shot belongs to exactly one event.
- Every shot has `duration_sec`.
- Shot durations sum to the event duration budget within 0.5 seconds.
- Every event has at least one bound visual asset.
- Every asset path exists or is explicitly marked as missing.
- The final HTML can display event-level and shot-level structure.

## 8. Milestones

### Milestone 1: JSON-only TMM Layer

Deliverables:

- `atomic_events.json`
- `atomic_events_with_assets.json`
- `composed_events.json`
- `event_shot_groups.json`

No video generation required.

### Milestone 2: Visual Preview Layer

Deliverables:

- event-to-multishot HTML
- asset binding preview
- event timeline view

No video generation required.

### Milestone 3: Video Execution Layer

Deliverables:

- shot-level video generation
- event-level segment assembly
- metadata for post-production

### Milestone 4: Full TMM Workflow

Deliverables:

- one command from story to event-level multi-shot final video
- traceable JSON artifacts
- visual comparison HTML

## 9. Relationship to Current CineWorldSim

This plan does not replace the existing agents. It inserts a new event-centric layer between the narrative/world simulation stage and the downstream video generation stage.

Current:

```text
step_logs -> path planner -> shot planner -> video
```

Extended:

```text
step_logs + world_states + event_keyframes
  -> atomic events
  -> asset-bound event packages
  -> multi-shot event storyboard
  -> shot planner
  -> video
```

This keeps the current architecture intact while making the generated video more cinematic and less like a one-event-one-clip prompt expansion system.

