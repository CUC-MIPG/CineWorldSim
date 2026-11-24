# Narrative World Video Pipeline Redesign Plan

> Scope: implement the requested redesign after the current pipeline has been
> backed up. This document is a plan only; production code is not changed by
> this phase.

## 1. Target Architecture

```text
story text
  -> story_bundle + character/world blueprint
  -> initial asset stage
       character portraits
       item reference images
       location {wide, medium, close, extreme_close}
  -> step-by-step narrative simulation
       role actions
       event/story descriptions
       world-state LLM decision
       selected world-view refresh
       event keyframe
  -> narrative node selection
       one role/step by default
       same-step same-location direct interaction may be grouped
  -> path orchestration + validation
  -> VideoAgent task planning
       reference mode
       establishing/transition insertion
       shot timing and shot count
       5s/124 or 10s/243 frame selection
  -> H3 prompt adapter/skill
  -> MiniMax H3 vLLM
  -> ordered concatenation + visualization
```

The central invariant is that simulation state, world image state, event
description, references, and video tasks all carry explicit `step`, `location`,
`characters`, `items`, and source-node identifiers.

## 2. Phase A: Shared Schemas and Utilities

### Files

- Create `narrative/pipeline_schemas.py`.
- Create `narrative/asset_reference_selector.py`.
- Modify `agents/llm_tool.py` only if a shared structured-output helper is
  required.
- Add schema and validation tests under `tests/`.

### Work

1. Define normalization helpers for asset IDs, names, paths, view names, node
   IDs, and timecodes.
2. Define allowed location views:
   `wide`, `medium`, `close`, `extreme_close`.
3. Define allowed video durations:
   `5 -> 124` and `10 -> 243`.
4. Add strict validation for world delta decisions, video reference modes,
   auxiliary tasks, and shot plans.
5. Preserve old fields such as `last_image_path`, `video_description`, and
   existing task JSON keys while adding new fields.

### Exit tests

- Invalid view names and frame counts are rejected.
- Old pipeline results can still be loaded.
- A valid plan round-trips through JSON without losing paths or source IDs.

## 3. Phase B: Initial Character, Item, and Location Assets

### Files

- Modify `scripts/run_dual_backend_narrative_world_pipeline.py`.
- Modify `narrative/world_renderer.py`.
- Modify `narrative/simulation_scaffold.py`.
- Add `narrative/initial_asset_renderer.py` if the rendering responsibilities
  become too large for the current modules.
- Add tests for prompts, paths, reuse, dry-run, and resume.

### Work

1. Run `OrchestratorAgent` before simulation and persist the character
   configurations.
2. Generate one initial portrait per character before step 1. The prompt must
   explicitly require:
   - the named character is the dominant visual subject;
   - maximum subject occupancy and clear identity;
   - minimal background;
   - no crowds, pedestrians, or unrelated people;
   - only the character-specific clothing, props, and attributes.
3. Generate one isolated initial image per item using the same
   maximum-subject/minimal-background rule.
4. Generate four initial images for every location:
   - wide: spatial layout and navigable geometry;
   - medium: central actionable area;
   - close: relevant surface/object interaction area;
   - extreme close: texture or detail area.
5. Save a manifest before simulation:

```text
narrative_artifacts/initial_assets/initial_assets.json
narrative_artifacts/portraits/<Character>_portrait.png
narrative_artifacts/world_model/items/<Item>/initial_<tool>.png
narrative_artifacts/world_model/locations/<Location>/<view>_initial_<tool>.png
```

6. Link the generated portraits into every character config and simulation
   checkpoint. Do not wait for a role action to create a portrait.
7. Keep existing files usable through aliases:
   `nano_banana.png` may point to or duplicate the `wide` location image, while
   the new manifest remains authoritative.

### Exit tests

- A non-dry initialization creates all expected character/item/location assets.
- Portrait and item prompts contain subject-dominance and minimal-background
  instructions.
- Every location has exactly the four declared view keys.
- Existing valid images are reused during resume.

## 4. Phase C: LLM World-State Decisions and View-Specific Updates

### Files

- Replace the rule-based decision path in `narrative/world_delta_judge.py`.
- Modify `narrative/world_state_tracker.py`.
- Modify `scripts/run_dual_backend_narrative_world_pipeline.py`.
- Add tests under `tests/test_world_delta_judge.py` and
  `tests/test_world_state_tracker.py`.

### LLM prompt contract

The judge receives:

- the story-level context;
- the current step's complete event/story descriptions for all roles;
- current location and item state;
- the previous image manifest and current view state.

The judge must follow these rules:

1. For a location or item, update only when the event explicitly changes its
   physical appearance: color, shape, surface, texture, visible structure,
   damage, opening/closing state, flooding, burning, or another observable
   visual property.
2. Movement, possession, intention, discovery, or reasoning alone is not a
   visual world-state change.
3. If there is no explicit physical change, return no update.
4. An updated image contains no people.
5. An updated image contains no elements not required by the stated change.
6. The image describes only the resulting state of that one location or item.
7. For a changed location, select only the view that best exposes the changed
   region: wide, medium, close, or extreme_close.
8. The renderer receives the previous image for that selected view; if there is
   no previous updated image, it receives the matching initial view.

### State model

Extend each asset row with:

```json
{
  "views": {
    "wide": {
      "current_state_text": "...",
      "last_image_path": "...",
      "image_history": []
    }
  },
  "changed_views_this_step": []
}
```

Items may use a single `isolated` view internally, while retaining
`last_image_path` for compatibility.

### Rendering order

```text
step role logs
  -> LLM delta decision
  -> validate decision
  -> render only changed assets/views with previous image reference
  -> update world_state_stepN.json
  -> render event keyframes
```

### Exit tests

- Keyword presence alone cannot trigger a world update.
- A purely mental/action event produces empty update lists.
- A physical change produces exactly the affected asset and selected view.
- The update prompt prohibits people and unrelated visual elements.
- Previous view image paths are passed to the image backend.

## 5. Phase D: Relevant Event Keyframes

### Files

- Modify `narrative/event_keyframe_renderer.py`.
- Modify `scripts/run_dual_backend_narrative_world_pipeline.py`.
- Reuse `narrative/asset_reference_selector.py`.
- Add or extend keyframe renderer tests.

### Work

1. Construct one canonical event record per role or interaction event with:
   `story_description`, `video_description`, action, location, characters,
   mentioned items, and world-state changes.
2. Render keyframes only after the current step's world updates have been
   applied.
3. Select references from event semantics:
   - the relevant character initialization portrait(s);
   - the event location's current best view;
   - only items explicitly mentioned or acted upon;
   - the current event keyframe predecessor only when a separate continuity
     policy permits it.
4. For a single-role event, use one character portrait.
5. For a same-location direct interaction event, allow the involved character
   portraits only.
6. Do not pass all current world images by default.
7. Store the exact reference decision in each keyframe record:

```json
{
  "reference_selection_reason": "...",
  "character_reference_paths": [],
  "location_reference_paths": [],
  "item_reference_paths": [],
  "excluded_assets": []
}
```

### Exit tests

- An event mentioning one item does not receive unrelated items.
- A single-role event does not receive other character portraits.
- A multi-person interaction receives all and only involved characters.
- Keyframe generation occurs after the world state checkpoint for that step.

## 6. Phase E: Node Selection and Path Orchestration

### Files

- Modify `agents/narrative_node_selector.py`.
- Modify `agents/path_orchestrator_agent.py`.
- Modify `scripts/run_narrative_world_path_pipeline.py`.
- Modify `narrative/visual_prompt_constraints.py` if interaction predicates
  need shared normalization.
- Add selector and path validation tests.

### Candidate contract

The selector receives one role action candidate for each `(step, role)` and
must produce:

- a default `single_role_step` node for each selected role action;
- a `multi_role_interaction` node only when all conditions hold:
  1. same step;
  2. same physical location;
  3. direct interaction or mutual causal action;
  4. the event text explicitly involves both characters;
  5. grouping improves event coherence rather than merely sharing a scene.

The selector prompt must explicitly prohibit merging:

```text
step::1::CharacterA + step::1::CharacterB
```

unless the above interaction conditions are satisfied. Three independent role
actions in one step must remain three nodes and therefore three video tasks.

### Path stage

The path planner receives node granularity and source-role metadata. It may
reorder nodes but may not merge or rewrite their role membership. A validator
runs after every LLM path response and rejects:

- duplicate node IDs;
- missing selected node IDs;
- illegal role grouping;
- cross-location interaction grouping;
- more than one video task for a single-role node.

### Exit tests

- Three independent role actions at one step produce three nodes.
- Two same-location interacting roles produce one grouped node.
- Path output preserves node granularity and source IDs.
- Invalid LLM grouping is repaired or rejected deterministically.

## 7. Phase F: VideoAgent Reference Modes

### Files

- Modify `agents/video_agent.py`.
- Modify `agents/h3_prompt_adapter.py` and `agents/skills/adapter.py`.
- Modify `scripts/run_full_narrative_world_video_pipeline.py`.
- Add tests for each mode and audit metadata.

### Modes

`VideoAgent` selects one mode from event semantics and available assets:

1. `event_keyframe_only`
   - event keyframe(s) only;
   - used when the event is already visually complete and extra references
     would add irrelevant content.
2. `semantic_assets`
   - character initialization portrait(s);
   - current location view;
   - explicitly mentioned item image(s);
   - no event/world update keyframe unless separately allowed.
3. `all_allowed`
   - event keyframe;
   - involved character portrait(s);
   - current location view;
   - explicitly mentioned item/world update images.

The mode is a decision, not a capacity fill. `max_reference_images` remains a
hard upper bound only. The task JSON records selected and excluded paths plus a
reason.

The H3 prompt must repeat the semantic policy in prose: use only references
that correspond to content stated in the event; do not add unmentioned
characters, props, crowds, pedestrians, or background elements.

### Tail-frame policy

Keep tail-frame continuity separate. Its default remains disabled. If enabled,
the continuity judge decides based on previous and current event text, role
identity, and location relationship; the selected reference mode does not
implicitly enable it.

### Exit tests

- No event receives four references merely because the cap is four.
- Unmentioned items and locations are excluded.
- Single-role events use at most one character portrait.
- Each task records the selected reference mode and reason.

## 8. Phase G: Establishing and World/Item Transition Videos

### Files

- Modify `agents/video_agent.py`.
- Add `agents/auxiliary_video_planner.py` if needed.
- Modify task ordering and concatenation logic.
- Add tests for insertion order and deduplication.

### Work

1. Identify the first event for every location in the ordered path.
2. Insert an `establishing` task before that event using the location wide or
   medium view and no characters unless the establishing event explicitly
   requires them.
3. For each world-state location/item update, identify the preceding and
   following event tasks.
4. Insert one `world_transition` or `item_transition` task between them when
   the changed visual state should be shown independently.
5. Transition prompts describe only the world/item change and do not add
   people unless the transition itself is explicitly character-driven.
6. Ensure task IDs are deterministic and reruns do not duplicate inserted
   tasks.

### Exit tests

- Every first location event has at most one establishing task.
- Every qualifying world/item update has a correctly positioned transition task.
- Existing event task order is preserved relative to inserted tasks.
- Resume does not regenerate completed auxiliary videos.

## 9. Phase H: LLM Shot and Frame Planning

### Files

- Modify `agents/video_agent.py`.
- Modify H3 prompt construction and skill adapter integration.
- Modify `agents/generation_backends.py` only where explicit frame validation
  is needed.
- Add shot-plan and frame-plan tests.

### LLM output

The planner receives the event `story_description`, dialogue, location, roles,
items, story phase, and neighboring event summaries. It returns:

```json
{
  "duration_sec": 5,
  "num_frames": 124,
  "shots": [
    {
      "shot_index": 1,
      "start_timecode": "00:00.000",
      "purpose": "...",
      "description": "...",
      "camera": "...",
      "dialogue": "none",
      "bgm": "..."
    }
  ]
}
```

Rules:

1. Choose only 5 or 10 seconds.
2. Map 5 seconds to 124 frames and 10 seconds to 243 frames.
3. Choose shot count based on event complexity, normally 1-4 and never merely
   because a fixed default exists.
4. Use increasing in-range start timecodes.
5. Keep the H3 prompt's shot timing explicit, for example:
   `[Shot 2] At 00:03.000`.
6. Preserve dialogue as `none` when no dialogue is required.
7. Keep `bgm` defined separately from dialogue.
8. If the LLM fails, use a deterministic one-shot fallback with a valid
   5-second/124-frame or 10-second/243-frame pair.

### Exit tests

- Arbitrary frame counts are rejected.
- A 5-second plan sends `124`; a 10-second plan sends `243`.
- H3 prompt contains valid shot timing and the story description.
- Fixed `--event-shot-count` is treated as a compatibility override only when
  explicitly supplied; the new default is LLM planning.

## 10. Phase I: Manifests, Visualization, and Resume

### Files

- Modify `scripts/generate_full_pipeline_stage_visualization.py`.
- Modify `scripts/generate_narrative_world_pipeline_html.py`.
- Modify `scripts/run_full_narrative_world_video_pipeline.py`.
- Modify `scripts/run_two_adventure_demo_pipelines.py` if its CLI needs new
  flags.

### Work

Expose the following in `pipeline_result.json`, `full_pipeline_manifest.json`,
and HTML visualization:

- initial assets and all location views;
- per-step world-state decisions;
- changed assets and changed views;
- event-to-reference selection;
- node granularity and grouped roles;
- establishing/transition tasks;
- shot/frame plan;
- generated task status and failure reason.

Use stage-level partial files so a failed Banana, LLM, or vLLM call can resume
without rebuilding valid initialization or simulation artifacts.

## 11. Verification Sequence

Run in this order:

1. Schema and utility tests.
2. Initialization and prompt tests.
3. World delta and tracker tests.
4. Keyframe and reference-selection tests.
5. Node/path validation tests.
6. VideoAgent shot/frame/reference tests.
7. Full existing regression suite.
8. Dry-run integration for both adventure demos.
9. One real single-event video using the local MiniMax vLLM service.
10. Full real demo run only after the single-event verification succeeds.

No completion claim should be made until the generated manifests, selected
reference records, frame counts, auxiliary task ordering, and final video
artifacts have been checked.
