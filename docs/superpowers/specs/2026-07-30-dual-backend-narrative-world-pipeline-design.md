# Dual-Backend Narrative World Pipeline Design

## Goal

Build a testable end-to-end narrative pipeline that starts from a custom story line, defines characters, defines the cinematic world, runs a 3-5 step narrative deduction loop inside that world, updates both character state and world state over time, generates event keyframes at every step, and finally reconstructs the result from both time and space perspectives.

The same input must run twice with identical logic:

- `all_flux`: all world images and event keyframes use FLUX
- `all_banana`: all world images and event keyframes use nano_banana

The output must support direct comparison between the two backends.

## Scope

### In Scope

- One unified pipeline runner for custom story testing
- Story parsing and character creation
- Initial world extraction for locations and important props
- Structured world state tracking through the narrative loop
- Per-step world delta judgment for both scene-state and object-state changes
- Conditional world image refresh when a world element changes significantly
- Event keyframe generation for each role event
- Final time-based and space-based reconstruction and visualization
- Dual-backend execution with the same story input

### Out of Scope for V1

- Video generation
- Audio generation
- Multi-story batch orchestration
- Human-in-the-loop editing inside the loop
- Fine-grained world physics simulation
- Mixed-backend modes in the same run

## User-Facing Behavior

The user provides one custom story line.

The system then:

1. extracts or defines characters
2. extracts the cinematic world
3. generates initial world images
4. runs a 3-5 step narrative loop
5. updates character state and public world state after each step
6. checks whether any location or object state changed enough to justify new images
7. generates keyframes for every event
8. writes structured JSON outputs
9. renders visualizations for:
   - temporal narrative progression
   - spatial world evolution
   - backend comparison

The exact same story is then rerun under the second backend.

## Architecture

The pipeline should be organized around one deterministic control loop that separates:

- story setup
- world initialization
- narrative deduction
- world-state evolution
- event visualization
- final reconstruction

The current codebase already contains the main building blocks:

- `OrchestratorAgent`
- `WorldModelAgent`
- `StateAnalysisMachine`
- narrative topology visualization
- world preview visualization

V1 should reuse these components and introduce a thin orchestration layer plus a small set of new world-state modules rather than rewriting the narrative system.

## Proposed Flow

### Phase 1: Story Setup

Input:

- custom story text

Process:

- `OrchestratorAgent` parses the story
- creates character configurations
- assigns goals, role profiles, relations, and initial environment

Output:

- `story_bundle.json`
- normalized character configuration list

### Phase 2: World Initialization

Input:

- original story text
- character list

Process:

- `WorldModelAgent` extracts:
  - locations
  - objects / clues
  - summary world context
- backend-specific image generation is run for every extracted location and object
- a structured world state is created for step 0

Output:

- `world_state_step0.json`
- `narrative_artifacts/world_model/...`
- world context text injected into character environments

### Phase 3: Narrative Loop

Input:

- modeled story prompt
- character agents
- world state

Process:

- `StateAnalysisMachine` runs 3-5 steps
- for each step:
  - each character proposes an action
  - the state analysis machine aggregates actions
  - story anchor, story phase, and remaining gap are updated
  - public world state is updated

Output per step:

- character action logs
- updated world summary
- public events
- story completion estimate

### Phase 4: World Delta Check

Input:

- previous world state
- current aggregated event state

Process:

- evaluate both:
  - location state changes
  - object state changes
- decide which world elements have materially changed
- only changed elements trigger image refresh

Examples:

- location changes:
  - dark corridor becomes flooded
  - greenhouse glass is broken
  - archive door is opened
- object changes:
  - key is stolen
  - recorder is broken
  - evidence box is opened

Output:

- `world_delta_stepN.json`
- refresh task list for locations and items

### Phase 5: Event Keyframe Generation

Input:

- each role event in each step
- prior keyframe for continuity
- role portrait / identity references
- optionally world references for the active location

Process:

- generate a keyframe for every role event
- preserve identity continuity across steps
- preserve local scene continuity when possible

Output:

- `narrative_artifacts/keyframes/...`
- keyframe history by actor

### Phase 6: Final Reconstruction

The run is reconstructed in two complementary ways.

#### Time Reconstruction

This is the narrative network view.

It must show:

- step-by-step role actions
- per-step keyframes
- story phase changes
- public event propagation

#### Space Reconstruction

This is the world evolution view.

It must show:

- initial locations and objects
- which elements changed at each step
- updated world images over time
- state text per world element

#### Comparison Reconstruction

This is the backend comparison view.

It must support side-by-side inspection of:

- initial world assets
- updated world assets
- event keyframes
- final summaries

## Backends

V1 must support two run modes with identical logic:

### `all_flux`

- world initialization images: FLUX
- world update images: FLUX
- event keyframes: FLUX

### `all_banana`

- world initialization images: nano_banana
- world update images: nano_banana
- event keyframes: nano_banana

The runner should not mix tools inside a single run.

## New Modules

### 1. Pipeline Runner

Responsibility:

- execute the whole test pipeline from one story input
- select backend mode
- own output directory layout

Suggested file:

- `scripts/run_dual_backend_narrative_world_pipeline.py`

### 2. World State Tracker

Responsibility:

- define structured state for:
  - locations
  - objects
  - world summary
- serialize step snapshots

Suggested file:

- `narrative/world_state_tracker.py`

### 3. World Delta Judge

Responsibility:

- compare previous and current world state
- mark elements as:
  - unchanged
  - text-changed only
  - image-refresh required

Suggested file:

- `narrative/world_delta_judge.py`

### 4. World Renderer

Responsibility:

- backend-specific image generation for world assets
- shared interface for both FLUX and nano_banana

Suggested file:

- `narrative/world_renderer.py`

### 5. Event Keyframe Renderer

Responsibility:

- backend-specific generation of per-event keyframes
- continuity-aware prompt assembly

Suggested file:

- `narrative/event_keyframe_renderer.py`

### 6. Narrative World Visualizer

Responsibility:

- render:
  - time network
  - space evolution view
  - comparison preview

Suggested files:

- `scripts/generate_narrative_world_timeline_html.py`
- `scripts/generate_narrative_world_space_html.py`
- or a single combined generator if the HTML structure stays manageable

## Reuse of Existing Code

### Reuse Without Major Change

- `OrchestratorAgent`
- `WorldModelAgent.extract_world_blueprint`
- `WorldModelAgent.build_world_model`
- `StateAnalysisMachine.run_narrative_deduction`
- current keyframe generation pattern
- current narrative topology visualizers
- current world preview HTML pattern

### Extend

- `WorldModelAgent`
  - must support world-state structured snapshots, not just initial extraction
- `StateAnalysisMachine`
  - must emit richer world-facing event signals for location/object updates
- current preview generators
  - should be extended or paralleled for stepwise space evolution

## Data Model

### Story Bundle

Must contain:

- raw story text
- modeled story prompt
- character definitions
- backend mode

### World State Snapshot

Must contain:

- step id
- summary
- locations:
  - id
  - name
  - description
  - current_state_text
  - last_image_path
  - changed_this_step
- items:
  - id
  - name
  - description
  - current_state_text
  - owner / holder if applicable
  - last_image_path
  - changed_this_step

### Step Log

Must contain:

- step id
- role logs
- aggregated world summary
- story anchor
- story phase
- remaining gap
- world delta result

## Output Directory Layout

Each backend run should live in its own directory, for example:

- `outputs/pipeline_dual_backend_test/all_flux/...`
- `outputs/pipeline_dual_backend_test/all_banana/...`

Inside each run:

- `story_bundle.json`
- `world_state_step0.json`
- `world_state_step1.json`
- `world_state_step2.json`
- `world_state_step3.json`
- `step_logs.json`
- `world_deltas/`
- `narrative_artifacts/world_model/`
- `narrative_artifacts/keyframes/`
- `visualizations/timeline.html`
- `visualizations/world_space.html`
- `visualizations/combined_overview.html`

At the comparison root:

- `comparison_summary.json`
- `comparison_overview.html`

## Error Handling

### Image Generation Failure

If one image generation call fails:

- keep the structured state update
- record the failed asset
- do not abort the whole run unless configured to fail-fast

### World Update Failure

If world update judgment fails:

- preserve the last valid world state
- record fallback reason

### Narrative Loop Failure

If a step fails after some progress:

- persist checkpointed step results
- keep generated keyframes already written

## Testing Strategy

### V1 Default Test Case

- one custom story
- same story used for both backends
- 3-5 steps

### Required Acceptance Checks

1. Same story input can run under `all_flux`
2. Same story input can run under `all_banana`
3. Each run emits:
   - character outputs
   - world state snapshots
   - world deltas
   - event keyframes
   - timeline visualization
   - world-space visualization
4. At least one location or object can be updated conditionally during the loop
5. Final comparison page can inspect both runs side by side

## CLI Shape

V1 should expose one primary test command:

```bash
python scripts/run_dual_backend_narrative_world_pipeline.py \
  --story-file /abs/path/to/story.txt \
  --backend-mode all_flux \
  --max-steps 5 \
  --output-dir /abs/path/to/output
```

and the same for:

```bash
--backend-mode all_banana
```

Later, a wrapper can run both sequentially:

```bash
python scripts/run_dual_backend_narrative_world_pipeline.py \
  --story-file /abs/path/to/story.txt \
  --run-both \
  --max-steps 5 \
  --output-dir /abs/path/to/output
```

## Open Design Decisions Deferred

These are intentionally deferred from V1:

- whether world updates should use image-to-image references from the previous state
- whether event keyframes should also consume current world image references
- whether the comparison UI should be one page or multiple linked pages
- whether story phase should directly control visual style

## Risks

### 1. Cost and Runtime

Two full runs with image generation at every step can become expensive.

Mitigation:

- keep V1 at 3-5 steps
- only refresh world images on significant delta

### 2. Over-coupling Between Narrative and World Layers

If world updates are embedded directly inside `StateAnalysisMachine`, the module may become too large.

Mitigation:

- isolate world-state tracking and delta judgment into separate modules

### 3. Visualization Complexity

Time and space views can become confusing if fused too early.

Mitigation:

- build them as separate views first
- then add a combined overview

## Recommendation

Implement V1 as a shared pipeline with two backend modes and a single custom story input.

Do not optimize for mixed backends yet.

Do not expand to 10 or 20 steps before the 3-5 step closed-loop test is stable.

Do not attempt video generation inside this testing pipeline.

The first milestone is a fully reproducible dual-run narrative-world test with:

- dynamic character actions
- dynamic world state
- conditional world image refresh
- event keyframes
- temporal and spatial visualization
