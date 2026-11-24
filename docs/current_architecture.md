# CineWorldSim Current Architecture

## 1. Project Goal

CineWorldSim is a multi-agent video content generation system built around narrative planning, world modeling, shot planning, video generation, and post-production.

The current implementation focuses on two parallel axes:

1. **Narrative axis**: story understanding, character-driven deduction, path orchestration, and shot ordering.
2. **World axis**: spatial world modeling, location/item updates, and visual consistency across steps.

The system is designed to support iterative experiments, backend comparison, resumable runs, and structured artifact logging.

## 2. High-Level Pipeline

Current production flow:

```text
User story
  -> OrchestratorAgent
  -> CharacterAgent creation
  -> WorldModelAgent / world blueprint extraction
  -> Initial world images
  -> StateAnalysisMachine narrative deduction
  -> World delta judgment and world updates
  -> Event keyframe generation
  -> Path orchestration / shot planning
  -> Video generation
  -> Post-production
  -> HTML visualization + info pool logging
```

The pipeline can also run in a reduced mode for experiments:

```text
story -> world model -> step logs -> world states -> event keyframes -> HTML
```

## 3. Core Modules

### 3.1 OrchestratorAgent

File:

- `narrative/simulation_scaffold.py`

Responsibility:

- Parse the user story.
- Identify characters.
- Create character configurations.
- Inject final goals, role profiles, known relations, and environment context.
- Build `CharacterAgent` instances.

Output:

- `character_configs`
- character agent objects

### 3.2 WorldModelAgent

File:

- `narrative/simulation_scaffold.py`

Responsibility:

- Extract a world blueprint from the story.
- Identify important `locations` and `items`.
- Build the world context text.
- Enrich the story prompt and character environments with spatial knowledge.

Output:

- `world_model`
- `context_text`
- initial prompts for spatial image generation

### 3.3 WorldRenderer

File:

- `narrative/world_renderer.py`

Responsibility:

- Dispatch image generation to the selected backend.
- Map backend mode to concrete tool names.
- Retry failed generations.
- Normalize image extensions when the tool returns an unexpected suffix.

Backends:

- `all_flux` -> `flux_local`
- `all_banana` -> `nano_banana`

### 3.4 StateAnalysisMachine

File:

- `narrative/simulation_scaffold.py`

Responsibility:

- Run the multi-step narrative deduction loop.
- Ask each character agent to propose an action.
- Aggregate role outputs into a step summary.
- Update the public environment.
- Track story phase and completion score.
- Broadcast updated environment back to all characters.

Output:

- `step_logs`
- step-level story anchors
- step-level completion state

### 3.5 WorldDeltaJudge

File:

- `narrative/world_delta_judge.py`

Responsibility:

- Determine whether the narrative event changes any location or item state.
- Mark assets that require a refreshed image.
- Produce per-step world deltas.

Current behavior:

- Rule-based matching over step text.
- If the step mentions an asset and a change keyword, the asset is treated as updated.

### 3.6 WorldStateTracker

File:

- `narrative/world_state_tracker.py`

Responsibility:

- Maintain step-indexed world snapshots.
- Persist the current text state and image history for each location and item.
- Apply deltas to the previous state.

Output:

- `world_state_step0.json`
- `world_state_stepN.json`

### 3.7 EventKeyframeRenderer

File:

- `narrative/event_keyframe_renderer.py`

Responsibility:

- Generate one keyframe per role per step.
- Use the selected backend image tool.
- Produce narrative event stills for the timeline.

Output:

- `narrative_artifacts/event_keyframes/step_XXX/<Role>_event_keyframe.*`

### 3.8 Path Orchestrator and Shot Planning

Files:

- `agents/path_orchestrator_agent.py`
- `agents/path_orchestrator_mcts_agent.py`
- `agents/shot_planner_agent.py`

Responsibility:

- Order narrative step nodes into a final playable shot path.
- Support both direct LLM ordering and MCTS-based search.
- Translate ordered nodes into shot-level prompts.

This layer bridges:

- the narrative deduction graph
- and downstream video generation

### 3.9 VideoAgent

File:

- `agents/video_agent.py`

Responsibility:

- Convert shot instructions into actual video generation calls.
- Support reference-driven or keyframe-driven video mode.

### 3.10 Post-Production Agent

Files:

- `agents/post_production_agent.py`
- `agents/post_production_tools.py`

Responsibility:

- Dubbing
- Narration
- Music
- Editing / clip assembly

Current design uses tool wrappers and keeps the actual media pipeline separate from the orchestration layer.

### 3.11 Info Pool

File:

- `agents/info_pool.py`

Responsibility:

- Record outputs from all agents.
- Persist timeline events and latest snapshots.
- Support debugging, resumable runs, and full traceability.

## 4. Main Runtime Entrypoints

### 4.1 Dual-backend narrative world pipeline

File:

- `scripts/run_dual_backend_narrative_world_pipeline.py`

Purpose:

- Run the narrative/world pipeline with `all_flux`, `all_banana`, or both.

Typical output:

- `pipeline_result.json`
- `step_logs.json`
- `world_state_step*.json`
- `world_deltas/*.json`
- `narrative_artifacts/*`
- `visualizations/combined_overview.html`

### 4.2 Full agent workflow

Files:

- `scripts/run_three_agents_only.py`
- `scripts/run_five_agents_full.py`

Purpose:

- Run the production workflow with director, shot, story supervisor, video, and post-production stages.

### 4.3 RAG and MovieProducer

Files:

- `scripts/rag_build.py`
- `scripts/rag_query.py`
- `scripts/movie_producer_query.py`

Purpose:

- Build and query the book-level knowledge base.
- Expose a multi-knowledge-base `MovieProducer` agent for narrative planning support.

## 5. Input / Output Contract

### Inputs

Typical pipeline input:

```json
{
  "storyline": "story text",
  "characters": ["A", "B", "C"],
  "backend_mode": "all_flux | all_banana",
  "max_steps": 3,
  "dry_run": false
}
```

Optional extended inputs:

- `story_file`
- `reference-assets-json`
- `prompt-library`
- video mode and post-production flags

### Outputs

The full pipeline produces:

- story bundle metadata
- character configs
- narrative step logs
- world states
- world deltas
- event keyframes
- rendered world images
- HTML visualization
- info pool event logs

## 6. Directory Layout

Representative output layout:

```text
outputs/<run_name>/
  pipeline_result.json
  story_bundle.json
  step_logs.json
  world_state_step0.json
  world_state_step1.json
  world_deltas/
  narrative_artifacts/
    world_model/
    world_updates/
    event_keyframes/
  visualizations/
    combined_overview.html
```

## 7. Design Principles

### 7.1 Separate narrative time from world space

The system treats story progression and spatial asset updates as two related but distinct dimensions.

- Narrative time is represented by `step_logs`.
- World space is represented by `world_states` and `world_deltas`.

This separation makes it easier to inspect plot logic independently from visual continuity.

### 7.2 Keep agent responsibilities narrow

Each agent has one main job:

- Orchestrator: understand and configure
- State analysis: deduce and broadcast
- World model: build spatial context
- Renderer: make assets
- Path orchestrator: order shots
- Video: generate motion
- Post-production: assemble final media

This reduces coupling and makes each stage easier to debug.

### 7.3 Preserve traceability

The workflow writes intermediate artifacts at every stage.

This makes it possible to:

- resume interrupted runs
- compare backends
- inspect step-by-step outputs
- debug failures without rerunning the entire pipeline

### 7.4 Prefer structured JSON over implicit state

The codebase relies on explicit JSON artifacts for:

- step logs
- world states
- world deltas
- character configs
- media metadata

This keeps the system inspectable and script-friendly.

### 7.5 Support backend experimentation

Image generation is backend-pluggable:

- local FLUX
- nano_banana

The same narrative pipeline can be run against either backend for comparison.

## 8. Current Status

Already in place:

- narrative deduction scaffold
- dual-backend world pipeline
- world state tracking
- event keyframe generation
- HTML visualization for single or dual runs
- RAG commands and MovieProducer query flow
- agent workflow entrypoints
- info pool logging

Known limitations:

- world delta judgment is still rule-based
- current HTML is a result viewer, not a fully interactive editor
- video / post-production integration is available as workflow code, but not all media backends are equally mature

## 9. Recommended Usage

For backend comparison:

```bash
python scripts/run_dual_backend_narrative_world_pipeline.py --run-both ...
```

For full agent workflow:

```bash
python scripts/run_five_agents_full.py ...
```

For knowledge retrieval:

```bash
python scripts/movie_producer_query.py ...
```

