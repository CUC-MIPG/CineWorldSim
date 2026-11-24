# Agents Pipeline

This module implements core agents by tracing CameraAtist workflows:

1. `DirectorAgent`
- Story + scene generation (`Fast_Story_Cot` style)
- Structured storyboard conversion (`Story_design_Prompt` style)
- Scene image generation via provider switch:
  - `IMAGE_GEN_PROVIDER=flux` -> Flux API style
  - `IMAGE_GEN_PROVIDER=nano_banana` -> direct nano_banana API call (no backend service)
- Input-asset aware image policy:
  - `character_image_policy=auto` (generate only missing character images)
  - `character_image_policy=always` (always regenerate character images)
  - `character_image_policy=never` (never generate character images)
  - `scene_image_policy=auto` (generate only missing scene images)
  - `scene_image_policy=always` (always regenerate scene images)
  - `scene_image_policy=never` (never generate scene images)
- Optional external asset attachment:
  - `--reference-assets-json` for unified character/scene reference mapping
  - supports one-to-many references for each character and scene

2. `ShotAgent`
- Iterative shot generation (`Slow_Shot_Cot` + `Loop_storyboard_generation` style)
- Terminates scene loop when `Scene End Point` is observed or max limit reached
- Shot JSON normalization + schema auto-repair:
  - normalizes `Shots` payload variants
  - fills `shot_type`, `camera_angle`, `camera_movement`, `character_names`, `scene_elements`

3. `VideoAgent`
- LLM video prompt generation from shot/scene/characters
- Video generation via provider switch:
  - `VIDEO_GEN_PROVIDER=magref` -> Magref API style
  - `VIDEO_GEN_PROVIDER=wan_r2v` -> direct wan2.6-r2v call (no backend service)
  - `VIDEO_GEN_PROVIDER=minimax_h3` -> local MiniMax-H3 via `conda run -n minimaxh3hf`
- Video mode switch:
  - `reference` mode: reference-image driven video generation
  - `text` mode: text-to-video generation
  - `image` mode: image-to-video generation from the first available image
  - `first_last` mode: first-and-last-frame-to-video generation from the first two available images
  - `last_frame` mode: last-frame-to-video generation
  - `keyframe` mode: keyframe pipeline (currently pseudocode placeholder)
- JSON task export and ffmpeg concat
- Task queue mode compatible with CameraAtist `video_info.json` list format
- Event-orchestration mode:
  - consumes `narrative_nodes.json` + `narrative_path_plan.json`
  - follows PathOrchestrator order
  - automatically selects character event keyframes and world-space update images
  - generates one multi-reference event video per selected narrative node

4. `StorySupervisorAgent`
- Story supervision and post-planning based on storyline/scene/shot content
- Generates scene-level plan JSON including:
  - shot_plans (per-shot duration planning and pacing notes)
  - narration text + duration
  - dialogue lines + durations
  - music description + duration
  - editing notes
- Output is directly consumable by `PostProductionAgent`

5. `PostProductionAgent`
- Post-production orchestration for:
  - dubbing
  - background music
  - narration
  - editing/mastering
- Tool status:
  - dubbing / narration / bgm: pseudocode placeholders (`*.todo.txt`)
  - editing/mastering: real implementation via OpenCV + ffmpeg

6. `InfoPool`
- Workflow-level output recorder
- Automatically records:
  - pipeline start / complete events
  - each agent's stage output payload
- Storage:
  - `output_dir/info_pool/events.jsonl` (append-only event log)
  - `output_dir/info_pool/latest_snapshot.json` (latest-by-agent/stage snapshot)

## Orchestration

- Uses LangGraph (`Director -> Shot -> StorySupervisor -> Video -> PostProduction`)
- Checkpoint file: `output_dir/pipeline_state.json`
- Supports resume execution

## Run

Recommended fixed-entry scripts:

Prompt template library argument:
- `--prompt-library` (default: `/data1/AgentG/hhb/CameraMaster/prompt_library/default`)

Reference assets argument:
- `--reference-assets-json` (optional, single JSON file for both character and scene refs)
- Example file: `/data1/AgentG/hhb/CameraMaster/configs/reference_assets.example.json`

3-agent only:

```bash
conda run -n mifage python scripts/run_three_agents_only.py \
  --storyline "A young detective uncovers a hidden conspiracy in a rainy cyberpunk city." \
  --output-dir outputs/three_agents_only_run \
  --dry-run-video \
  --no-strict-online
```

```bash
conda run -n mifage python scripts/resume_three_agents_only.py \
  --output-dir outputs/three_agents_only_run \
  --dry-run-video \
  --no-strict-online
```

5-agent full:

```bash
conda run -n mifage python scripts/run_five_agents_full.py \
  --storyline "A young detective uncovers a hidden conspiracy in a rainy cyberpunk city." \
  --output-dir outputs/five_agents_full_run \
  --dry-run-video \
  --no-strict-online
```

```bash
conda run -n mifage python scripts/resume_five_agents_full.py \
  --output-dir outputs/five_agents_full_run \
  --dry-run-video \
  --no-strict-online
```

Flexible entry scripts:

Strict online mode (default):

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "A young detective uncovers a hidden conspiracy in a rainy cyberpunk city." \
  --output-dir outputs/three_agents_run \
  --enable-story-supervisor \
  --no-enable-post-production \
  --strict-online
```

Fallback mode (for offline/dev):

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "A young detective uncovers a hidden conspiracy in a rainy cyberpunk city." \
  --output-dir outputs/three_agents_run \
  --enable-story-supervisor \
  --no-enable-post-production \
  --no-strict-online \
  --dry-run-video
```

Event-orchestration video generation:

```bash
python scripts/run_orchestrated_event_video_generation.py \
  --narrative-nodes-json outputs/current_full_built_pipeline_demo_dual/all_banana/path_orchestration/narrative_nodes.json \
  --path-plan-json outputs/current_full_built_pipeline_demo_dual/all_banana/path_orchestration/path_llm_direct/narrative_path_plan.json \
  --output-dir outputs/current_full_built_pipeline_demo_dual/all_banana/event_videos_minimax_h3 \
  --max-reference-images 8 \
  --no-dry-run \
  --log-level INFO
```

Resume from checkpoint:

```bash
conda run -n mifage python scripts/resume_three_agents.py \
  --output-dir outputs/three_agents_run \
  --enable-story-supervisor \
  --no-enable-post-production \
  --strict-online
```

Run with post-production pseudo stage:

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "A young detective uncovers a hidden conspiracy in a rainy cyberpunk city." \
  --output-dir outputs/three_agents_run \
  --enable-story-supervisor \
  --enable-post-production \
  --dry-run-video \
  --no-strict-online
```

Run story supervisor only:

```bash
conda run -n mifage python scripts/run_story_supervisor.py \
  --storyboard-json outputs/three_agents_run/storyboard.json \
  --output-dir outputs/three_agents_run/story_supervisor \
  --no-strict-online
```

Output files:
- `storyboard.json`
- `story_supervisor/story_supervisor_plan.json` (if enabled)
- `videos/video_info.json`
- `final_concat.mp4` (if video clips exist)
- `post_production/post_production_info.json` (if enabled)
- `info_pool/events.jsonl`
- `info_pool/latest_snapshot.json`

Run queued tasks from `video_info.json`:

```bash
conda run -n mifage python scripts/process_video_tasks.py \
  --task-json outputs/three_agents_run/videos/video_info.json \
  --strict-online
```

## Env Keys

Each LLM module reads its own `.env` variables first, then falls back to `OPENAI_*`:

- Director: `DIRECTOR_API_KEY`, `DIRECTOR_BASE_URL`, `DIRECTOR_MODEL_NAME`
- Shot: `SHOT_API_KEY`, `SHOT_BASE_URL`, `SHOT_MODEL_NAME`
- Video: `VIDEO_API_KEY`, `VIDEO_BASE_URL`, `VIDEO_MODEL_NAME`
- Story Supervisor: `SUPERVISOR_API_KEY`, `SUPERVISOR_BASE_URL`, `SUPERVISOR_MODEL_NAME`
- Story Supervisor duration controls: `SUPERVISOR_SHOT_MIN_SEC`, `SUPERVISOR_SHOT_MAX_SEC`, `SUPERVISOR_SCENE_SEC_PER_SHOT`, `SUPERVISOR_DURATION_TOLERANCE_SEC`
- Fallback: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`

Generation backends:

- Image provider: `IMAGE_GEN_PROVIDER=flux|nano_banana`; narrative pipelines additionally support `--backend-mode all_flux2_klein`
- Flux: `FLUX_API_URL`, `FLUX_WIDTH`, `FLUX_HEIGHT`
- FLUX.2 Klein local service: `FLUX2_KLEIN_API_URL`, `FLUX2_KLEIN_HEALTH_URL`, `FLUX2_KLEIN_WIDTH`, `FLUX2_KLEIN_HEIGHT`, `FLUX2_KLEIN_STEPS`, `FLUX2_KLEIN_SEED`, `FLUX2_KLEIN_TIMEOUT_SEC`
- Nano Banana: `NANO_BANANA_API_KEY`, `NANO_BANANA_BASE_URL`, `NANO_BANANA_MODEL_NAME`
- Video provider: `VIDEO_GEN_PROVIDER=magref|wan_r2v|minimax_h3`
- Magref: `MAGREF_API_URL`
- Wan2.6-r2v: `WAN_R2V_API_KEY`, `WAN_R2V_BASE_HTTP_API_URL`, `WAN_R2V_MODEL_NAME`
- MiniMax-H3: `MINIMAX_H3_MODEL_DIR`, `MINIMAX_H3_CONDA_ENV`, `MINIMAX_H3_SCRIPT`, `MINIMAX_H3_TASK`
- MiniMax-H3 task modes: `MINIMAX_H3_TASK=auto|t2va|fl2va|fl2va_last_frame|fl2va_first_last|ref2va`
- Timeout override (optional): `MAGREF_TIMEOUT_SEC`, `WAN_R2V_TIMEOUT_SEC`, `MINIMAX_H3_TIMEOUT_SEC` (`0`/`None` disables timeout)
