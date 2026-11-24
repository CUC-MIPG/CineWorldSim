# MovieBench Narrative-to-Video Pipeline

## Experimental Tracks

The pipeline uses the restructured MovieBench hierarchy: L4 is the movie-level global story, L3 is a local story arc, L2 is an event, and L1 is a source shot. One L4-driven narrative simulation is run per invocation; Track A and Track B share the resulting narrative network so the comparison does not depend on two independently sampled simulations. No MCTS component is used.

- **Track A, anchored:** select one or more L3 arcs as anchors. Resolve the L2 events and L1 shots assigned to each L3. For each L2 candidate, retrieve up to three related generated simulation nodes using character overlap and lexical overlap as network context. `PathOrchestratorAgent` orders the L2 events against the L3 storyline. Expand that order into the associated L1 shots, render one keyframe per L1, and generate one event video per L1 shot.
- **Track B, unanchored:** use only the L4 global story and its character profiles to run simulation. Order the generated simulation nodes against L4 with `PathOrchestratorAgent`, then generate one event video per selected node. The runner does not load Track B withheld/reference-answer fields.

The L3/L2/L1 MovieBench annotations condition Track A only. They do not replace the L4 global story used to create the shared simulation network. `--sample-id` filters Track A; Track B remains one movie-level run.

## Inputs

Use the restructured movie folder containing:

```text
<movie>_llm_l4/
  l4_global/movie_bible.json
  l3_arcs/story_arcs.json
  l2_events/units/*.json
  l1_shots/units/*.json
```

For the current American Beauty sample:

```text
outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4
```

## Dry-Run Wiring Test

This checks both tracks and writes task plans without online simulation, image generation, story/dialogue backfill, or video generation:

```bash
python -u scripts/run_moviebench_narrative_video_pipeline.py \
  --dataset-root outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --track both \
  --sample-id 0001_American_Beauty_L3ARC_001 \
  --output-dir outputs/moviebench_video_pipeline/0001_American_Beauty/arc_001_dryrun \
  --backend all_banana \
  --max-steps 2 \
  --max-nodes 2 \
  --max-l1-nodes 1 \
  --dry-run \
  --video-dry-run \
  --no-strict-online
```

Dry-run images are not materialized, so Track A may list a video task as skipped for missing reference images. This is expected; inspect task JSON and ordering artifacts, not generated media.

## Real Single-Arc Evaluation

Run from the `mifage` environment after the configured LLM and Banana endpoints are available. This generates both tracks for the same L4 narrative simulation and one anchored L3 arc:

```bash
mkdir -p outputs/moviebench_video_pipeline/0001_American_Beauty/arc_001_real
set -o pipefail
PYTHONUNBUFFERED=1 conda run -n mifage python -u scripts/run_moviebench_narrative_video_pipeline.py \
  --dataset-root outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --track both \
  --sample-id 0001_American_Beauty_L3ARC_001 \
  --output-dir outputs/moviebench_video_pipeline/0001_American_Beauty/arc_001_real \
  --backend all_banana \
  --max-steps 15 \
  --max-nodes 45 \
  --strict-online \
  --retries 3 \
  --log-level INFO \
  2>&1 | tee outputs/moviebench_video_pipeline/0001_American_Beauty/arc_001_real/run.log
```

Omit `--sample-id` to run Track A for every L3 arc. This can generate a large number of keyframes and clips. Use a separate output directory for each run.

## Output Layout

```text
<output-dir>/
  narrative_simulation/<backend>/
    pipeline_result.json
    narrative_checkpoint.json
    narrative_artifacts/...
  track_A/<L3-id>/
    l2_anchor_order/path_plan.json
    l1_keyframes/...
    narrative_nodes.json
    path_plan.json
    video/event_video_tasks.json
    video/event_video_summary.json
  track_B/
    narrative_nodes.json
    llm_node_order/path_plan.json
    path_plan.json
    video/event_video_tasks.json
    video/event_video_summary.json
  moviebench_pipeline_manifest.json
```

`event_video_tasks.json` records each L1/Track B node's rank, reference images, prompt, source IDs, and output video path. The manifest summarizes the shared simulation result and per-track outputs.

## Script

`scripts/run_moviebench_narrative_video_pipeline.py` orchestrates L4 loading, one shared narrative simulation, node selection, Track A anchor retrieval and ordering, Track B unanchored ordering, L1 keyframe generation, and `VideoAgent` video generation. `agents/path_orchestrator_agent.py` performs the LLM ordering; there is no MCTS fallback.
