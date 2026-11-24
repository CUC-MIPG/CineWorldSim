# Pre-Video Pipeline Recovery

`recover_pre_video_pipeline.py` provides a unified `check -> plan -> repair`
workflow for all stages before VideoAgent:

```text
world_initialization
  -> narrative_simulation
  -> world_delta
  -> event_keyframe
  -> node_selection
  -> path_llm
  -> path_mcts
```

## Check

Read-only health check:

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir outputs/example/all_banana \
  --mode check \
  --report-dir outputs/example/pre_video_reports
```

The command writes:

```text
pre_video_health_report.json
pre_video_recovery_plan.json
```

The checker validates:

- world model locations, items, portraits, and image files
- narrative step continuity, checkpoint, and role logs
- `world_state_stepN.json` and `world_deltas/world_delta_stepN.json`
- event keyframes for every simulated role
- narrative node selection and candidate IDs
- LLM and MCTS path plans
- fallback/error markers in logs, JSON status fields, and call logs

`healthy` means the checked pre-video artifacts are complete and no fallback
marker was found. It does not mean that generated video files exist.

## Plan

Generate only the recovery plan:

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir outputs/example/all_banana \
  --mode plan \
  --report-dir outputs/example/pre_video_reports
```

The plan identifies:

- `earliest_failed_stage`
- `affected_stages`
- `affected_steps`
- repair tasks
- artifacts that can be reused

Dependency propagation is conservative:

- initialization, simulation, world delta, or event keyframe failure rebuilds
  the remaining downstream pre-video stages
- node selection failure rebuilds both path planners
- LLM path failure only rebuilds the LLM path
- MCTS path failure only rebuilds the MCTS path

## Repair

Repair always writes to a new, empty directory and never overwrites the source:

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir outputs/example/all_banana \
  --mode repair \
  --output-dir outputs/example_pre_video_repaired/all_banana \
  --backend-mode all_banana \
  --strict-online \
  --retries 5
```

Repair behavior:

- world-stage failures rerun the world/narrative/world-state/keyframe pipeline
- downstream node selection and path stages are rebuilt in the same new output
- path-only failures copy the source snapshot and rebuild only the affected path
- the source directory is protected from writes
- historical logs are retained for audit, but post-repair health is evaluated
  from regenerated artifacts and JSON status fields

Repair writes:

```text
pre_video_recovery_manifest.json
pre_video_health_report.json
pre_video_recovery_plan.json
```

Do not use `--mode repair` as a video retry mechanism. Video generation is
intentionally outside this recovery tool; the repaired directory is the input
snapshot for a later VideoAgent run.

## Existing Narrow Tool

For a known narrative node, the older tool remains available:

```bash
python scripts/rerun_fallback_narrative_nodes.py \
  --pipeline-dir outputs/example/all_banana \
  --node 13:Mara_Venn:evaluation \
  --rerun-downstream \
  --output-dir outputs/example_fallback_rerun/all_banana
```

Use `recover_pre_video_pipeline.py` when the failed task or stage is not known
in advance.
