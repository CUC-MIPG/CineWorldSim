# ViMax MovieBench arc_01 Baseline

This runner evaluates the ViMax MiniMax H3 video adapter against the paired conditions in CameraMaster's MovieBench pilot. It consumes the exact prompts, task names, and reference-image paths in `video_comparison_plan.json`; it does not re-plan events, regenerate images, or alter prompts. This isolates the video-generation adapter and is not a full Script2Video pipeline benchmark.

## Inputs

Default input plan:

```text
/data1/AgentG/hhb/CameraMaster/outputs/moviebench_video_eval/0001_American_Beauty/arc_01/video_comparison_plan.json
```

It contains four chronologically ordered nodes and two conditions per node, for eight video requests:

- `fl2va_event_keyframe_only`: only the shared event keyframe, using `fl2va`.
- `ref2va_semantic_assets`: the shared semantic asset bundle, using `ref2va`.

The protocol sets 5 seconds and 124 frames per node. Each condition is concatenated separately. Video seed is omitted so the run follows the protocol's uncontrolled-seed setting.

## Dry Run

The default command makes no API calls and writes task manifests plus `baseline_summary.json`:

```bash
conda run -n Vimax python -u scripts/run_vimax_moviebench_arc01_baseline.py
```

Check `missing_references` in the summary. All planned event keyframes and asset images must exist at their exact plan paths before generation. Do not create replacement images for this baseline.

## Real Generation

Load the local environment and enable the synchronous vLLM endpoint:

```bash
set -a
source .env
set +a
export MINIMAX_H3_VLLM_ENABLED=true
export MINIMAX_H3_SEED=none
conda run -n Vimax python -u scripts/run_vimax_moviebench_arc01_baseline.py --run --retries 3
```

The runner preflights every selected reference before sending any request, processes jobs serially, resumes from existing non-empty video files, writes per-condition `event_video_tasks.json` manifests, and exits nonzero if generation or concatenation is incomplete. Use `--condition` to run one condition or `--limit N` for the first N paired nodes. These options reduce the request count but should not be used for the full protocol result.

## Outputs

Default output root:

```text
outputs/vimax_moviebench/0001_American_Beauty/arc_01/
```

It contains the aggregate `baseline_summary.json`, separate condition directories with task manifests, per-node MP4 files and MiniMax request reports, plus a separate `final_comparison.mp4` for each completed condition. A successful exit alone is insufficient; confirm four successful node videos in each condition and both concatenations in the summary.

The summary records resolved endpoint and generation settings, output paths, failures, retries, and missing-reference information. It does not contain API keys. Keep CameraMaster's original reference files in place so both systems use identical assets.
