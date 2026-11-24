# MovieBench All-Arc Baseline Test Protocol

## 1. Scope and Current Status

This protocol covers all 16 L4 story arcs for the processed `0001_American_Beauty` sample. Each selected L3 scene occurrence is a video node. The paired conditions compare a Banana event keyframe used alone against semantic character/location/item references.

The verified dry-run at `outputs/moviebench_video_eval/0001_American_Beauty/all_arcs_l1_evidence_20261018/` resolves all 16 arcs into 91 L3 nodes, plans 91 event keyframes and 182 video jobs, and plans 257 unique asset images. No Banana or video-generation API calls were made; these are plans, not generated results.

Runner: [`scripts/run_moviebench_arc_video_eval.py`](../scripts/run_moviebench_arc_video_eval.py)

Data-construction background: [`docs/CineWorldSim_Bench_Data_Processing_Specification.md`](CineWorldSim_Bench_Data_Processing_Specification.md)

## 2. Data and Provenance

| Input | Path | Use |
|---|---|---|
| L4 story arcs | `outputs/moviebench_cineworldsim/0001_American_Beauty/l4_story/story_arcs.json` | Storylines, essential events, source L3/L2 references, key objects |
| L3 scenes | `outputs/moviebench_cineworldsim/0001_American_Beauty/l3_scenes/final.json` | Scene narrative, setting, characters, items, time ranges, L1/L2 evidence |
| L2 units | `outputs/moviebench_cineworldsim/0001_American_Beauty/l2_events/units/` | Resolve L4 L2 event IDs to their source L1 shot IDs |
| L1 descriptions | `outputs/moviebench_cineworldsim/0001_American_Beauty/l1_shots/` | Provenance only; not directly sent to image/video generation |
| Track A/B | `outputs/moviebench_cineworldsim/0001_American_Beauty/track_a.json`, `track_b.json` | Not used in these generation conditions |

The data folder reports 749 L1 shots, 598 L2 events, 143 L3 scene records and 16 L4 arcs. L3 IDs can repeat across occurrences, and some consolidated L4 references do not match the current L3 L2 segmentation. The runner first resolves by L3 ID plus exact L2 evidence. When those links do not resolve, it uses the chain `L4 source_l2_ids -> L2 source_shot_ids -> L3 evidence_trace.source_l1_ids`. The selected scenes and the actual shared L1 shot IDs are recorded under each node's `l3_resolution`. If no shared L1 evidence exists, that event fails rather than guessing.

## 3. Dry-Run Scope

The node/job counts in the verified dry-run are:

| Arc | L3 nodes | Video jobs (2 conditions) |
|---|---:|---:|
| `arc_01` | 4 | 8 |
| `arc_02` | 5 | 10 |
| `arc_03` | 3 | 6 |
| `arc_04` | 6 | 12 |
| `arc_05` | 6 | 12 |
| `arc_06` | 4 | 8 |
| `arc_07` | 1 | 2 |
| `arc_08` | 10 | 20 |
| `arc_09` | 8 | 16 |
| `arc_10` | 4 | 8 |
| `arc_11` | 12 | 24 |
| `arc_12` | 9 | 18 |
| `arc_13` | 3 | 6 |
| `arc_14` | 7 | 14 |
| `arc_15` | 7 | 14 |
| `arc_16` | 2 | 4 |
| **Total** | **91** | **182** |

The per-arc `narrative_nodes.json` is authoritative for each experiment's selected L3 scenes, order, event text and provenance. The aggregate manifest is `moviebench_all_arcs_eval_summary.json`.

## 4. Experimental Conditions

Both conditions use the same node order, VideoAgent prompt, duration and frame count. Only the video-stage references differ.

| Condition | References | MiniMax-H3 task |
|---|---|---|
| `fl2va_event_keyframe_only` | One Banana-generated event keyframe; no semantic assets passed to the video request | `fl2va` for one keyframe; `ref2va` if a node has multiple event keyframes |
| `ref2va_semantic_assets` | Character portrait(s), one location image and only mentioned item assets; excludes the event keyframe | `ref2va` |

The semantic reference cap is six; unused slots remain empty. The event keyframe is generated using available character, location and mentioned-item assets. Therefore, this is a comparison of video-stage reference packaging, not a comparison where the keyframe condition had no semantic image conditioning at any stage. Reuse the exact generated image files across baselines.

## 5. Shared Settings

- Duration: 5 seconds per node.
- Frames: 124 per node.
- Video requests: 91 per condition, 182 total.
- Concatenation: enabled by default, separately per condition and arc.
- Prompt style: unset by default; if changed, hold it constant across conditions and baselines.
- Provider/model and provider-specific parameters come from the runtime environment; record their resolved values.
- No video seed is set by this runner. Treat outputs as stochastic and repeat runs for stronger comparisons.

Use the exact prompts and reference lists in each arc's `video_comparison_plan.json`. Other baselines should map the two conditions to their closest supported I2V and multi-reference modes, marking unsupported modes rather than silently substituting another condition.

## 6. Execution

No-call planning command:

```bash
python -u scripts/run_moviebench_arc_video_eval.py \
  --all-arcs \
  --all-arcs-output-dir outputs/moviebench_video_eval/0001_American_Beauty/all_arcs_l1_evidence_20261018
```

Actual image and video generation, after reviewing all per-arc plans and confirming `.env` provider settings:

```bash
set -o pipefail
conda run -n mifage python -u scripts/run_moviebench_arc_video_eval.py \
  --all-arcs \
  --all-arcs-output-dir outputs/moviebench_video_eval/0001_American_Beauty/all_arcs_l1_evidence_20261018 \
  --run \
  --retries 3 \
  2>&1 | tee outputs/moviebench_video_eval/0001_American_Beauty/all_arcs_l1_evidence_20261018/run.log
```

The real command requests 257 unique Banana asset images, 91 event keyframes and 182 videos, subject to existing-image reuse and request failures. The runner does not print API keys.

## 7. Output Layout

```text
outputs/moviebench_video_eval/0001_American_Beauty/all_arcs_l1_evidence_20261018/
  moviebench_all_arcs_eval_summary.json
  run.log                         # created by the tee command
  <movie_id>_arc_<NN>/
    narrative_nodes.json
    narrative_path_plan.json
    banana_image_generation_plan.json
    video_comparison_plan.json
    moviebench_arc_video_eval_summary.json
    model_tool_calls.jsonl        # populated during real tool calls
    assets/characters/
    assets/locations/
    assets/items/
    event_keyframes/
    video_comparison/
      fl2va_event_keyframe_only/
      ref2va_semantic_assets/
      comparison_summary.json
```

Each condition directory contains task manifests and node videos; successful concatenations are stored separately by condition. A zero process exit is not sufficient: verify every task manifest, all 182 expected node videos, all per-arc summaries and concatenations.

## 8. Baseline Fairness and Limits

1. Preserve the node IDs, ordering, event text, prompts and image files from the plans. Record hashes when copying assets.
2. Keep duration, frame count, resolution/aspect policy and audio policy consistent where supported.
3. Record model/version, endpoint/provider, parameters, run date, retries, failures and outputs.
4. Keep conditions and arcs separate; do not concatenate across conditions or arcs.
5. Blind condition labels for human evaluation. Suggested dimensions are event/action fidelity, character and location consistency, relevant-object fidelity, temporal coherence and overall quality. The runner does not calculate these metrics.

This is one film and one processed sample. It is not a cross-film generalization result or a statistically powered benchmark. Track A/B and original L1 images are not inputs to this comparison. Character/location/item assets are generated from annotations and are not recovered film frames.
