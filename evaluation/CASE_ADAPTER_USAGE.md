# CameraMaster Case Adapter

This adapter prepares inputs from the read-only case `arc_001_real` for ConStory-Bench and DirectorBench. It does not call judge APIs or run either benchmark.

## Prepare this case

From the `evaluation` directory:

```bash
python prepare_case_benchmark_inputs.py \
  --case-root /data1/AgentG/hhb/CameraMaster/outputs/moviebench_video_pipeline/0001_American_Beauty/arc_001_real \
  --output-dir adapted_cases/0001_American_Beauty_arc_001_real
```

The output directory must be empty or absent. The script refuses to overwrite an existing non-empty directory unless `--overwrite` is supplied. Source case files are only read.

## Mapping

- Track order comes exclusively from each `path_plan.json`'s `ordered_node_ids`; generated video records and `selected_nodes` are joined by `node_id`. No filesystem sorting is used to infer chronology.
- ConStory receives one row per track in `constory/cameramaster_stories.parquet`: `id`, `generated_story`, and provenance fields. `generated_story` is the ordered concatenation of each generated record's `story_description`, with available non-empty dialogue appended. `prompt` is retained as metadata, but ConStory's judge currently scores only the story text and does not condition on the prompt.
- DirectorBench receives the ordered, concatenated video, the context prompt, script text, and per-node storyboard in `directorbench/`. The context prompt is taken from the recorded storyline (`track_b_context.storyline` for B when available, otherwise `storyline`). This is a context fallback, not the original runtime user prompt.
- Track A and Track B are different granularities (L3 arc sequence vs. global-story sequence), not paired experimental conditions. Do not interpret their scores as a controlled A/B comparison.

## Prepared outputs

- `constory/cameramaster_stories.parquet`: two story rows, one for each track.
- `directorbench/track_A.mp4`, `track_B.mp4`: concatenated videos in planned order, re-encoded with H.264/AAC.
- `directorbench/track_*_script.txt` and `track_*_storyboard.json`: text references derived from generated descriptions and narrative-node metadata.
- `directorbench/inputs.json`: provenance, duration/audio checks, applicability notes, and exact commands.
- `directorbench/run_commands.sh`: commands to run DirectorBench using the environment that has DirectorBench installed.
- `case_adapter_manifest.json`: source/output paths, clip counts, durations, and caveats.

## Run ConStory-Bench

Activate the ConStory environment and run from this repository:

```bash
cd /data1/AgentG/hhb/CameraMaster/evaluation/ConStory-Bench-main
python -m constory.judge \
  --input ../adapted_cases/0001_American_Beauty_arc_001_real/constory/cameramaster_stories.parquet \
  --story-column generated_story \
  --model-name cameramaster_moviebench_arc001 \
  --output-dir ../adapted_cases/0001_American_Beauty_arc_001_real/constory/evaluations \
  --concurrent 2
```

Set the judge API credentials/configuration required by the deployed ConStory environment before running. Then compute CED from the resulting judge CSV with `python -m constory.metrics --mode ced --eval-dir <evaluation-output-dir>`. CED is error count normalized per 10,000 words (lower is better). The benchmark also defines GRR for comparing model groups; with this adapter's two different tracks and no matched set of competing systems, GRR is not a meaningful standalone result. Report the five category counts and CED per track, plus story word count. This is a case-level diagnostic, not a reproduction of ConStory's 2,000-prompt leaderboard protocol.

## Run DirectorBench

Use the deployed DirectorBench environment and execute the generated commands:

```bash
bash /data1/AgentG/hhb/CameraMaster/evaluation/adapted_cases/0001_American_Beauty_arc_001_real/directorbench/run_commands.sh
```

The commands provide `--video`, `--prompt`, `--script`, `--storyboard`, profile ID 1, and per-track output paths. DirectorBench's standard evaluation runs may call configured VLM/LLM, ASR, or audio services; confirm credentials and expected cost before running. Track A is about 67.7 seconds and is close to the benchmark's minute-long scope. Track B is about 234.3 seconds, so its result is an out-of-range long-video diagnostic and must be reported separately, not as a nominal DirectorBench comparison. Preserve the same profile and model/service configuration across samples.

## Run Story Semantic Similarity (SSS)

The evaluator supports three protocols. `v1` is the low-cost baseline: uniformly sample 12 full-video midpoint frames, reconstruct and length-normalize one global story, then calculate embedding cosine. `v2` detects shots automatically and reconstructs L1 shot captions, L2 micro-events, L3 scene paragraphs, and L4 global story. `v3` uses generated clip boundaries for Track A when a Track A adapter mapping is supplied; Track B still uses automatic shot detection. `v2` and `v3` perform the same L1-L4 reconstruction. All modes use raw cosine between `text-embedding-3-small` embeddings as primary `SSS_emb`; structured alignment, event coverage, and caption grounding are separate diagnostics.

The API wrapper loads the project environment file by default:

```bash
bash run_sss_api.sh
```

The wrapper loads `/data1/AgentG/hhb/CameraMaster/.env`, including `OPENAI_API_KEY`, `OPENAI_BASE_URL`/`OPENAI_API_BASE`, and `OPENAI_MODEL`. Use `SSS_ENV_FILE=/path/to/other.env` to select another environment file. The reference defaults to `movie_synopsis.short_synopsis` from the L4 `movie_bible.json`; this is a proxy storyline unless provenance establishes it is the original generation-time prompt. Use a vision-capable model. The wrapper never prints the API key and refuses to overwrite an existing result directory.

Useful overrides:

```bash
SSS_MODEL='your-vision-model' \
OPENAI_BASE_URL='http://localhost:8000/v1' \
SSS_MODE=v2 \
SSS_FRAMES_PER_SHOT=1 \
SSS_SHOT_THRESHOLD=0.3 \
SSS_OUTPUT_DIR="$PWD/sss_results/american_beauty_track_b" \
bash run_sss_api.sh
```

For v3 Track A, set `SSS_MODE=v3`, `SSS_TRACK=A`, and `SSS_TRACK_ADAPTER_INPUTS=/path/to/track_adapter_inputs.json`. The mapping must include ordered clip paths and node IDs and match the selected concatenated video. `SSS_MODE=v3 SSS_TRACK=B` selects automatic detection. If PySceneDetect and ffmpeg scene detection both fail, the evaluator stops unless `SSS_UNIFORM_FALLBACK_SEC` is explicitly set; fallback use is recorded in the manifest. Set `SSS_L4_TEXT=global_storyline` to use that L4 field, `SSS_EMBEDDING_MODEL` to change the embedding model, `SSS_SKIP_DIAGNOSTICS=1` to skip structured alignment and event coverage, or `SSS_OVERWRITE=1` to replace an existing output directory.

The output includes shot/frame manifests, prompt and raw-response artifacts for each reconstruction level, `hierarchical_reconstruction.json`, grounding results, `sss_result.json`, and `run_manifest.json`. Audio is not transcribed. Reconstruction and grounding are video-only; the reference is loaded only after reconstruction and grounding complete. Do not interpret `SSS_emb` as a benchmark leaderboard score until the prompt protocol and LLM judge have been calibrated against human judgments. The old category-balanced claim metric is retained only behind `--legacy-claim-sss` and is not the primary SSS definition.

## Interpretation limits

- ConStory-Bench is designed to judge long-form text-story consistency. Here the story is reconstructed from generated per-node descriptions; its score measures consistency of that text, not direct video understanding or prompt adherence.
- DirectorBench's storyboard is constructed from CameraMaster node descriptions rather than an independently authored ground-truth storyboard. Reference-mode fidelity scores therefore measure alignment to this proxy plan.
- The runtime user prompt was not found in the case manifest. Do not label the storyline fallback as the original prompt.
- Keep Track A and Track B results separate. Track B exceeds the nominal minute-scale duration substantially.
