# MovieBench ViMax-Native Pipeline Experiment

This runner uses the restructured MovieBench hierarchy as input and ViMax's own pipelines for planning and generation. It does not consume CameraMaster-generated prompts, narrative simulations, images, videos, or Track B answer files.

## Tracks

- **Track A:** selects an L3 arc, resolves its linked L2 events and L1 shot annotations, formats them into a screenplay, and runs ViMax `Script2VideoPipeline`.
- **Track B:** reads only `l4_global/movie_bible.json`, formats its global storyline, synopsis, and global character profiles as an idea, and runs ViMax `Idea2VideoPipeline`. It does not load `track_b.json` or other evaluation answers.
- `--track both` runs each pipeline independently. No generated narrative network or artifacts are shared between tracks.

Both configs default to the repository's ViMax settings: `configs/script2video.yaml` and `configs/idea2video.yaml`. Each run receives its own ViMax `working_dir`, preserving native intermediate JSON, character portraits, frames, shot videos, and final video outputs.

## Offline Input Dry-Run

Dry-run is the default and makes no API calls. It validates the MovieBench hierarchy, writes the derived ViMax input files, and emits a manifest:

```bash
conda run -n Vimax python -u scripts/run_vimax_moviebench_pipeline.py \
  --dataset-root /data1/AgentG/hhb/CameraMaster/outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --track both \
  --sample-id 0001_American_Beauty_L3ARC_001 \
  --max-l1-nodes 2 \
  --output-dir /tmp/vimax_moviebench_smoke
```

Inspect `track_A/input_script.txt`, `track_A/input_characters.json`, `track_B/input_idea.txt`, and `moviebench_vimax_manifest.json`.

## Real Generation

Use a fresh output directory. `--run` enables online ViMax LLM, image, and video generation. `--max-l1-nodes 0` (the default) uses the whole arc.

```bash
conda run -n Vimax python -u scripts/run_vimax_moviebench_pipeline.py \
  --dataset-root /data1/AgentG/hhb/CameraMaster/outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --track both \
  --sample-id 0001_American_Beauty_L3ARC_001 \
  --output-dir outputs/vimax_moviebench/0001_American_Beauty/arc_001_native \
  --run
```

The runner refuses to write into a non-empty output directory unless `--resume` is explicitly supplied. Track B is movie-level and runs once even when `--sample-id` filters Track A.

## Output Layout

```text
<output-dir>/
  moviebench_vimax_manifest.json
  track_A/
    input_script.txt
    input_characters.json
    vimax_config.yaml
    vimax_pipeline/...
  track_B/
    input_idea.txt
    vimax_config.yaml
    vimax_pipeline/...
```
