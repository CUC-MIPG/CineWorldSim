# ViMax Pipeline Output Evaluation Design

## Goal

Add a standalone evaluation command for outputs produced by the current ViMax
benchmark pipeline. It evaluates generated shot clips with ViCLIP embeddings
and reports visual consistency metrics without requiring CameraMaster's output
adapter or changing online video generation.

This evaluates custom cases from `vimax_benchmark`; reports must not claim to be
official ViMax-Bench results.

## Existing Pipeline Artifacts

- Track A clips are stored at `track_A/vimax_pipeline/shots/<idx>/video.mp4`.
- Track B clips are stored at
  `track_B/vimax_pipeline/scene_<n>/shots/<id>/video.mp4`.
- Track A source cases are listed in `vimax_benchmark/benchmark_index.json`.
- Existing manifests may be stale after interrupted or resumed runs. Video
  artifacts and their indices are the evaluation input of record; manifest
  status is reported as provenance but does not establish completeness.

## Proposed Command

Add `scripts/run_vimax_pipeline_eval.py` as an offline evaluator, separate from
`scripts/run_vimax_benchmark_pipeline.py`. It discovers cases from the
benchmark index and outputs from the configured output root, with selectors for
case and track. It accepts a ViCLIP `/embed` URL, frame count, resolution,
timeout, cache directory, and output JSON path. A deterministic fake encoder
is available only for tests and smoke checks.

The evaluator must not edit source benchmark cases, Pipeline outputs, or
CameraMaster files. Defaults follow the existing ViCLIP evaluation guide:
`http://172.17.34.43:8890/embed`, 8 sampled frames, resolution 224, and a
600-second request timeout. All are overridable from the CLI.

## Track Mapping and Metrics

### Track A

Sort source shots by numeric `(scene_num, shot_id)` and generated clips by
numeric `<idx>`. Assign source scene IDs positionally only if generated clip
indices are exactly contiguous from zero and the generated clip count equals
the source shot count. Record that the mapping is positional. If either check
fails, preserve available clips for provenance, mark the sample incomplete, and
do not report sample metrics; do not silently infer scene IDs or score a
partial subset as the full sample.

For a complete mapping, compute:

- IC over all same-source-scene shot pairs.
- CC over all cross-source-scene shot pairs.
- GC over all shot pairs.

### Track B

Sort generated clips by numeric scene index and then numeric shot index. Do not
map these clips to source benchmark scenes because Track B receives only the
story overview. Treat each clip identity as `(scene_idx, shot_idx)` because
shot indices restart within each generated scene. Determine expected scene
indices from the Pipeline root `script.json` and expected shot indices from
each scene's `storyboard.json`; require the scene and shot clip inventory to
match these artifacts before scoring. If these planning artifacts are missing
or inconsistent, mark the sample incomplete instead of scoring the available
subset. These artifacts establish expected output inventory only and are not
source-scene annotations. Compute GC as the primary result without scene
labels.

An opt-in `--track-b-scene-diagnostics` option may compute IC/CC using the
Pipeline's generated `scene_<n>` directories. Store these separately under a
diagnostic field and explicitly label the scene source as `pipeline_generated`;
never mix these diagnostic scores with protocol IC/CC or dataset aggregates.

## ViCLIP and Cache

Use the HTTP ViCLIP `/embed` contract by default. Expect a 768-dimensional
embedding and validate that it is finite, non-zero, approximately unit-norm,
and consistent with response metadata (`dimension`, `sampled_frames`,
`resolution`, and `normalized=true`) before computing cosine similarities.
Reuse the existing metric definitions: normalize vectors, clip cosine scores
to `[-1, 1]`, and compute pair means, standard deviations, and pair counts.

Cache embeddings using the resolved video path, file size, modification time,
endpoint, frame count, and resolution. Report cache hits and encoder settings.
The fake encoder must be marked as non-reportable.

## Output and Error Handling

Write a versioned JSON report with one record per selected case/track. Each
record includes sample status, source case and output paths, expected and found
shot counts, missing/extra/non-contiguous indices, scene-label source,
protocol-structure checks, metrics, per-shot provenance, and feature-extraction
settings. The top level includes dataset aggregates, encoder configuration,
creation time, and `results_are_official_vimax_bench: false`.

Dataset aggregates are reported independently for each track and metric. For
each metric, `story_macro_mean` is the arithmetic mean of scores from complete
samples where that metric is defined; `pair_weighted_mean` is the mean weighted
by each sample's `pair_count`. Reports also include the contributing sample and
pair counts. Samples without a defined metric (for example, Track B IC/CC
without source scene labels) are excluded from that metric's aggregate. Track B
scene diagnostics are aggregated separately under a diagnostic namespace and
never contribute to protocol aggregates.

For incomplete samples (including missing, zero-byte, or unexpected clip
artifacts), record actionable diagnostics and omit metrics for that sample. A
video-decode, ViCLIP request, or embedding-validation failure should identify
the affected sample and `(scene_idx, shot_idx)` where applicable, and fail the
run rather than emitting a misleading partial aggregate. If every selected
sample is incomplete, still write the diagnostic report and return a non-zero
status.

Protocol structure checks report whether the sample has 8-16 shots and, when
reliable source scene labels exist, 2-4 scenes with complete labels. Passing
these checks is not sufficient to claim official benchmark results.

## Verification

Unit tests use temporary benchmark/output trees and a fake encoder. Cover:

- Track A numeric ordering, positional scene mapping, and count/index mismatch.
- Track B nested discovery and the absence of source-scene labels.
- Optional Track B generated-scene diagnostics remain separate from primary
  metrics and aggregates.
- Missing or zero-byte videos make the sample incomplete without partial
  scores; corrupt videos fail the run with the sample and shot identified.
- Embedding caching and JSON report serialization.
- CLI selectors, smoke mode, and non-official result markers.

No network access, GPU, ViCLIP checkpoint, or generated videos are required for
the unit test suite.

## Out of Scope

- Text-video semantic alignment or prompt adherence scoring.
- Audio quality, narrative quality, or temporal continuity metrics.
- Concatenated-video embeddings.
- Changing how either ViMax Pipeline generates shots or videos.
- Treating generated Track B scene groups as source ground truth.
