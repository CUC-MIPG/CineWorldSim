# CameraMaster Evaluation Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-resistant evaluation pipeline for CameraMaster/CineWorldSim that evaluates Track A faithful story visualization and Track B constrained story expansion, while exposing compatible adapters for ConStory-Bench, DirectorBench, and LoCoT2V-Bench.

**Architecture:** Use a canonical manifest/schema layer first, then build annotation generation and blind reconstruction/alignment evaluators on top of it. Keep benchmark adapters independent from the CineWorldSim-specific metrics, and never collapse heterogeneous metrics into an uncalibrated single score. Every result must retain source evidence, prompt/model versions, run configuration, and per-sample outputs.

**Tech Stack:** Python 3, JSONL/JSON, Pydantic or dataclass schemas, pandas/pyarrow for tabular manifests, ffprobe/ffmpeg, existing benchmark scripts, OpenAI-compatible or local VLM/LLM clients, pytest.

---

## Phase 0: Freeze Evaluation Contract

### Task 0.1: Define run manifest and ID contract

**Files:**
- Create: `schemas/evaluation_manifest.schema.json`
- Create: `schemas/evaluation_types.py`
- Create: `docs/evaluation_contract.md`
- Test: `tests/test_evaluation_contract.py`

- [ ] Define stable IDs: `movie_id`, `shot_id`, `sequence_id`, `l2_event_id`, `l3_scene_id`, `l4_arc_id`, `benchmark_item_id`, `sample_id`, and `run_id`.
- [ ] Require parent IDs, source paths, time ranges, input hash, dataset/version, model/prompt version, status, failure reason, review status, confidence, and evidence IDs for every derived record.
- [ ] Define Track A fields: full story, ordered anchors, essential events, FDC units, anchor constraints, references, evidence map.
- [ ] Define Track B fields: short storyline, hard constraints, allowed expansion dimensions, non-unique valid expansion policy, references, evidence map.
- [ ] Add schema tests for valid records, missing parent IDs, invalid time ranges, and unknown review/status values.

### Task 0.2: Create experiment configuration and output layout

**Files:**
- Create: `configs/evaluation_default.yaml`
- Create: `evaluation/run_context.py`
- Test: `tests/test_run_context.py`

- [ ] Freeze dataset split, backend, seed, resolution, frame budget, shot budget, context mode (`causal` or `full_story`), reference strategy, judge model, temperature, and retry policy.
- [ ] Write `run_manifest.json`, `metrics/*.json`, `failures.jsonl`, `prompts/`, `references/`, and `intermediate/` under each `run_id`.
- [ ] Redact API keys and secrets from all logs.
- [ ] Make config hash and input hash part of every result.

## Phase 1: Dataset Audit and Hierarchical Narrative Data

### Task 1.1: Audit MovieBench assets

**Files:**
- Create: `scripts/audit_moviebench_assets.py`
- Create: `tests/fixtures/moviebench_audit/`
- Test: `tests/test_moviebench_audit.py`

- [ ] Enumerate movie directories, source JSON, auxiliary JSON, valid/invalid files, shots, sequences, and duplicate IDs.
- [ ] Emit `manifest.jsonl`, `invalid_files_manifest.json`, summary counts, and a machine-readable audit report.
- [ ] Verify the mismatch between the nominal 160 movies and observed directories instead of assuming the true count.
- [ ] Preserve raw files; do not overwrite source annotations.

### Task 1.2: Refine L1 to L2 micro-events

**Files:**
- Create: `scripts/refine_l2_micro_events.py`
- Create: `schemas/l2_micro_event.schema.json`
- Test: `tests/test_l2_refinement.py`

- [ ] Treat `<1000 ms` temporal groups as candidates only.
- [ ] Trigger review for spatial changes, long sequences, phase changes, time jumps, independent conflicts, and contradictory descriptions.
- [ ] Preserve empty establishing shots when they carry setting/time/transition evidence.
- [ ] Separate `observed_facts`, `inferred_details`, and `uncertainties`; never promote an inference to ground truth without review.
- [ ] Support dry-run, cache, resume, bounded concurrency, retry, cost logging, and deterministic output ordering.

### Task 1.3: Build L3 scene segments and L4 story arcs/movie bible

**Files:**
- Create: `scripts/build_l3_scene_segments.py`
- Create: `scripts/build_l4_story_arcs_and_movie_bible.py`
- Create: `schemas/l3_scene_segment.schema.json`
- Create: `schemas/l4_story_arc.schema.json`
- Test: `tests/test_narrative_hierarchy.py`

- [ ] Aggregate L2 units using evidence-supported location, local goal/conflict, character continuity, state transitions, and explicit transitions.
- [ ] Store entry state, key actions/conflicts, exit state, objects/states, evidence L2 IDs, uncertainties, and review status.
- [ ] Keep movie bible/global synopsis separate from locally supported story arcs.
- [ ] Ensure all parent-child mappings are bidirectional and lossless.

## Phase 2: Build the CineWorldSim Dual-Track Bench

### Task 2.1: Generate and review Track A annotations

**Files:**
- Create: `scripts/build_track_a_annotations.py`
- Create: `schemas/track_a.schema.json`
- Test: `tests/test_track_a_annotations.py`

- [ ] Select ordered storyboard anchors without deleting necessary anchor semantics or leaking all target details into the anchor.
- [ ] Label essential events, observable FDC units, anchor constraints, evidence shots, observability, confidence, and reviewer status.
- [ ] Define each unit so a blind evaluator can decide pass/fail or graded fulfillment from generated output.
- [ ] Keep source evidence and evaluator-facing text separate.

### Task 2.2: Generate and review Track B annotations

**Files:**
- Create: `scripts/build_track_b_annotations.py`
- Create: `schemas/track_b.schema.json`
- Test: `tests/test_track_b_annotations.py`

- [ ] Produce 1-3 sentence storylines containing characters, initial situation, core conflict, and broad direction.
- [ ] Label hard constraints and allowed expansion dimensions without specifying one unique continuation.
- [ ] Define valid expansion as new, observable, causally connected, non-redundant, constraint-compatible content.
- [ ] Require independent review for hard constraints and expansion criteria to prevent LLM self-labeling loops.

### Task 2.3: Build reference asset bindings and split checks

**Files:**
- Create: `scripts/build_reference_asset_manifest.py`
- Create: `scripts/check_split_leakage.py`
- Create: `schemas/reference_asset.schema.json`
- Test: `tests/test_reference_and_split.py`

- [ ] Bind character, location, and object references with source movie/shot/time, selection reason, visibility, licensing, and benchmark sample IDs.
- [ ] Split by movie/story arc, never by individual shot or derived unit.
- [ ] Fail validation if the same story/reference asset crosses train/dev/test or if blind evaluation receives source annotations.

## Phase 3: Blind Evaluation and CineWorldSim Metrics

### Task 3.1: Implement blind story reconstruction interface

**Files:**
- Create: `evaluation/blind_reconstruction.py`
- Create: `schemas/reconstruction_output.schema.json`
- Test: `tests/test_blind_reconstruction.py`

- [ ] Accept only generated video/images plus permitted prompt-level information according to Track A/B protocol.
- [ ] Produce structured reconstruction of observed events, characters, locations, objects, state changes, ordering, and confidence.
- [ ] Prevent source story, benchmark labels, hidden anchors, and internal world state from entering evaluator context.
- [ ] Save exact evaluator input, model response, parser result, and failure state.

### Task 3.2: Implement Track A metrics

**Files:**
- Create: `evaluation/metrics/track_a.py`
- Test: `tests/test_track_a_metrics.py`

- [ ] Implement `EF`/event fidelity or `KER` as essential-event recall with explicit denominator and abstention handling.
- [ ] Implement `FDC`/fine-detail coverage over independently labeled observable units.
- [ ] Implement `NSP`/state correctness for character, location, and object state at event time.
- [ ] Implement `LNSC`/long-range state consistency across non-adjacent events.
- [ ] Implement anchor constraint fulfillment and order/causal checks separately from event recall.
- [ ] Report per-event, per-category, per-story, and aggregate values with confidence intervals where applicable.

### Task 3.3: Implement Track B metrics

**Files:**
- Create: `evaluation/metrics/track_b.py`
- Test: `tests/test_track_b_metrics.py`

- [ ] Implement hard-constraint compatibility.
- [ ] Implement valid expansion count and valid expansion ratio (`EVR`) using independently reviewed expansion units.
- [ ] Implement causal linkage, novelty/non-redundancy, visual observability, and state consistency for new content.
- [ ] Do not compare open-ended outputs to one reference continuation as if it were the only correct answer.
- [ ] Report both raw expansion quantity and quality-adjusted expansion ratio.

### Task 3.4: Add controlled human calibration

**Files:**
- Create: `evaluation/human_review_schema.json`
- Create: `evaluation/export_review_tasks.py`
- Create: `evaluation/agreement.py`
- Test: `tests/test_agreement.py`

- [ ] Sample easy, ambiguous, high-score, and low-score cases across both tracks.
- [ ] Collect at least two independent judgments for critical labels; retain adjudication and disagreement reasons.
- [ ] Report agreement separately for event presence, state correctness, causal order, and valid expansion.
- [ ] Use calibration to revise prompts/rubrics before full-scale scoring.

## Phase 4: External Benchmark Adapters

### Task 4.1: ViStoryBench adapter

**Files:**
- Create: `adapters/vistorybench_adapter.py`
- Create: `scripts/run_vistorybench_eval.py`
- Test: `tests/test_vistorybench_adapter.py`

- [ ] Keep official `Shots[]` fixed: no node merging, deletion, reordering, or replacement by CineWorldSim node selection.
- [ ] Support `anchor_only`, `anchor_plus_sim`, `causal`, and `full_story` as explicit modes.
- [ ] Export one PNG per official shot and invoke official metrics without changing their input contract.
- [ ] Record backend, seed, resolution, prompt context, reference assets, and official result files.

### Task 4.2: ConStory-Bench adapter

**Files:**
- Create: `adapters/constory_adapter.py`
- Create: `scripts/run_constory_eval.py`
- Test: `tests/test_constory_adapter.py`

- [ ] Use only for long-form text/storyboard outputs, not as a video quality metric.
- [ ] Preserve CED and GRR definitions and report the implementation caveat that CED counts error-bearing subtype cells.
- [ ] Store judge model, prompt version, API settings, raw CSV, and parsed summary.

### Task 4.3: DirectorBench and LoCoT2V adapters

**Files:**
- Create: `adapters/directorbench_adapter.py`
- Create: `adapters/locot2v_adapter.py`
- Create: `scripts/run_external_video_benches.py`
- Test: `tests/test_external_video_adapters.py`

- [ ] DirectorBench: fix profile/config and preserve reference/reference-free mode; export per-metric scores, confidence, skipped dimensions, bottlenecks, and tool traces.
- [ ] LoCoT2V: generate sample-id-to-video JSON, use `purified_prompt` for overall alignment, retain FGA/HERD metadata alignment, and preserve per-sample outputs.
- [ ] Normalize only for display; never reinterpret or average heterogeneous scores into a CameraMaster score.
- [ ] Record exact conda environment and checkpoint paths.

## Phase 5: Experiment Matrix and Reporting

### Task 5.1: Implement baseline/ablation runner

**Files:**
- Create: `scripts/run_evaluation_matrix.py`
- Create: `configs/experiment_matrix.yaml`
- Test: `tests/test_experiment_matrix.py`

- [ ] Define B0 Direct, B1 Storyboard-only, B2 Visual-memory, B3 Agentic-without-state, and Ours Full where implementations exist.
- [ ] Define core ablations: no state tracker, no local update, no node selector, no path planner, no reference gating, event-keyframe-only, no auxiliary shots, no recovery.
- [ ] Ensure every cell uses the same stories, backend, seed policy, resolution, output budget, and evaluation protocol.
- [ ] Support smoke (5-10 samples), pilot (20-50 films/events), and full runs as separate configs.

### Task 5.2: Build aggregate reports

**Files:**
- Create: `evaluation/reporting.py`
- Create: `scripts/build_evaluation_report.py`
- Test: `tests/test_reporting.py`

- [ ] Produce per-sample JSON, per-story summaries, baseline tables, ablation tables, cost/failure tables, and human agreement tables.
- [ ] Report metrics by track, story length, scene count, character count, and difficulty strata.
- [ ] Include metric direction (`higher_is_better`/`lower_is_better`), denominator, skipped/N/A counts, and confidence intervals.
- [ ] Keep external benchmark tables separate from CineWorldSim-specific metric tables.

## Acceptance Criteria

- [ ] Two sample movies can run end-to-end from asset audit through L4 hierarchy, Track A/B annotation, blind evaluation, and report generation.
- [ ] A Track A run cannot access source story/hidden labels during reconstruction; leakage test fails closed.
- [ ] All derived records are traceable to source shots and run configuration.
- [ ] At least one baseline, Full, and three core ablations produce comparable outputs.
- [ ] External benchmark adapters run in smoke mode and retain official output semantics.
- [ ] Human calibration reports agreement and unresolved ambiguity categories.
- [ ] No final report claims universal improvement without matched baseline/ablation evidence.

## Known Risks

| Risk | Control |
|---|---|
| Temporal grouping mistaken for semantic event | LLM/label review and boundary audit |
| LLM hallucination becomes ground truth | observed/inferred/unknown separation and human review |
| Anchor leaks target details | separate anchor constraints from evaluator target units |
| Open-ended Track B judged against one continuation | constraint/expansion rubric, not exact-reference matching |
| External scores mixed into one number | report separately; calibrate before any composite |
| Reference/story leakage across splits | movie-level split and automated checks |
| Missing optional audio/API dependencies | explicit skipped/fallback status and confidence |

