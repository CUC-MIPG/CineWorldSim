# CineWorldSim ICLR Boundary And Main Axis Research Note

Date: 2026-08-25

## Research Brief

This note evaluates the CineWorldSim pipeline as a possible ICLR submission.
The key question is not whether the system can produce a long story video, but
whether it contributes a defensible research claim beyond existing work on
long-video generation, agentic video storytelling, and world-model-based video
control.

Research questions:

1. What should be the paper's main scientific axis?
2. Which claims are too broad or already covered by nearby work?
3. What experiments are required to make the contribution credible for ICLR?

## Literature Position

### Retrieval Caveat

The local pipeline documentation is concrete enough to assess the system
boundary, but the current web retrieval did not reliably verify a complete
paper corpus for every nearby story-to-video and world-model work. Therefore,
this note uses method families for the landscape judgment and does not treat
unverified paper titles as evidence. Before submission, every claimed
comparison must be checked against the final published or official preprint
metadata.

The current field already covers several nearby territories:

- Long-video generation methods improve duration and coherence using
  hierarchical autoregressive planning, rolling context, visual memory, or
  diffusion rendering.
- Story-to-video systems use multi-agent or storyboard pipelines to convert
  high-level prompts into multi-shot outputs.
- World-model approaches increasingly decouple "what to render" from "how to
  render", using explicit intermediate representations for layout, motion,
  camera, and scene state.
- Narrative evaluation papers show that ordinary video metrics miss event-level
  story presentation, narrative coverage, and coherence.

This means CineWorldSim should not claim novelty as:

- the first long-video generation pipeline;
- the first multi-agent video storytelling system;
- the first world-model video generation framework;
- the first character-consistent multi-shot system;
- a new video foundation model.

The viable novelty is narrower: CineWorldSim can be positioned as a
state-grounded narrative planning framework that treats a simulated story world
as the source of truth, then selects and renders cinematic events from that
state. This is a hypothesis about the paper's defensible boundary, not a
completed novelty proof.

## Recommended Main Axis

Use this as the core paper thesis:

> Explicit narrative world state is a useful intermediate representation for
> long-horizon story-to-video generation because it separates story evolution
> from cinematic rendering and makes event selection, continuity control, and
> failure diagnosis more observable.

The strongest title-level framing is:

> CineWorldSim: State-Grounded Narrative Planning for Long-Horizon
> Story-to-Video Generation

The main contribution should be "state-grounded narrative planning", not
"agentic movie generation". The latter is too crowded and will collide with
Co-Director, MovieAgent-style systems, FilmWorld, and related pipeline papers.

## Contribution Boundary

### In Scope

CineWorldSim should claim:

1. A narrative world-state representation that tracks characters, locations,
   objects, relations, and evolving story status across steps.
2. A simulation-to-cinema bridge that converts open-ended multi-character
   evolution into selectable narrative nodes.
3. A state-anchored rendering policy that decides which references are allowed
   for each event, including character references, world references, object
   references, and optional previous-tail-frame references.
4. A diagnostic evaluation protocol for testing whether world-state grounding
   improves event coverage, temporal continuity, character consistency, and
   scene-object persistence.

### Out of Scope

Avoid claiming:

1. New video model architecture.
2. General physical world simulation.
3. Full film generation from arbitrary novels.
4. Guaranteed cinematic quality.
5. Human-level directing or autonomous filmmaking.
6. Better low-level motion/visual fidelity than video foundation models.

## Taxonomy And Where CineWorldSim Fits

| Branch | Direction to verify in the final related-work search | What It Solves | What It May Miss | CineWorldSim Boundary |
|---|---|---|---|---|
| Long-video model architectures | Hierarchical, autoregressive, diffusion, or rolling-context video models | Longer generation and visual coherence | May not expose symbolic story state | Do not compete on backbone architecture |
| Multi-shot/storyboard pipelines | Agentic, storyboard, or director-style systems | Planning, style, shot decomposition, refinement | May treat story as a prompt/script artifact | Compete only on state-grounded event planning |
| Visual memory methods | Character, scene, or history-reference methods | Subject and style consistency | Memory may be visual/semantic rather than causal | Use as baselines/components |
| World-model / previsualization systems | Scene, 4D, camera, or interactive world representations | Control, layout, motion, and editable representation | May emphasize physical controllability over narrative causality | Focus on narrative state and event selection |
| Narrative evaluation | Story, event, consistency, and video-quality benchmarks | Story presentation and video quality | Metrics may not test explicit state faithfulness | Add state-to-video faithfulness metrics |

## Required Core Claims

The paper should make only three primary claims:

1. **State grounding improves continuity.**
   Generated videos better preserve character identity, location, object state,
   and event order when prompts are grounded in explicit evolving world state.

2. **Narrative-node selection improves cinematic relevance.**
   Selecting event nodes from simulation logs produces more complete and less
   redundant story videos than rendering every step or directly chunking the
   original story.

3. **State records make failures diagnosable.**
   Because every rendered event can be traced to world state, node selection,
   path ordering, and reference decisions, failures can be localized to
   simulation, selection, ordering, or rendering.

## Experimental Design

### Main Baselines

Use at least these:

1. Direct story-to-video prompt: one long story prompt or chunked prompts.
2. Script/storyboard-only pipeline: no explicit world state.
3. Visual-memory baseline: keep character/scene references but remove
   simulated state.
4. Agentic pipeline baseline: Co-Director-style pre-production/storyboard
   without state-grounded simulation, if reproducible.
5. CineWorldSim full system.

### Ablations

Minimum required:

1. No world-state tracker.
2. No narrative-node selector.
3. No world/item visual references.
4. No character reference gating.
5. No previous-tail-frame policy.
6. LLM ordering vs MCTS ordering vs chronological ordering.

### Metrics

The key metric family should be "state faithfulness":

1. Character continuity: identity, clothing, role, presence/absence correctness.
2. Location continuity: whether the scene matches the current world state.
3. Object-state continuity: whether important objects persist, change, or
   disappear according to the state log.
4. Event coverage: whether selected video events cover the required narrative
   nodes.
5. Event order: whether displayed events preserve causal/temporal dependencies.
6. Redundancy: whether adjacent generated events repeat the same semantic action.
7. Prompt-to-video alignment and ordinary video quality: use existing metrics as
   secondary support, not the main story.

### Human Evaluation

Ask annotators to answer structured questions, not broad preference questions:

1. Does the video show the event described by the node?
2. Is the correct character present, and are unrelated people absent?
3. Is the object/location state compatible with the previous event?
4. Does the event order make sense?
5. Which output better preserves the story world?

## Paper Structure

Recommended section spine:

1. Introduction: long story-to-video fails because prompts do not maintain a
   persistent causal story world.
2. Related Work: long video generation, agentic storytelling, visual memory,
   world-model video generation, narrative evaluation.
3. Task Definition: state-grounded story-to-video generation.
4. Method: world initialization, state simulation, node selection, path
   orchestration, state-anchored video prompting.
5. Evaluation Protocol: state-faithfulness metrics and baselines.
6. Experiments: main comparison, ablation, robustness, human study.
7. Failure Analysis: simulation error, node selection error, reference error,
   video rendering error.

## Killer Figure

The paper needs one figure that makes the distinction obvious:

Story text -> simulated world state over time -> narrative node selection ->
state-anchored prompt/reference package -> generated event video.

Show that the world state is not a prompt cache. It is the source-of-truth
ledger for characters, locations, objects, and event dependencies.

## Submission Risk

The highest risk is that reviewers see the work as a pipeline engineering paper.
The defense is to make "state faithfulness" the experimental object and show
that removing explicit state harms measurable continuity. If the experiments do
not show that, the paper should be redirected to a systems/demo venue rather
than ICLR main conference.

## Final Recommendation

Proceed only if the project can produce quantitative evidence for the state
grounding hypothesis. The strongest ICLR submission is not "we built an AI film
director", but:

> We define and evaluate state-grounded story-to-video generation, where a
> persistent narrative world state mediates between story evolution and visual
> rendering.
