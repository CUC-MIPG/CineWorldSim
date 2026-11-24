# MovieBench RAG Narrative Retrieval And Ordering Design

## 1. Purpose

This document specifies a new retrieval-augmented narrative ordering mode for the MovieBench narrative-to-video pipeline. The current Path Orchestrator workflow remains unchanged and serves as the original baseline. The new mode retrieves evidence from the shared narrative simulation, enriches each benchmark or simulation node without changing its identity, and then passes the enriched atomic candidates to the existing Path Orchestrator ordering contract.

The design covers Track A (L3-anchored) and Track B (unanchored, L4-global). It also evaluates whether GraphRAG can be added on top of the existing narrative graph.

## 2. Current Pipeline Baseline

The current MovieBench runner creates one shared narrative simulation and uses it for both tracks. Track A uses L3/L2/L1 annotations as its anchor and generation units; Track B uses simulation nodes and the L4 global story.

### Track A today

1. Select an L3 arc.
2. Build a candidate list from the L2 IDs assigned to that arc.
3. For each L2, find up to three related simulation nodes using lexical token overlap and character overlap. Their IDs, summaries, and step fields are included in that L2 candidate JSON sent to the ordering LLM, but they are not turned into separate candidates or used to synthesize a new L2 description. The lexical fallback scores only the L2 description text.
4. Ask `PathOrchestratorAgent` to order the L2 IDs against the L3 title and `full_story_S`, with the related-node rows available as candidate context to the LLM.
5. Expand the ordered L2 list into its source L1 shots, preserving source shot order within an event, then generate L1 keyframes and videos.

### Track B today

1. Select simulation nodes from the shared narrative simulation.
2. Give all selected nodes to `PathOrchestratorAgent` with the L4 global storyline.
3. Generate videos in the returned order.

### Existing ordering contract

The Path Orchestrator receives a storyline objective and candidate text, and returns a full permutation of candidate IDs plus `storyline_fit_summary`, `transition_notes`, and `quality_checks` (`coherence`, `coverage`, `pacing`). It must preserve candidate boundaries and cannot merge or split candidates. Code normalizes the output into a full permutation: invalid and duplicate IDs are discarded and omitted IDs are appended in source order. In non-strict mode, a failed model call can fall back to lexical overlap ranking; strict-online mode propagates the error.

The existing MovieBench L2-to-simulation association is not semantic vector retrieval and does not use graph edges. It is a lightweight lexical/character matching heuristic. The existing `DirectedCausalGraph` is built from role actions and has same-role, target-character, and public-event dependency edges, but the MovieBench runner does not currently use those edges as hard ordering constraints.

## 3. Goals And Non-Goals

### Goals

- Add a reproducible RAG mode alongside the current ordering baseline.
- Use simulation content to improve event context, state continuity, and narrative relevance.
- Make retrieval evidence and transformations auditable by recording source IDs, scores, and selected fields.
- Preserve each original L2 event or simulation node as an atomic candidate through retrieval, synthesis, and ordering.
- Keep the current Path Orchestrator output contract and quality-check fields.
- Permit later comparison of flat RAG and GraphRAG without changing video generation semantics.

### Non-goals

- Do not replace L1/L2/L3/L4 source data or overwrite source annotations.
- Do not generate new cross-event nodes by merging neighboring candidates.
- Do not treat retrieval similarity as proof that two events are causally identical.
- Do not make graph constraints mandatory in the first RAG implementation.
- Do not change image generation, video generation, prompt style, or Track A/B reference-image policies as part of this ordering work.

## 4. Proposed Modes

Keep the current behavior under an explicit baseline mode, for example `ordering_mode=path_orchestrator`. Add a new mode, for example `ordering_mode=rag_path_orchestrator`. The runner should write the selected mode and configuration into the manifest and each ordering artifact. The current default should remain the baseline until a separate experiment explicitly opts into RAG.

Both modes use the same Path Orchestrator contract. In RAG mode, retrieval and node-local enrichment happen first; Path Orchestrator then orders the enriched candidates. This isolates the experimental variable to evidence retrieval/enrichment rather than replacing the ordering model at the same time.

## 5. Shared RAG Architecture

### 5.1 Retrieval corpus

Build a read-only retrieval corpus from the shared simulation result. Each source record should retain its immutable provenance and only include fields that are available in the source artifact:

- Stable source ID, normally `step::<step>::<role>` or the selected narrative node ID.
- Step and role/actor, where available.
- Action text, action type, target character, reason, and location.
- Step summary, story phase, story anchor, and relevant environment description.
- Character, location, and item names/entities.
- Public events or explicit dependency references, when present.
- World-state delta and state-at-step references, where available.
- Event keyframe and asset paths as optional metadata, not embedding text.

The initial implementation should use a hybrid retriever: lexical/BM25-style matching plus dense text embeddings, followed by deterministic metadata filters and a weighted reranker. Character/location/item exact matches should be high-value signals; dense similarity should find paraphrases; chronology and state compatibility should resolve ties rather than dominate topical relevance. The exact weights and `top_k` must be configurable and persisted.

Each retrieved evidence record must contain `source_id`, `step`, `score`, `match_reasons`, and a concise source excerpt. Do not pass image paths or full checkpoint payloads as natural-language evidence. Asset paths may be copied into a separate visual-reference field after relevance and existence checks.

### 5.2 Node-local synthesis

The retrieval result enriches one candidate at a time. It does not combine candidate IDs. The synthesis output should distinguish:

- `base_node`: immutable original candidate identity and text.
- `retrieved_evidence`: source IDs, excerpts, scores, and match reasons.
- `enriched_context`: a concise, grounded description of relevant prior state, action, consequence, and continuity.
- `uncertainties`: retrieved facts that are weakly supported or conflicting.
- `ordering_text`: compact text passed to Path Orchestrator, combining base text and grounded context.

The synthesizer must not invent an event, alter who performed the source action, or claim a world-state change not supported by evidence. If retrieval is empty or below threshold, keep the original candidate text unchanged and record `retrieval_status=empty_or_below_threshold`.

### 5.3 Atomicity and ordering invariants

- Candidate ID and source boundary never change during retrieval or synthesis.
- One base L2 event remains one Track A ordering candidate.
- One selected simulation node remains one Track B ordering candidate.
- Multiple retrieved records may enrich one candidate, but retrieved records do not become additional candidates by default.
- The Path Orchestrator must return a permutation of the candidate IDs; no IDs may be created, merged, or dropped.
- Existing `storyline_fit_summary`, `transition_notes`, and `quality_checks` are retained.
- Source chronology is evidence and a tie-break signal, not a universal hard constraint; the Path Orchestrator may construct a non-chronological presentation when that better serves the supplied storyline, unless a configured causal constraint forbids it.

## 6. Track A: L3-Anchored Retrieval And L2 Enrichment

### 6.1 Query construction

For each L3 anchor, construct a retrieval query from:

- L3 title and `full_story_S` as the local narrative objective.
- The current L2 event description/title.
- L2 main characters, locations, items, and source time range when present.
- Relevant state-transition terms from L3/L2 annotations, without including unrelated L3 arcs.

Search the shared simulation corpus for role-action nodes that support, precede, explain, or provide a state transition relevant to this L2 event. Prefer exact entity matches, then semantic relevance, then compatible chronology. Retrieve a configurable top K per L2 and retain a global cap to control prompt length.

### 6.2 L2 enrichment and ordering

For each L2, create an enriched L2 view whose identity remains the original L2 ID. It may add grounded context such as:

- Which simulated action or interaction is relevant to this annotated event.
- What state existed before the action and what changed afterward.
- Which participating characters, locations, or items establish continuity.
- Evidence-supported transition information useful for ordering.

`PathOrchestratorAgent` receives the L3 objective and one enriched candidate per original L2. It returns the ordered original L2 IDs and the existing summaries/quality checks. The ordered L2 IDs then expand to their existing L1 source shots. L1 IDs and shot content are not fused with simulation records.

If one L2 retrieves multiple simulation nodes, they stay attached as evidence to that L2. They are not expanded into multiple L2 candidates, and one simulation node may support multiple L2 candidates only when its recorded evidence supports both; this many-to-many relation must be explicit in the audit artifact.

### 6.3 Track A output artifacts

Suggested layout under each L3 run:

```text
track_A/<L3-id>/
  rag_retrieval/
    corpus_manifest.json
    l2_retrieval.jsonl
    l2_enriched_candidates.json
    retrieval_config.json
  l2_anchor_order/
    path_plan.json
  narrative_nodes.json
  video/...
```

Each enriched candidate should record `l2_id`, original source description, retrieved simulation node IDs, evidence scores/reasons, synthesized context, final ordering text, and retrieval status. `path_plan.json` should retain original `ordered_node_ids` while adding `ordering_mode`, `retrieval_run_id`, and any model metadata.

## 7. Track B: Multi-Level Retrieval And Simulation-Node Enrichment

Track B has no L3/L2 benchmark anchor. L4 global storyline is the global objective, and selected simulation nodes remain the atomic output candidates.

### 7.1 Multi-level retrieval stages

For each selected Track B node, run retrieval in bounded stages:

1. **Entity retrieval:** identify people, locations, and items from the node and retrieve simulation/world-state records matching those entities.
2. **Event retrieval:** use action, target, event summary, and semantic query text to retrieve related actions and consequences.
3. **Narrative-clue retrieval:** use L4 themes, unresolved goals, story phase, explicit story anchors, and remaining-gap information to retrieve evidence relevant to the larger storyline.
4. **Relation/continuity expansion:** optionally follow explicit references from retrieved records, such as same-character prior actions, target-character dependencies, public-event production/consumption, and relevant state deltas. Limit hop count and result count; record every traversed edge/reference.
5. **Fusion:** deduplicate evidence by source ID, rerank it, and synthesize one enriched view for the original candidate node.

These stages are retrieval paths into the corpus, not permissions to merge multiple candidate nodes. A retrieved predecessor or consequence is explanatory evidence attached to the current node. It remains a separately orderable candidate if it was already selected by the node selector.

### 7.2 Track B ordering

The L4 global storyline and each node's `ordering_text` are passed to Path Orchestrator. It returns a permutation of the original selected simulation node IDs and the standard narrative-fit, transition, coherence, coverage, and pacing fields. The orderer must be told that retrieved context supports interpretation but must not expand the candidate set or merge candidate boundaries.

### 7.3 Track B output artifacts

Suggested layout:

```text
track_B/
  rag_retrieval/
    corpus_manifest.json
    node_retrieval.jsonl
    enriched_candidates.json
    retrieval_config.json
  narrative_nodes.json
  path_plan.json
  video/...
```

Store base node fields unchanged and place additions under a separate `rag_context` object. This allows downstream `VideoAgent` to use either the original node text or an explicitly selected enriched description without losing source provenance.

## 8. Retrieval And Fusion Configuration

The implementation should expose and persist at least:

- `ordering_mode`: baseline or RAG.
- `retrieval_backend`: lexical, dense, hybrid, or graph-enhanced hybrid.
- `top_k_per_stage`, `max_total_evidence`, and maximum graph hops.
- Entity match weights and lexical/dense reranking weights.
- Minimum relevance threshold.
- Synthesis model/scope and prompt version.
- Whether evidence may affect only ordering text or also downstream video description.
- Strict-online behavior and fallback policy.

Prompt/token budgets should be bounded by compact excerpts and evidence count. Full simulation logs, all world snapshots, and raw visual assets must not be inserted into every ordering prompt.

## 9. GraphRAG Feasibility Assessment

### Assessment: feasible as a second-stage retrieval backend, not as the first implementation dependency

The repository already has a `DirectedCausalGraph` over role-action nodes. It currently represents same-role temporal chains, target-character dependencies, and public-event dependencies. These relations can support graph expansion after an initial lexical/dense retrieval. The graph is therefore a useful foundation for GraphRAG.

However, the existing graph is not yet a complete narrative knowledge graph:

- Nodes are primarily role-action events, rather than normalized people/location/item/state entities.
- Entity resolution across aliases and mentions is not guaranteed by the graph interface.
- Edges are heuristic dependency links, not necessarily verified causal relations.
- World-state deltas and transitions are not uniformly represented as typed graph edges.
- The MovieBench runner currently performs lexical/character matching separately and does not use graph edges in Path Orchestrator ordering.

### Recommended GraphRAG scope

Implement graph-enhanced retrieval in stages:

1. Keep hybrid lexical/dense retrieval as the initial candidate retriever.
2. Seed graph traversal from retrieved event IDs and exact entity matches.
3. Expand only over typed/known relations: `same_character_previous_action`, `targets_character`, `produces_public_event`, `consumes_public_event`, `updates_location`, `updates_item`, and `same_location` when that relation is explicitly available.
4. Use bounded hops, edge-specific weights, temporal filters, and per-node evidence caps.
5. Return the same evidence-record schema as flat RAG so the synthesis and ordering layers remain backend-agnostic.

Do not treat inferred graph connectivity alone as evidence of causality. Record edge type, source/target IDs, edge provenance, traversal depth, and confidence. If an edge type or confidence is missing, use it only as a low-weight retrieval hint.

### Expected value and risks

GraphRAG is most likely to help Track B, where a local simulation event must be situated within character arcs, environment state, and longer-range narrative clues. In Track A it can improve the match between an L2 event and non-adjacent simulation actions or state transitions, but the L3/L2 metadata already narrows the search and may make flat hybrid RAG sufficient.

Main risks are incorrect entity linking, heuristic edges being mistaken for causality, retrieval loops/redundant evidence, and GraphRAG increasing implementation and prompt complexity without measurable gains. Therefore GraphRAG should be an ablation after flat hybrid RAG, not a prerequisite for validating the core RAG hypothesis.

## 10. Evaluation Plan

Compare at least these variants using the same simulation output, selected nodes, source annotations, video model settings, and random seeds where available:

1. **Baseline:** current Path Orchestrator inputs and ordering.
2. **Lexical RAG:** lexical retrieval plus node-local evidence attachment and synthesis.
3. **Hybrid RAG:** lexical+dense retrieval, reranking, synthesis, then the unchanged Path Orchestrator.
4. **GraphRAG:** hybrid RAG plus bounded typed graph expansion.

Keep video generation fixed when measuring ordering quality. Evaluate:

- Candidate coverage and permutation validity.
- Retrieval precision/recall on manually judged relevance for a sampled set.
- Evidence grounding: fraction of synthesized claims traceable to retrieved source IDs.
- Track A: L2 order relevance to L3, preservation of L2-to-L1 coverage, and consistency with source time/causal evidence.
- Track B: global-storyline fit, character/state continuity, event coverage, and transition quality.
- Human pairwise preference for narrative order and generated clips, scored separately.
- Runtime, retrieval size, LLM token usage, failure rate, and reproducibility.

Do not collapse all quality into a single score. Report retrieval quality, ordering quality, and generated-video quality separately to identify where an observed improvement originates.

## 11. Failure Handling And Reproducibility

- In strict-online mode, retrieval embedding or synthesis failures should fail the RAG run with a recorded error; they must not silently label an unaugmented run as RAG.
- In permissive mode, a candidate with failed retrieval can retain its original text, but its status must be recorded. The run manifest must show the number and fraction of fallback candidates.
- Ordering LLM failures follow the existing Path Orchestrator strict/fallback behavior and are recorded independently from retrieval status.
- Cache retrieval and synthesis by corpus fingerprint, candidate ID, retrieval configuration, and prompt/model version. Never reuse a cached enrichment when any of these inputs differ.
- Preserve baseline artifacts and outputs; use separate output directories or explicit mode subdirectories.

## 12. Proposed Implementation Phases

1. Add mode/config plumbing and stable retrieval-corpus serialization, without changing the baseline.
2. Implement lexical retrieval and auditable Track A L2 enrichment; verify atomicity and output compatibility.
3. Add dense retrieval and hybrid reranking, then run baseline-vs-RAG ordering experiments.
4. Add Track B staged entity/event/narrative-clue retrieval and node-local synthesis.
5. Add optional bounded graph expansion and evaluate GraphRAG as an ablation.
6. After offline ordering results are stable, optionally test whether enriched context should also condition video prompts; keep this separate from the ordering evaluation.

## 13. Current Code Touchpoints

- `scripts/run_moviebench_narrative_video_pipeline.py`: mode selection, Track A/Track B orchestration, L2 candidate construction, and artifact paths.
- `agents/path_orchestrator_agent.py`: unchanged final ordering contract and output normalization.
- `agents/narrative_node_selector.py`: selected simulation-node identity and source fields.
- `narrative/causal_graph.py`: current event-node graph and heuristic dependency edges; candidate base for optional GraphRAG expansion.
- `narrative/world_state_tracker.py` and `pipeline_result.json`: state snapshots and per-step state provenance.
- `agents/video_agent.py`: consumes the final ordered nodes; should remain unchanged in the first ordering-only experiment.

## 14. Decisions Captured

- The current Path Orchestrator pipeline remains the original baseline.
- RAG precedes ordering; it enriches candidates but does not replace the ordering model.
- Track A retrieves simulation evidence per L2 within an L3 anchor and orders enriched L2 candidates before L1 expansion.
- Track B uses multi-stage entity, event, and narrative-clue retrieval to enrich each selected simulation node before ordering against L4.
- Retrieved evidence never merges or splits candidate nodes.
- Existing Path Orchestrator summary and quality-check fields remain required.
- GraphRAG is feasible as a bounded optional retrieval expansion, but flat hybrid RAG should be implemented and evaluated first.
