# Evaluation Planning Findings

## Source documents

- `docs/20261015.md` defines the dual-track research problem, L1-L4 narrative hierarchy, MovieBench data gaps, Track A/B annotation requirements, blind evaluation, and priority order.
- `docs/cameramaster_research_framework.md` defines CineWorldSim's world-state method, baseline/ablation matrix, expected claims, existing pipeline outputs, and external benchmark boundaries.
- `three_bench_evaluation_guide.md` documents the local ConStory-Bench, DirectorBench, and LoCoT2V-Bench implementations.

## Decisions

1. Build a canonical CameraMaster schema and manifest before metric code.
2. Keep Track A (faithful visualization) and Track B (constrained expansion) separate.
3. Evaluate with a blind reconstruction stage before comparing reconstruction to source annotations.
4. Treat L1-L4 data as an evidence hierarchy; time adjacency and LLM output are not ground truth by themselves.
5. Keep ConStory, DirectorBench, and LoCoT2V as adapters/auxiliary tables rather than a shared composite score.
6. Use movie/story-arc splits to prevent character, location, style, and narrative leakage.
7. Start with two engineering movies and 5-10 reviewed benchmark items before scaling LLM calls.

## Metric mapping

| Research claim | CameraMaster metric family | External support |
|---|---|---|
| Existing events are preserved | KER/EF, FDC, anchor fulfillment | LoCoT2V overall/FGA, DirectorBench text-video |
| Character/world state remains correct | NSP, LNSC, character/location/object state | LoCoT2V character/background consistency, DirectorBench temporal coherence |
| Events retain time/causal order | order/causal accuracy | DirectorBench script reasonableness/temporal coherence |
| Open-ended additions are useful | hard constraints, valid expansion count/ratio, novelty, causal linkage | HERD/narrative alignment as secondary evidence |
| Video remains technically usable | quality, warping, smoothness, stability | LoCoT2V PQ/DQ/warping; DirectorBench video/stability/audio |

## Important implementation caveats

- ConStory CED implementation counts error-bearing subtype columns, not every error object in an array.
- DirectorBench's final overall aggregation uses profile priority weights and omits unavailable dimensions; the default YAML display weights are not the same aggregation contract.
- LoCoT2V overall alignment expects `purified_prompt`; FGA/HERD require sample-aligned `prompt.json` metadata.
- LoCoT2V Dynamic Degree measures amount of motion, not quality; interpret with Motion Smoothness and DEVIL.
- All external benchmark results require exact environment/checkpoint/version recording.

