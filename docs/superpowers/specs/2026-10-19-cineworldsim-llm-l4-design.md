# CineWorldSim LLM L4 Global Story Design

## Goal

Extend `scripts/restructure_cineworldsim_hierarchy.py` with an explicit opt-in
mode that generates a new movie-level L4 global story using the configured LLM.
The current offline restructuring behavior remains unchanged unless the new
flag is provided.

## Scope

### In Scope

- Read the existing Movie Bible as a global prior.
- Build compact, lineage-preserving summaries from L3 scenes and L2 events.
- Call the configured OpenAI-compatible chat-completions endpoint.
- Validate and normalize the returned L4 JSON.
- Write the generated L4 Movie Bible and request/response audit artifacts.
- Preserve the existing L3/L2/L1 restructuring and unresolved-reference logic.
- Add tests for opt-in behavior, prompt compaction, JSON validation, and
  source immutability.

### Out of Scope

- Re-generating L3 arcs, L2 events, or L1 shot annotations.
- Modifying the source dataset.
- Calling the LLM in the default mode.
- Sending every raw L1 payload directly to the model.
- Changing downstream video evaluators.

## Existing Behavior

The current script copies `l4_story/movie_bible.json` to
`l4_global/movie_bible.json`, marks it as `L4_global_movie_story`, and maps
the legacy story arcs to L3 local arcs. It is an offline file transformation
and does not use `.env` or an API.

## Proposed CLI

Existing invocation remains valid:

```bash
python -u scripts/restructure_cineworldsim_hierarchy.py \
  --input-root SOURCE_ROOT \
  --output-root OUTPUT_ROOT \
  --allow-unresolved-references
```

LLM generation is explicitly enabled with:

```text
--generate-l4
```

Optional controls:

```text
--llm-scope OPENAI
--llm-model <model-name>
--llm-retries 3
--llm-timeout <seconds>
--fallback-existing-bible
```

The implementation uses the existing environment resolution convention:
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from `.env`, with the
scope-specific variables supported by `agents.client.resolve_llm_env`.

## Input Projection

The LLM prompt contains four bounded components:

1. Existing Movie Bible fields as the global prior.
2. Compact L3 scene projections containing IDs, time ranges, summaries,
   characters, locations, actions, state changes, and source L2 IDs.
3. Compact L2 event projections containing IDs, source L1 IDs, event title,
   description, participants, setting, actions, interactions, and state
   changes.
4. A strict output schema and evidence-preservation instructions.

Repeated raw evidence, long camera descriptions, and full L1 payloads are not
sent unless required by the compact projection. Every generated global claim
must retain source IDs where the input supports them.

## L4 Output Contract

The model must return a JSON object containing at least:

```json
{
  "movie_id": "...",
  "movie_synopsis": "...",
  "global_storyline": "...",
  "global_character_arcs": [],
  "recurring_locations": [],
  "recurring_items": [],
  "global_constraints": [],
  "evidence_trace": {
    "source_l3_scene_ids": [],
    "source_l2_event_ids": [],
    "source_l1_shot_ids": []
  }
}
```

The validator requires non-empty `movie_synopsis` and `global_storyline`,
preserves the source `movie_id`, and requires evidence-trace fields to be
lists when present. The output is marked with:

```json
"hierarchy_level": "L4_global_movie_story",
"generation_method": "llm_compaction"
```

The script must not silently accept a non-object response or missing required
story fields.

## Audit Artifacts

When `--generate-l4` is enabled, the target contains:

```text
l4_global/
  movie_bible.json
  l4_generation_request.json
  l4_generation_response.json
```

The request artifact records the model, scope, source paths, compact input
counts, and prompt text. The response artifact records the parsed JSON and
generation metadata. API keys are never written to these artifacts.

## Failure Handling

- Missing API key, network failure, malformed JSON, or schema failure causes
  the command to fail by default and prevents a misleading generated L4 from
  being written.
- `--fallback-existing-bible` explicitly permits using the source Movie Bible
  when LLM generation fails; the report records that fallback and does not
  label it as newly generated.
- `--allow-unresolved-references` remains independent: it controls source
  reference validation only and does not suppress LLM failures.
- Existing output-directory safety is unchanged: a non-empty target is not
  overwritten.

## Data Flow

1. Validate the source tree and collect IDs.
2. Convert source story arcs to L3 local arcs in memory.
3. Build compact L3/L2 projections for the L4 request.
4. If `--generate-l4` is absent, use the source Movie Bible unchanged.
5. If enabled, call the JSON LLM tool and validate the response.
6. Write L4, L3, copied L2/L1 data, and the restructure report.
7. Record LLM metadata and fallback status in the report.

## Testing

Tests will verify:

- default mode performs no LLM call and preserves current output;
- generation mode calls an injected client and writes the new L4;
- compact projections retain IDs while omitting repeated raw payloads;
- malformed or incomplete model output fails validation;
- explicit fallback uses the source Movie Bible and records the fallback;
- source files remain byte-for-byte unchanged;
- existing unresolved-reference tests continue to pass.

## Acceptance Criteria

- The existing command remains offline and behavior-compatible.
- `--generate-l4` produces one new L4 global story from the Movie Bible plus
  compact L3/L2 context.
- The generated L4 is saved at `l4_global/movie_bible.json` with audit files.
- The output identifies whether the L4 was LLM-generated or source-fallback.
- No API credential is persisted.
- Tests cover both modes and pass.
