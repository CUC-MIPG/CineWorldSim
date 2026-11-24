# Dialogue Backfill and Prompt Audio Design

## Goal

Add an explicit dialogue-generation step to `VideoAgent` so that every event can first produce a structured dialogue JSON, then feed that JSON back into the pre-skill prompt, and finally preserve the same dialogue content in the MiniMax-H3 full-reference prompt path.

The feature must be opt-out, not opt-in:

- default behavior: generate dialogue JSON, backfill the pre-skill prompt, then pass the result through the existing skill layer
- opt-out behavior: keep the current fallback dialogue handling when the new step is disabled

The current `bgm` behavior must remain unchanged.

## Scope

### In Scope

- A new `VideoAgent` pre-prompt dialogue generation tool
- Structured per-event dialogue JSON output
- Prompt backfilling from the generated dialogue JSON
- Clear `dialogue` semantics in the pre-skill prompt
- Skill-path compatibility so the same dialogue content is visible in `h3-prompt-writing`
- Per-event artifact persistence for inspection and replay
- Tests for prompt construction, JSON validation, and disabled-mode fallback

### Out of Scope

- Changing the MiniMax-H3 prompt format itself
- Changing `bgm` generation logic
- Changing tail-frame continuity logic
- Reworking the narrative state machine
- Adding a new prompt format for non-video agents

## Current State

Today `VideoAgent` builds event prompts by combining:

- the event node summary
- the current story context
- the selected prompt style
- shot planning metadata for multi-shot events

The fallback prompt path in `agents/skills/adapter.py` and `agents/h3_prompt_adapter.py` currently derives audio text from the event node itself:

- `dialogue` is inferred from event text or role monologues
- `bgm` is inferred from event type and story phase

There is no separate structured dialogue artifact. The pre-skill prompt does not distinguish between:

- a real spoken line
- a silent event
- a placeholder description of how the event should sound

That makes prompt behavior harder to control and makes it difficult to align the pre-skill prompt with the `detailed_description` structure used by `h3-prompt-writing`.

## Requirements

### 1. Explicit Dialogue Semantics

- Every event prompt must contain a `dialogue` field.
- If no spoken dialogue exists for the event, `dialogue` must be exactly `none`.
- If dialogue exists, `dialogue` must contain only the generated dialogue text, not a free-form explanation.
- `bgm` must remain as the existing natural-language music description.

### 2. Structured Dialogue Generation

Before skill selection, `VideoAgent` must call a dedicated LLM-based dialogue generator using:

- the raw event node
- the story node description
- the current event context
- the current character list
- the pre-skill visual / narrative prompt context

The generator must return a JSON object with stable default character slots as keys and dialogue text as values:

```json
{
  "Character1": "xxxx",
  "Character2": "xxxx",
  "Character3": "none"
}
```

Rules:

- keys must be stable default slots derived from the event character order, using `Character1`, `Character2`, etc.
- the artifact metadata must record the slot mapping, for example `Character1 -> LateShiftChef`, so the generated dialogue remains traceable to the real event characters
- values must be strings
- `none` means no spoken content for that character
- empty strings are not allowed
- invalid JSON must be rejected and retried or fall back safely

### 3. Prompt Backfilling

The generated JSON is an intermediate artifact, not the final prompt representation.

The final pre-skill prompt should render dialogue deterministically from the JSON:

- keep `bgm` unchanged
- render the JSON content into a compact dialogue block
- preserve character order from the event node
- omit characters whose value is `none`
- if every value is `none`, render `dialogue: none`

The exact rendering format must be deterministic so repeated runs produce the same prompt text for the same JSON.

### 4. Skill Compatibility

The same backfilled dialogue content must flow into the skill layer:

- compact prompt style continues to use the backfilled dialogue in the old prompt template
- `h3-prompt-writing` should receive the same dialogue content so it can be surfaced in `detailed_description` as spoken lines

This feature does not change the MiniMax-H3 skill file format. It only changes what `VideoAgent` feeds into the skill adapter.

### 5. Optional Disable Switch

The new step must be controllable by configuration.

Recommended behavior:

- default: enabled
- disable flag: `--disable-dialogue-backfill`
- optional env mirror: `VIDEO_AGENT_DIALOGUE_BACKFILL=false`

When disabled:

- `VideoAgent` must preserve current dialogue fallback behavior
- no dialogue JSON needs to be generated
- `bgm` remains unchanged

## Proposed Design

### Component 1: Dialogue Generator

Add a dedicated helper in `VideoAgent` that produces event dialogue JSON before prompt building.

The raw LLM request must be encapsulated in a reusable tool instead of being written inline in `VideoAgent`:

- `agents/llm_tool.py`: generic JSON-returning LLM wrapper
- `agents/event_dialogue_tool.py`: event-specific dialogue generation, validation, rendering, and artifact writing

Responsibilities:

- inspect the event node and story context
- decide which characters speak
- produce one dialogue line per speaking character
- write a JSON artifact to disk
- return both the structured object and a rendered text representation

Suggested output artifact location:

- `.../dialogue_artifacts/event_dialogue.json`

Suggested companion metadata:

- `.../dialogue_artifacts/event_dialogue.meta.json`

### Component 2: Dialogue Renderer

Add a deterministic renderer that converts the JSON into prompt text.

Target rendering rules:

- `dialogue: none` when every value is `none`
- otherwise a compact multi-line block:
  - `Character1: ...`
  - `Character2: ...`
  - `Character3: ...`

This renderer should not infer new content. It only formats what the JSON generator returned.

### Component 3: Prompt Integration

Update the existing prompt construction path so `dialogue` is assigned from the rendered dialogue text instead of from `_dialogue_from_event()` when dialogue backfill is enabled.

For the skill path:

- `agents/skills/adapter.py` continues to build the prompt structure
- `agents/h3_prompt_adapter.py` should receive the backfilled dialogue text through the event node or shot context
- the H3 prompt output should express that same spoken content in the `detailed_description` section

### Component 4: Fallback Behavior

If the dialogue generator fails:

- log the failure
- fall back to the current dialogue inference path
- do not block video generation unless strict mode is explicitly configured

If the generator returns malformed JSON:

- retry once or twice according to the existing LLM retry policy
- then fall back to the current behavior

If a character has no spoken content:

- keep its key in the JSON only if the generator emitted it
- do not invent dialogue
- render `dialogue: none` only when no character speaks

## Data Flow

1. `VideoAgent` loads the event node.
2. `VideoAgent` runs the dialogue generator.
3. The generator writes `event_dialogue.json`.
4. `VideoAgent` renders dialogue text from the JSON.
5. `VideoAgent` injects the rendered dialogue into the pre-skill prompt.
6. The skill resolver selects either:
   - the compact CineWorldSim prompt path, or
   - `h3-prompt-writing`
7. The selected prompt builder consumes the same dialogue content.
8. The resulting task JSON stores the dialogue artifact path and prompt metadata.

## Prompt Semantics

### Pre-skill Prompt

- `dialogue` is always explicit
- `dialogue: none` means silent event
- `dialogue: ...` means the dialogue was generated upstream and then rendered into the prompt

### Skill Path

- `h3-prompt-writing` should use the same spoken content when building `detailed_description`
- the spoken content should be treated as the textual source for `<d>[Language] ...</d>` style lines
- the skill layer should not need to re-generate dialogue unless the upstream JSON is missing

## Testing Plan

### Unit Tests

- dialogue generator returns valid JSON for a speaking event
- dialogue renderer outputs `none` for silent events
- dialogue renderer preserves character order
- invalid JSON falls back safely
- disabled mode preserves the old prompt behavior

### Integration Tests

- `VideoAgent` writes a dialogue JSON artifact for an event
- the pre-skill prompt contains the rendered dialogue text
- the MiniMax-H3 full-reference prompt includes the same spoken content
- `bgm` remains unchanged

### Regression Tests

- current non-dialogue events still generate valid prompts
- existing multi-shot event generation still succeeds
- the task JSON schema remains compatible

## Risks

- A dialogue generator that is too verbose could leak plot detail into the prompt.
- Rendering JSON too literally could make the prompt clunky or too long.
- If the dialogue generator and skill adapter drift apart, the skill path could stop matching the pre-skill prompt.
- Silent-event handling must stay strict so the system does not invent dialogue where none exists.

## Rollout Plan

1. Add the dialogue generator and artifact writer behind a default-enabled flag.
2. Update pre-skill prompt construction to consume the generated dialogue.
3. Wire the same dialogue content into the skill path.
4. Add tests and verify both compact and H3 prompt modes.
5. Keep the old fallback path available behind the disable switch.
