# ViMax Track B Character Discovery

## Goal

Make Track B initialize the full relevant cast from a ViMax case by default, instead of silently using only `Protagonist`.

## Design

- When `--track-b-characters` is omitted, discover character names with the configured LLM from `story_overview` and textual shot fields (`first_frame`, `video_prompt`).
- Use these shot fields only to identify the cast. Track B simulation continues to receive `story_overview` as its narrative input; shot descriptions do not become anchors or simulation events.
- Pass the discovered names to the existing `OrchestratorAgent` initialization flow, which derives each character's role profile, goal, and relations.
- Preserve `--track-b-characters` as an explicit comma-separated override for controlled experiments.
- Reject an empty/invalid discovery result with a clear error rather than silently falling back to a single generic protagonist.
- Record the names and whether they were extracted or explicitly overridden in the run manifest for reproducibility.

## Validation

- Unit-test extraction input assembly, parsing/deduplication, explicit override precedence, and empty-result handling without making network calls.
- Test that the Track B orchestration still passes only `story_overview` to narrative simulation.
- Run the focused ViMax pipeline tests.

## Scope

This changes Track B cast selection only. Track A shot handling, narrative node selection, asset generation, and video generation remain unchanged.
