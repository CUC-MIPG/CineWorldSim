---
name: h3-prompt-writing
description: Convert CineWorldSim event descriptions and references into MiniMax-H3 full-reference prompt format for ref2va video generation.
---

# MiniMax-H3 Prompt Writing for CineWorldSim

This local project skill follows MiniMax-H3's official full-reference prompt guide from:
`/data1/AgentG/hhb/minimax_h3/MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`.

## Required Sections

Generate prompts with six sections in order:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

## CineWorldSim Mapping

- Character event keyframes become character `<Subject N>` entries.
- Location/world images become environment `<Subject N>` entries.
- Important item images become object `<Subject N>` entries.
- Event summary becomes the shot action in `detailed_description`.
- Use `[reference generation]` for event-level ref2va unless a reference is explicitly a first/last/keyframe anchor.

## Implementation

CineWorldSim implements this adapter in:
`agents/h3_prompt_adapter.py`

Enable it with:

```bash
MINIMAX_H3_PROMPT_STYLE=official_ref
```

Fallback compact prompt style:

```bash
MINIMAX_H3_PROMPT_STYLE=compact
```
