"""Prompt construction for the two ViStoryBench experiment modes."""
from __future__ import annotations

import json
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def build_vistorybench_prompt(
    node: dict[str, Any],
    *,
    mode: str,
    language: str = "en",
) -> str:
    if mode not in {"vistorybench_anchor_only", "vistorybench_anchor_plus_sim"}:
        raise ValueError("mode must be vistorybench_anchor_only or vistorybench_anchor_plus_sim")
    anchor = node.get("benchmark_anchor") or {}
    characters = anchor.get("characters") or []
    lines = [
        "[ViStoryBench Shot Requirement]",
        f"Shot index: {_text(node.get('shot_index'))}",
        f"Camera: {_text(anchor.get('camera'))}",
        f"Scene: {_text(anchor.get('scene'))}",
        f"Action: {_text(anchor.get('action'))}",
        f"Plot: {_text(anchor.get('plot'))}",
        "",
        "[Required Characters]",
        ", ".join(_text(item) for item in characters) if characters else "None",
    ]
    if mode == "vistorybench_anchor_plus_sim":
        context = node.get("simulation_context") or {}
        lines.extend([
            "",
            "[Continuity Context]",
            f"Context mode: {_text(context.get('context_mode') or 'causal')}",
            f"Related prior events: {_text(context.get('related_events') or 'None')}",
            f"World state: {_text(context.get('world_state') or 'None')}",
            f"Continuity constraints: {_text(context.get('continuity_constraints') or 'None')}",
        ])
    lines.extend([
        "",
        "[Generation Constraints]",
        "The benchmark shot requirement has priority over all supplemental context.",
        "Continuity context may supplement state and visual details only.",
        "Do not change the camera type, character count, setting, or core action.",
        "Do not add major characters absent from the benchmark shot.",
        "Do not add unrelated people, pedestrians, or crowds.",
        "Preserve character identity and generate exactly one cinematic still image.",
        f"Output language: {language}",
    ])
    return "\n".join(lines)
