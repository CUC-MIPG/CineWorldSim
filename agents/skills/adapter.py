from __future__ import annotations

import re
from typing import Any

from agents.h3_prompt_adapter import build_h3_ref_prompt, reference_selection_policy, referenced_event_items
from agents.prompts import load_skill_prompt_template_text
from agents.skills.types import SkillManifest


def _clean(value: object, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(".,; ") + "."
    return text


def _story_description(node: dict[str, Any]) -> str:
    text = _clean(
        node.get("story_description", "") or node.get("video_description", "") or node.get("event_summary", ""),
        limit=360,
    )
    return "" if text.lower() == "none" else text


def _build_compact_event_prompt(node: dict[str, Any], storyline: str = "") -> str:
    story_description = _story_description(node)
    storyline_summary = _clean(storyline, limit=260)
    dialogue = _clean(node.get("dialogue", ""), limit=360) or "none"
    bgm = _clean(node.get("bgm", ""), limit=260) or "Subtle atmospheric underscore matched to the event."
    items = referenced_event_items(node)
    ref_policy = reference_selection_policy(items)
    pieces = [
        "Cinematic ref2video shot.",
        f"Storyline: {storyline_summary}" if storyline_summary else "",
        f"Story description: {story_description}" if story_description else "",
        f"Action: {story_description}" if story_description else "",
        f"Characters: {', '.join(node.get('characters', []) or [])}",
        f"Location: {', '.join(node.get('locations', []) or [])}",
        f"Objects: {', '.join(items) if items else 'none'}",
        ref_policy,
        f"dialogue: {dialogue}",
        f"bgm: {bgm}",
        "Preserve character identity and references, smooth camera motion, stable lighting, no slideshow cuts.",
    ]
    return " ".join(str(x).strip() for x in pieces if str(x).strip())


def _build_multi_shot_segment_prompt(node: dict[str, Any], shot: dict[str, Any], storyline: str = "") -> str:
    shot_index = int(shot.get("shot_index", 1) or 1)
    prefix = f"[Shot {shot_index}]"
    if shot_index > 1:
        prefix = f"[Shot {shot_index}] At {shot.get('start_timecode', '00:00.000')},"
    dialogue = _clean(shot.get("dialogue", "") or node.get("dialogue", ""), limit=360) or "none"
    return (
        "integrated_multimodal description: "
        f"{prefix} "
        f"video_description: {shot.get('video_description', '')} "
        f"cinematography: {shot.get('cinematography', '')}. "
        f"dialogue: {dialogue} "
        f"bgm: {shot.get('bgm', '')} "
        f"Preserve referenced character identity, world-space layout, lighting continuity, object placement, and coherent motion. "
        f"Generate only this shot as one continuous cinematic clip, not a slideshow. "
        f"Event type: {node.get('event_type', '')}. Story phase: {node.get('story_phase', '')}. "
        f"Storyline: {_clean(storyline, 300)}"
    )


def _build_multi_shot_event_prompt(node: dict[str, Any], shot_plan: list[dict[str, Any]], storyline: str = "") -> str:
    shot_segments: list[str] = []
    for shot in shot_plan:
        segment = _build_multi_shot_segment_prompt(node, shot, storyline=storyline)
        segment = segment.replace("integrated_multimodal description: ", "", 1)
        shot_segments.append(segment)
    return (
        "integrated_multimodal description: "
        + " ".join(shot_segments)
        + " Preserve referenced character identity, world-space layout, lighting continuity, object placement, and coherent motion across the full video. "
        + "Generate one complete continuous video containing all listed shots in order; do not split this into separate videos and do not make a slideshow. "
        + f"Event type: {node.get('event_type', '')}. Story phase: {node.get('story_phase', '')}."
    )


class SkillAdapter:
    def build(self, agent_name: str, skill: SkillManifest, context: dict[str, Any]) -> dict[str, Any]:
        del agent_name  # Reserved for future agent-specific adaptation.

        node = context.get("node") if isinstance(context.get("node"), dict) else {}
        shot = context.get("shot") if isinstance(context.get("shot"), dict) else {}
        storyline = str(context.get("storyline") or "")
        prompt_style = str(context.get("prompt_style") or "").strip().lower()
        mode = str(context.get("mode") or "").strip().lower()

        if mode == "multi_shot_segment":
            prompt = _build_multi_shot_segment_prompt(node, shot, storyline=storyline)
        elif skill.name == "h3-prompt-writing":
            shot_plan = context.get("shot_plan") if isinstance(context.get("shot_plan"), list) else []
            prompt = build_h3_ref_prompt(node, storyline=storyline, shot_plan=shot_plan)
        elif mode == "multi_shot_event":
            shot_plan = context.get("shot_plan") if isinstance(context.get("shot_plan"), list) else []
            prompt = _build_multi_shot_event_prompt(node, shot_plan, storyline=storyline)
        else:
            template_text = load_skill_prompt_template_text(skill)
            if template_text:
                prompt = template_text
            else:
                prompt = _build_compact_event_prompt(node, storyline=storyline)

        return {
            "prompt": prompt,
            "skill_name": skill.name,
            "skill_kind": skill.kind,
            "skill_source": skill.source_root,
            "skill_path": skill.skill_path,
            "prompt_style": prompt_style or ("official_ref" if skill.name == "h3-prompt-writing" else "compact"),
        }
