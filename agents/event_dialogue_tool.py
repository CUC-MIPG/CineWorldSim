from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .llm_tool import LLMJsonTool


def _clean(value: object, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(".,; ") + "."
    return text


def character_slots(characters: list[Any]) -> dict[str, str]:
    names = [str(name).strip() for name in characters or [] if str(name).strip()]
    return {f"Character{i}": name for i, name in enumerate(names, start=1)}


def normalize_dialogue_payload(payload: dict[str, Any], slot_mapping: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for slot in slot_mapping:
        value = payload.get(slot, "none")
        text = _clean(value, limit=320)
        normalized[slot] = text if text else "none"
    return normalized


def render_dialogue_for_prompt(dialogue: dict[str, str], slot_mapping: dict[str, str] | None = None) -> str:
    lines: list[str] = []
    slots = list(slot_mapping or dialogue)
    for slot in slots:
        value = _clean(dialogue.get(slot, ""), limit=320)
        if not value or value.lower() == "none":
            continue
        lines.append(f"{slot}: {value}")
    return "\n".join(lines) if lines else "none"


class EventDialogueTool:
    """Generate per-event dialogue JSON and deterministic prompt text."""

    def __init__(self, *, llm_tool: LLMJsonTool | None = None, retries: int = 3) -> None:
        self.llm_tool = llm_tool or LLMJsonTool(scope="VIDEO", retries=retries)

    @staticmethod
    def fallback_result(
        *,
        node: dict[str, Any],
        output_dir: Path | None = None,
        rank: int = 0,
        node_id: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        slot_mapping = character_slots(node.get("characters", []) or [])
        dialogue = {slot: "none" for slot in slot_mapping}
        rendered = render_dialogue_for_prompt(dialogue, slot_mapping)
        result = {
            "status": "fallback",
            "dialogue": dialogue,
            "slot_mapping": slot_mapping,
            "rendered_dialogue": rendered,
            "artifact_path": "",
            "meta_path": "",
            "error": error,
        }
        if output_dir is not None:
            paths = EventDialogueTool.write_artifacts(
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                dialogue=dialogue,
                slot_mapping=slot_mapping,
                rendered_dialogue=rendered,
                status="fallback",
                error=error,
            )
            result.update(paths)
        return result

    @staticmethod
    def write_artifacts(
        *,
        output_dir: Path,
        rank: int,
        node_id: str,
        dialogue: dict[str, str],
        slot_mapping: dict[str, str],
        rendered_dialogue: str,
        status: str,
        error: str = "",
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_node_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_id or "node")).strip("_") or "node"
        prefix = f"event_dialogue_{int(rank):03d}_{safe_node_id}"
        artifact_path = output_dir / f"{prefix}.json"
        meta_path = output_dir / f"{prefix}.meta.json"
        artifact_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "slot_mapping": slot_mapping,
                    "rendered_dialogue": rendered_dialogue,
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"artifact_path": str(artifact_path), "meta_path": str(meta_path)}

    def generate(
        self,
        *,
        node: dict[str, Any],
        storyline: str,
        output_dir: Path,
        rank: int,
        node_id: str,
        base_prompt_context: str = "",
    ) -> dict[str, Any]:
        slot_mapping = character_slots(node.get("characters", []) or [])
        if not slot_mapping:
            return self.fallback_result(
                node=node,
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                error="event has no character slots",
            )

        system_prompt = (
            "You are a dialogue writer for a cinematic event-video generation pipeline. "
            "Return only a JSON object. Use exactly the provided CharacterN keys. "
            "Each value must be one concise spoken line for that character, or exactly \"none\" if the character should not speak. "
            "Do not add stage directions, camera instructions, explanations, markdown, or extra keys."
        )
        user_prompt = (
            "Generate event dialogue JSON for the following event.\n\n"
            f"Character slot mapping:\n{json.dumps(slot_mapping, ensure_ascii=False, indent=2)}\n\n"
            f"Storyline:\n{_clean(storyline, 1200)}\n\n"
            f"Story description:\n{_clean(node.get('story_description', ''), 1200)}\n\n"
            f"Event node:\n{json.dumps(node, ensure_ascii=False, indent=2)}\n\n"
            f"Pre-skill prompt context:\n{_clean(base_prompt_context, 1200)}\n\n"
            "Return JSON only."
        )
        payload = self.llm_tool.run_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        dialogue = normalize_dialogue_payload(payload, slot_mapping)
        rendered = render_dialogue_for_prompt(dialogue, slot_mapping)
        paths = self.write_artifacts(
            output_dir=output_dir,
            rank=rank,
            node_id=node_id,
            dialogue=dialogue,
            slot_mapping=slot_mapping,
            rendered_dialogue=rendered,
            status="done",
        )
        return {
            "status": "done",
            "dialogue": dialogue,
            "slot_mapping": slot_mapping,
            "rendered_dialogue": rendered,
            "error": "",
            **paths,
        }
