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


def _normalize_story_payload(payload: dict[str, Any], fallback_text: str) -> str:
    text = ""
    if isinstance(payload.get("story_description"), str):
        text = _clean(payload.get("story_description"), limit=1200)
        if text.lower() == "none":
            text = ""
    if not text:
        for value in payload.values():
            if isinstance(value, str):
                text = _clean(value, limit=1200)
                if text.lower() == "none":
                    text = ""
                    continue
                if text:
                    break
    return text or _clean(fallback_text, limit=1200) or "none"


class EventStoryTool:
    """Generate a prompt-only narrative story description for a single event."""

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
        story_description = _clean(
            node.get("video_description", "") or node.get("event_summary", "") or node.get("storyline", ""),
            limit=1200,
        )
        if not story_description:
            story_description = "none"
        result = {
            "status": "fallback",
            "story_description": story_description,
            "artifact_path": "",
            "meta_path": "",
            "error": error,
        }
        if output_dir is not None:
            paths = EventStoryTool.write_artifacts(
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                story_description=story_description,
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
        story_description: str,
        status: str,
        error: str = "",
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_node_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_id or "node")).strip("_") or "node"
        prefix = f"event_story_{int(rank):03d}_{safe_node_id}"
        artifact_path = output_dir / f"{prefix}.json"
        meta_path = output_dir / f"{prefix}.meta.json"
        artifact_path.write_text(
            json.dumps({"story_description": story_description}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "story_description": story_description,
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
        system_prompt = (
            "You are a narrative rewrite assistant for a cinematic event-video pipeline. "
            "Return only a JSON object with one key: story_description. "
            "Write a concrete story-style description of what happens in the event, in time order, focusing on character actions, dialogue, and spatial context. "
            "Do not write reasoning, analysis, or production notes. Do not add extra keys."
        )
        user_prompt = (
            "Rewrite the event into a prompt-ready story description.\n\n"
            f"Storyline:\n{_clean(storyline, 1200)}\n\n"
            f"Event summary:\n{_clean(node.get('event_summary', ''), 1200)}\n\n"
            f"Video description:\n{_clean(node.get('video_description', ''), 1200)}\n\n"
            f"Characters:\n{json.dumps(node.get('characters', []) or [], ensure_ascii=False, indent=2)}\n\n"
            f"Locations:\n{json.dumps(node.get('locations', []) or [], ensure_ascii=False, indent=2)}\n\n"
            f"Items:\n{json.dumps(node.get('items', []) or [], ensure_ascii=False, indent=2)}\n\n"
            f"Base prompt context:\n{_clean(base_prompt_context, 1200)}\n\n"
            "Return JSON only."
        )
        try:
            payload = self.llm_tool.run_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            return self.fallback_result(
                node=node,
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        story_description = _normalize_story_payload(
            payload if isinstance(payload, dict) else {},
            fallback_text=node.get("video_description", "") or node.get("event_summary", ""),
        )
        paths = self.write_artifacts(
            output_dir=output_dir,
            rank=rank,
            node_id=node_id,
            story_description=story_description,
            status="done",
        )
        return {
            "status": "done",
            "story_description": story_description,
            "error": "",
            **paths,
        }
