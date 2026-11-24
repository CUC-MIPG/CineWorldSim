from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.llm_tool import LLMJsonTool
from narrative.pipeline_schemas import normalize_duration_frame_plan


def _clean(value: Any, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(".,; ") + "."
    return text


class VideoShotPlanningTool:
    """LLM-backed planner for H3 event duration, frame count, and shot timing."""

    def __init__(self, *, llm_tool: LLMJsonTool | None = None, retries: int = 3) -> None:
        self.llm_tool = llm_tool or LLMJsonTool(scope="VIDEO", retries=retries)

    @staticmethod
    def fallback_plan(node: dict[str, Any], *, duration_sec: int | None = None) -> dict[str, Any]:
        event = _clean(node.get("story_description") or node.get("video_description") or node.get("event_summary"), 900)
        plan = normalize_duration_frame_plan({"duration_sec": 10}, default_duration_sec=10)
        plan.update(
            {
                "status": "fallback",
                "shots": [
                    {
                        "shot_index": 1,
                        "start_timecode": "00:00.000",
                        "start_sec": 0.0,
                        "duration_sec": plan["duration_sec"],
                        "video_description": event or "A concise cinematic event beat.",
                        "cinematography": "controlled cinematic framing with stable spatial continuity",
                        "dialogue": "none",
                        "bgm": "subtle atmospheric underscore matched to the event",
                    }
                ],
                "artifact_path": "",
                "meta_path": "",
                "error": "",
            }
        )
        return plan

    @staticmethod
    def _normalize_shots(shots: Any, total_duration: int, *, max_shots: int = 4) -> list[dict[str, Any]]:
        if not isinstance(shots, list) or not shots:
            return []
        rows: list[dict[str, Any]] = []
        last_start = 0.0
        kept_shots = [raw for raw in shots if isinstance(raw, dict)][:max(1, min(4, int(max_shots)))]
        weights: list[float] = []
        for raw in kept_shots:
            try:
                weights.append(max(0.1, float(raw.get("duration_sec", 0) or 0)))
            except (TypeError, ValueError):
                weights.append(1.0)
        total_weight = sum(weights) or float(len(kept_shots) or 1)
        durations = [round(10.0 * weight / total_weight, 3) for weight in weights]
        if durations:
            durations[-1] = round(10.0 - sum(durations[:-1]), 3)
        cursor = 0.0
        for idx, (raw, duration) in enumerate(zip(kept_shots, durations), start=1):
            rows.append(
                {
                    "shot_index": idx,
                    "start_timecode": f"00:{int(cursor):02d}.{int(round((cursor % 1) * 1000)):03d}",
                    "start_sec": round(cursor, 3),
                    "duration_sec": duration,
                    "video_description": _clean(raw.get("video_description") or raw.get("description"), 900),
                    "cinematography": _clean(raw.get("cinematography") or raw.get("camera"), 360),
                    "dialogue": _clean(raw.get("dialogue"), 500) or "none",
                    "narration": _clean(raw.get("narration") or raw.get("voiceover"), 700),
                    "bgm": _clean(raw.get("bgm"), 260) or "subtle atmospheric underscore matched to the event",
                }
            )
            cursor += duration
        return rows

    def generate(
        self,
        *,
        node: dict[str, Any],
        storyline: str,
        output_dir: Path | None = None,
        rank: int = 0,
        node_id: str = "",
        max_shots: int = 4,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are VideoAgent's H3 shot planner. Return JSON only. "
            "Plan one MiniMax H3 multishot event. The total video duration is exactly 10 seconds and 243 frames. "
            f"Choose 1-{max(1, min(4, int(max_shots)))} shots based on the event/storyboard complexity; never output more than 4 shots. "
            "Each shot must have shot_index, start_timecode, duration_sec, video_description, cinematography, dialogue, and bgm. "
            "Use dialogue='none' when no spoken line is needed."
        )
        user_prompt = json.dumps(
            {
                "storyline": storyline,
                "event_node": node,
                "required_schema": {
                    "duration_sec": 10,
                    "num_frames": 243,
                    "shots": [
                        {
                            "shot_index": 1,
                            "start_timecode": "00:00.000",
                            "duration_sec": "portion of the 10-second event",
                            "video_description": "story action for this shot",
                            "cinematography": "camera/framing/motion",
                            "dialogue": "none or spoken line",
                            "narration": "empty or explicit voiceover line",
                            "bgm": "music cue",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            payload = self.llm_tool.run_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
            plan = normalize_duration_frame_plan({**payload, "duration_sec": 10}, default_duration_sec=10)
            shots = self._normalize_shots(payload.get("shots", []), 10, max_shots=max_shots)
            if not shots:
                return self.fallback_plan(node, duration_sec=10)
            result = {**plan, "status": "done", "shots": shots, "artifact_path": "", "meta_path": "", "error": ""}
        except Exception as exc:  # noqa: BLE001
            result = self.fallback_plan(node)
            result["error"] = f"{type(exc).__name__}: {exc}"

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_node = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_id or "node").strip("_") or "node"
            path = output_dir / f"shot_plan_{int(rank):03d}_{safe_node}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result["artifact_path"] = str(path)
        return result
