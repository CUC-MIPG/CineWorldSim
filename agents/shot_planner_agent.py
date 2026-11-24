from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import make_client, pick_model
from .retry import call_with_retries

logger = logging.getLogger(__name__)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _clamp_duration(value: Any, default: float = 3.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if v < 0.8:
        return 0.8
    if v > 12.0:
        return 12.0
    return v


def _parse_node_id(node_id: str) -> tuple[int, str]:
    m = re.match(r"^step::(\d+)::(.+)$", str(node_id or ""))
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2)


def _extract_json_object(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError("invalid json response")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("json root is not object")
    return obj


def _default_visual_heuristic(action_text: str, same_actor_streak: int) -> dict[str, Any]:
    action_l = (action_text or "").lower()
    shot_type = "medium shot"
    camera_angle = "eye-level"
    camera_movement = "static"
    editing_heuristic = "causal_progression"
    transition = "hard_cut"

    if any(k in action_l for k in ["run", "rush", "chase", "sprint", "attack", "grab"]):
        shot_type = "medium full shot"
        camera_movement = "tracking"
        editing_heuristic = "tracking_sequence"
    elif any(k in action_l for k in ["look", "inspect", "scan", "observe", "check"]):
        shot_type = "close-up"
        camera_movement = "slow_push_in"
        editing_heuristic = "investigation_focus"
    elif any(k in action_l for k in ["talk", "ask", "question", "reply", "whisper"]):
        shot_type = "over-the-shoulder"
        camera_movement = "subtle_handheld"
        editing_heuristic = "dialogue_exchange"

    if same_actor_streak >= 2:
        editing_heuristic = "tracking_sequence"
        transition = "match_on_action"

    return {
        "shot_type": shot_type,
        "camera_angle": camera_angle,
        "camera_movement": camera_movement,
        "editing_heuristic": editing_heuristic,
        "transition_to_next": transition,
    }


@dataclass(slots=True)
class ShotPlannerConfig:
    model: str
    strict_online: bool
    retries: int


class ShotPlannerAgent:
    def __init__(
        self,
        *,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ) -> None:
        self.client = make_client(scope="SHOT_PLANNER")
        self.model = model or pick_model("gpt-5.5", scope="SHOT_PLANNER")
        self.strict_online = bool(strict_online)
        self.retries = max(1, int(retries))

    def _query_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return _extract_json_object(content)

        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)

    @staticmethod
    def _index_narrative_step_logs(step_logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        idx: dict[str, dict[str, Any]] = {}
        for step_item in step_logs:
            step = int(step_item.get("step", 0) or 0)
            role_logs = step_item.get("role_logs", [])
            if not isinstance(role_logs, list):
                continue
            for role_item in role_logs:
                if not isinstance(role_item, dict):
                    continue
                role = _safe_text(role_item.get("role_name"))
                if not role:
                    continue
                nid = f"step::{step}::{role}"
                idx[nid] = {
                    "step": step,
                    "role_name": role,
                    "sam_summary": _safe_text(step_item.get("sam_summary")),
                    "sam_story_phase": _safe_text(step_item.get("sam_story_phase")),
                    "sam_current_environment": _safe_text(step_item.get("sam_current_environment")),
                    "role_output_text": _safe_text(role_item.get("role_output_text")),
                    "environment_description": role_item.get("environment_description", {}),
                    "next_action": role_item.get("next_action", {}),
                    "action_keyframe_path": _safe_text(role_item.get("action_keyframe_path")),
                }
        return idx

    def _compile_single_shot(
        self,
        *,
        storyline: str,
        node_id: str,
        node_data: dict[str, Any],
        prev_node_data: dict[str, Any] | None,
        same_actor_streak: int,
    ) -> dict[str, Any]:
        next_action = node_data.get("next_action", {})
        if not isinstance(next_action, dict):
            next_action = {}
        action_type = _safe_text(next_action.get("action_type"))
        target = _safe_text(next_action.get("target_character"))
        reason = _safe_text(next_action.get("reason"))
        monologue = _safe_text(next_action.get("monologue"))
        location = _safe_text(next_action.get("location"))
        role_name = _safe_text(node_data.get("role_name"))
        role_output = _safe_text(node_data.get("role_output_text"))
        world_summary = _safe_text(node_data.get("sam_summary"))
        story_phase = _safe_text(node_data.get("sam_story_phase"))
        env = _safe_text(node_data.get("sam_current_environment"))

        fallback_visual = _default_visual_heuristic(action_type or role_output, same_actor_streak)
        fallback = {
            **fallback_visual,
            "shot_prompt": (
                f"{role_name} {action_type or 'acts'} at {location or 'the scene'}, "
                f"cinematic lighting, coherent continuity, story phase: {story_phase or 'development'}."
            ),
            "duration_sec": 3.0,
            "narrative_purpose": f"Advance {role_name}'s action trajectory toward storyline objective.",
        }

        prev_context = ""
        if prev_node_data:
            prev_context = (
                f"Previous shot role={_safe_text(prev_node_data.get('role_name'))}, "
                f"action={_safe_text((prev_node_data.get('next_action') or {}).get('action_type'))}, "
                f"summary={_safe_text(prev_node_data.get('sam_summary'))}\n"
            )

        system_prompt = (
            "You are a cinematic shot planner. "
            "Generate one shot instruction that is visually continuous and narratively causal. "
            "Prefer tracking-sequence continuity for same-role streaks, "
            "and meaningful cross-cutting only when conflict tension increases. "
            "Avoid surveillance-style round-robin cuts. Return JSON only."
        )
        user_prompt = (
            f"Storyline objective:\n{storyline}\n\n"
            f"Current node id: {node_id}\n"
            f"Role: {role_name}\n"
            f"Action type: {action_type}\n"
            f"Target character: {target or 'none'}\n"
            f"Reason: {reason}\n"
            f"Monologue: {monologue}\n"
            f"Location: {location}\n"
            f"Role output: {role_output}\n"
            f"World summary: {world_summary}\n"
            f"Story phase: {story_phase}\n"
            f"Current environment: {env}\n"
            f"Same-role streak count: {same_actor_streak}\n"
            f"{prev_context}\n"
            "Return strict JSON:\n"
            "{\n"
            '  "shot_type": "<string>",\n'
            '  "camera_angle": "<string>",\n'
            '  "camera_movement": "<string>",\n'
            '  "editing_heuristic": "<tracking_sequence|cross_cutting|causal_progression|dialogue_exchange|investigation_focus>",\n'
            '  "shot_prompt": "<video generation prompt>",\n'
            '  "duration_sec": 0.8_to_12,\n'
            '  "transition_to_next": "<hard_cut|match_on_action|motivated_cut|cutaway>",\n'
            '  "narrative_purpose": "<short reason>"\n'
            "}\n"
        )

        payload: dict[str, Any] = {}
        try:
            payload = self._query_json(system_prompt, user_prompt)
        except Exception as exc:
            if self.strict_online:
                raise
            logger.warning("shot_planner llm failed node=%s fallback=%s", node_id, exc)

        out = {
            "shot_type": _safe_text(payload.get("shot_type")) or fallback["shot_type"],
            "camera_angle": _safe_text(payload.get("camera_angle")) or fallback["camera_angle"],
            "camera_movement": _safe_text(payload.get("camera_movement")) or fallback["camera_movement"],
            "editing_heuristic": _safe_text(payload.get("editing_heuristic")) or fallback["editing_heuristic"],
            "shot_prompt": _safe_text(payload.get("shot_prompt")) or fallback["shot_prompt"],
            "duration_sec": _clamp_duration(payload.get("duration_sec"), default=float(fallback["duration_sec"])),
            "transition_to_next": _safe_text(payload.get("transition_to_next")) or fallback["transition_to_next"],
            "narrative_purpose": _safe_text(payload.get("narrative_purpose")) or fallback["narrative_purpose"],
        }
        return out

    def compile_narrative_path(
        self,
        *,
        storyline: str,
        step_logs: list[dict[str, Any]],
        ordered_node_ids: list[str],
    ) -> dict[str, Any]:
        node_map = self._index_narrative_step_logs(step_logs)
        timeline: list[dict[str, Any]] = []
        prev_node_id = ""
        same_actor_streak = 0

        for rank, node_id in enumerate(ordered_node_ids, start=1):
            node_data = node_map.get(node_id)
            if not node_data:
                continue
            _step, role = _parse_node_id(node_id)
            if role and prev_node_id.endswith(f"::{role}"):
                same_actor_streak += 1
            else:
                same_actor_streak = 1

            prev_data = node_map.get(prev_node_id) if prev_node_id else None
            shot_plan = self._compile_single_shot(
                storyline=storyline,
                node_id=node_id,
                node_data=node_data,
                prev_node_data=prev_data,
                same_actor_streak=same_actor_streak,
            )

            timeline.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "step": int(node_data.get("step", 0) or 0),
                    "role_name": role,
                    "source_action": _safe_text((node_data.get("next_action") or {}).get("action_type")),
                    "source_target": _safe_text((node_data.get("next_action") or {}).get("target_character")),
                    "source_keyframe_path": _safe_text(node_data.get("action_keyframe_path")),
                    **shot_plan,
                }
            )
            prev_node_id = node_id

        return {
            "agent": "ShotPlannerAgent",
            "model": self.model,
            "mode": "narrative_path_to_video_shot_plan",
            "storyline": storyline,
            "shot_count": len(timeline),
            "timeline": timeline,
        }

    @staticmethod
    def save_json(payload: dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
