from __future__ import annotations

import json
import os
from typing import Any

from agents.llm_tool import LLMJsonTool
from narrative.pipeline_schemas import normalize_location_view


class WorldDeltaJudge:
    def __init__(
        self,
        model_config: Any = None,
        *,
        llm_tool: Any | None = None,
        llm_enabled: bool | None = None,
        strict_online: bool = False,
        retries: int = 3,
    ) -> None:
        self.model_config = model_config
        self.llm_tool = llm_tool
        if llm_enabled is None:
            llm_enabled = os.getenv("NARRATIVE_WORLD_DELTA_LLM_ENABLED", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.llm_enabled = bool(llm_enabled)
        self.strict_online = bool(strict_online)
        self.retries = max(1, int(retries or 1))

    @staticmethod
    def _step_text(step_log: dict[str, Any]) -> str:
        parts = [str(step_log.get("sam_summary", "") or "")]
        for role in step_log.get("role_logs", []) or []:
            action = role.get("next_action", {}) or {}
            parts.extend(
                str(action.get(key, "") or "")
                for key in ["action", "action_type", "location", "reason", "monologue"]
            )
            parts.append(str(role.get("role_output_text", "") or ""))
        return " ".join(parts).lower()

    @staticmethod
    def _changed(name: str, text: str) -> bool:
        name_tokens = [part.lower() for part in str(name).replace("_", " ").split() if part]
        if not any(token in text for token in name_tokens):
            return False
        change_words = [
            "open",
            "opened",
            "break",
            "broken",
            "take",
            "taken",
            "steal",
            "stolen",
            "move",
            "moved",
            "discover",
            "discovered",
            "flood",
            "flooded",
            "burn",
            "burned",
            "destroy",
            "destroyed",
            "unlock",
            "unlocked",
            "reveal",
            "revealed",
        ]
        return any(word in text for word in change_words)

    @staticmethod
    def _empty_delta(*, previous_state: dict[str, Any], step_log: dict[str, Any]) -> dict[str, Any]:
        step = int(step_log.get("step", previous_state.get("step", 0)) or 0)
        return {
            "step": step,
            "summary": str(step_log.get("sam_summary") or previous_state.get("summary") or ""),
            "location_updates": [],
            "item_updates": [],
            "character_updates": [],
            "decision_source": "disabled_or_fallback",
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a strict world-state visual-change judge for a narrative simulation pipeline. "
            "For locations and items, update world_state only when the current step's event descriptions explicitly change physical appearance: "
            "color, shape, visible structure, surface, texture, damage, open/closed physical configuration, flooding, burning, or other observable material state. "
            "Do not update for possession, discovery, movement, intention, reasoning, attention, emotional state, or narrative importance alone. "
            "For changed locations, choose exactly one view_to_update from wide, medium, close, extreme_close: the region whose image should be modified. "
            "For characters, update only explicit visible appearance changes such as clothing, colors, shape, hairstyle, or carried objects; "
            "do not update for emotion, intention, action, or pose alone. Include the source event roles and changed properties. "
            "Updated location and item images must contain no people, no characters, no pedestrians, no crowds, and no unrelated elements. "
            "The new_state_text must describe only the updated visual world_state after the change. Return JSON only."
        )

    @staticmethod
    def _user_prompt(*, story: str, previous_state: dict[str, Any], step_log: dict[str, Any]) -> str:
        payload = {
            "story": story,
            "previous_world_state": previous_state,
            "current_step_log": step_log,
            "required_output_schema": {
                "summary": "string",
                "location_updates": [
                    {
                        "id": "existing location id",
                        "changed": True,
                        "physical_change": "observable visual change only",
                        "changed_properties": ["color|shape|appearance|texture|structure|damage|configuration"],
                        "view_to_update": "wide|medium|close|extreme_close",
                        "new_state_text": "resulting physical visual state, no people",
                        "image_refresh_required": True,
                        "source_event_roles": ["role names"],
                    }
                ],
                "item_updates": [
                    {
                        "id": "existing item id",
                        "changed": True,
                        "physical_change": "observable visual change only",
                        "changed_properties": ["color|shape|appearance|texture|structure|damage|configuration"],
                        "new_state_text": "resulting physical visual state, no people",
                        "image_refresh_required": True,
                        "source_event_roles": ["role names"],
                    }
                ],
                "character_updates": [
                    {
                        "id": "existing character id or name",
                        "changed": True,
                        "physical_change": "explicit visible appearance change",
                        "changed_properties": ["clothing|color|shape|hair|carried_object"],
                        "new_state_text": "resulting character appearance",
                        "image_refresh_required": True,
                        "source_event_roles": ["role names"],
                    }
                ],
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool(self) -> Any:
        if self.llm_tool is None:
            self.llm_tool = LLMJsonTool(scope="NARRATIVE_WORLD_DELTA", retries=self.retries)
        return self.llm_tool

    @staticmethod
    def _valid_ids(previous_state: dict[str, Any], group: str) -> set[str]:
        ids: set[str] = set()
        for row in previous_state.get(group, []) or []:
            if not isinstance(row, dict):
                continue
            for key in ["id", "name"]:
                value = str(row.get(key, "") or "").strip()
                if value:
                    ids.add(value)
        return ids

    @staticmethod
    def _normalize_updates(
        raw_updates: Any,
        *,
        step: int,
        valid_ids: set[str],
        is_location: bool,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_updates, list):
            return []
        rows: list[dict[str, Any]] = []
        for raw in raw_updates:
            if not isinstance(raw, dict):
                continue
            if raw.get("changed") is False or raw.get("image_refresh_required") is False:
                continue
            update_id = str(raw.get("id", "") or "").strip()
            if not update_id or (valid_ids and update_id not in valid_ids):
                continue
            physical_change = str(raw.get("physical_change") or raw.get("new_state_text") or "").strip()
            new_state = str(raw.get("new_state_text") or physical_change).strip()
            if not physical_change and not new_state:
                continue
            row = {
                "step": step,
                "id": update_id,
                "changed": True,
                "physical_change": physical_change or new_state,
                "changed_properties": [
                    str(x)
                    for x in (raw.get("changed_properties", []) if isinstance(raw.get("changed_properties"), list) else [])
                    if str(x).strip()
                ],
                "new_state_text": new_state or physical_change,
                "image_refresh_required": True,
                "reason": "llm_physical_world_state_change",
                "source_event_roles": [
                    str(x)
                    for x in (raw.get("source_event_roles", []) if isinstance(raw.get("source_event_roles"), list) else [])
                    if str(x).strip()
                ],
            }
            if is_location:
                row["view_to_update"] = normalize_location_view(raw.get("view_to_update"), default="wide")
            rows.append(row)
        return rows

    def judge_step_delta(
        self,
        *,
        previous_state: dict[str, Any],
        step_log: dict[str, Any],
        story: str = "",
    ) -> dict[str, Any]:
        step = int(step_log.get("step", previous_state.get("step", 0)) or 0)
        if not self.llm_enabled:
            return self._empty_delta(previous_state=previous_state, step_log=step_log)

        try:
            payload = self._tool().run_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(story=story, previous_state=previous_state, step_log=step_log),
                temperature=0.1,
            )
        except Exception:
            if self.strict_online:
                raise
            return self._empty_delta(previous_state=previous_state, step_log=step_log)

        return {
            "step": step,
            "summary": str(payload.get("summary") or step_log.get("sam_summary") or previous_state.get("summary") or ""),
            "location_updates": self._normalize_updates(
                payload.get("location_updates", []),
                step=step,
                valid_ids=self._valid_ids(previous_state, "locations"),
                is_location=True,
            ),
            "item_updates": self._normalize_updates(
                payload.get("item_updates", []),
                step=step,
                valid_ids=self._valid_ids(previous_state, "items"),
                is_location=False,
            ),
            "character_updates": self._normalize_updates(
                payload.get("character_updates", []),
                step=step,
                valid_ids=self._valid_ids(previous_state, "characters"),
                is_location=False,
            ),
            "decision_source": "llm",
        }
