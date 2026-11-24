"""Fixed-anchor adapter for ViStoryBench shots.

This module deliberately does not use the narrative node selector. ViStoryBench
defines the number and order of evaluation shots, so CineWorldSim can only
attach bounded continuity context to those anchors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value if _text(item))
    return str(value).strip()


def _localized(value: Any, language: str) -> str:
    if isinstance(value, dict):
        return _text(value.get(language) or value.get("en") or next(iter(value.values()), ""))
    return _text(value)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", value) if len(token) > 2}


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _text(value)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


@dataclass(frozen=True)
class ViStoryBenchSelectorConfig:
    context_mode: str = "causal"
    max_prior_events: int = 3
    max_future_events: int = 0
    max_world_states: int = 1
    max_event_keyframes_per_character: int = 1

    def __post_init__(self) -> None:
        if self.context_mode not in {"causal", "full_story"}:
            raise ValueError("context_mode must be 'causal' or 'full_story'")
        if self.max_prior_events < 0 or self.max_future_events < 0:
            raise ValueError("event bounds must be non-negative")


class ViStoryBenchAnchorSelector:
    def __init__(self, config: ViStoryBenchSelectorConfig | None = None) -> None:
        self.config = config or ViStoryBenchSelectorConfig()

    def build_nodes(
        self,
        story: dict[str, Any],
        simulation: dict[str, Any] | None = None,
        *,
        story_id: str = "",
        language: str = "en",
    ) -> list[dict[str, Any]]:
        simulation = simulation or {}
        nodes: list[dict[str, Any]] = []
        for raw_shot in story.get("Shots", []) or []:
            if not isinstance(raw_shot, dict):
                continue
            if "index" not in raw_shot:
                raise ValueError("ViStoryBench shot is missing index")
            index = int(raw_shot["index"])
            anchor = self._anchor(raw_shot, language)
            context = self._context(anchor, index, simulation, language)
            nodes.append(
                {
                    "node_id": f"VB_{story_id or 'STORY'}_SHOT_{index:03d}",
                    "story_id": str(story_id),
                    "shot_index": index,
                    "benchmark_anchor": anchor,
                    "simulation_context": context,
                    "visual_assets": self._assets(anchor, context),
                }
            )
        return nodes

    @staticmethod
    def _anchor(raw: dict[str, Any], language: str) -> dict[str, Any]:
        characters = raw.get("Characters Appearing", {})
        if isinstance(characters, dict):
            character_names = characters.get(language) or characters.get("en") or []
        else:
            character_names = characters
        if isinstance(character_names, str):
            character_names = [character_names]
        return {
            "scene": _localized(raw.get("Setting Description"), language),
            "plot": _localized(raw.get("Plot Correspondence"), language),
            "camera": _localized(raw.get("Shot Perspective Design"), language),
            "action": _localized(raw.get("Static Shot Description"), language),
            "characters": [_text(item) for item in character_names or [] if _text(item)],
        }

    def _context(
        self,
        anchor: dict[str, Any],
        index: int,
        simulation: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        events = self._events(simulation, language)
        anchor_tokens = _tokens(" ".join(str(value) for value in anchor.values()))
        characters = {name.lower() for name in anchor.get("characters", [])}
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for position, event in enumerate(events):
            step = int(event.get("step", 0) or 0)
            if self.config.context_mode == "causal" and step > index:
                continue
            if step == index:
                distance_score = 2.0
            else:
                distance_score = 1.0 / (1.0 + abs(step - index))
            event_text = " ".join(_text(event.get(key)) for key in (
                "summary", "story_description", "role_output_text", "action", "location", "items"
            ))
            overlap = len(anchor_tokens & _tokens(event_text))
            role = _text(event.get("role_name")).lower()
            character_score = 2.0 if role in characters else 0.0
            score = character_score + overlap * 0.25 + distance_score
            ranked.append((score, position, event))
        prior = [item[2] for item in sorted(ranked, key=lambda item: (-item[0], item[1])) if int(item[2].get("step", 0) or 0) <= index]
        future = [item[2] for item in sorted(ranked, key=lambda item: (-item[0], item[1])) if int(item[2].get("step", 0) or 0) > index]
        related = prior[: self.config.max_prior_events]
        if self.config.context_mode == "full_story":
            related += future[: self.config.max_future_events]
        world_states = [
            state for state in simulation.get("world_states", []) or []
            if isinstance(state, dict) and int(state.get("step", 0) or 0) <= index
        ]
        world_states = sorted(world_states, key=lambda item: int(item.get("step", 0) or 0), reverse=True)
        keyframes: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        for event in related:
            role = _text(event.get("role_name"))
            if role.lower() in seen_roles and self.config.max_event_keyframes_per_character <= 1:
                continue
            if role:
                seen_roles.add(role.lower())
            keyframes.extend(event.get("event_keyframes", []) or [])
        return {
            "context_mode": self.config.context_mode,
            "source_steps": _unique([str(event.get("step")) for event in related]),
            "source_nodes": _unique([_text(event.get("node_id")) for event in related]),
            "related_events": related,
            "world_state": world_states[: self.config.max_world_states],
            "continuity_constraints": [],
            "event_keyframes": _unique([_text(path) for path in keyframes]),
        }

    @staticmethod
    def _events(simulation: dict[str, Any], language: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for step in simulation.get("step_logs", []) or []:
            if not isinstance(step, dict):
                continue
            step_number = int(step.get("step", 0) or 0)
            for role in step.get("role_logs", []) or []:
                if not isinstance(role, dict):
                    continue
                action = role.get("next_action") if isinstance(role.get("next_action"), dict) else {}
                events.append({
                    "node_id": f"step::{step_number}::{_text(role.get('role_name'))}",
                    "step": step_number,
                    "role_name": _text(role.get("role_name")),
                    "summary": _text(step.get("sam_summary")),
                    "story_description": _text(role.get("story_description")),
                    "role_output_text": _text(role.get("role_output_text")),
                    "action": _text(action.get("action") or action.get("action_type")),
                    "location": _text(action.get("location")),
                    "items": _text(action.get("items") or action.get("target")),
                    "event_keyframes": [],
                })
        for keyframe in simulation.get("event_keyframes", []) or []:
            if not isinstance(keyframe, dict):
                continue
            step = int(keyframe.get("step", 0) or 0)
            role = _text(keyframe.get("role_name"))
            for event in events:
                if int(event["step"]) == step and event["role_name"] == role:
                    path = _text(keyframe.get("output_path"))
                    if path:
                        event["event_keyframes"].append(path)
        return events

    @staticmethod
    def _assets(anchor: dict[str, Any], context: dict[str, Any]) -> dict[str, list[str]]:
        return {
            "character_refs": [],
            "location_refs": [],
            "event_keyframes": list(context.get("event_keyframes", [])),
        }
