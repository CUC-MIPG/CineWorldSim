from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .retry import call_with_retries
from narrative.visual_prompt_constraints import (
    build_node_selection_prompt_policy,
    candidates_have_direct_interaction,
    event_needs_multi_person_references,
)

logger = logging.getLogger(__name__)


@dataclass
class NarrativeNodeSelectorConfig:
    selection_mode: str = "hybrid"
    score_threshold: float = 0.55
    min_nodes: int = 3
    max_nodes: int = 8
    llm_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(text: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", _safe_text(text)).strip("_")
    return slug or fallback


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


class NarrativeNodeSelector:
    """Select story-worthy events from stage-1 narrative/world simulation artifacts."""

    def __init__(self, config: NarrativeNodeSelectorConfig | None = None) -> None:
        self.config = config or NarrativeNodeSelectorConfig()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("invalid json response")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("json root is not object")
        return parsed

    @staticmethod
    def load_pipeline_result(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid pipeline_result JSON root: {path}")
        return payload

    @staticmethod
    def save_json(payload: dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def build_role_action_candidates(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        idx = 0
        for step_item in sorted(
            [x for x in payload.get("step_logs", []) or [] if isinstance(x, dict)],
            key=lambda x: int(x.get("step", 0) or 0),
        ):
            step = int(step_item.get("step", 0) or 0)
            if step <= 0:
                continue
            for role in step_item.get("role_logs", []) or []:
                if not isinstance(role, dict):
                    continue
                role_name = _safe_text(role.get("role_name"))
                if not role_name:
                    continue
                idx += 1
                action = role.get("next_action", {})
                action = action if isinstance(action, dict) else {}
                node_id = f"step::{step}::{role_name}"
                candidates.append(
                    {
                        "candidate_id": f"CAND_{idx:03d}",
                        "source_node_id": node_id,
                        "step": step,
                        "role_name": role_name,
                        "role_output_text": _safe_text(role.get("role_output_text")),
                        "action_type": _safe_text(action.get("action_type") or action.get("action")),
                        "target_character": _safe_text(action.get("target_character")),
                        "location": _safe_text(action.get("location")),
                        "reason": _safe_text(action.get("reason")),
                        "monologue": _safe_text(action.get("monologue")),
                        "sam_summary": _safe_text(step_item.get("sam_summary")),
                        "story_phase": _safe_text(step_item.get("sam_story_phase")),
                        "completion_score": self._completion_score(step_item),
                    }
                )
        return candidates

    @staticmethod
    def _completion_score(step_item: dict[str, Any]) -> float:
        raw = step_item.get("storyline_completion", {})
        if not isinstance(raw, dict):
            return 0.0
        try:
            return max(0.0, min(1.0, float(raw.get("completion_score", 0.0) or 0.0)))
        except Exception:
            return 0.0

    @staticmethod
    def _world_delta_for_step(payload: dict[str, Any], step: int) -> dict[str, Any]:
        for delta in payload.get("world_deltas", []) or []:
            if isinstance(delta, dict) and int(delta.get("step", 0) or 0) == step:
                return delta
        return {}

    @staticmethod
    def _world_state_for_step(payload: dict[str, Any], step: int) -> dict[str, Any]:
        exact: dict[str, Any] = {}
        fallback: dict[str, Any] = {}
        for state in payload.get("world_states", []) or []:
            if not isinstance(state, dict):
                continue
            state_step = int(state.get("step", 0) or 0)
            if state_step <= step:
                fallback = state
            if state_step == step:
                exact = state
                break
        return exact or fallback

    @staticmethod
    def _story_control_for_step(payload: dict[str, Any], step: int) -> dict[str, Any]:
        for item in payload.get("step_logs", []) or []:
            if isinstance(item, dict) and int(item.get("step", 0) or 0) == step:
                control = item.get("story_control")
                return control if isinstance(control, dict) else {}
        return {}

    @staticmethod
    def _delta_names(delta: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        locations: list[str] = []
        items: list[str] = []
        images: list[str] = []
        for key, bucket in [("location_updates", locations), ("item_updates", items)]:
            for row in delta.get(key, []) or []:
                if not isinstance(row, dict):
                    continue
                if row.get("id"):
                    bucket.append(str(row["id"]).replace("_", " "))
                if row.get("new_image_path"):
                    images.append(str(row["new_image_path"]))
        return _ordered_unique(locations), _ordered_unique(items), _ordered_unique(images)

    @staticmethod
    def _state_asset_images(state: dict[str, Any], names: list[str], group_name: str) -> list[str]:
        wanted = {_safe_id(x, x).lower() for x in names} | {x.lower() for x in names}
        paths: list[str] = []
        for row in state.get(group_name, []) or []:
            if not isinstance(row, dict):
                continue
            rid = _safe_text(row.get("id")).lower()
            rname = _safe_text(row.get("name")).lower()
            if wanted and rid not in wanted and rname not in wanted:
                continue
            path = _safe_text(row.get("last_image_path"))
            if path:
                paths.append(path)
        return _ordered_unique(paths)

    @staticmethod
    def _character_reference_images(payload: dict[str, Any], names: list[str]) -> list[str]:
        portrait_map = payload.get("character_portraits", {}) or {}
        paths: list[str] = []
        for name in names or []:
            path = portrait_map.get(name)
            if path:
                paths.append(str(path))
        return _ordered_unique(paths)

    @staticmethod
    def _keyframes_for_step_roles(payload: dict[str, Any], step: int, roles: list[str]) -> list[str]:
        role_set = set(roles)
        paths: list[str] = []
        for row in payload.get("event_keyframes", []) or []:
            if not isinstance(row, dict):
                continue
            if int(row.get("step", 0) or 0) != step:
                continue
            if _safe_text(row.get("role_name")) not in role_set:
                continue
            path = _safe_text(row.get("output_path"))
            if path:
                paths.append(path)
        return _ordered_unique(paths)

    def group_candidates_into_event_candidates(
        self,
        payload: dict[str, Any],
        role_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for cand in role_candidates:
            by_step.setdefault(int(cand.get("step", 0) or 0), []).append(cand)

        events: list[dict[str, Any]] = []
        event_idx = 0
        for step in sorted(by_step):
            step_group = list(by_step[step])
            used_candidate_ids: set[str] = set()
            clusters: list[list[dict[str, Any]]] = []
            for cand in step_group:
                cand_id = str(cand.get("candidate_id", ""))
                if cand_id in used_candidate_ids:
                    continue
                cluster = [cand]
                used_candidate_ids.add(cand_id)
                for other in step_group:
                    other_id = str(other.get("candidate_id", ""))
                    if other_id in used_candidate_ids:
                        continue
                    if candidates_have_direct_interaction(cand, other):
                        cluster.append(other)
                        used_candidate_ids.add(other_id)
                clusters.append(cluster)

            for cluster in clusters:
                event_idx += 1
                roles = _ordered_unique([_safe_text(x.get("role_name")) for x in cluster])
                locations = _ordered_unique([_safe_text(x.get("location")) for x in cluster if _safe_text(x.get("location"))])
                delta = self._world_delta_for_step(payload, step)
                delta_locations, delta_items, world_update_images = self._delta_names(delta)
                if not locations:
                    locations = delta_locations
                items = self._infer_items(payload=payload, step=step, group=cluster, delta_items=delta_items)
                summary = self._build_event_summary(cluster)
                source_node_ids = [str(x["source_node_id"]) for x in cluster]
                state = self._world_state_for_step(payload, step)
                event = {
                    "event_id": f"EVT_{event_idx:03d}",
                    "source_steps": [step],
                    "source_node_ids": source_node_ids,
                    "source_roles": roles,
                    "characters": roles,
                    "locations": locations,
                    "items": items,
                    "event_summary": summary,
                    "event_type": self._event_type(summary, len(roles), bool(delta_locations or delta_items)),
                    "story_phase": _safe_text(cluster[0].get("story_phase")),
                    "world_snapshot_ref": f"world_state_step{step}.json",
                    "world_delta_refs": [f"world_delta_step{step}.json"] if delta else [],
                    "world_update_images": world_update_images,
                    "completion_score": max(float(x.get("completion_score", 0.0) or 0.0) for x in cluster),
                    "node_granularity": "multi_role_interaction" if len(roles) > 1 else "single_role_step",
                }
                control = self._story_control_for_step(payload, step)
                event["covered_beat_ids"] = [str(x) for x in control.get("newly_covered_beat_ids", []) or [] if str(x).strip()]
                event["terminal_distance"] = control.get("distance_to_terminal")
                event["information_scope"] = _safe_text(control.get("recommended_scope"))
                event["visual_assets"] = self.bind_visual_assets(
                    payload=payload,
                    event=event,
                    state=state,
                    step=step,
                )
                events.append(event)
        return events

    @staticmethod
    def _build_event_summary(group: list[dict[str, Any]]) -> str:
        parts = []
        for cand in group:
            role = _safe_text(cand.get("role_name"))
            text = _safe_text(cand.get("role_output_text"))
            action = _safe_text(cand.get("action_type"))
            reason = _safe_text(cand.get("reason"))
            phrase = text or f"{role} {action}".strip()
            if reason:
                phrase = f"{phrase} Reason: {reason}."
            parts.append(phrase)
        return " ".join(parts).strip()

    @staticmethod
    def _event_type(summary: str, role_count: int, has_world_delta: bool) -> str:
        low = summary.lower()
        if role_count > 1 and any(x in low for x in ["block", "confront", "stop", "hide", "fight"]):
            return "conflict"
        if any(x in low for x in ["discover", "reveal", "uncover", "find"]):
            return "information_reveal"
        if has_world_delta:
            return "world_state_change"
        return "character_action"

    @staticmethod
    def _infer_items(
        *,
        payload: dict[str, Any],
        step: int,
        group: list[dict[str, Any]],
        delta_items: list[str],
    ) -> list[str]:
        items = list(delta_items)
        text = " ".join(
            [_safe_text(x.get("role_output_text")) + " " + _safe_text(x.get("action_type")) for x in group]
        ).lower()
        state = NarrativeNodeSelector._world_state_for_step(payload, step)
        for row in state.get("items", []) or []:
            if not isinstance(row, dict):
                continue
            name = _safe_text(row.get("name"))
            rid = _safe_text(row.get("id")).replace("_", " ")
            if (name and name.lower() in text) or (rid and rid.lower() in text):
                items.append(name or rid)
        return _ordered_unique(items)

    def bind_visual_assets(
        self,
        *,
        payload: dict[str, Any],
        event: dict[str, Any],
        state: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        multi_person_reference_allowed = event_needs_multi_person_references(event)
        character_reference_images = self._character_reference_images(payload, event.get("characters", []) or [])
        location_images = self._state_asset_images(state, event.get("locations", []) or [], "locations")
        item_images = self._state_asset_images(state, event.get("items", []) or [], "items")
        world_update_images = [x for x in event.get("world_update_images", []) or [] if x]
        event_keyframes = self._keyframes_for_step_roles(payload, step, event.get("characters", []) or [])
        return {
            "multi_person_reference_allowed": multi_person_reference_allowed,
            "character_reference_images": character_reference_images,
            "character_keyframes": list(event_keyframes),
            "location_images": _ordered_unique(location_images + [x for x in world_update_images if "/locations/" in x]),
            "item_images": _ordered_unique(item_images + [x for x in world_update_images if "/items/" in x]),
            "world_update_images": _ordered_unique(world_update_images),
            "event_keyframes": event_keyframes,
        }

    def score_event_candidate(self, event: dict[str, Any]) -> dict[str, Any]:
        text = _safe_text(event.get("event_summary")).lower()
        has_conflict = len(event.get("characters", []) or []) > 1 or any(
            x in text for x in ["block", "confront", "stop", "fight", "threat"]
        )
        has_reveal = any(x in text for x in ["discover", "reveal", "uncover", "find", "truth", "clue"])
        has_world_delta = bool(event.get("world_delta_refs"))
        has_assets = any(event.get("visual_assets", {}).get(k) for k in ["location_images", "item_images", "event_keyframes"])
        breakdown = {
            "dramatic_conflict": 0.85 if has_conflict else 0.25,
            "goal_progress": max(0.35, min(0.9, float(event.get("completion_score", 0.0) or 0.0) + (0.25 if has_reveal else 0.0))),
            "emotional_turning_point": 0.7 if has_conflict and has_reveal else (0.45 if has_conflict else 0.25),
            "information_reveal": 0.9 if has_reveal else 0.25,
            "visual_potential": 0.8 if has_assets else 0.35,
            "causal_importance": 0.85 if has_world_delta else 0.45,
            "redundancy_penalty": 0.1,
        }
        score = (
            0.20 * breakdown["dramatic_conflict"]
            + 0.20 * breakdown["goal_progress"]
            + 0.15 * breakdown["emotional_turning_point"]
            + 0.20 * breakdown["information_reveal"]
            + 0.15 * breakdown["visual_potential"]
            + 0.10 * breakdown["causal_importance"]
            - 0.15 * breakdown["redundancy_penalty"]
        )
        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "selection_reason": self._selection_reason(has_conflict, has_reveal, has_world_delta, has_assets),
            "score_source": "rule",
        }

    def _score_event_candidate_llm(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.llm_enabled:
            return None
        try:
            import importlib

            client_mod = importlib.import_module("agents.client")
        except Exception as exc:  # noqa: BLE001
            logger.warning("narrative_node_selector llm disabled: failed to import client helpers: %s", exc)
            return None

        system_prompt = (
            "You are a film story editor and director doing Stage 2 Narrative Node Selection. "
            "You walk through the spatiotemporal co-simulation results like a director reviewing rehearsal footage. "
            "Your job is to decide whether an event should become a kept Narrative Node for later "
            "event-to-multishot cinematography. An event is not a shot: it is a dramatic unit that may later "
            "be expanded into multiple camera setups. Reward dramatic conflict, goal progress, emotional turning "
            "points, information reveal, visual potential, and causal importance. Penalize mundane transitional "
            "actions, repetitive walking/looking, low-drama connective tissue, and events that do not deserve "
            "screen time. Default to one role action per step. Only merge multiple role actions when they share the same step, the same location, and explicit direct interaction. Return JSON only with 1-10 integer scores."
        )
        user_prompt = (
            "Narrative selection task for a story editor and director:\n"
            f"{json.dumps(event, ensure_ascii=False, indent=2)}\n\n"
            f"{build_node_selection_prompt_policy()}\n\n"
            "Selection standard:\n"
            "- Keep events that can serve as Key Events / Narrative Nodes.\n"
            "- High scores should indicate the event deserves to be adapted into an event-to-multishot sequence.\n"
            "- Low scores should indicate mundane transitional actions or redundant simulation residue.\n"
            "- Remember: this event is not a shot; it is a narrative unit for later multi-shot expansion.\n\n"
            "- When reading an event candidate, treat it as a single-role step unless it already states a same-step, same-location direct interaction between multiple characters.\n\n"
            "Return JSON fields:\n"
            "{\n"
            '  "dramatic_conflict": 1-10,\n'
            '  "goal_progress": 1-10,\n'
            '  "emotional_turning_point": 1-10,\n'
            '  "information_reveal": 1-10,\n'
            '  "visual_potential": 1-10,\n'
            '  "causal_importance": 1-10,\n'
            '  "redundancy_penalty": 1-10,\n'
            '  "selection_reason": "string"\n'
            "}\n"
            "Higher score means the node is more narratively valuable. "
            "Penalize repetitive, low-drama, or purely transitional content."
        )

        def _call() -> dict[str, Any]:
            client = client_mod.make_client(scope="OPENAI")
            model = client_mod.pick_model("gpt-5.5", scope="OPENAI")
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else "{}"
            if not isinstance(content, str):
                content = str(content)
            return self._extract_json_object(content)

        try:
            payload = call_with_retries(_call, attempts=3, delay_sec=1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("narrative_node_selector llm scoring failed, fallback to rule: %s", exc)
            return None

        def _norm_10(value: Any, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                v = default
            if v <= 1.0:
                return max(0.0, min(1.0, v))
            return max(0.0, min(1.0, v / 10.0))

        breakdown = {
            "dramatic_conflict": _norm_10(payload.get("dramatic_conflict"), 5.0),
            "goal_progress": _norm_10(payload.get("goal_progress"), 5.0),
            "emotional_turning_point": _norm_10(payload.get("emotional_turning_point"), 5.0),
            "information_reveal": _norm_10(payload.get("information_reveal"), 5.0),
            "visual_potential": _norm_10(payload.get("visual_potential"), 5.0),
            "causal_importance": _norm_10(payload.get("causal_importance"), 5.0),
            "redundancy_penalty": _norm_10(payload.get("redundancy_penalty"), 1.0),
        }
        score = (
            0.20 * breakdown["dramatic_conflict"]
            + 0.20 * breakdown["goal_progress"]
            + 0.15 * breakdown["emotional_turning_point"]
            + 0.20 * breakdown["information_reveal"]
            + 0.15 * breakdown["visual_potential"]
            + 0.10 * breakdown["causal_importance"]
            - 0.15 * breakdown["redundancy_penalty"]
        )
        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "selection_reason": _safe_text(payload.get("selection_reason")) or "llm scored narrative value",
            "score_source": "llm",
        }

    @staticmethod
    def _selection_reason(has_conflict: bool, has_reveal: bool, has_world_delta: bool, has_assets: bool) -> str:
        parts = []
        if has_conflict:
            parts.append("contains character conflict")
        if has_reveal:
            parts.append("reveals information or clue")
        if has_world_delta:
            parts.append("changes the world state")
        if has_assets:
            parts.append("has visual assets")
        return "; ".join(parts) if parts else "kept as baseline story coverage"

    def select_narrative_nodes(
        self,
        event_candidates: list[dict[str, Any]],
        *,
        required_beat_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scored: list[dict[str, Any]] = []
        for event in event_candidates:
            score_payload = self._score_event_candidate_llm(event) or self.score_event_candidate(event)
            scored.append({**event, **score_payload})

        ordered = sorted(scored, key=lambda x: (-float(x.get("score", 0.0) or 0.0), int((x.get("source_steps") or [0])[0])))
        required = _ordered_unique(required_beat_ids or [])
        coverage_selected: list[dict[str, Any]] = []
        coverage_ids: set[str] = set()
        for beat_id in required:
            candidate = next((row for row in ordered if beat_id in (row.get("covered_beat_ids") or [])), None)
            if candidate is not None and candidate["event_id"] not in coverage_ids:
                coverage_selected.append(candidate)
                coverage_ids.add(candidate["event_id"])
        budget = max(1, self.config.max_nodes, len(coverage_selected))
        if self.config.selection_mode == "top_k":
            selected = list(coverage_selected)
            selected.extend(row for row in ordered if row["event_id"] not in coverage_ids)
            selected = selected[:budget]
        elif self.config.selection_mode == "threshold":
            selected = [x for x in ordered if float(x.get("score", 0.0) or 0.0) >= self.config.score_threshold]
        else:
            selected = [x for x in ordered if float(x.get("score", 0.0) or 0.0) >= self.config.score_threshold]
            if len(selected) < self.config.min_nodes:
                selected = ordered[: min(len(ordered), self.config.min_nodes)]
            selected = selected[:budget]

        selected_ids = {x["event_id"] for x in selected}
        rejected = [x for x in scored if x["event_id"] not in selected_ids]
        selected = sorted(selected, key=lambda x: int((x.get("source_steps") or [0])[0]))
        for i, node in enumerate(selected, start=1):
            node["node_id"] = f"NN_{i:03d}"
        return selected, rejected

    def select(self, payload: dict[str, Any], *, source_pipeline_result_json: str = "") -> dict[str, Any]:
        role_candidates = self.build_role_action_candidates(payload)
        event_candidates = self.group_candidates_into_event_candidates(payload, role_candidates)
        control = payload.get("story_control") if isinstance(payload.get("story_control"), dict) else {}
        contract = control.get("contract") if isinstance(control.get("contract"), dict) else {}
        required = [
            _safe_text(item.get("beat_id"))
            for item in contract.get("hard_beats", []) or []
            if isinstance(item, dict) and _safe_text(item.get("beat_id"))
        ]
        selected, rejected = self.select_narrative_nodes(event_candidates, required_beat_ids=required)
        covered = _ordered_unique([beat for node in selected for beat in node.get("covered_beat_ids", []) or []])
        return {
            "storyline": _safe_text(payload.get("story")),
            "source_pipeline_result_json": source_pipeline_result_json,
            "selector_config": self.config.to_dict(),
            "role_candidate_count": len(role_candidates),
            "candidate_count": len(event_candidates),
            "selected_count": len(selected),
            "selected_nodes": selected,
            "rejected_nodes": rejected,
            "coverage_report": {
                "required_beat_ids": required,
                "selected_beat_ids": covered,
                "missing_beat_ids": [beat for beat in required if beat not in covered],
            },
        }

    def select_from_file(self, pipeline_result_json: Path, output_dir: Path) -> Path:
        payload = self.load_pipeline_result(pipeline_result_json)
        result = self.select(payload, source_pipeline_result_json=str(pipeline_result_json))
        return self.save_json(result, output_dir / "narrative_nodes.json")
