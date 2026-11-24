"""Hidden story control for adaptive, diegetic narrative simulation.

The controller may read the whole story outline, but its output to characters is
strictly an in-world observation.  Beat IDs, future events, coverage and the
distance to the ending stay in the controller/audit artifacts.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip()


def _unique_text(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, list) else []:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _scope(distance: float, urgency: float) -> str:
    effective = max(0.0, min(1.0, distance - 0.25 * urgency))
    if effective > 0.72:
        return "local"
    if effective > 0.42:
        return "scene"
    if effective > 0.16:
        return "causal"
    return "closure"


def _without_control_markers(value: str) -> str:
    """Remove accidental audit identifiers before an observation reaches an actor."""
    return re.sub(r"\bB\d{1,3}\b", "", _text(value), flags=re.IGNORECASE).strip()


class StoryController:
    """Tracks hidden beat coverage and compiles safe observations for agents."""

    # A checkpoint produced by v1 could mark every beat covered from the
    # outline alone.  Do not silently resume it after tightening the evidence
    # contract below.
    VERSION = 6

    def __init__(
        self,
        *,
        story_outline: str,
        model_config: Any,
        saved_state: dict[str, Any] | None = None,
    ) -> None:
        self.story_outline = _text(story_outline)
        self.model_config = model_config
        restored = saved_state if isinstance(saved_state, dict) else {}
        if restored and int(restored.get("version", 0) or 0) != self.VERSION:
            raise ValueError(
                "Story-control checkpoint format is obsolete. Start Track B with a new output directory."
            )
        self.contract = self._normalise_contract(restored.get("contract")) if restored.get("contract") else self._build_contract()
        self.scene_plan = (self._normalise_scene_plan(restored["scene_plan"])
                           if isinstance(restored.get("scene_plan"), list) else self._build_scene_plan())
        restored_index = restored.get("active_scene_index")
        if restored_index is None and restored:
            restored_index = max((i for i, scene in enumerate(self.scene_plan)
                                  if scene["beat_id"] in restored.get("covered_beat_ids", [])), default=0)
        self.active_scene_index = max(0, min(len(self.scene_plan) - 1, int(restored_index or 0)))
        self.scene_memory = _unique_text(restored.get("scene_memory"))
        self.covered_beat_ids = {
            item for item in restored.get("covered_beat_ids", []) or []
            if item in self._beat_ids
        }
        self.contradictions = _unique_text(restored.get("contradictions"))
        self.stagnant_steps = max(0, int(restored.get("stagnant_steps", 0) or 0))
        self.last_assessment = dict(restored.get("last_assessment") or {})
        raw_evidence = restored.get("beat_evidence", {})
        self.beat_evidence = {
            beat_id: dict(value)
            for beat_id, value in raw_evidence.items()
            if beat_id in self.covered_beat_ids and isinstance(value, dict)
        } if isinstance(raw_evidence, dict) else {}

    @property
    def _beat_ids(self) -> set[str]:
        return {str(item["beat_id"]) for item in self.contract["hard_beats"]}

    @property
    def unmet_beat_ids(self) -> list[str]:
        return [item["beat_id"] for item in self.contract["hard_beats"] if item["beat_id"] not in self.covered_beat_ids]

    @property
    def terminal_ready(self) -> bool:
        """The last required beat has accepted evidence and no open conflict."""
        terminal_id = self.contract["hard_beats"][-1]["beat_id"]
        return (not self.unmet_beat_ids and not self.contradictions
                and terminal_id in self.beat_evidence)

    def _fallback_contract(self) -> dict[str, Any]:
        return {
            "immutable_entities": [],
            "hard_beats": [{
                "beat_id": "B01",
                "title": "Resolve the supplied story premise",
                "requirement": self.story_outline or "Advance the supplied story premise to a causal resolution.",
                "preconditions": [],
                "terminal": True,
            }],
            "terminal_constraints": ["Do not declare the story complete until every hard beat is evidenced in the world state."],
            "contract_source": "fallback",
        }

    def _normalise_contract(self, value: Any) -> dict[str, Any]:
        fallback = self._fallback_contract()
        raw = value if isinstance(value, dict) else fallback
        beats: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, row in enumerate(raw.get("hard_beats", []) or [], start=1):
            if not isinstance(row, dict):
                continue
            beat_id = _text(row.get("beat_id")) or f"B{index:02d}"
            if beat_id in seen:
                continue
            requirement = _text(row.get("requirement"))
            if not requirement:
                continue
            seen.add(beat_id)
            beats.append({
                "beat_id": beat_id,
                "title": _text(row.get("title")) or requirement[:96],
                "requirement": requirement,
                "preconditions": [x for x in _unique_text(row.get("preconditions")) if x in seen],
                "terminal": bool(row.get("terminal", False)),
            })
        if not beats:
            return fallback
        beats[-1]["terminal"] = True
        return {
            "immutable_entities": _unique_text(raw.get("immutable_entities")),
            "hard_beats": beats,
            "terminal_constraints": _unique_text(raw.get("terminal_constraints")) or fallback["terminal_constraints"],
            "contract_source": _text(raw.get("contract_source")) or "llm",
        }

    def _chat_json(self, *, label: str, prompt: str, fallback: dict[str, Any], temperature: float = 0.1) -> dict[str, Any]:
        try:
            payload = self.model_config.chat_json(
                user_prompt=prompt,
                fallback=fallback,
                temperature=temperature,
                call_label=label,
            )
            return payload if isinstance(payload, dict) else fallback
        except Exception:
            return fallback

    def _build_contract(self) -> dict[str, Any]:
        fallback = self._fallback_contract()
        prompt = (
            "You are the hidden story-control layer for a simulation. Extract a compact, ordered story contract from the supplied outline. "
            "Hard beats are facts that must be visibly evidenced before the story may end. Do not add unsupported plot facts. "
            "Return JSON only: {immutable_entities:string[], hard_beats:[{beat_id,title,requirement,preconditions:string[],terminal:boolean}], terminal_constraints:string[]}.\n\n"
            f"Story outline:\n{self.story_outline}"
        )
        return self._normalise_contract(self._chat_json(label="story_contract", prompt=prompt, fallback=fallback))

    def _normalise_scene_plan(self, value: Any) -> list[dict[str, str]]:
        rows = value if isinstance(value, list) else []
        by_beat = {str(row.get("beat_id")): row for row in rows if isinstance(row, dict)}
        result = []
        for index, beat in enumerate(self.contract["hard_beats"]):
            row = by_beat.get(beat["beat_id"], {})
            relation = _text(row.get("relationship_to_previous")).lower()
            if relation not in {"continuous", "independent"} or index == 0:
                relation = "opening" if index == 0 else "continuous"
            result.append({
                "beat_id": beat["beat_id"],
                "relationship_to_previous": relation,
                "scene_setting": _text(row.get("scene_setting")) or beat["title"],
                "stage_goal": _text(row.get("stage_goal")) or beat["requirement"],
            })
        return result

    def _build_scene_plan(self) -> list[dict[str, str]]:
        """Classify scene boundaries without making planned stages public facts."""
        prompt = (
            "Classify the relationship of consecutive stages in this story. "
            "Use independent only when the next stage has a distinct place/time and "
            "requires a scene handoff. Use continuous when action and physical state "
            "flow through the same scene. Preserve shared characters and objects in "
            "either case. A setting is not necessarily the subject depicted in a prop. "
            "Return JSON {scenes:[{beat_id,relationship_to_previous:" 
            "continuous|independent,scene_setting,stage_goal}]}. Stage goals must "
            "describe the next action in the original story, without inventing another "
            "painting, task, or outcome.\n"
            f"Story: {self.story_outline}\nBeats: {self.contract['hard_beats']}"
        )
        raw = self._chat_json(label="story_scene_plan", prompt=prompt, fallback={"scenes": []})
        return self._normalise_scene_plan(raw.get("scenes"))

    def apply_scene_transition(self, *, world: Any, agents: list[Any], world_tick: dict[str, Any]) -> dict[str, Any]:
        """Commit a stage boundary only after the world has established it."""
        next_index = self.active_scene_index + 1
        if next_index >= len(self.scene_plan) or not world_tick.get("scene_transition"):
            return {}
        scene = self.scene_plan[next_index]
        if scene["beat_id"] != world_tick.get("target_beat_id"):
            return {}
        prior = self.scene_plan[self.active_scene_index]
        handoff = (
            f"Previous stage: {prior['scene_setting']}. "
            f"Last established state: {_text(getattr(world, 'summary', ''))[-500:]}"
        )
        self.scene_memory.append(handoff)
        self.scene_memory = self.scene_memory[-8:]
        self.active_scene_index = next_index
        for agent in agents:
            # The local objective changes; identity, personal memory and clues persist.
            agent.state.final_goal = scene["stage_goal"]
            agent.model_config.system_prompt = (
                f"You are {agent.state.name}. Your current personal goal is: {scene['stage_goal']}. "
                "Reason from your present observations and memory. Decide for yourself how to handle obstacles. "
                "Do not assume knowledge of later story events or settings."
            )
            agent.state.state_summary = (
                f"{agent.state.state_summary}\nScene handoff: {handoff}"
            ).strip()[-2000:]
        return {"from": prior["beat_id"], "to": scene["beat_id"],
                "relationship": scene["relationship_to_previous"],
                "stage_goal": scene["stage_goal"], "memory": handoff}

    def _fallback_assessment(self) -> dict[str, Any]:
        unmet = self.unmet_beat_ids
        distance = len(unmet) / max(1, len(self.contract["hard_beats"]))
        urgency = min(1.0, self.stagnant_steps / 3.0)
        return {
            "newly_covered_beat_ids": [],
            "covered_beat_ids": sorted(self.covered_beat_ids),
            "active_beat_id": unmet[0] if unmet else "",
            "blocked_preconditions": [],
            "contradictions": list(self.contradictions),
            "distance_to_terminal": distance,
            "urgency": urgency,
            "recommended_scope": _scope(distance, urgency),
        }

    def opening_public_state(self) -> dict[str, str]:
        """Compile an opening without copying the full outline into public state."""
        first = self.contract["hard_beats"][0]["requirement"]
        fallback = {"environment": "The character is in the opening setting, with only nearby conditions observable.",
                    "goal": "Pursue the character's current personal objective.",
                    "role": "A participant in the present scene."}
        raw = self._chat_json(
            label="story_opening_public_state",
            prompt=("Create the present opening scene and a short personal goal from ONLY the first story stage below. "
                    "Do not name or foreshadow later settings, events, weather, outcomes, or a final action. "
                    "The goal may describe the character's intent but may not declare it achieved. "
                    "Return JSON {environment:string, goal:string, role:string}.\nFirst stage: " + first),
            fallback=fallback,
        )
        return {"environment": _without_control_markers(raw.get("environment")) or fallback["environment"],
                "goal": _without_control_markers(raw.get("goal")) or fallback["goal"],
                "role": _without_control_markers(raw.get("role")) or fallback["role"]}

    def assess(
        self,
        *,
        step: int,
        world_summary: str,
        current_environment: str,
        public_events: list[str],
        step_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = self._fallback_assessment()
        before = set(self.covered_beat_ids)
        terminal_beat = self.contract["hard_beats"][-1]
        terminal_id = terminal_beat["beat_id"]
        active_id = self.unmet_beat_ids[0] if self.unmet_beat_ids else ""
        terminal_eligible = (
            active_id == terminal_id
            and set(terminal_beat.get("preconditions", [])).issubset(before)
        )
        fallback["beat_evidence"] = []
        fallback["terminal_decision"] = {
            "beat_id": terminal_id, "matched": False,
            "reason": "Terminal assessment unavailable; no event was accepted as proof.",
            "evidence_text": "",
        }
        compact_actions = [
            {"actor": _text(item.get("role_name")),
             "action": _text((item.get("next_action") or {}).get("action_type")),
             "event": _unique_text(item.get("actual_step_events")),
             "prior_environment": _text((item.get("environment_description") or {}).get("current_environment"))}
            for item in step_actions if isinstance(item, dict) and item.get("action_accepted", True)
        ]
        opening_state = ""
        if step == 1:
            opening_state = " ".join(
                f"{_text((item.get('environment_description') or {}).get('current_environment'))} "
                f"{_text(((item.get('environment_description') or {}).get('visible_observation') or {}).get('visible_environment'))}"
                for item in step_actions if isinstance(item, dict)
            ).strip()
        prompt = (
            "You are the hidden story-control layer. Assess whether exactly one hard beat became evidenced by an accepted action in THIS step. "
            "At step 1 only, the publicly established opening state is also admissible evidence for the FIRST beat, if it explicitly depicts the required ongoing activity. "
            "The story contract is only a plan, never evidence. Do not infer an event from a requirement, a future plan, a world summary, or a desired ending. "
            "A beat may be covered only after all of its listed preconditions were covered before this step. "
            "Return JSON only: {beat_evidence:[{beat_id:string,evidence_text:string}], "
            "terminal_decision:{beat_id:string,matched:boolean,reason:string,evidence_text:string}, "
            "contradictions:string[]}. The evidence_text must quote or closely paraphrase the accepted "
            "action/event that proves the beat; return an empty beat_evidence list when no current action proves a beat. "
            "When the active beat is terminal, decide whether THIS STEP's accepted action and resulting event "
            "semantically establish its full requirement. Judge what the actor did and what visibly happened, "
            "including paraphrases such as 'make a brush stroke'; do not require a fixed verb or exact wording. "
            "An attempted, planned, remembered, or merely described earlier event does not count. "
            "A corrective stroke does not count unless the event actually establishes the required final completion. "
            "Set terminal_decision.matched to true only when the terminal action occurred and the resulting "
            "event proves completion; cite this step's evidence_text and explain why in reason. "
            "For a nonmatch, explain the missing fact in reason and leave evidence_text empty. "
            "When the terminal beat is not yet eligible, set matched=false and explain why.\n\n"
            f"Story contract (hidden): {self.contract}\n"
            f"Already covered before step {step}: {sorted(self.covered_beat_ids)}\n"
            f"Active beat: {active_id}; terminal eligible: {terminal_eligible}\n"
            f"Opening state (first beat only, step 1 only): {opening_state}\n"
            f"Accepted actions/events at step {step}: {compact_actions}"
        )
        raw = self._chat_json(label="story_control_assessment", prompt=prompt, fallback=fallback)
        beats_by_id = {str(item["beat_id"]): item for item in self.contract["hard_beats"]}
        action_text = " ".join(
            f"{item.get('action', '')} {' '.join(item.get('event', []))}" for item in compact_actions
        ).lower()
        if step == 1:
            action_text += " " + opening_state.lower()
        evidence_rows = raw.get("beat_evidence", [])
        if not isinstance(evidence_rows, list):
            evidence_rows = []
        # Terminal coverage comes only from the explicit, reasoned decision.
        evidence_rows = [row for row in evidence_rows if not isinstance(row, dict) or _text(row.get("beat_id")) != terminal_id]
        decision_raw = raw.get("terminal_decision")
        terminal_decision = {
            "beat_id": terminal_id, "matched": False, "accepted": False,
            "reason": "Terminal beat is not the eligible active beat.",
            "evidence_text": "", "validation_reason": "",
        }
        if terminal_eligible:
            terminal_decision["reason"] = "Model returned no valid terminal decision."
            if isinstance(decision_raw, dict) and _text(decision_raw.get("beat_id")) == terminal_id:
                reason = _text(decision_raw.get("reason"))
                matched = decision_raw.get("matched")
                if type(matched) is bool and reason:
                    terminal_decision.update({
                        "matched": matched, "reason": reason,
                        "evidence_text": _text(decision_raw.get("evidence_text")),
                    })
                    if matched and not terminal_decision["evidence_text"]:
                        terminal_decision["validation_reason"] = "Matched decision has no current-step evidence."
                else:
                    terminal_decision["validation_reason"] = "Terminal decision needs a boolean match and nonempty reason."
            else:
                terminal_decision["validation_reason"] = "Terminal decision is missing or has the wrong beat ID."
            if terminal_decision["matched"] and terminal_decision["evidence_text"]:
                evidence_rows = [*evidence_rows, {
                    "beat_id": terminal_id,
                    "evidence_text": terminal_decision["evidence_text"],
                }]
        # One newly proven beat per simulation step is deliberate: it prevents
        # an LLM from collapsing a multi-stage storyline into one broad action.
        for row in evidence_rows:
            if self.covered_beat_ids != before or not isinstance(row, dict):
                break
            beat_id = _text(row.get("beat_id"))
            evidence_text = _text(row.get("evidence_text"))
            beat = beats_by_id.get(beat_id)
            if not beat or beat_id in before or not evidence_text:
                continue
            if not set(beat.get("preconditions", [])).issubset(before):
                continue
            if not compact_actions and not (step == 1 and beat_id == self.contract["hard_beats"][0]["beat_id"]):
                continue
            if beat.get("terminal") and (
                not terminal_eligible or not terminal_decision["matched"]
                or not terminal_decision["evidence_text"]
            ):
                continue
            if beat.get("terminal") and not any(item["event"] for item in compact_actions):
                terminal_decision["validation_reason"] = "No accepted resulting event was recorded for this step."
                continue
            if "painting" in beat["requirement"].lower() and "effort toward completion" in beat["requirement"].lower():
                if any(re.search(r"(?:painting|landscape).{0,30}(?:is |already |: |')?complete(?:d)?\b", item["prior_environment"].lower()) for item in compact_actions):
                    continue
            # Evidence must remain grounded in text emitted for this step,
            # rather than being an unsupported restatement of the contract.
            evidence_tokens = set(re.findall(r"[a-zA-Z]{3,}", evidence_text.lower()))
            action_tokens = set(re.findall(r"[a-zA-Z]{3,}", action_text))
            if not beat.get("terminal") and evidence_tokens and action_tokens and not (evidence_tokens & action_tokens):
                continue
            self.covered_beat_ids.add(beat_id)
            self.beat_evidence[beat_id] = {
                "evidence_step": step,
                "evidence_event_id": f"step::{step}",
                "evidence_text": evidence_text,
                "source_actions": compact_actions,
            }
            if beat.get("terminal"):
                self.beat_evidence[beat_id]["reason"] = terminal_decision["reason"]
                terminal_decision["accepted"] = True
        if terminal_eligible and terminal_decision["matched"] and not terminal_decision["accepted"] and not terminal_decision["validation_reason"]:
            terminal_decision["validation_reason"] = "Terminal decision did not pass the current-step evidence checks."
        if self.covered_beat_ids == before:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        contradictions = _unique_text(raw.get("contradictions"))
        self.contradictions = contradictions
        unmet = self.unmet_beat_ids
        active = unmet[0] if unmet else ""
        # Distance is controller-owned, so an LLM cannot report a near-terminal
        # state while required beats are still open.
        distance = len(unmet) / max(1, len(self.contract["hard_beats"]))
        urgency = min(1.0, self.stagnant_steps / 3.0)
        blocked = [
            f"{item['beat_id']} requires {', '.join(item.get('preconditions', []))}"
            for item in self.contract["hard_beats"]
            if item["beat_id"] in unmet and item.get("preconditions")
            and not set(item["preconditions"]).issubset(self.covered_beat_ids)
        ]
        assessment = {
            "newly_covered_beat_ids": sorted(self.covered_beat_ids - before),
            "covered_beat_ids": sorted(self.covered_beat_ids),
            "unmet_beat_ids": unmet,
            "active_beat_id": active,
            "beat_evidence": {
                beat_id: self.beat_evidence[beat_id]
                for beat_id in sorted(self.covered_beat_ids - before)
                if beat_id in self.beat_evidence
            },
            "terminal_decision": terminal_decision if terminal_eligible else None,
            "blocked_preconditions": blocked,
            "contradictions": contradictions,
            "distance_to_terminal": distance,
            "urgency": urgency,
            "recommended_scope": _scope(distance, urgency),
            "stagnant_steps": self.stagnant_steps,
        }
        self.last_assessment = assessment
        return assessment

    def build_agent_observation(self, *, agent_state: Any, world: Any) -> dict[str, Any]:
        assessment = self.last_assessment or self._fallback_assessment()
        scope = str(assessment.get("recommended_scope") or "local")
        fallback = {
            "visible_environment": _text(getattr(agent_state, "current_environment", "")) or _text(getattr(world, "current_environment", "")),
            "salient_changes": [],
            "immediate_risks": [],
            "condition_details": [],
            "available_affordances": ["inspect the immediate surroundings"],
            "perceived_events": list(getattr(world, "public_events", []) or [])[-6:],
            "private_memory_summary": _text(getattr(agent_state, "state_summary", "")) or _text(getattr(agent_state, "final_goal", "")),
        }
        stagnant = int(assessment.get("stagnant_steps", self.stagnant_steps) or 0)
        prompt = (
            "You are a world model compiling an in-world observation for one character. The hidden control state tells you how broadly to present present-world information and what kind of opportunity the world should naturally make possible. "
            "Never mention beat IDs, hidden requirements, completion scores, future plot events, the story outline, or an instruction to satisfy a narrative plan. "
            "Describe the degree of each present condition, its physical effects and what it could prevent; do not rank risks or prescribe a next action. "
            "Available affordances are for the world's private feasibility check; they must NOT be narrated to the character. "
            "Never portray a contemplated action as already performed. Return only facts currently perceived or established by accepted actions. "
            "An accepted, directly observable bodily action has a bounded physical result unless a concrete blocker prevented it. "
            "If a prior strategy repeatedly failed, describe its observed failure and whether a short attempt toward the personal goal is physically possible despite residual risk; do not prescribe the attempt or invent resources. "
            "Return JSON only: {visible_environment:string, salient_changes:string[], condition_details:string[], immediate_risks:string[], available_affordances:string[], perceived_events:string[], private_memory_summary:string}.\n\n"
            f"Information scale: {scope}\n"
            f"Consecutive steps without a new story event: {stagnant}. Give the character concrete feedback about repeated attempts when this is high.\n"
            f"Hidden world-steering requirement (do not quote, name, or reveal it): {next((item['requirement'] for item in self.contract['hard_beats'] if item['beat_id'] == assessment.get('active_beat_id')), '')}\n"
            f"Current world summary: {_text(getattr(world, 'summary', ''))}\n"
            f"Current world environment: {_text(getattr(world, 'current_environment', ''))}\n"
            f"Character: {getattr(agent_state, 'name', '')}; profile: {getattr(agent_state, 'role_profile', '')}; private state: {getattr(agent_state, 'state_summary', '')}; clues: {getattr(agent_state, 'known_clues', [])}\n"
            f"Past public events: {list(getattr(world, 'public_events', []) or [])[-8:]}"
        )
        raw = self._chat_json(label="world_observation", prompt=prompt, fallback=fallback, temperature=0.2)
        observation = {
            "visible_environment": _without_control_markers(raw.get("visible_environment")) or fallback["visible_environment"],
            "salient_changes": [_without_control_markers(item) for item in _unique_text(raw.get("salient_changes"))],
            "condition_details": [_without_control_markers(item) for item in _unique_text(raw.get("condition_details"))],
            "immediate_risks": [_without_control_markers(item) for item in _unique_text(raw.get("immediate_risks"))],
            "available_affordances": [_without_control_markers(item) for item in _unique_text(raw.get("available_affordances"))] or fallback["available_affordances"],
            "perceived_events": [_without_control_markers(item) for item in _unique_text(raw.get("perceived_events"))] or fallback["perceived_events"],
            "private_memory_summary": _without_control_markers(raw.get("private_memory_summary")) or fallback["private_memory_summary"],
            "information_scope": scope,
        }
        return observation

    def advance_world(self, *, world: Any) -> dict[str, Any]:
        """Allow a bounded external event before the character observes a turn."""
        unmet = self.unmet_beat_ids
        if not unmet:
            return {"changed": False, "public_event": "", "current_environment": "", "reason": "story complete"}
        active = next(item for item in self.contract["hard_beats"] if item["beat_id"] == unmet[0])
        next_index = next((i for i, row in enumerate(self.scene_plan) if row["beat_id"] == active["beat_id"]), self.active_scene_index)
        boundary = next_index > self.active_scene_index
        relation = self.scene_plan[next_index]["relationship_to_previous"]
        fallback = {"changed": False, "public_event": "", "current_environment": "", "reason": "no established external change"}
        prompt = (
            "You are the hidden world model advancing external conditions by at most ONE plausible, observable change before the character acts. "
            "Use the hidden upcoming story stage as a direction for causal opportunities, never as public information. "
            "A new place requires an established path, travel, or a clear temporal transition; do not teleport the character or painting. "
            "Weather may change or a nearby route/resource may become observable, but do not invent a completed character action or a finished goal. "
            "When prior risk responses have stalled, provide a concrete new fact or a bounded change supported by the setting. "
            "When a previous stage is complete, create a credible handoff to the next stage if the environment supports it. "
            "For an independent scene, explicitly establish travel or a time transition while carrying forward the same character and important props. "
            "For a continuous scene, retain the same location and incrementally update its state. "
            "Return JSON {changed:boolean, public_event:string, current_environment:string, reason:string, scene_transition:boolean}. "
            "If nothing changes, changed=false and leave both public fields empty.\n"
            f"Hidden next stage: {active['requirement']}\n"
            f"Stage boundary pending: {boundary}; relation: {relation}; next setting: {self.scene_plan[next_index]['scene_setting']}\n"
            f"Current location and conditions: {getattr(world, 'current_environment', '')}\n"
            f"Recent public events: {list(getattr(world, 'public_events', []) or [])[-6:]}\n"
            f"Steps without new story evidence: {self.stagnant_steps}"
        )
        raw = self._chat_json(label="world_external_tick", prompt=prompt, fallback=fallback)
        changed = raw.get("changed") is True
        event = _without_control_markers(raw.get("public_event")) if changed else ""
        environment = _without_control_markers(raw.get("current_environment")) if changed else ""
        if not event or not environment:
            changed = False
            event = environment = ""
        transition = changed and boundary and raw.get("scene_transition") is True
        if transition and relation == "independent":
            transition = any(word in event.lower() for word in ("travel", "arriv", "reach", "later", "after", "move", "journey", "transition"))
        return {"changed": changed, "public_event": event, "current_environment": environment,
                "scene_transition": transition, "target_beat_id": active["beat_id"] if transition else "",
                "reason": _text(raw.get("reason"))}

    def match_action(self, *, action: Any, observation: dict[str, Any], world: Any) -> dict[str, Any]:
        """Privately compare the character's choice to world affordances and state."""
        fallback = {"alignment": "unverified", "matched_affordance": "", "feasible": False,
                    "reason": "Action feasibility could not be verified.", "condition_effects": [],
                    "permitted_state_changes": []}
        prompt = (
            "You are the hidden world model. Compare the character's OWN action and obstacle strategy with the current physical conditions and private affordances. "
            "Affordances are examples, not an exhaustive menu: allow a novel feasible solution. "
            "Assess whether the action addresses or safely works around conditions that materially obstruct the character's goal. "
            "No event has happened merely because this prompt mentions it; never claim the action succeeded here. "
            "Preserve causal ordering: a terminal act and completion cannot occur before earlier required stages, and a planned action cannot retroactively change the world. "
            "Return JSON {alignment:string, matched_affordance:string, feasible:boolean, reason:string, condition_effects:string[], permitted_state_changes:string[]}.\n"
            f"Hidden contract: {self.contract}\nCovered stages: {sorted(self.covered_beat_ids)}\n"
            f"Current world: {getattr(world, 'current_environment', '')}\n"
            f"Observed conditions: {observation.get('condition_details', [])}\n"
            f"Private affordances: {observation.get('available_affordances', [])}\n"
            f"Character's obstacle assessment: {getattr(action, 'obstacles', [])}\n"
            f"Character's strategy: {getattr(action, 'strategy', '')}\n"
            f"Chosen action: {getattr(action, 'action_type', '')}\nReason: {getattr(action, 'reason', '')}"
        )
        raw = self._chat_json(label="world_action_alignment", prompt=prompt, fallback=fallback)
        action_text = _text(getattr(action, "action_type", "")).lower()
        terminal = self.contract["hard_beats"][-1]
        terminal_text = f"{terminal['title']} {terminal['requirement']}".lower()
        premature_final_stroke = (
            terminal["beat_id"] not in self.unmet_beat_ids[:1]
            and "stroke" in terminal_text
            and "stroke" in action_text
            and ("final" in action_text or "finishing" in action_text)
        )
        if premature_final_stroke:
            return {"alignment": "premature_terminal_action", "matched_affordance": "", "feasible": False,
                    "reason": "The decisive final stroke requires the intervening story stages to occur first.",
                    "condition_effects": [], "permitted_state_changes": []}
        return {"alignment": _text(raw.get("alignment")) or fallback["alignment"],
                "matched_affordance": _text(raw.get("matched_affordance")),
                "feasible": raw.get("feasible") is True,
                "reason": _text(raw.get("reason")) or fallback["reason"],
                "condition_effects": _unique_text(raw.get("condition_effects")),
                "permitted_state_changes": _unique_text(raw.get("permitted_state_changes"))}

    def gate_completion(self, completion: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(completion)
        unmet = self.unmet_beat_ids
        if unmet or self.contradictions:
            result["raw_is_story_complete"] = bool(completion.get("is_story_complete", False))
            result["is_story_complete"] = False
            result["completion_score"] = min(float(completion.get("completion_score", 0.0) or 0.0), 0.95)
            result["completion_reason"] = "Hidden story-control gate kept the simulation open because required story evidence is incomplete or contradictory."
            result["remaining_gap_text"] = "The world still contains unresolved narrative conditions."
            result["next_focus"] = "Allow the world to present a concrete, causally appropriate next opportunity."
        elif self.terminal_ready:
            # The controller's accepted event evidence is authoritative. A
            # free-form completion judge must not reopen a proven final beat.
            result["raw_is_story_complete"] = bool(completion.get("is_story_complete", False))
            result["is_story_complete"] = True
            result["completion_score"] = 1.0
            result["completion_reason"] = "The final required beat has accepted event evidence and all prior beats are covered."
            result["remaining_gap_text"] = ""
            result["next_focus"] = ""
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "contract": self.contract,
            "scene_plan": self.scene_plan,
            "active_scene_index": self.active_scene_index,
            "scene_memory": self.scene_memory,
            "covered_beat_ids": sorted(self.covered_beat_ids),
            "beat_evidence": self.beat_evidence,
            "unmet_beat_ids": self.unmet_beat_ids,
            "contradictions": list(self.contradictions),
            "stagnant_steps": self.stagnant_steps,
            "last_assessment": self.last_assessment,
        }
