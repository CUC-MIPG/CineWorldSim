from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrative.causal_graph import CausalNode, DirectedCausalGraph

from .client import make_client, pick_model
from .retry import call_with_retries
from .schemas import Shot, Storyboard
from narrative.visual_prompt_constraints import build_path_orchestration_prompt_policy

logger = logging.getLogger(__name__)


def _extract_number(text: str) -> int:
    m = re.search(r"(\d+)", str(text or ""))
    return int(m.group(1)) if m else 0


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(slots=True)
class _LLMSequenceScore:
    narrative_causality: float
    visual_transition: float
    rhythmic_pacing: float
    coverage: float
    coherence: float
    pacing: float
    goal_completion: float
    constraint_satisfaction: float
    total_score: float
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "narrative_causality": float(self.narrative_causality),
            "visual_transition": float(self.visual_transition),
            "rhythmic_pacing": float(self.rhythmic_pacing),
            "coverage": float(self.coverage),
            "coherence": float(self.coherence),
            "pacing": float(self.pacing),
            "goal_completion": float(self.goal_completion),
            "constraint_satisfaction": float(self.constraint_satisfaction),
            "total_score": float(self.total_score),
            "summary": self.summary,
        }


class _MCTSNode:
    __slots__ = (
        "ordered_ids",
        "remaining_ids",
        "parent",
        "action_from_parent",
        "prior",
        "children",
        "visits",
        "value_sum",
    )

    def __init__(
        self,
        *,
        ordered_ids: tuple[str, ...],
        remaining_ids: tuple[str, ...],
        parent: _MCTSNode | None,
        action_from_parent: str | None,
        prior: float,
    ) -> None:
        self.ordered_ids = ordered_ids
        self.remaining_ids = remaining_ids
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = float(prior)
        self.children: dict[str, _MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def is_terminal(self) -> bool:
        return len(self.remaining_ids) == 0

    @property
    def q_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / float(self.visits)


class PathOrchestratorMCTSAgent:
    """
    LLM-scored MCTS path planner:
    - prior P(a|s): LLM next-action scoring
    - rollout reward: LLM full-sequence scoring
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
        budget_iters: int = 300,
        topk_actions: int = 8,
        cpuct: float = 1.2,
        rollout_horizon: int = 32,
        rollout_topk: int = 5,
        rollout_epsilon: float = 0.2,
        progressive_widening_base: float = 1.5,
        progressive_widening_alpha: float = 0.55,
        pov_switch_penalty: float = 0.05,
        over_focus_threshold: int = 4,
        over_focus_penalty: float = 0.08,
        reward_weight_narrative_causality: float = 0.4,
        reward_weight_visual_transition: float = 0.3,
        reward_weight_rhythmic_pacing: float = 0.3,
        random_seed: int = 42,
    ) -> None:
        self.client = make_client(scope="PATH_ORCHESTRATOR_MCTS")
        self.model = model or pick_model("gpt-5.5", scope="PATH_ORCHESTRATOR_MCTS")
        self.strict_online = bool(strict_online)
        self.retries = max(1, int(retries))

        self.budget_iters = max(1, int(budget_iters))
        self.topk_actions = max(2, int(topk_actions))
        self.cpuct = float(cpuct)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.rollout_topk = max(1, int(rollout_topk))
        self.rollout_epsilon = _clamp01(rollout_epsilon, default=0.2)
        self.progressive_widening_base = max(1.0, float(progressive_widening_base))
        self.progressive_widening_alpha = max(0.1, min(1.0, float(progressive_widening_alpha)))
        self.pov_switch_penalty = max(0.0, float(pov_switch_penalty))
        self.over_focus_threshold = max(1, int(over_focus_threshold))
        self.over_focus_penalty = max(0.0, float(over_focus_penalty))
        self.reward_weight_narrative_causality = max(0.0, float(reward_weight_narrative_causality))
        self.reward_weight_visual_transition = max(0.0, float(reward_weight_visual_transition))
        self.reward_weight_rhythmic_pacing = max(0.0, float(reward_weight_rhythmic_pacing))
        rw_sum = (
            self.reward_weight_narrative_causality
            + self.reward_weight_visual_transition
            + self.reward_weight_rhythmic_pacing
        )
        if rw_sum <= 0:
            self.reward_weight_narrative_causality = 0.4
            self.reward_weight_visual_transition = 0.3
            self.reward_weight_rhythmic_pacing = 0.3
        else:
            self.reward_weight_narrative_causality /= rw_sum
            self.reward_weight_visual_transition /= rw_sum
            self.reward_weight_rhythmic_pacing /= rw_sum
        self.random = random.Random(int(random_seed))

        self._action_score_cache: dict[str, dict[str, float]] = {}
        self._sequence_score_cache: dict[str, _LLMSequenceScore] = {}
        self._trace_events: list[dict[str, Any]] = []
        self._trace_enabled = True
        self._trace_file_path: Path | None = None
        self._checkpoint_file_path: Path | None = None

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
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
            raise ValueError("Invalid JSON response")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("JSON root is not object")
        return obj

    def _query_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return self._extract_json_object(content)

        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)

    def _trace(self, event: str, **fields: Any) -> None:
        if not self._trace_enabled:
            return
        row = {
            "ts": time.time(),
            "event": event,
            **fields,
        }
        self._trace_events.append(row)
        if self._trace_file_path is not None:
            with self._trace_file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _flush_trace(self, output_dir: Path | None) -> str:
        if output_dir is None:
            return ""
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "mcts_trace.jsonl"
        if not trace_path.exists():
            with trace_path.open("w", encoding="utf-8"):
                pass
        return str(trace_path)

    def _init_run_files(self, output_dir: Path | None) -> None:
        self._trace_file_path = None
        self._checkpoint_file_path = None
        if output_dir is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file_path = output_dir / "mcts_trace.jsonl"
        self._trace_file_path.write_text("", encoding="utf-8")
        self._checkpoint_file_path = output_dir / "mcts_live_checkpoint.json"
        self._checkpoint_file_path.write_text(
            json.dumps(
                {
                    "status": "initialized",
                    "ts": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_live_checkpoint(
        self,
        *,
        storyline: str,
        iter_idx: int,
        best_seq: list[str],
        best_score: _LLMSequenceScore,
        nodes_created: int,
        root_visits: int,
    ) -> None:
        if self._checkpoint_file_path is None:
            return
        payload = {
            "status": "running",
            "ts": time.time(),
            "iter": int(iter_idx),
            "storyline": storyline,
            "best_seq_len": len(best_seq),
            "best_seq": best_seq,
            "best_score": best_score.to_dict(),
            "tree_nodes": int(nodes_created),
            "root_visits": int(root_visits),
            "action_cache_size": len(self._action_score_cache),
            "sequence_cache_size": len(self._sequence_score_cache),
            "trace_file": str(self._trace_file_path) if self._trace_file_path is not None else "",
        }
        self._checkpoint_file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_storyboard_candidates(self, storyboard: Storyboard) -> list[dict[str, Any]]:
        ordered = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
        rows: list[dict[str, Any]] = []
        for idx, shot in enumerate(ordered, start=1):
            rows.append(
                {
                    "idx": idx,
                    "shot_id": shot.shot_id,
                    "scene_id": shot.scene_id,
                    "scene_sequence": int(shot.sequence),
                    "character_names": list(shot.character_names),
                    "text": self._shot_text(storyboard, shot),
                }
            )
        return rows

    def _shot_text(self, storyboard: Storyboard, shot: Shot) -> str:
        scene = storyboard.scenes.get(shot.scene_id)
        scene_desc = scene.plot_description if scene else ""
        scene_loc = scene.location if scene else ""
        scene_time = scene.time_of_day if scene else ""
        elem = self._safe_text(shot.scene_elements)
        return " | ".join(
            [
                f"scene={shot.scene_id}",
                f"shot={shot.shot_id}",
                f"prompt={self._safe_text(shot.prompt)}",
                f"elements={elem}",
                f"scene_plot={scene_desc}",
                f"location={scene_loc}",
                f"time={scene_time}",
                f"characters={','.join(shot.character_names)}",
            ]
        )

    def _normalize_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        id_field: str,
        text_field: str,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        valid_ids: list[str] = []
        data_map: dict[str, dict[str, Any]] = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cid = str(c.get(id_field, "")).strip()
            if not cid or cid in data_map:
                continue
            txt = self._safe_text(c.get(text_field, ""))
            scene_id = str(c.get("scene_id", "")).strip()
            step = int(c.get("step", 0) or 0)
            idx = int(c.get("idx", 0) or 0)
            scene_seq = int(c.get("scene_sequence", 0) or 0)
            role_name = str(c.get("role_name", "")).strip()
            action_type = str(c.get("action_type", "")).strip()
            chars = c.get("character_names", [])
            if not isinstance(chars, list):
                chars = []
            data_map[cid] = {
                "id": cid,
                "idx": idx,
                "scene_id": scene_id,
                "scene_num": _extract_number(scene_id),
                "scene_sequence": scene_seq,
                "step": step,
                "role_name": role_name,
                "action_type": action_type,
                "character_names": [str(x) for x in chars if str(x).strip()],
                "text": txt,
            }
            valid_ids.append(cid)
        return valid_ids, data_map

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _serialize_items(ids: list[str], data_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cid in ids:
            item = data_map[cid]
            rows.append(
                {
                    "id": cid,
                    "idx": int(item.get("idx", 0) or 0),
                    "scene_id": str(item.get("scene_id", "") or ""),
                    "step": int(item.get("step", 0) or 0),
                    "role_name": str(item.get("role_name", "") or ""),
                    "action_type": str(item.get("action_type", "") or ""),
                    "text": str(item.get("text", "") or ""),
                }
            )
        return rows

    def _llm_next_action_scores(
        self,
        *,
        storyline: str,
        prefix_ids: list[str],
        remaining_ids: list[str],
        data_map: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        if not remaining_ids:
            return {}

        cache_key = self._cache_key(
            {
                "type": "next_action",
                "storyline": storyline,
                "prefix": prefix_ids,
                "remaining": remaining_ids,
            }
        )
        if cache_key in self._action_score_cache:
            self._trace(
                "llm_next_action_cache_hit",
                cache_key=cache_key,
                prefix_len=len(prefix_ids),
                remaining_len=len(remaining_ids),
            )
            return self._action_score_cache[cache_key]

        system_prompt = (
            "You are a cinematic path-scoring model for MCTS search. "
            "Score each candidate next node for narrative progression quality. "
            "Return JSON only. "
            "Treat each candidate as an atomic narrative node and do not infer merges across separate candidates."
        )
        user_prompt = (
            f"Storyline objective:\n{storyline}\n\n"
            "Current prefix sequence:\n"
            f"{json.dumps(self._serialize_items(prefix_ids, data_map), ensure_ascii=False)}\n\n"
            "Remaining candidate next nodes:\n"
            f"{json.dumps(self._serialize_items(remaining_ids, data_map), ensure_ascii=False)}\n\n"
            f"{build_path_orchestration_prompt_policy()}\n\n"
            "Return strict JSON object:\n"
            "{\n"
            '  "next_action_scores": [\n'
            '    {"id": "<candidate_id>", "score": 0.0_to_1.0, "reason": "short reason"}\n'
            "  ],\n"
            '  "global_comment": "short summary"\n'
            "}\n"
            "Rules:\n"
            "- Score range must be [0,1].\n"
            "- Cover all candidate ids exactly once if possible.\n"
            "- Higher score means better immediate continuation for storyline, coherence, pacing.\n"
        )

        self._trace(
            "llm_next_action_request",
            cache_key=cache_key,
            storyline=storyline,
            prefix_ids=prefix_ids,
            remaining_ids=remaining_ids,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        t0 = time.time()
        try:
            payload = self._query_json(system_prompt, user_prompt)
            self._trace(
                "llm_next_action_response",
                cache_key=cache_key,
                elapsed_sec=time.time() - t0,
                ok=True,
                payload=payload,
            )
        except Exception:
            if self.strict_online:
                raise
            payload = {}
            self._trace(
                "llm_next_action_response",
                cache_key=cache_key,
                elapsed_sec=time.time() - t0,
                ok=False,
                payload=payload,
            )
        raw_rows = payload.get("next_action_scores", [])
        score_map: dict[str, float] = {}
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("id", "")).strip()
                if not cid or cid not in remaining_ids:
                    continue
                score_map[cid] = _clamp01(row.get("score", 0.0))

        # Fill missing ids with tiny floor, but no rule-based value.
        for cid in remaining_ids:
            if cid not in score_map:
                score_map[cid] = 0.01

        self._action_score_cache[cache_key] = score_map
        self._trace(
            "llm_next_action_score_map",
            cache_key=cache_key,
            score_map=score_map,
        )
        return score_map

    def _llm_sequence_score(
        self,
        *,
        storyline: str,
        ordered_ids: list[str],
        data_map: dict[str, dict[str, Any]],
    ) -> _LLMSequenceScore:
        cache_key = self._cache_key(
            {
                "type": "full_sequence",
                "storyline": storyline,
                "ordered_ids": ordered_ids,
            }
        )
        if cache_key in self._sequence_score_cache:
            self._trace(
                "llm_sequence_cache_hit",
                cache_key=cache_key,
                ordered_len=len(ordered_ids),
            )
            return self._sequence_score_cache[cache_key]

        system_prompt = (
            "You are a film editor-level sequence evaluator for MCTS reward. "
            "Evaluate the full ordered sequence using cinematic editing intent. "
            "Strongly reward: "
            "(1) continuous single-character tracking across multiple progressive beats, "
            "(2) meaningful cross-cutting between conflicting roles, "
            "(3) causal escalation toward conflict and resolution. "
            "Strongly penalize: round-robin surveillance style sequencing "
            "(A1->B1->C1->A2->B2->C2) and arbitrary viewpoint jumps without causal payoff. "
            "Return JSON only. "
            "Treat each candidate node as atomic and preserve candidate boundaries."
        )
        user_prompt = (
            f"Storyline objective:\n{storyline}\n\n"
            "Ordered sequence to evaluate:\n"
            f"{json.dumps(self._serialize_items(ordered_ids, data_map), ensure_ascii=False)}\n\n"
            f"{build_path_orchestration_prompt_policy()}\n\n"
            "Return strict JSON object:\n"
            "{\n"
            '  "narrative_causality": 0.0_to_1.0,\n'
            '  "visual_transition": 0.0_to_1.0,\n'
            '  "rhythmic_pacing": 0.0_to_1.0,\n'
            '  "coverage": 0.0_to_1.0,\n'
            '  "coherence": 0.0_to_1.0,\n'
            '  "goal_completion": 0.0_to_1.0,\n'
            '  "constraint_satisfaction": 0.0_to_1.0,\n'
            '  "summary": "short summary"\n'
            "}\n"
            "Rules:\n"
            "- All scores in [0,1].\n"
            "- narrative_causality: causal progression and story logic.\n"
            "- visual_transition: shot-to-shot continuity and viewpoint smoothness.\n"
            "- rhythmic_pacing: cinematic rhythm, including tracking runs and cross-cutting quality.\n"
            f"- Final total score is computed by system as weighted sum: "
            f"{self.reward_weight_narrative_causality:.3f}*narrative_causality + "
            f"{self.reward_weight_visual_transition:.3f}*visual_transition + "
            f"{self.reward_weight_rhythmic_pacing:.3f}*rhythmic_pacing.\n"
        )

        self._trace(
            "llm_sequence_request",
            cache_key=cache_key,
            storyline=storyline,
            ordered_ids=ordered_ids,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        t0 = time.time()
        try:
            payload = self._query_json(system_prompt, user_prompt)
            self._trace(
                "llm_sequence_response",
                cache_key=cache_key,
                elapsed_sec=time.time() - t0,
                ok=True,
                payload=payload,
            )
        except Exception:
            if self.strict_online:
                raise
            payload = {}
            self._trace(
                "llm_sequence_response",
                cache_key=cache_key,
                elapsed_sec=time.time() - t0,
                ok=False,
                payload=payload,
            )
        narrative_causality = _clamp01(payload.get("narrative_causality", payload.get("coherence", 0.5)), default=0.5)
        visual_transition = _clamp01(payload.get("visual_transition", payload.get("constraint_satisfaction", 0.5)), default=0.5)
        rhythmic_pacing = _clamp01(payload.get("rhythmic_pacing", payload.get("pacing", 0.5)), default=0.5)
        total_score = (
            self.reward_weight_narrative_causality * narrative_causality
            + self.reward_weight_visual_transition * visual_transition
            + self.reward_weight_rhythmic_pacing * rhythmic_pacing
        )
        score = _LLMSequenceScore(
            narrative_causality=narrative_causality,
            visual_transition=visual_transition,
            rhythmic_pacing=rhythmic_pacing,
            coverage=_clamp01(payload.get("coverage", narrative_causality), default=narrative_causality),
            coherence=_clamp01(payload.get("coherence", narrative_causality), default=narrative_causality),
            pacing=_clamp01(payload.get("pacing", rhythmic_pacing), default=rhythmic_pacing),
            goal_completion=_clamp01(payload.get("goal_completion", narrative_causality), default=narrative_causality),
            constraint_satisfaction=_clamp01(payload.get("constraint_satisfaction", visual_transition), default=visual_transition),
            total_score=_clamp01(total_score, default=0.5),
            summary=self._safe_text(payload.get("summary", "")),
        )
        self._sequence_score_cache[cache_key] = score
        self._trace(
            "llm_sequence_score",
            cache_key=cache_key,
            score=score.to_dict(),
        )
        return score

    def _action_rankings(
        self,
        *,
        storyline: str,
        ordered_ids: list[str],
        remaining_ids: list[str],
        data_map: dict[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        score_map = self._llm_next_action_scores(
            storyline=storyline,
            prefix_ids=ordered_ids,
            remaining_ids=remaining_ids,
            data_map=data_map,
        )
        ranked = [(cid, float(score_map.get(cid, 0.01))) for cid in remaining_ids]
        ranked.sort(key=lambda x: (-x[1], int(data_map[x[0]].get("idx", 0) or 0), x[0]))
        self._trace(
            "action_rankings",
            ordered_ids=ordered_ids,
            remaining_ids=remaining_ids,
            ranked_actions=ranked[: min(20, len(ranked))],
        )
        return ranked

    def _build_causal_graph(
        self,
        *,
        valid_ids: list[str],
        data_map: dict[str, dict[str, Any]],
    ) -> DirectedCausalGraph:
        graph = DirectedCausalGraph()
        for idx, cid in enumerate(valid_ids, start=1):
            item = data_map[cid]
            graph.add_node(
                CausalNode(
                    node_id=cid,
                    step=int(item.get("step", 0) or 0),
                    role_name=str(item.get("role_name", "") or ""),
                    action_type=str(item.get("action_type", "") or ""),
                    text=str(item.get("text", "") or ""),
                    source_index=idx,
                )
            )

        # Same-role chain is the core dependency used to break strict round-robin.
        role_to_nodes: dict[str, list[str]] = {}
        for cid in valid_ids:
            role = str(data_map[cid].get("role_name", "") or "").strip()
            if role:
                role_to_nodes.setdefault(role, []).append(cid)
        for role_nodes in role_to_nodes.values():
            role_nodes.sort(key=lambda nid: (int(data_map[nid].get("step", 0) or 0), int(data_map[nid].get("idx", 0) or 0), nid))
            for i in range(1, len(role_nodes)):
                graph.add_edge(role_nodes[i - 1], role_nodes[i])
        return graph

    def _available_actions(
        self,
        *,
        graph: DirectedCausalGraph,
        ordered_ids: list[str],
        remaining_ids: list[str],
    ) -> list[str]:
        available = graph.get_available_candidates(ordered_ids)
        available_set = set(available)
        constrained = [cid for cid in remaining_ids if cid in available_set]
        if constrained:
            return constrained
        # Fallback to avoid deadlock when graph has cycles or missing edges.
        return list(remaining_ids)

    def _pick_expand_action(
        self,
        node: _MCTSNode,
        ranked_actions: list[tuple[str, float]],
    ) -> tuple[str, float] | None:
        if not ranked_actions:
            return None
        widen_cap = int(self.progressive_widening_base * ((node.visits + 1) ** self.progressive_widening_alpha))
        widen_cap = max(1, min(len(ranked_actions), widen_cap))
        candidate_actions = [a for a in ranked_actions[:widen_cap] if a[0] not in node.children]
        self._trace(
            "expand_candidates",
            prefix_ids=list(node.ordered_ids),
            node_visits=node.visits,
            widen_cap=widen_cap,
            candidate_actions=candidate_actions,
        )
        if not candidate_actions:
            return None
        return candidate_actions[0]

    @staticmethod
    def _current_pov_state(ordered_ids: list[str], data_map: dict[str, dict[str, Any]]) -> tuple[str, int]:
        if not ordered_ids:
            return "", 0
        last = ordered_ids[-1]
        pov = str(data_map.get(last, {}).get("role_name", "") or "")
        if not pov:
            return "", 0
        count = 0
        for cid in reversed(ordered_ids):
            role = str(data_map.get(cid, {}).get("role_name", "") or "")
            if role == pov:
                count += 1
            else:
                break
        return pov, count

    def _apply_pov_bias(
        self,
        *,
        ranked_actions: list[tuple[str, float]],
        ordered_ids: list[str],
        data_map: dict[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        if not ranked_actions:
            return ranked_actions
        cur_pov, consec = self._current_pov_state(ordered_ids, data_map)
        if not cur_pov:
            return ranked_actions
        adjusted: list[tuple[str, float]] = []
        for cid, base in ranked_actions:
            role = str(data_map.get(cid, {}).get("role_name", "") or "")
            score = float(base)
            if role and role != cur_pov:
                score -= self.pov_switch_penalty
            elif role and role == cur_pov and consec > self.over_focus_threshold:
                extra = float(consec - self.over_focus_threshold)
                score -= self.over_focus_penalty * extra
            adjusted.append((cid, score))
        adjusted.sort(key=lambda x: (-x[1], int(data_map[x[0]].get("idx", 0) or 0), x[0]))
        self._trace(
            "pov_bias_applied",
            current_pov=cur_pov,
            current_consecutive=consec,
            switch_penalty=self.pov_switch_penalty,
            over_focus_threshold=self.over_focus_threshold,
            over_focus_penalty=self.over_focus_penalty,
            ranked_actions=adjusted[: min(20, len(adjusted))],
        )
        return adjusted

    def _select_child_ucb(
        self,
        node: _MCTSNode,
        ranked_actions: list[tuple[str, float]],
    ) -> _MCTSNode | None:
        if not node.children:
            return None
        prior_map = {cid: sc for cid, sc in ranked_actions}
        best_child: _MCTSNode | None = None
        best_score = -1e9
        parent_visits = max(1, node.visits)
        for cid, child in node.children.items():
            q = child.q_value
            prior = max(0.01, float(child.prior if child.prior > 0 else prior_map.get(cid, 0.01)))
            u = self.cpuct * prior * math.sqrt(float(parent_visits)) / float(1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        self._trace(
            "select_child_ucb",
            prefix_ids=list(node.ordered_ids),
            parent_visits=parent_visits,
            selected_action=best_child.action_from_parent if best_child else "",
            selected_score=best_score,
        )
        return best_child

    def _rollout(
        self,
        *,
        storyline: str,
        ordered_ids: list[str],
        remaining_ids: list[str],
        data_map: dict[str, dict[str, Any]],
        graph: DirectedCausalGraph,
    ) -> tuple[list[str], _LLMSequenceScore]:
        seq = list(ordered_ids)
        rem = list(remaining_ids)
        steps = 0
        self._trace(
            "rollout_start",
            ordered_ids=ordered_ids,
            remaining_ids=remaining_ids,
        )
        while rem and steps < self.rollout_horizon:
            allowed = self._available_actions(
                graph=graph,
                ordered_ids=seq,
                remaining_ids=rem,
            )
            ranked = self._action_rankings(
                storyline=storyline,
                ordered_ids=seq,
                remaining_ids=allowed,
                data_map=data_map,
            )
            ranked = self._apply_pov_bias(
                ranked_actions=ranked,
                ordered_ids=seq,
                data_map=data_map,
            )
            if not ranked:
                break
            if self.random.random() < self.rollout_epsilon:
                k = min(len(ranked), self.rollout_topk)
                pick = ranked[self.random.randint(0, k - 1)][0]
            else:
                pick = ranked[0][0]
            seq.append(pick)
            rem.remove(pick)
            steps += 1
            self._trace(
                "rollout_step",
                step=steps,
                picked=pick,
                seq=seq,
                remaining=rem,
            )

        while rem:
            allowed = self._available_actions(
                graph=graph,
                ordered_ids=seq,
                remaining_ids=rem,
            )
            ranked = self._action_rankings(
                storyline=storyline,
                ordered_ids=seq,
                remaining_ids=allowed,
                data_map=data_map,
            )
            ranked = self._apply_pov_bias(
                ranked_actions=ranked,
                ordered_ids=seq,
                data_map=data_map,
            )
            if not ranked:
                seq.extend(rem)
                break
            pick = ranked[0][0]
            seq.append(pick)
            rem.remove(pick)
            self._trace(
                "rollout_greedy_fill",
                picked=pick,
                seq=seq,
                remaining=rem,
            )

        score = self._llm_sequence_score(
            storyline=storyline,
            ordered_ids=seq,
            data_map=data_map,
        )
        self._trace(
            "rollout_done",
            sequence=seq,
            reward=score.to_dict(),
        )
        return seq, score

    def _mcts_search(
        self,
        *,
        valid_ids: list[str],
        data_map: dict[str, dict[str, Any]],
        storyline: str,
        graph: DirectedCausalGraph,
        output_dir: Path | None = None,
    ) -> tuple[list[str], _LLMSequenceScore, dict[str, Any]]:
        root = _MCTSNode(
            ordered_ids=tuple(),
            remaining_ids=tuple(valid_ids),
            parent=None,
            action_from_parent=None,
            prior=1.0,
        )

        topo_ids = [cid for cid in graph.topological_order() if cid in set(valid_ids)]
        best_seq = list(topo_ids or valid_ids)
        best_score = self._llm_sequence_score(
            storyline=storyline,
            ordered_ids=best_seq,
            data_map=data_map,
        )
        nodes_created = 1

        for iter_idx in range(self.budget_iters):
            self._trace("mcts_iter_start", iter=iter_idx + 1)
            node = root
            path: list[_MCTSNode] = [node]

            while not node.is_terminal:
                remaining = list(node.remaining_ids)
                allowed = self._available_actions(
                    graph=graph,
                    ordered_ids=list(node.ordered_ids),
                    remaining_ids=remaining,
                )
                ranked = self._action_rankings(
                    storyline=storyline,
                    ordered_ids=list(node.ordered_ids),
                    remaining_ids=allowed,
                    data_map=data_map,
                )
                ranked = self._apply_pov_bias(
                    ranked_actions=ranked,
                    ordered_ids=list(node.ordered_ids),
                    data_map=data_map,
                )
                if not ranked:
                    break

                expand = self._pick_expand_action(node, ranked)
                if expand is not None:
                    action_id, prior = expand
                    new_order = tuple(list(node.ordered_ids) + [action_id])
                    new_remaining = tuple(x for x in node.remaining_ids if x != action_id)
                    child = _MCTSNode(
                        ordered_ids=new_order,
                        remaining_ids=new_remaining,
                        parent=node,
                        action_from_parent=action_id,
                        prior=prior,
                    )
                    node.children[action_id] = child
                    node = child
                    path.append(node)
                    nodes_created += 1
                    self._trace(
                        "mcts_expand",
                        iter=iter_idx + 1,
                        action_id=action_id,
                        prior=prior,
                        prefix_ids=list(node.ordered_ids),
                    )
                    break

                selected = self._select_child_ucb(node, ranked)
                if selected is None:
                    self._trace("mcts_select_none", iter=iter_idx + 1, prefix_ids=list(node.ordered_ids))
                    break
                node = selected
                path.append(node)
                self._trace(
                    "mcts_select",
                    iter=iter_idx + 1,
                    selected_action=node.action_from_parent or "",
                    prefix_ids=list(node.ordered_ids),
                )

            rollout_seq, rollout_score = self._rollout(
                storyline=storyline,
                ordered_ids=list(node.ordered_ids),
                remaining_ids=list(node.remaining_ids),
                data_map=data_map,
                graph=graph,
            )
            reward = float(rollout_score.total_score)
            self._trace(
                "mcts_rollout_result",
                iter=iter_idx + 1,
                reward=reward,
                rollout_sequence=rollout_seq,
            )
            for n in reversed(path):
                n.visits += 1
                n.value_sum += reward
                self._trace(
                    "mcts_backprop",
                    iter=iter_idx + 1,
                    node_prefix=list(n.ordered_ids),
                    visits=n.visits,
                    value_sum=n.value_sum,
                    q_value=n.q_value,
                )

            if rollout_score.total_score > best_score.total_score:
                best_seq = rollout_seq
                best_score = rollout_score
                self._trace(
                    "mcts_best_update",
                    iter=iter_idx + 1,
                    best_score=best_score.total_score,
                    best_seq=best_seq,
                )
            self._write_live_checkpoint(
                storyline=storyline,
                iter_idx=iter_idx + 1,
                best_seq=best_seq,
                best_score=best_score,
                nodes_created=nodes_created,
                root_visits=root.visits,
            )

        stats = {
            "iterations": int(self.budget_iters),
            "tree_nodes": int(nodes_created),
            "root_visits": int(root.visits),
            "best_score": float(best_score.total_score),
            "action_cache_size": len(self._action_score_cache),
            "sequence_cache_size": len(self._sequence_score_cache),
        }
        return best_seq, best_score, stats

    @staticmethod
    def _normalize_order(raw_order: list[str], valid_ids: list[str]) -> list[str]:
        valid = set(valid_ids)
        out: list[str] = []
        seen: set[str] = set()
        for x in raw_order:
            sid = str(x)
            if sid in valid and sid not in seen:
                out.append(sid)
                seen.add(sid)
        for sid in valid_ids:
            if sid not in seen:
                out.append(sid)
        return out

    def arrange_candidate_path(
        self,
        *,
        storyline: str,
        candidates: list[dict[str, Any]],
        id_field: str = "node_id",
        text_field: str = "text",
        output_list_field: str = "ordered_node_ids",
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        valid_ids, data_map = self._normalize_candidates(
            candidates=candidates,
            id_field=id_field,
            text_field=text_field,
        )
        if not valid_ids:
            raise RuntimeError("No valid candidates found for MCTS path orchestrator.")
        self._trace_events = []
        self._init_run_files(output_dir)
        self._trace(
            "run_start",
            storyline=storyline,
            model=self.model,
            candidate_count=len(valid_ids),
            config={
                "strict_online": self.strict_online,
                "retries": self.retries,
                "budget_iters": self.budget_iters,
                "topk_actions": self.topk_actions,
                "cpuct": self.cpuct,
                "rollout_horizon": self.rollout_horizon,
                "rollout_topk": self.rollout_topk,
                "rollout_epsilon": self.rollout_epsilon,
                "progressive_widening_base": self.progressive_widening_base,
                "progressive_widening_alpha": self.progressive_widening_alpha,
            },
        )
        graph = self._build_causal_graph(valid_ids=valid_ids, data_map=data_map)
        self._trace(
            "causal_graph_ready",
            node_count=len(graph.nodes),
            edge_count=int(sum(len(v) for v in graph.successors.values())),
        )

        ordered_raw, best_score, stats = self._mcts_search(
            valid_ids=valid_ids,
            data_map=data_map,
            storyline=storyline,
            graph=graph,
            output_dir=output_dir,
        )
        ordered_ids = self._normalize_order(ordered_raw, valid_ids)

        summary = {
            "agent": "PathOrchestratorMCTSAgent",
            "mode": "mcts_llm_factored_scoring",
            "model": self.model,
            "storyline": storyline,
            "candidate_count": len(valid_ids),
            output_list_field: ordered_ids,
            "storyline_fit_summary": best_score.summary,
            "transition_notes": (
                "MCTS with LLM next-action priors and factored LLM full-sequence reward "
                "(narrative_causality, visual_transition, rhythmic_pacing)."
            ),
            "quality_checks": {
                "narrative_causality": f"{best_score.narrative_causality:.3f}",
                "visual_transition": f"{best_score.visual_transition:.3f}",
                "rhythmic_pacing": f"{best_score.rhythmic_pacing:.3f}",
                "coverage": f"{best_score.coverage:.3f}",
                "coherence": f"{best_score.coherence:.3f}",
                "pacing": f"{best_score.pacing:.3f}",
                "goal_completion": f"{best_score.goal_completion:.3f}",
                "constraint_satisfaction": f"{best_score.constraint_satisfaction:.3f}",
            },
            "value_breakdown": best_score.to_dict(),
            "search_stats": stats,
            "config": {
                "strict_online": self.strict_online,
                "retries": self.retries,
                "budget_iters": self.budget_iters,
                "topk_actions": self.topk_actions,
                "cpuct": self.cpuct,
                "rollout_horizon": self.rollout_horizon,
                "rollout_topk": self.rollout_topk,
                "rollout_epsilon": self.rollout_epsilon,
                "progressive_widening_base": self.progressive_widening_base,
                "progressive_widening_alpha": self.progressive_widening_alpha,
                "pov_switch_penalty": self.pov_switch_penalty,
                "over_focus_threshold": self.over_focus_threshold,
                "over_focus_penalty": self.over_focus_penalty,
                "reward_weight_narrative_causality": self.reward_weight_narrative_causality,
                "reward_weight_visual_transition": self.reward_weight_visual_transition,
                "reward_weight_rhythmic_pacing": self.reward_weight_rhythmic_pacing,
            },
        }
        trace_path = self._flush_trace(output_dir)
        summary["trace_file"] = trace_path

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "path_plan_mcts.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if self._checkpoint_file_path is not None:
                self._checkpoint_file_path.write_text(
                    json.dumps(
                        {
                            "status": "done",
                            "ts": time.time(),
                            "summary_file": str(output_dir / "path_plan_mcts.json"),
                            "trace_file": trace_path,
                            "candidate_count": len(valid_ids),
                            "best_score": best_score.to_dict(),
                            "ordered_ids": ordered_ids,
                            "search_stats": stats,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        logger.info(
            "path_orchestrator_mcts status=done model=%s count=%d best_score=%.3f first=%s last=%s",
            self.model,
            len(ordered_ids),
            best_score.total_score,
            ordered_ids[0] if ordered_ids else "",
            ordered_ids[-1] if ordered_ids else "",
        )
        return summary

    def arrange_story_path(
        self,
        *,
        storyline: str,
        storyboard: Storyboard,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        candidates = self._build_storyboard_candidates(storyboard)
        summary = self.arrange_candidate_path(
            storyline=storyline,
            candidates=candidates,
            id_field="shot_id",
            text_field="text",
            output_list_field="ordered_shot_ids",
            output_dir=output_dir,
        )
        ordered_ids = [str(x) for x in summary.get("ordered_shot_ids", [])]
        storyboard.path_sequence = list(ordered_ids)
        shot_map = storyboard.shots
        for i, sid in enumerate(ordered_ids, start=1):
            if sid in shot_map:
                shot_map[sid].path_sequence = i
        next_rank = len(ordered_ids) + 1
        for shot in storyboard.ordered_shots_scene_first(unique_by_scene_sequence=False):
            if int(getattr(shot, "path_sequence", 0) or 0) <= 0:
                shot.path_sequence = next_rank
                next_rank += 1
        storyboard.path_metadata = dict(summary)
        return summary
