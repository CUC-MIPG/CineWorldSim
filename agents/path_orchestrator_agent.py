from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .client import make_client, pick_model
from .retry import call_with_retries
from .schemas import Shot, Storyboard
from narrative.visual_prompt_constraints import build_path_orchestration_prompt_policy

logger = logging.getLogger(__name__)


class PathOrchestratorAgent:
    """
    Path Orchestrator:
    - Collect all shot text content.
    - Re-order shots to best satisfy user storyline objective.
    - Write a global `path_sequence` for final video assembly.
    """

    def __init__(
        self,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ):
        self.client = make_client(scope="PATH_ORCHESTRATOR")
        self.model = model or pick_model("gpt-5.5", scope="PATH_ORCHESTRATOR")
        self.strict_online = strict_online
        self.retries = retries

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

    @staticmethod
    def _norm_tokens(text: str) -> set[str]:
        toks = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", (text or "").lower())
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "to",
            "of",
            "and",
            "in",
            "on",
            "for",
            "with",
            "this",
            "that",
            "it",
        }
        return {t for t in toks if len(t) > 1 and t not in stop}

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
            ]
        )

    def _build_candidates(self, storyboard: Storyboard) -> list[dict[str, Any]]:
        ordered = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
        rows: list[dict[str, Any]] = []
        for idx, shot in enumerate(ordered, start=1):
            rows.append(
                {
                    "idx": idx,
                    "shot_id": shot.shot_id,
                    "scene_id": shot.scene_id,
                    "scene_sequence": int(shot.sequence),
                    "text": self._shot_text(storyboard, shot),
                }
            )
        return rows

    def _fallback_sequence(self, storyline: str, storyboard: Storyboard, candidates: list[dict[str, Any]]) -> list[str]:
        story_tokens = self._norm_tokens(storyline)
        if not story_tokens:
            return [str(c["shot_id"]) for c in candidates]
        scored: list[tuple[float, int, str]] = []
        for c in candidates:
            sid = str(c["shot_id"])
            text_tokens = self._norm_tokens(str(c.get("text", "")))
            overlap = len(story_tokens & text_tokens)
            scored.append((float(overlap), int(c.get("idx", 0)), sid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [x[2] for x in scored]

    def _fallback_sequence_from_candidates(
        self,
        *,
        storyline: str,
        candidates: list[dict[str, Any]],
        id_field: str,
        text_field: str = "text",
        idx_field: str = "idx",
    ) -> list[str]:
        story_tokens = self._norm_tokens(storyline)
        if not story_tokens:
            return [str(c.get(id_field, "")) for c in candidates if str(c.get(id_field, "")).strip()]
        scored: list[tuple[float, int, str]] = []
        for c in candidates:
            cid = str(c.get(id_field, ""))
            if not cid:
                continue
            text_tokens = self._norm_tokens(str(c.get(text_field, "")))
            overlap = len(story_tokens & text_tokens)
            scored.append((float(overlap), int(c.get(idx_field, 0) or 0), cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [x[2] for x in scored]

    def _query_model(self, storyline: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        system_prompt = (
            "You are PathOrchestrator, a cinematic sequencing planner. "
            "Given user storyline objective and all candidate shot texts, "
            "return the best full shot order as JSON only. "
            "Treat every candidate as an already-formed narrative node; do not merge or split candidates during ordering."
        )
        user_prompt = (
            f"User storyline objective:\n{storyline}\n\n"
            "Candidate shots (JSON array):\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"{build_path_orchestration_prompt_policy()}\n\n"
            "Return JSON fields:\n"
            "- ordered_shot_ids: string[] (must be permutation of candidate shot_ids, no duplicates)\n"
            "- storyline_fit_summary: string\n"
            "- transition_notes: string\n"
            "- quality_checks: {coherence: string, coverage: string, pacing: string}\n"
        )

        def _call() -> dict[str, Any]:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return self._extract_json_object(content)

        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)

    def _query_model_generic(
        self,
        *,
        storyline: str,
        candidates: list[dict[str, Any]],
        id_field: str,
        output_list_field: str,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are PathOrchestrator, a cinematic sequencing planner. "
            "Given user storyline objective and all candidate nodes, "
            "return the best full order as JSON only. "
            "Treat each candidate node as atomic; preserve its boundary exactly as provided."
        )
        user_prompt = (
            f"User storyline objective:\n{storyline}\n\n"
            f"Candidates (JSON array, unique id field '{id_field}'):\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"{build_path_orchestration_prompt_policy()}\n\n"
            "Return JSON fields:\n"
            f"- {output_list_field}: string[] (must be permutation of candidate ids, no duplicates)\n"
            "- storyline_fit_summary: string\n"
            "- transition_notes: string\n"
            "- quality_checks: {coherence: string, coverage: string, pacing: string}\n"
        )

        def _call() -> dict[str, Any]:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return self._extract_json_object(content)

        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)

    @staticmethod
    def _normalize_order(raw_order: Any, valid_ids: list[str]) -> list[str]:
        valid = set(valid_ids)
        out: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_order, list):
            for x in raw_order:
                sid = str(x)
                if sid in valid and sid not in seen:
                    seen.add(sid)
                    out.append(sid)
        for sid in valid_ids:
            if sid not in seen:
                out.append(sid)
        return out

    def arrange_story_path(
        self,
        *,
        storyline: str,
        storyboard: Storyboard,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        candidates = self._build_candidates(storyboard)
        valid_ids = [str(c["shot_id"]) for c in candidates]

        payload: dict[str, Any] | None = None
        if valid_ids:
            try:
                payload = self._query_model(storyline=storyline, candidates=candidates)
            except Exception:
                if self.strict_online:
                    raise
                payload = None

        if payload is None:
            ordered_ids = self._fallback_sequence(storyline, storyboard, candidates)
            payload = {
                "ordered_shot_ids": ordered_ids,
                "storyline_fit_summary": "fallback lexical ranking by storyline overlap.",
                "transition_notes": "fallback sequencing; model unavailable.",
                "quality_checks": {
                    "coherence": "fallback",
                    "coverage": "fallback",
                    "pacing": "fallback",
                },
                "mode": "fallback",
            }
        else:
            payload["mode"] = "llm"

        ordered_ids = self._normalize_order(payload.get("ordered_shot_ids", []), valid_ids)
        storyboard.path_sequence = list(ordered_ids)

        shot_map = storyboard.shots
        for i, sid in enumerate(ordered_ids, start=1):
            if sid in shot_map:
                shot_map[sid].path_sequence = i

        # Ensure every shot has path_sequence.
        next_rank = len(ordered_ids) + 1
        for shot in storyboard.ordered_shots_scene_first(unique_by_scene_sequence=False):
            if int(getattr(shot, "path_sequence", 0) or 0) <= 0:
                shot.path_sequence = next_rank
                next_rank += 1

        summary = {
            "agent": "PathOrchestratorAgent",
            "mode": payload.get("mode", "unknown"),
            "storyline": storyline,
            "shot_count": len(valid_ids),
            "ordered_shot_ids": ordered_ids,
            "storyline_fit_summary": self._safe_text(payload.get("storyline_fit_summary", "")),
            "transition_notes": self._safe_text(payload.get("transition_notes", "")),
            "quality_checks": payload.get("quality_checks", {}),
        }
        storyboard.path_metadata = dict(summary)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "path_plan.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        logger.info(
            "path_orchestrator status=done mode=%s shot_count=%d first=%s last=%s",
            summary.get("mode", ""),
            len(ordered_ids),
            ordered_ids[0] if ordered_ids else "",
            ordered_ids[-1] if ordered_ids else "",
        )
        return summary

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
        valid_ids = [str(c.get(id_field, "")) for c in candidates if str(c.get(id_field, "")).strip()]
        payload: dict[str, Any] | None = None
        if valid_ids:
            try:
                payload = self._query_model_generic(
                    storyline=storyline,
                    candidates=candidates,
                    id_field=id_field,
                    output_list_field=output_list_field,
                )
            except Exception:
                if self.strict_online:
                    raise
                payload = None

        if payload is None:
            ordered_ids = self._fallback_sequence_from_candidates(
                storyline=storyline,
                candidates=candidates,
                id_field=id_field,
                text_field=text_field,
            )
            payload = {
                output_list_field: ordered_ids,
                "storyline_fit_summary": "fallback lexical ranking by storyline overlap.",
                "transition_notes": "fallback sequencing; model unavailable.",
                "quality_checks": {
                    "coherence": "fallback",
                    "coverage": "fallback",
                    "pacing": "fallback",
                },
                "mode": "fallback",
            }
        else:
            payload["mode"] = "llm"

        ordered_ids = self._normalize_order(payload.get(output_list_field, []), valid_ids)
        summary = {
            "agent": "PathOrchestratorAgent",
            "mode": payload.get("mode", "unknown"),
            "storyline": storyline,
            "candidate_count": len(valid_ids),
            output_list_field: ordered_ids,
            "storyline_fit_summary": self._safe_text(payload.get("storyline_fit_summary", "")),
            "transition_notes": self._safe_text(payload.get("transition_notes", "")),
            "quality_checks": payload.get("quality_checks", {}),
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "path_plan.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        logger.info(
            "path_orchestrator candidates status=done mode=%s count=%d first=%s last=%s",
            summary.get("mode", ""),
            len(ordered_ids),
            ordered_ids[0] if ordered_ids else "",
            ordered_ids[-1] if ordered_ids else "",
        )
        return summary
