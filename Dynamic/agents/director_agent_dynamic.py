from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.client import make_client, pick_model
from agents.director_agent import DirectorAgent
from agents.prompts import load_prompt_library_query_dict
from agents.retry import call_with_retries
from agents.schemas import PlotPoint, Storyboard

from .co_director_prompts import (
    CREATIVE_BRIEF_SYSTEM,
    CREATIVE_BRIEF_USER,
    STORYLINE_GEN_SYSTEM,
    STORYLINE_GEN_USER,
    STORYLINE_EVAL_SYSTEM,
    STORYLINE_EVAL_USER,
    STORYLINE_REFINE_SYSTEM,
    STORYLINE_REFINE_USER,
)

logger = logging.getLogger(__name__)


class DirectorAgentDynamic:
    """Co-Director inspired pre-production wrapper on top of CineWorldSim DirectorAgent."""

    def __init__(
        self,
        prompt_library: Path,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ) -> None:
        self.prompt_library = prompt_library
        self.query_dict = load_prompt_library_query_dict(prompt_library)
        self.client = make_client(scope="DIRECTOR")
        self.model = model or pick_model("gpt-5.5", scope="DIRECTOR")
        self.strict_online = strict_online
        self.retries = retries
        self.base_director = DirectorAgent(
            prompt_library=prompt_library,
            model=self.model,
            strict_online=strict_online,
            retries=retries,
        )

    @staticmethod
    def _safe_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError("No valid JSON object found in model response")

    def _call_llm(self, *, system: str, user: str, json_mode: bool = False) -> str:
        def _run() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            return content if isinstance(content, str) else str(content)

        return call_with_retries(_run, attempts=self.retries, delay_sec=1.0)

    def _select_creative_configuration(self, storyline: str) -> dict[str, str]:
        text = (storyline or "").lower()
        strategy = "informational"
        if any(k in text for k in ["对比", "vs", "compare", "竞争"]):
            strategy = "comparative"
        elif any(k in text for k in ["情感", "温暖", "梦想", "lifestyle", "emotion"]):
            strategy = "transformational"

        narrative_mode = "narrative_drama"
        if any(k in text for k in ["教程", "分析", "步骤", "technical", "how to"]):
            narrative_mode = "analytical"
        elif any(k in text for k in ["片段", "氛围", "vignette"]):
            narrative_mode = "vignette"

        aesthetic_archetype = "cinematic_premium"
        if any(k in text for k in ["快节奏", "运动", "极限", "fast", "kinetic"]):
            aesthetic_archetype = "kinetic_grit"
        elif any(k in text for k in ["极简", "clean", "minimal"]):
            aesthetic_archetype = "minimalist_focus"
        elif any(k in text for k in ["明亮", "活力", "energy"]):
            aesthetic_archetype = "clarity_energy"

        return {
            "creative_strategy": strategy,
            "narrative_mode": narrative_mode,
            "aesthetic_archetype": aesthetic_archetype,
        }

    @staticmethod
    def _synthesize_creative_direction(cfg: dict[str, str]) -> dict[str, str]:
        strategy_map = {
            "informational": "Focus on utility, capability proof, and practical value clarity.",
            "transformational": "Focus on emotional transformation, aspiration, and identity resonance.",
            "comparative": "Highlight relative advantages with clear contrast and decisive outcomes.",
        }
        mode_map = {
            "analytical": "Use argument-driven scene flow with explicit evidence and progression.",
            "vignette": "Use atmospheric slices of life and soft transitions around a central feeling.",
            "narrative_drama": "Use setup-conflict-resolution arc with clear character motivation.",
        }
        aa_map = {
            "clarity_energy": "Bright lighting, clean framing, dynamic but readable motion.",
            "cinematic_premium": "Layered lighting, dramatic depth, polished cinematic composition.",
            "minimalist_focus": "Simple backgrounds, focused subject separation, subtle controlled motion.",
            "kinetic_grit": "Low-key, handheld/FPV energy, aggressive motion emphasis.",
        }
        return {
            "storyline_instruction": strategy_map.get(cfg.get("creative_strategy", ""), "Keep storyline coherent and actionable."),
            "shot_instruction": mode_map.get(cfg.get("narrative_mode", ""), "Preserve narrative continuity scene by scene."),
            "visual_instruction": aa_map.get(cfg.get("aesthetic_archetype", ""), "Maintain stable visual identity and style continuity."),
        }

    def _build_structured_constraints(self, storyline: str) -> dict[str, Any]:
        return {
            "campaign_goal": "narrative video generation",
            "user_storyline": storyline,
            "target_duration_hint_sec": 12,
        }

    def _annotate_reference_visuals(self, storyboard: Storyboard) -> dict[str, dict[str, str]]:
        annotated: dict[str, dict[str, str]] = {}
        for scene in storyboard.scenes.values():
            for p in scene.reference_images:
                name = Path(p).name
                if not Path(p).exists():
                    continue
                role = "scene"
                lower = name.lower()
                if "logo" in lower:
                    role = "logo"
                elif "product" in lower:
                    role = "product"
                annotated[name] = {
                    "file_name": name,
                    "caption": f"Scene reference for {scene.scene_id}",
                    "semantic_role": role,
                }

        for cname, c in storyboard.screenplay.characters.items():
            all_refs = []
            if c.reference_image:
                all_refs.append(c.reference_image)
            all_refs.extend(c.reference_images)
            for p in all_refs:
                if not p or not Path(p).exists():
                    continue
                name = Path(p).name
                annotated[name] = {
                    "file_name": name,
                    "caption": f"Character reference for {cname}",
                    "semantic_role": "character",
                }
        return annotated

    def _generate_creative_brief(
        self,
        *,
        structured_constraints: dict[str, Any],
        annotated_reference_visuals: dict[str, Any],
        creative_configuration: dict[str, str],
        creative_direction: dict[str, str],
        user_storyline: str,
    ) -> str:
        user = CREATIVE_BRIEF_USER.format(
            structured_constraints=json.dumps(structured_constraints, ensure_ascii=False, indent=2),
            annotated_reference_visuals=json.dumps(annotated_reference_visuals, ensure_ascii=False, indent=2),
            creative_configuration=json.dumps(creative_configuration, ensure_ascii=False, indent=2),
            creative_direction=json.dumps(creative_direction, ensure_ascii=False, indent=2),
            user_storyline=user_storyline,
        )
        try:
            return self._call_llm(system=CREATIVE_BRIEF_SYSTEM, user=user, json_mode=False).strip()
        except Exception:
            if self.strict_online:
                raise
            return f"Creative brief fallback: {user_storyline}"

    def _generate_storyline_json(self, *, creative_brief: str, creative_direction: dict[str, str], user_storyline: str) -> dict[str, Any]:
        user = STORYLINE_GEN_USER.format(
            creative_brief=creative_brief,
            creative_direction=json.dumps(creative_direction, ensure_ascii=False, indent=2),
            user_storyline=user_storyline,
        )
        content = self._call_llm(system=STORYLINE_GEN_SYSTEM, user=user, json_mode=True)
        return self._extract_json_object(content)

    def _evaluate_storyline_json(self, *, storyline_json: dict[str, Any], user_storyline: str) -> dict[str, Any]:
        user = STORYLINE_EVAL_USER.format(
            user_storyline=user_storyline,
            storyline_json=json.dumps(storyline_json, ensure_ascii=False, indent=2),
        )
        content = self._call_llm(system=STORYLINE_EVAL_SYSTEM, user=user, json_mode=True)
        return self._extract_json_object(content)

    def _refine_storyline_json(
        self,
        *,
        creative_brief: str,
        creative_direction: dict[str, str],
        storyline_json: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        user = STORYLINE_REFINE_USER.format(
            creative_brief=creative_brief,
            creative_direction=json.dumps(creative_direction, ensure_ascii=False, indent=2),
            storyline_json=json.dumps(storyline_json, ensure_ascii=False, indent=2),
            evaluator_feedback=self._safe_text(evaluation.get("feedback", "")),
            actionable_feedback=self._safe_text(evaluation.get("actionable_feedback", "")),
        )
        content = self._call_llm(system=STORYLINE_REFINE_SYSTEM, user=user, json_mode=True)
        return self._extract_json_object(content)

    @staticmethod
    def _storyline_to_scene_rows(storyline_json: dict[str, Any]) -> list[dict[str, str]]:
        rows = storyline_json.get("scenes", []) if isinstance(storyline_json, dict) else []
        out: list[dict[str, str]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                topic = str(row.get("topic", "")).strip()
                action = str(row.get("action", "")).strip()
                if topic or action:
                    out.append({"topic": topic or "Scene", "action": action or topic})
        return out

    def _apply_storyline_to_storyboard(self, storyboard: Storyboard, storyline_json: dict[str, Any]) -> Storyboard:
        rows = self._storyline_to_scene_rows(storyline_json)
        if not rows:
            return storyboard
        scene_ids = sorted(storyboard.scenes.keys())
        for idx, sid in enumerate(scene_ids):
            if idx >= len(rows):
                break
            row = rows[idx]
            topic = row.get("topic", "").strip()
            action = row.get("action", "").strip()
            scene = storyboard.scenes[sid]
            scene.plot_description = action or scene.plot_description
            if topic and (not scene.location or scene.location.lower().startswith("scene")):
                scene.location = topic

        storyboard.screenplay.plot_points = []
        for row in rows:
            storyboard.screenplay.plot_points.append(
                PlotPoint(description=row.get("action", ""), impact=row.get("topic", ""))
            )
        return storyboard

    @staticmethod
    def _verify_storyboard_assets(storyboard: Storyboard) -> Storyboard:
        for scene in storyboard.scenes.values():
            scene.reference_images = [p for p in scene.reference_images if p and Path(p).exists()]
        for c in storyboard.screenplay.characters.values():
            c.reference_images = [p for p in c.reference_images if p and Path(p).exists()]
            if c.reference_image and not Path(c.reference_image).exists():
                c.reference_image = c.reference_images[0] if c.reference_images else None
        return storyboard

    def run_preproduction(
        self,
        *,
        storyline: str,
        reference_assets_json: Path | None = None,
        max_storyline_refine_attempts: int = 2,
        storyline_score_threshold: int = 85,
    ) -> tuple[Storyboard, dict[str, Any]]:
        logger.info("dynamic_preproduction status=start")
        storyboard = self.base_director.build_storyboard(storyline)
        if reference_assets_json and reference_assets_json.exists():
            storyboard = self.base_director.attach_reference_assets_json(
                storyboard,
                reference_assets_json=reference_assets_json,
            )

        structured_constraints = self._build_structured_constraints(storyline)
        annotated_reference_visuals = self._annotate_reference_visuals(storyboard)
        creative_configuration = self._select_creative_configuration(storyline)
        creative_direction = self._synthesize_creative_direction(creative_configuration)

        creative_brief = self._generate_creative_brief(
            structured_constraints=structured_constraints,
            annotated_reference_visuals=annotated_reference_visuals,
            creative_configuration=creative_configuration,
            creative_direction=creative_direction,
            user_storyline=storyline,
        )

        refinement_history: list[dict[str, Any]] = []
        try:
            current_storyline_json = self._generate_storyline_json(
                creative_brief=creative_brief,
                creative_direction=creative_direction,
                user_storyline=storyline,
            )
            for attempt in range(1, max(1, int(max_storyline_refine_attempts)) + 1):
                evaluation = self._evaluate_storyline_json(
                    storyline_json=current_storyline_json,
                    user_storyline=storyline,
                )
                score = int(evaluation.get("score", 0) or 0)
                refinement_history.append(
                    {
                        "attempt": attempt,
                        "score": score,
                        "evaluation": evaluation,
                        "storyline": current_storyline_json,
                    }
                )
                if score >= int(storyline_score_threshold):
                    break
                if attempt < int(max_storyline_refine_attempts):
                    current_storyline_json = self._refine_storyline_json(
                        creative_brief=creative_brief,
                        creative_direction=creative_direction,
                        storyline_json=current_storyline_json,
                        evaluation=evaluation,
                    )
        except Exception:
            if self.strict_online:
                raise
            current_storyline_json = {"scenes": [{"topic": s.scene_id, "action": s.plot_description} for s in storyboard.scenes.values()]}

        storyboard = self._apply_storyline_to_storyboard(storyboard, current_storyline_json)
        storyboard = self._verify_storyboard_assets(storyboard)

        bundle = {
            "structured_constraints": structured_constraints,
            "annotated_reference_visuals": annotated_reference_visuals,
            "creative_configuration": creative_configuration,
            "creative_direction": creative_direction,
            "creative_brief": creative_brief,
            "storyline": current_storyline_json,
            "refinement_history": refinement_history,
        }
        logger.info("dynamic_preproduction status=done scenes=%d", len(storyboard.scenes))
        return storyboard, bundle
