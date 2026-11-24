from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .client import make_client, pick_model
from .retry import call_with_retries
from .skills.resolver import SkillResolver
from .skills.runtime import build_skill_resolver_from_env, skill_name_override
from .prompts import load_skill_prompt_template_text
from .schemas import Scene, Shot, Storyboard

logger = logging.getLogger(__name__)


class StorySupervisorAgent:
    """
    剧情监制智能体：
    - 基于剧情生成片段级后期规划（旁白、对话、音乐描述、时长等）
    - 输出结构可直接接入 PostProductionAgent
    """

    def __init__(
        self,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
        shot_duration_min_sec: float | None = None,
        shot_duration_max_sec: float | None = None,
        scene_duration_per_shot_sec: float | None = None,
        duration_tolerance_sec: float | None = None,
        skill_resolver: SkillResolver | None = None,
    ):
        self.client = make_client(scope="SUPERVISOR")
        self.model = model or pick_model("gpt-5.5", scope="SUPERVISOR")
        self.strict_online = strict_online
        self.retries = retries
        self.skill_resolver = skill_resolver or build_skill_resolver_from_env()
        self.shot_duration_min_sec = max(
            0.2,
            self._env_float("SUPERVISOR_SHOT_MIN_SEC", shot_duration_min_sec, 1.5),
        )
        self.shot_duration_max_sec = max(
            self.shot_duration_min_sec,
            self._env_float("SUPERVISOR_SHOT_MAX_SEC", shot_duration_max_sec, 8.0),
        )
        self.scene_duration_per_shot_sec = max(
            0.2,
            self._env_float("SUPERVISOR_SCENE_SEC_PER_SHOT", scene_duration_per_shot_sec, 3.2),
        )
        self.duration_tolerance_sec = max(
            0.1,
            self._env_float("SUPERVISOR_DURATION_TOLERANCE_SEC", duration_tolerance_sec, 0.5),
        )

    @staticmethod
    def _env_float(name: str, override: float | None, default: float) -> float:
        if override is not None:
            try:
                return float(override)
            except Exception:
                return default
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _safe_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)

        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Invalid JSON response: {t[:240]}")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("JSON root must be object")
        return obj

    def _estimate_scene_duration(self, shots: list[Shot]) -> float:
        if not shots:
            return 4.0
        n = len(shots)
        low = n * self.shot_duration_min_sec
        high = n * self.shot_duration_max_sec
        base = n * self.scene_duration_per_shot_sec
        return round(self._clamp(base, low, high), 2)

    def _duration_constraints(self, shots: list[Shot]) -> dict[str, float]:
        return {
            "min_sec": round(self.shot_duration_min_sec, 2),
            "max_sec": round(self.shot_duration_max_sec, 2),
            "target_scene_duration_sec": round(self._estimate_scene_duration(shots), 2),
            "tolerance_sec": round(self.duration_tolerance_sec, 2),
        }

    def _allocate_durations(
        self,
        requested_values: list[float | None],
        target_total_sec: float,
        count: int,
    ) -> list[float]:
        if count <= 0:
            return []

        min_sec = self.shot_duration_min_sec
        max_sec = self.shot_duration_max_sec
        min_total = count * min_sec
        max_total = count * max_sec
        target = self._clamp(float(target_total_sec), min_total, max_total)

        default_each = self._clamp(target / count, min_sec, max_sec)
        values: list[float] = []
        for idx in range(count):
            val = requested_values[idx] if idx < len(requested_values) else None
            if not isinstance(val, (int, float)) or float(val) <= 0:
                values.append(default_each)
            else:
                values.append(self._clamp(float(val), min_sec, max_sec))

        total = sum(values)
        if total <= 0:
            values = [default_each] * count
            total = sum(values)

        scale = target / total if total > 0 else 1.0
        values = [self._clamp(v * scale, min_sec, max_sec) for v in values]

        remaining = target - sum(values)
        for _ in range(count * 6):
            if abs(remaining) <= 0.005:
                break
            if remaining > 0:
                idx = max(range(count), key=lambda i: (max_sec - values[i], -values[i]))
                room = max_sec - values[idx]
                if room <= 1e-6:
                    break
                delta = min(remaining, room)
                values[idx] += delta
                remaining -= delta
            else:
                idx = max(range(count), key=lambda i: (values[i] - min_sec, values[i]))
                room = values[idx] - min_sec
                if room <= 1e-6:
                    break
                delta = min(-remaining, room)
                values[idx] -= delta
                remaining += delta

        rounded = [round(v, 2) for v in values]
        target_rounded = round(target, 2)
        diff = round(target_rounded - sum(rounded), 2)
        if abs(diff) >= 0.01:
            for i in range(count - 1, -1, -1):
                candidate = round(rounded[i] + diff, 2)
                if min_sec <= candidate <= max_sec:
                    rounded[i] = candidate
                    break

        return rounded

    def _build_fallback_shot_plans(self, shots: list[Shot], target_scene_duration_sec: float) -> list[dict[str, Any]]:
        ordered = sorted(shots, key=lambda x: x.sequence)
        if not ordered:
            return []

        durations = self._allocate_durations(
            requested_values=[self.scene_duration_per_shot_sec for _ in ordered],
            target_total_sec=target_scene_duration_sec,
            count=len(ordered),
        )
        rows: list[dict[str, Any]] = []
        for idx, sh in enumerate(ordered):
            purpose = self._safe_text(sh.auto_prompt()) or "推进当前场景信息与情绪。"
            if len(purpose) > 96:
                purpose = purpose[:93] + "..."
            rows.append(
                {
                    "shot_id": sh.shot_id,
                    "sequence": int(sh.sequence),
                    "duration_sec": durations[idx],
                    "purpose": purpose,
                    "transition_to_next": "保持叙事连续，顺切到下一个镜头。",
                }
            )
        return rows

    def _normalize_shot_plans(
        self,
        raw_shot_plans: Any,
        shots: list[Shot],
        target_scene_duration_sec: float,
        fallback_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float]:
        ordered = sorted(shots, key=lambda x: x.sequence)
        if not ordered:
            return [], 0.0

        shot_map = {s.shot_id: s for s in ordered}
        fallback_map = {
            str(row.get("shot_id", "")): row
            for row in fallback_rows
            if isinstance(row, dict)
        }
        parsed_map: dict[str, dict[str, Any]] = {}
        if isinstance(raw_shot_plans, list):
            for row in raw_shot_plans:
                if not isinstance(row, dict):
                    continue
                shot_id = self._safe_text(row.get("shot_id", ""))
                if not shot_id or shot_id not in shot_map or shot_id in parsed_map:
                    continue
                dur = row.get("duration_sec")
                parsed_map[shot_id] = {
                    "shot_id": shot_id,
                    "sequence": int(shot_map[shot_id].sequence),
                    "duration_sec": float(dur) if isinstance(dur, (int, float)) and float(dur) > 0 else None,
                    "purpose": self._safe_text(row.get("purpose", "")),
                    "transition_to_next": self._safe_text(row.get("transition_to_next", "")),
                }

        rows: list[dict[str, Any]] = []
        duration_seed: list[float | None] = []
        for sh in ordered:
            base = (
                parsed_map.get(sh.shot_id)
                or fallback_map.get(sh.shot_id)
                or {
                    "shot_id": sh.shot_id,
                    "sequence": int(sh.sequence),
                    "duration_sec": None,
                    "purpose": "",
                    "transition_to_next": "",
                }
            )
            row = {
                "shot_id": sh.shot_id,
                "sequence": int(sh.sequence),
                "duration_sec": base.get("duration_sec"),
                "purpose": self._safe_text(base.get("purpose", "")),
                "transition_to_next": self._safe_text(base.get("transition_to_next", "")),
            }
            if not row["purpose"]:
                prompt_text = self._safe_text(sh.auto_prompt())
                row["purpose"] = (prompt_text[:93] + "...") if len(prompt_text) > 96 else (prompt_text or "推进剧情。")
            if not row["transition_to_next"]:
                row["transition_to_next"] = "保持叙事连续，顺切到下一个镜头。"
            rows.append(row)
            duration_seed.append(row["duration_sec"] if isinstance(row["duration_sec"], (int, float)) else None)

        durations = self._allocate_durations(
            requested_values=duration_seed,
            target_total_sec=target_scene_duration_sec,
            count=len(rows),
        )
        for idx, row in enumerate(rows):
            row["duration_sec"] = durations[idx]

        return rows, round(sum(durations), 2)

    def _scene_prompt_payload(self, scene: Scene, shots: list[Shot]) -> dict[str, Any]:
        constraints = self._duration_constraints(shots)
        return {
            "scene_id": scene.scene_id,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "scene_plot_description": scene.plot_description,
            "duration_constraints": constraints,
            "shots": [
                {
                    "shot_id": sh.shot_id,
                    "sequence": sh.sequence,
                    "shot_prompt": sh.auto_prompt(),
                }
                for sh in shots
            ],
        }

    def _resolve_system_prompt(self, skill_names: list[str], fallback_prompt: str) -> str:
        resolver = self.skill_resolver or build_skill_resolver_from_env()
        if resolver is None:
            return fallback_prompt
        for skill_name in skill_names:
            if not skill_name:
                continue
            try:
                manifest = resolver.resolve_exact(self.__class__.__name__, skill_name, {})
            except Exception:
                manifest = None
            if manifest is None:
                continue
            return load_skill_prompt_template_text(manifest) or fallback_prompt
        return fallback_prompt

    def _fallback_scene_plan(self, scene: Scene, shots: list[Shot]) -> dict[str, Any]:
        constraints = self._duration_constraints(shots)
        est_duration = float(constraints["target_scene_duration_sec"])
        narration = (
            f"在{scene.location or '未知地点'}的{scene.time_of_day or '未知时间'}，"
            f"故事围绕“{scene.plot_description or '该场景事件'}”展开。"
        )
        dialogue = [
            {
                "speaker": "角色A",
                "text": f"我们得尽快处理这件事。{scene.plot_description or ''}".strip(),
                "duration_sec": round(est_duration * 0.25, 2),
            },
            {
                "speaker": "角色B",
                "text": "我同意，但先确认现场情况。",
                "duration_sec": round(est_duration * 0.2, 2),
            },
        ]
        return {
            "scene_id": scene.scene_id,
            "scene_storyline_summary": self._safe_text(scene.plot_description) or "场景推进剧情发展。",
            "estimated_scene_duration_sec": round(est_duration, 2),
            "shot_duration_constraints": constraints,
            "shot_plans": self._build_fallback_shot_plans(shots, target_scene_duration_sec=est_duration),
            "narration": {
                "text": narration,
                "duration_sec": round(est_duration * 0.3, 2),
                "tone": "客观、电影化",
            },
            "dialogue": dialogue,
            "music": {
                "description": (
                    f"Cinematic underscore for {scene.location or 'unknown location'}, "
                    "subtle tension and emotional continuity, no vocals."
                ),
                "mood": "dramatic",
                "duration_sec": round(est_duration, 2),
            },
            "editing_notes": "对白优先清晰度，旁白在转场处进入，音乐做ducking。",
            "source": "fallback",
        }

    def _generate_scene_plan_online(self, scene: Scene, shots: list[Shot]) -> dict[str, Any]:
        payload = self._scene_prompt_payload(scene, shots)
        system = self._resolve_system_prompt(
            [skill_name_override(self.__class__.__name__), "cameraatist-shot-planning"],
            (
                "你是电影剧情监制（Story Supervisor）。"
                "请基于剧情信息，为后期制作给出可执行规划。"
                "返回严格JSON，不要输出额外解释。"
            ),
        )
        user = (
            "输入场景数据：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "请输出JSON，字段必须包含：\n"
            "{\n"
            '  "scene_id": "string",\n'
            '  "scene_storyline_summary": "string",\n'
            '  "estimated_scene_duration_sec": number,\n'
            '  "shot_plans": [\n'
            '    {"shot_id":"string","sequence":number,"duration_sec":number,"purpose":"string","transition_to_next":"string"}\n'
            "  ],\n"
            '  "narration": {"text":"string","duration_sec":number,"tone":"string"},\n'
            '  "dialogue": [{"speaker":"string","text":"string","duration_sec":number}],\n'
            '  "music": {"description":"string","mood":"string","duration_sec":number},\n'
            '  "editing_notes": "string"\n'
            "}\n"
            "要求：\n"
            "1) 旁白与对话要贴合剧情推进；\n"
            "2) 音乐描述包含情绪与节奏导向；\n"
            "3) shot_plans 必须覆盖输入中的每个shot_id，不能缺失或重复；\n"
            "4) 每个 shot.duration_sec 必须在 duration_constraints.min_sec 与 max_sec 之间；\n"
            "5) shot_plans.duration_sec 总和要与 estimated_scene_duration_sec 基本一致（误差不超过 tolerance_sec）；\n"
            "6) 所有duration_sec需为正数。\n"
        )

        def _call() -> dict[str, Any]:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content if resp.choices else "{}"
            if not isinstance(content, str):
                content = str(content)
            return self._extract_json_object(content)

        raw = call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
        raw["source"] = "llm"
        return raw

    def _normalize_scene_plan(
        self,
        scene: Scene,
        shots: list[Shot],
        raw: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        plan = dict(fallback)
        plan["source"] = raw.get("source", "llm")
        plan["scene_id"] = str(raw.get("scene_id", scene.scene_id)) or scene.scene_id

        ssum = self._safe_text(raw.get("scene_storyline_summary", ""))
        if ssum:
            plan["scene_storyline_summary"] = ssum

        est = raw.get("estimated_scene_duration_sec")
        if isinstance(est, (int, float)) and est > 0:
            plan["estimated_scene_duration_sec"] = round(float(est), 2)

        shot_rows, total_shot_duration = self._normalize_shot_plans(
            raw_shot_plans=raw.get("shot_plans"),
            shots=shots,
            target_scene_duration_sec=float(plan.get("estimated_scene_duration_sec", 0.0) or 0.0),
            fallback_rows=list(plan.get("shot_plans", [])),
        )
        if shot_rows:
            plan["shot_plans"] = shot_rows
            plan["estimated_scene_duration_sec"] = total_shot_duration
            constraints = plan.get("shot_duration_constraints")
            if isinstance(constraints, dict):
                constraints["target_scene_duration_sec"] = total_shot_duration

        narration = raw.get("narration")
        if isinstance(narration, dict):
            text = self._safe_text(narration.get("text", ""))
            tone = self._safe_text(narration.get("tone", ""))
            dur = narration.get("duration_sec")
            if text:
                plan["narration"]["text"] = text
            if tone:
                plan["narration"]["tone"] = tone
            if isinstance(dur, (int, float)) and dur > 0:
                plan["narration"]["duration_sec"] = round(float(dur), 2)

        dialogue = raw.get("dialogue")
        if isinstance(dialogue, list):
            rows = []
            for row in dialogue:
                if not isinstance(row, dict):
                    continue
                sp = self._safe_text(row.get("speaker", "角色"))
                tx = self._safe_text(row.get("text", ""))
                dur = row.get("duration_sec")
                if not tx:
                    continue
                rows.append(
                    {
                        "speaker": sp or "角色",
                        "text": tx,
                        "duration_sec": round(float(dur), 2) if isinstance(dur, (int, float)) and dur > 0 else 2.0,
                    }
                )
            if rows:
                plan["dialogue"] = rows

        music = raw.get("music")
        if isinstance(music, dict):
            desc = self._safe_text(music.get("description", ""))
            mood = self._safe_text(music.get("mood", ""))
            dur = music.get("duration_sec")
            if desc:
                plan["music"]["description"] = desc
            if mood:
                plan["music"]["mood"] = mood
            if isinstance(dur, (int, float)) and dur > 0:
                plan["music"]["duration_sec"] = round(float(dur), 2)

        edit_notes = self._safe_text(raw.get("editing_notes", ""))
        if edit_notes:
            plan["editing_notes"] = edit_notes

        music_obj = plan.get("music")
        if isinstance(music_obj, dict):
            dur = music_obj.get("duration_sec")
            if not isinstance(dur, (int, float)) or float(dur) <= 0:
                music_obj["duration_sec"] = round(float(plan["estimated_scene_duration_sec"]), 2)

        return plan

    def generate_scene_plan(self, scene: Scene, shots: list[Shot]) -> dict[str, Any]:
        logger.info("story_supervisor scene status=start scene_id=%s shots=%d", scene.scene_id, len(shots))
        fallback = self._fallback_scene_plan(scene, shots)
        try:
            raw = self._generate_scene_plan_online(scene, shots)
            plan = self._normalize_scene_plan(scene, shots, raw, fallback=fallback)
            logger.info(
                "story_supervisor scene status=done scene_id=%s source=%s duration_sec=%.2f",
                scene.scene_id,
                plan.get("source", "unknown"),
                float(plan.get("estimated_scene_duration_sec", 0.0) or 0.0),
            )
            return plan
        except Exception:
            if self.strict_online:
                raise
            logger.info("story_supervisor scene status=fallback scene_id=%s", scene.scene_id)
            return fallback

    def generate_story_plan(self, storyboard: Storyboard, output_dir: Path | None = None) -> dict[str, Any]:
        logger.info("story_supervisor generate_story_plan status=start scenes=%d", len(storyboard.scenes))
        scene_plans: list[dict[str, Any]] = []
        for scene_id in sorted(storyboard.scenes.keys()):
            scene = storyboard.scenes[scene_id]
            shots = storyboard.get_scene_shots(scene_id)
            scene_plans.append(self.generate_scene_plan(scene, shots))

        result = {
            "agent": "StorySupervisorAgent",
            "movie_title": storyboard.title,
            "scene_count": len(scene_plans),
            "scene_plans": scene_plans,
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out = output_dir / "story_supervisor_plan.json"
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("story_supervisor output saved path=%s", str(out))

        logger.info("story_supervisor generate_story_plan status=done scene_count=%d", len(scene_plans))
        return result
