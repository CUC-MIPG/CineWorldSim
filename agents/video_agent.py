from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
import re
from typing import Any

from .client import make_client, pick_model
from .generation_backends import (
    video_provider_from_env,
    magref_settings_from_env,
    wan_r2v_settings_from_env,
    minimax_h3_settings_from_env,
    minimax_h3_vllm_settings_from_env,
    generate_video_magref,
    generate_video_minimax_h3,
    generate_video_minimax_h3_vllm,
    generate_video_wan_r2v,
    generate_keyframe_pseudocode,
    generate_i2v_from_keyframe_pseudocode,
)
from .event_dialogue_tool import EventDialogueTool
from .event_story_tool import EventStoryTool
from .video_shot_planning_tool import VideoShotPlanningTool
from .retry import call_with_retries
from .schemas import Character, Scene, Shot, Storyboard
from .skills.adapter import SkillAdapter
from .skills.resolver import SkillResolver
from .skills.runtime import build_skill_resolver_from_env, skill_name_override
from .skills.types import SkillManifest
from narrative.pipeline_schemas import normalize_duration_frame_plan
from narrative.visual_prompt_constraints import build_people_control_clause, event_needs_multi_person_references
from .video_timeline import write_video_timeline

logger = logging.getLogger(__name__)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
MINIMAX_EVENT_DURATION_SEC = 10
MINIMAX_EVENT_NUM_FRAMES = 243
_REFERENCE_NAME_STOPWORDS = {
    "black",
    "blue",
    "broken",
    "cracked",
    "dark",
    "gold",
    "gray",
    "green",
    "hidden",
    "large",
    "left",
    "light",
    "little",
    "new",
    "old",
    "red",
    "right",
    "sealed",
    "silver",
    "small",
    "true",
    "white",
}


class VideoAgent:
    """Generate shot videos and concatenate final movie."""

    def __init__(
        self,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
        skill_resolver: SkillResolver | None = None,
    ):
        self.client = None
        self.model = model or pick_model("gpt-5.5", scope="VIDEO")
        self.strict_online = strict_online
        self.retries = retries
        self.skill_resolver = skill_resolver or build_skill_resolver_from_env()
        self.dialogue_tool = EventDialogueTool(retries=retries)
        self.story_tool = EventStoryTool(retries=retries)
        self.shot_planning_tool = VideoShotPlanningTool(retries=retries)

    @staticmethod
    def _safe_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    @staticmethod
    def _dedupe_existing(paths: list[str], limit: int | None = None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in paths:
            p = str(value or "").strip()
            if not p:
                continue
            resolved = VideoAgent._resolve_existing_media_path(p)
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
            if limit is not None and len(out) >= limit:
                break
        return out

    @staticmethod
    def _reference_selection_text(node: dict[str, Any]) -> str:
        fields = [
            "event_summary",
            "story_description",
            "video_description",
            "dialogue",
            "rendered_dialogue",
            "event_type",
            "story_phase",
        ]
        return " ".join(VideoAgent._safe_text(node.get(field)) for field in fields).lower()

    @staticmethod
    def _name_parts(name: str) -> list[str]:
        text = re.sub(r"[_\\-]+", " ", VideoAgent._safe_text(name)).lower()
        return [
            part
            for part in re.findall(r"[a-z0-9]+", text)
            if len(part) >= 4 and part not in _REFERENCE_NAME_STOPWORDS
        ]

    @staticmethod
    def _mentions_name(text: str, name: str) -> bool:
        normalized = re.sub(r"[_\\-]+", " ", VideoAgent._safe_text(name)).lower().strip()
        if not normalized:
            return False
        if normalized in text:
            return True
        parts = VideoAgent._name_parts(normalized)
        return bool(parts) and any(part in text for part in parts)

    @staticmethod
    def _path_matches_name(path: str, name: str) -> bool:
        path_text = re.sub(r"[_\\-]+", " ", VideoAgent._safe_text(path)).lower()
        normalized = re.sub(r"[_\\-]+", " ", VideoAgent._safe_text(name)).lower().strip()
        if normalized and normalized in path_text:
            return True
        parts = VideoAgent._name_parts(normalized)
        return bool(parts) and all(part in path_text for part in parts)

    @staticmethod
    def _filter_named_reference_paths(
        paths: list[str],
        names: list[str],
        *,
        mention_text: str,
        require_text_mention: bool,
    ) -> list[str]:
        selected: list[str] = []
        for path in paths:
            for name in names:
                if not VideoAgent._path_matches_name(path, name):
                    continue
                if require_text_mention and not VideoAgent._mentions_name(mention_text, name):
                    continue
                selected.append(path)
                break
        return VideoAgent._dedupe_existing(selected)

    @staticmethod
    def _limit_character_reference_paths(paths: list[str], *, allow_multi: bool, limit: int | None = None) -> list[str]:
        cap = limit if limit is not None else (len(paths) if allow_multi else 1)
        if not allow_multi:
            cap = min(cap, 1)
        return VideoAgent._dedupe_existing(paths, limit=max(0, cap))

    @staticmethod
    def _select_character_reference_paths(
        paths: list[str],
        character_names: list[str],
        *,
        allow_multi: bool,
    ) -> list[str]:
        candidates = VideoAgent._dedupe_existing(paths)
        if not candidates:
            return []
        desired_names = character_names if allow_multi else character_names[:1]
        if desired_names:
            matched = VideoAgent._filter_named_reference_paths(
                candidates,
                desired_names,
                mention_text="",
                require_text_mention=False,
            )
            if matched:
                return VideoAgent._limit_character_reference_paths(matched, allow_multi=allow_multi)
        return VideoAgent._limit_character_reference_paths(candidates, allow_multi=allow_multi)

    @staticmethod
    def _select_scene_reference_paths(paths: list[str], location_names: list[str]) -> list[str]:
        candidates = VideoAgent._dedupe_existing(paths)
        if not candidates:
            return []
        if location_names:
            matched = VideoAgent._filter_named_reference_paths(
                candidates,
                location_names,
                mention_text="",
                require_text_mention=False,
            )
            if matched:
                return VideoAgent._dedupe_existing(matched, limit=1)
        return VideoAgent._dedupe_existing(candidates, limit=1)

    @staticmethod
    def _select_mentioned_item_reference_paths(
        paths: list[str],
        item_names: list[str],
        *,
        mention_text: str,
    ) -> list[str]:
        if not item_names:
            return []
        candidates = VideoAgent._dedupe_existing(paths)
        matched = VideoAgent._filter_named_reference_paths(
            candidates,
            item_names,
            mention_text=mention_text,
            require_text_mention=True,
        )
        if matched:
            return matched
        if len(item_names) == 1 and len(candidates) == 1 and VideoAgent._mentions_name(mention_text, item_names[0]):
            return candidates[:1]
        return []

    @staticmethod
    def referenced_event_items(node: dict[str, Any]) -> list[str]:
        mention_text = VideoAgent._reference_selection_text(node)
        items = [str(x).strip() for x in node.get("items", []) or [] if str(x).strip()]
        return [item for item in items if VideoAgent._mentions_name(mention_text, item)]

    @staticmethod
    def _resolve_existing_media_path(path_value: str) -> str:
        path = Path(path_value)
        if path.exists() and path.stat().st_size > 0:
            return str(path)
        if path.suffix.lower() in _IMAGE_EXTS:
            for suffix in _IMAGE_EXTS:
                candidate = path.with_suffix(suffix)
                if candidate.exists() and candidate.stat().st_size > 0:
                    return str(candidate)
        return ""

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSON root must be an object: {path}")
        return payload

    @staticmethod
    def _event_skill_candidates(prompt_style: str) -> list[str]:
        style = (prompt_style or "compact").strip().lower()
        if style in {"official_ref", "h3_ref", "full_reference"}:
            return ["h3-prompt-writing", "cameraatist-event-video-prompt"]
        return ["cameraatist-event-video-prompt", "h3-prompt-writing"]

    @staticmethod
    def _env_skill_name() -> str:
        return skill_name_override("VideoAgent")

    @staticmethod
    def _build_event_skill_manifest(prompt_style: str) -> SkillManifest:
        repo_root = Path(__file__).resolve().parents[1]
        style = (prompt_style or "compact").strip().lower()
        if style in {"official_ref", "h3_ref", "full_reference"}:
            skill_root = repo_root / "docs" / "skills"
            return SkillManifest(
                name="h3-prompt-writing",
                description="MiniMax-H3 full-reference prompt writing for CineWorldSim events.",
                kind="prompt",
                source_root=str(skill_root),
                skill_path=str(skill_root / "h3-prompt-writing"),
            )
        return SkillManifest(
            name="cameraatist-event-video-prompt",
            description="CineWorldSim built-in compact event video prompt.",
            kind="prompt",
            source_root="builtin",
            skill_path="builtin://cameraatist-event-video-prompt",
        )

    def _resolve_prompt_artifact(
        self,
        node: dict[str, Any],
        *,
        storyline: str = "",
        prompt_style: str | None = None,
        shot: dict[str, Any] | None = None,
        shot_plan: list[dict[str, Any]] | None = None,
        mode: str = "",
    ) -> dict[str, Any]:
        style = prompt_style or self._event_prompt_style()
        resolver = getattr(self, "skill_resolver", None) or build_skill_resolver_from_env()
        candidates = [self._env_skill_name(), *self._event_skill_candidates(style)]
        if resolver is not None:
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    manifest = resolver.resolve_exact("VideoAgent", candidate, {})
                except Exception:
                    manifest = None
                if manifest is not None:
                    return SkillAdapter().build(
                        "VideoAgent",
                        manifest,
                        {
                            "node": node,
                            "shot": shot or {},
                            "shot_plan": shot_plan or [],
                            "storyline": storyline,
                            "prompt_style": style,
                            "mode": mode,
                        },
                    )
        manifest = self._build_event_skill_manifest(style)
        return SkillAdapter().build(
            "VideoAgent",
            manifest,
            {
                "node": node,
                "shot": shot or {},
                "shot_plan": shot_plan or [],
                "storyline": storyline,
                "prompt_style": style,
                "mode": mode,
            },
        )

    @staticmethod
    def _event_video_prompt(node: dict[str, Any], storyline: str = "") -> str:
        agent = VideoAgent.__new__(VideoAgent)
        agent.skill_resolver = build_skill_resolver_from_env()
        return str(agent._resolve_prompt_artifact(node, storyline=storyline)["prompt"])

    @staticmethod
    def _event_prompt_style() -> str:
        return os.getenv("MINIMAX_H3_PROMPT_STYLE", "compact").strip().lower() or "compact"

    @staticmethod
    def _dialogue_backfill_enabled(value: bool | None = None) -> bool:
        if value is not None:
            return bool(value)
        raw = os.getenv("VIDEO_AGENT_DIALOGUE_BACKFILL", "true").strip().lower()
        return raw not in {"0", "false", "no", "off", "disabled"}

    @staticmethod
    def _dialogue_base_prompt_context(node: dict[str, Any], storyline: str = "") -> str:
        payload = {
            "storyline": storyline,
            "event_summary": node.get("event_summary", ""),
            "video_description": node.get("video_description", ""),
            "story_description": node.get("story_description", ""),
            "event_type": node.get("event_type", ""),
            "story_phase": node.get("story_phase", ""),
            "characters": node.get("characters", []) or [],
            "locations": node.get("locations", []) or [],
            "items": node.get("items", []) or [],
            "source_steps": node.get("source_steps", []) or [],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _backfill_event_story(
        self,
        node: dict[str, Any],
        *,
        storyline: str,
        output_dir: Path,
        rank: int,
        node_id: str,
        enabled: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_node = dict(node)
        if not enabled:
            description = self._safe_text(node.get("story_description") or node.get("event_summary"))
            prompt_node["story_description"] = description
            return prompt_node, {
                "status": "disabled",
                "story_description": description,
                "artifact_path": "",
                "meta_path": "",
                "error": "",
            }
        story_dir = output_dir / "story_artifacts"
        tool = getattr(self, "story_tool", None)
        if tool is None:
            tool = EventStoryTool(retries=max(1, int(getattr(self, "retries", 1))))
            self.story_tool = tool
        try:
            result = tool.generate(
                node=node,
                storyline=storyline,
                output_dir=story_dir,
                rank=rank,
                node_id=node_id,
                base_prompt_context=self._dialogue_base_prompt_context(node, storyline=storyline),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("video story_backfill status=fallback node_id=%s error=%s", node_id, type(exc).__name__)
            result = EventStoryTool.fallback_result(
                node=node,
                output_dir=story_dir,
                rank=rank,
                node_id=node_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        story_description = str(result.get("story_description") or "").strip() or "none"
        prompt_node["story_description"] = story_description
        prompt_node["_story_backfill"] = result
        return prompt_node, result

    def _backfill_event_dialogue(
        self,
        node: dict[str, Any],
        *,
        storyline: str,
        output_dir: Path,
        rank: int,
        node_id: str,
        enabled: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_node = dict(node)
        if not self._dialogue_backfill_enabled(enabled):
            prompt_node.setdefault("dialogue", prompt_node.get("dialogue", "") or prompt_node.get("dialogue_text", ""))
            return prompt_node, {"status": "disabled", "dialogue": {}, "slot_mapping": {}, "rendered_dialogue": "", "artifact_path": "", "meta_path": "", "error": ""}

        dialogue_dir = output_dir / "dialogue_artifacts"
        base_prompt_context = self._dialogue_base_prompt_context(node, storyline=storyline)
        tool = getattr(self, "dialogue_tool", None)
        if tool is None:
            tool = EventDialogueTool(retries=max(1, int(getattr(self, "retries", 1))))
            self.dialogue_tool = tool
        try:
            result = tool.generate(
                node=node,
                storyline=storyline,
                output_dir=dialogue_dir,
                rank=rank,
                node_id=node_id,
                base_prompt_context=base_prompt_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("video dialogue_backfill status=fallback node_id=%s error=%s", node_id, type(exc).__name__)
            result = EventDialogueTool.fallback_result(
                node=node,
                output_dir=dialogue_dir,
                rank=rank,
                node_id=node_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        rendered = str(result.get("rendered_dialogue") or "").strip() or "none"
        prompt_node["dialogue"] = rendered
        prompt_node["_dialogue_backfill"] = result
        return prompt_node, result

    def _event_prompt_artifact(
        self,
        node: dict[str, Any],
        *,
        storyline: str = "",
        prompt_style: str | None = None,
    ) -> dict[str, Any]:
        return self._resolve_prompt_artifact(node, storyline=storyline, prompt_style=prompt_style)

    def _multi_shot_prompt_artifact(
        self,
        node: dict[str, Any],
        shot: dict[str, Any],
        *,
        storyline: str = "",
        prompt_style: str | None = None,
    ) -> dict[str, Any]:
        return self._resolve_prompt_artifact(
            node,
            storyline=storyline,
            prompt_style=prompt_style,
            shot=shot,
            mode="multi_shot_segment",
        )

    @staticmethod
    def _timecode(seconds: float) -> str:
        total_ms = max(0, int(round(float(seconds) * 1000)))
        sec = total_ms // 1000
        ms = total_ms % 1000
        return f"00:{sec:02d}.{ms:03d}"

    @staticmethod
    def _build_event_multi_shot_plan(
        node: dict[str, Any],
        *,
        storyline: str = "",
        shot_count: int = 4,
        event_duration_sec: float = 8.0,
    ) -> list[dict[str, Any]]:
        count = max(1, min(4, int(shot_count or 4)))
        total = float(MINIMAX_EVENT_DURATION_SEC)
        duration = round(total / count, 3)
        story_text = str(node.get("story_description", "") or "").strip()
        if story_text.lower() == "none":
            story_text = ""
        video_text = str(node.get("video_description", "") or "").strip()
        if video_text.lower() == "none":
            video_text = ""
        event = re.sub(
            r"\s+",
            " ",
            story_text or video_text or str(node.get("event_summary", "") or ""),
        ).strip()
        characters = ", ".join(str(x) for x in node.get("characters", []) or [])
        locations = ", ".join(str(x) for x in node.get("locations", []) or [])
        items = ", ".join(str(x) for x in node.get("items", []) or [])
        event_type = str(node.get("event_type", "") or "narrative action")
        phase = str(node.get("story_phase", "") or "ongoing development")

        templates = [
            (
                "Establish the spatial situation and character blocking before the event action begins.",
                "medium-wide establishing shot, stable screen direction, slow push-in through the environment",
            ),
            (
                "Cut into the main character action and make the causal movement readable.",
                "medium shot or over-the-shoulder framing, shoulder-height tracking move, preserve hand-object contact",
            ),
            (
                "Isolate the key decision, clue, or emotional reaction that changes the event.",
                "close-up detail or reaction shot, rack focus from object/action to face, controlled shallow depth",
            ),
            (
                "Resolve the event beat and prepare visual continuity into the next event.",
                "wider resolving composition, gentle diagonal camera move, hold final screen direction for tail-frame continuity",
            ),
        ]
        rows: list[dict[str, Any]] = []
        event_dialogue = str(node.get("dialogue", "") or "").strip()
        for i in range(count):
            purpose, camera = templates[i] if i < len(templates) else templates[-1]
            start = round(i * duration, 3)
            dialogue = event_dialogue or "No spoken dialogue; use facial expression, body movement, ambient sound, and camera emphasis."
            bgm = "A restrained cinematic underscore supports the event rhythm without overpowering dialogue or ambient sound."
            if "climax" in phase.lower() or "escalat" in phase.lower() or "reveal" in event_type.lower():
                bgm = "Low suspense pulse with restrained strings and distant percussion, rising slightly across the shot."
            rows.append(
                {
                    "shot_index": i + 1,
                    "start_timecode": VideoAgent._timecode(start),
                    "start_sec": start,
                    "duration_sec": duration,
                    "video_description": (
                        f"{purpose} Event: {event}. Characters: {characters or 'visible story subjects'}. "
                        f"Location: {locations or 'current environment'}. Key items: {items or 'none'}."
                    ),
                    "cinematography": camera,
                    "dialogue": dialogue,
                    "bgm": bgm,
                    "event_type": event_type,
                    "story_phase": phase,
                    "storyline_context": storyline,
                }
            )
        return rows

    @staticmethod
    def _normalize_shot_plan_rows(node: dict[str, Any], shots: list[dict[str, Any]], *, duration_sec: int) -> list[dict[str, Any]]:
        if not shots:
            return VideoAgent._build_event_multi_shot_plan(
                node,
                shot_count=1,
                event_duration_sec=float(duration_sec),
            )
        rows = VideoShotPlanningTool._normalize_shots(shots, MINIMAX_EVENT_DURATION_SEC)
        for row in rows:
            row["event_type"] = str(node.get("event_type", "") or "")
            row["story_phase"] = str(node.get("story_phase", "") or "")
        return rows

    def _build_event_video_plan(
        self,
        node: dict[str, Any],
        *,
        storyline: str,
        output_dir: Path,
        rank: int,
        node_id: str,
        event_shot_count: int,
        event_duration_sec: float | None,
    ) -> dict[str, Any]:
        tool = getattr(self, "shot_planning_tool", None)
        if tool is None:
            tool = VideoShotPlanningTool(retries=max(1, int(getattr(self, "retries", 1))))
            self.shot_planning_tool = tool
        try:
            plan = tool.generate(
                node=node,
                storyline=storyline,
                output_dir=output_dir / "shot_plan_artifacts",
                rank=rank,
                node_id=node_id,
                max_shots=max(1, min(4, int(event_shot_count or 4))),
            )
        except Exception as exc:  # noqa: BLE001
            plan = VideoShotPlanningTool.fallback_plan(node, duration_sec=MINIMAX_EVENT_DURATION_SEC)
            plan["error"] = f"{type(exc).__name__}: {exc}"
        normalized = normalize_duration_frame_plan({**plan, "duration_sec": MINIMAX_EVENT_DURATION_SEC}, default_duration_sec=MINIMAX_EVENT_DURATION_SEC)
        planned_shots = VideoShotPlanningTool._normalize_shots(
            list(plan.get("shots", []) or []),
            MINIMAX_EVENT_DURATION_SEC,
            max_shots=max(1, min(4, int(event_shot_count or 4))),
        )
        shots = self._normalize_shot_plan_rows(node, planned_shots, duration_sec=MINIMAX_EVENT_DURATION_SEC)
        return {
            **plan,
            "duration_sec": normalized["duration_sec"],
            "num_frames": normalized["num_frames"],
            "shots": shots,
        }

    @staticmethod
    def _multi_shot_prompt_from_plan(
        node: dict[str, Any],
        shot: dict[str, Any],
        *,
        storyline: str = "",
    ) -> str:
        shot_index = int(shot.get("shot_index", 1) or 1)
        prefix = f"[Shot {shot_index}]"
        if shot_index > 1:
            prefix = f"[Shot {shot_index}] At {shot.get('start_timecode', '00:00.000')},"
        return (
            "integrated_multimodal description: "
            f"{prefix} "
            f"video_description: {shot.get('video_description', '')} "
            f"cinematography: {shot.get('cinematography', '')}. "
            f"dialogue: {shot.get('dialogue', '')} "
            f"narration: {shot.get('narration') or shot.get('voiceover') or 'none'} "
            f"bgm: {shot.get('bgm', '')} "
            f"Preserve referenced character identity, world-space layout, lighting continuity, object placement, and coherent motion. "
            "Do not introduce pedestrians, crowds, or unrelated people. "
            f"Generate only this shot as one continuous cinematic clip, not a slideshow. "
            f"Event type: {node.get('event_type', '')}. Story phase: {node.get('story_phase', '')}. "
            f"Storyline: {re.sub(r'\\s+', ' ', str(storyline or '')).strip()[:300]}"
        )

    def _multi_shot_event_prompt_from_plan(
        self,
        node: dict[str, Any],
        shot_plan: list[dict[str, Any]],
        *,
        storyline: str = "",
        prompt_style: str | None = None,
    ) -> str:
        artifact = self._resolve_prompt_artifact(
            node,
            storyline=storyline,
            prompt_style=prompt_style,
            shot_plan=shot_plan,
            mode="multi_shot_event",
        )
        return str(artifact["prompt"])

    @staticmethod
    def _explicit_reference_mode(node: dict[str, Any]) -> str:
        allowed = {"event_keyframe_only", "semantic_assets", "all_allowed"}
        assets = node.get("visual_assets", {}) if isinstance(node.get("visual_assets"), dict) else {}
        for key in ["video_reference_mode", "reference_mode"]:
            mode = str(node.get(key, "") or "").strip().lower()
            if mode in allowed:
                return mode
        mode = str(assets.get("video_reference_mode", "") or assets.get("reference_mode", "") or "").strip().lower()
        return mode if mode in allowed else "all_allowed"

    @staticmethod
    def _node_reference_bundle(
        node: dict[str, Any],
        *,
        include_character_keyframes: bool = True,
        include_world_images: bool = True,
        max_reference_images: int = 4,
    ) -> dict[str, Any]:
        assets = node.get("visual_assets", {}) or {}
        allow_multi_character_refs = bool(assets.get("multi_person_reference_allowed")) or event_needs_multi_person_references(node)
        mention_text = VideoAgent._reference_selection_text(node)
        character_names = [str(x).strip() for x in node.get("characters", []) or [] if str(x).strip()]
        location_names = [str(x).strip() for x in node.get("locations", []) or [] if str(x).strip()]
        item_names = VideoAgent.referenced_event_items(node)
        cap = max(1, int(max_reference_images or 1))
        reference_mode = VideoAgent._explicit_reference_mode(node)

        event_keyframes: list[str] = []
        character_refs: list[str] = []
        location_refs: list[str] = []
        item_refs: list[str] = []
        if include_character_keyframes:
            keyframe_candidates = [str(x) for x in assets.get("character_keyframes", []) or []]
            keyframe_candidates.extend([str(x) for x in assets.get("event_keyframes", []) or []])
            event_keyframes = VideoAgent._select_character_reference_paths(
                keyframe_candidates,
                character_names,
                allow_multi=allow_multi_character_refs,
            )
            portrait_candidates = [str(x) for x in assets.get("character_reference_images", []) or []]
            character_refs = VideoAgent._select_character_reference_paths(
                portrait_candidates,
                character_names,
                allow_multi=allow_multi_character_refs,
            )
        if include_world_images:
            world_updates = [str(x) for x in assets.get("world_update_images", []) or []]
            world_updates.extend([str(x) for x in node.get("world_update_images", []) or []])
            location_candidates = [str(x) for x in assets.get("location_images", []) or []]
            location_candidates.extend([x for x in world_updates if "/locations/" in x])
            location_candidates.extend([x for x in world_updates if "/locations/" not in x and "/items/" not in x])
            item_candidates = [str(x) for x in assets.get("item_images", []) or []]
            item_candidates.extend([x for x in world_updates if "/items/" in x])
            location_refs = VideoAgent._select_scene_reference_paths(
                location_candidates,
                location_names,
            )
            item_refs = VideoAgent._select_mentioned_item_reference_paths(
                item_candidates,
                item_names,
                mention_text=mention_text,
            )

        if reference_mode == "event_keyframe_only":
            selected = event_keyframes
        elif reference_mode == "semantic_assets":
            selected = [*character_refs, *location_refs, *item_refs]
        else:
            selected = [*event_keyframes, *character_refs, *location_refs, *item_refs]
        selected = VideoAgent._dedupe_existing(selected, limit=cap)
        all_candidates = VideoAgent._dedupe_existing([*event_keyframes, *character_refs, *location_refs, *item_refs])
        excluded = [path for path in all_candidates if path not in selected]
        return {
            "reference_mode": reference_mode,
            "reference_image_paths": selected,
            "event_keyframe_paths": [p for p in event_keyframes if p in selected],
            "character_reference_paths": [p for p in character_refs if p in selected],
            "location_reference_paths": [p for p in location_refs if p in selected],
            "item_reference_paths": [p for p in item_refs if p in selected],
            "excluded_reference_paths": excluded,
            "mentioned_items": item_names,
            "multi_person_reference_allowed": allow_multi_character_refs,
            "reference_selection_reason": "selected by explicit or default VideoAgent reference mode without filling unused slots",
        }

    @staticmethod
    def _node_reference_images(
        node: dict[str, Any],
        *,
        include_character_keyframes: bool,
        include_world_images: bool,
        max_reference_images: int,
    ) -> list[str]:
        return VideoAgent._node_reference_bundle(
            node,
            include_character_keyframes=include_character_keyframes,
            include_world_images=include_world_images,
            max_reference_images=max_reference_images,
        )["reference_image_paths"]

    @staticmethod
    def _extract_video_continuity_assets(
        video_path: str,
        tail_frame_path: str,
        audio_path: str,
    ) -> dict[str, str]:
        """Extract reusable visual and audio continuity assets from a completed event video."""
        source = Path(video_path)
        tail = Path(tail_frame_path)
        audio = Path(audio_path)
        result = {"tail_frame_path": "", "audio_path": ""}
        if not source.exists():
            return result

        tail.parent.mkdir(parents=True, exist_ok=True)
        audio.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-vf",
                    "reverse",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(tail),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if tail.exists() and tail.stat().st_size > 0:
                result["tail_frame_path"] = str(tail)
        except Exception:
            logger.exception("video continuity tail_frame status=failed source=%s", source)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    str(audio),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if audio.exists() and audio.stat().st_size > 0:
                result["audio_path"] = str(audio)
        except Exception:
            # Video backends can return video-only clips. Tail-frame continuity remains usable.
            logger.info("video continuity audio status=unavailable source=%s", source)
        return result

    @staticmethod
    def _continuity_node_context(node: dict[str, Any], label: str) -> str:
        payload = {
            "event_summary": node.get("event_summary", ""),
            "event_type": node.get("event_type", ""),
            "story_phase": node.get("story_phase", ""),
            "characters": node.get("characters", []) or [],
            "locations": node.get("locations", []) or [],
            "items": node.get("items", []) or [],
        }
        return f"{label}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _judge_previous_tail_frame(
        self,
        previous_node: dict[str, Any],
        current_node: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask the LLM whether the previous event tail frame helps current continuity."""
        fallback = {
            "use_previous_tail_frame": False,
            "continuity_score": 0.0,
            "reason": "Continuity judge failed or returned an ambiguous result; tail frame is not used.",
        }
        system_prompt = (
            "You are a continuity editor for an event-based cinematic video pipeline. "
            "Decide whether the previous event video's final frame should be used as a visual reference "
            "for the current event. Compare the previous event description with the current event description, "
            "character identity and relationships, location and spatial relationships, and important objects. "
            "Use the previous tail frame only when the events are narratively and visually continuous and the "
            "tail frame is likely to preserve useful state. Do not use it for a clear scene change, time jump, "
            "character change, location change without visual continuity, or conflicting object state. "
            "The default decision is NO. Return only a JSON object with exactly these fields: "
            "use_previous_tail_frame (boolean), continuity_score (number from 0 to 1), and reason (short string)."
        )
        user_prompt = (
            "Judge whether to use the previous event tail frame for the current event.\n\n"
            f"{self._continuity_node_context(previous_node, 'Previous event')}\n\n"
            f"{self._continuity_node_context(current_node, 'Current event')}\n\n"
            "Return JSON only."
        )
        try:
            if getattr(self, "client", None) is None:
                self.client = make_client(scope="VIDEO")
            if not getattr(self, "model", None):
                self.model = pick_model("gpt-5.5", scope="VIDEO")

            def _call():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

            response = call_with_retries(
                _call,
                attempts=max(1, int(getattr(self, "retries", 1))),
                delay_sec=1.0,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                return fallback
            text = content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\s*```$", "", text).strip()
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("use_previous_tail_frame"), bool):
                return fallback
            score = parsed.get("continuity_score", 0.0)
            try:
                score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                score = 0.0
            reason = str(parsed.get("reason") or "").strip()
            return {
                "use_previous_tail_frame": parsed["use_previous_tail_frame"] is True,
                "continuity_score": score,
                "reason": reason or "The LLM returned no explanation.",
            }
        except Exception as exc:
            logger.info("video continuity judge status=fallback error=%s", type(exc).__name__)
            return {
                **fallback,
                "reason": f"Continuity judge failed with {type(exc).__name__}; tail frame is not used.",
            }

    def _match_characters_for_shot(self, shot: Shot, characters: dict[str, Character]) -> list[Character]:
        if not characters:
            return []

        selected: list[Character] = []
        seen: set[str] = set()

        # 1) Preferred: explicit shot.character_names
        for name in shot.character_names:
            n = self._normalize_name(self._safe_text(name))
            if not n:
                continue
            for k, c in characters.items():
                if self._normalize_name(k) == n and k not in seen:
                    selected.append(c)
                    seen.add(k)

        # 2) Fallback: match by prompt + scene elements text.
        if not selected:
            haystack = " ".join(
                [
                    shot.auto_prompt().lower(),
                    self._safe_text(shot.scene_elements).lower(),
                ]
            )
            for k, c in characters.items():
                key = self._normalize_name(k)
                if not key:
                    continue
                if key in haystack and k not in seen:
                    selected.append(c)
                    seen.add(k)
                    continue
                tokens = [t for t in key.split(" ") if len(t) > 2]
                if tokens and any(t in haystack for t in tokens) and k not in seen:
                    selected.append(c)
                    seen.add(k)

        return selected

    def generate_llm_video_prompt(
        self,
        shot: Shot,
        scene: Scene,
        characters: dict[str, Character],
    ) -> str:
        system_prompt = (
            "You are a professional storyboard artist and VFX expert. "
            "Generate one high-quality English image-to-video prompt paragraph. "
            "Keep human subjects limited to the named characters and avoid pedestrians, crowds, and unrelated people."
        )
        character_info = "\n".join(f"{c.name}: {c.appearance}" for c in characters.values())
        elements_info = "\n".join(f"{k}: {v}" for k, v in shot.scene_elements.items())
        people_clause = build_people_control_clause(list(characters.keys()), allow_multi_people=len(characters) > 1)

        user_prompt = f"""
Scene:
- scene_id: {scene.scene_id}
- location: {scene.location}
- time: {scene.time_of_day}
- plot: {scene.plot_description}

Characters:
{character_info}

Shot:
- shot_id: {shot.shot_id}
- sequence: {shot.sequence}
- shot_type: {shot.shot_type}
- camera_angle: {shot.camera_angle}
- scene_elements:
{elements_info}
- camera_movement: {shot.camera_movement}

Visual constraints:
{people_clause}

Return only one polished English paragraph for image-to-video generation.
"""
        def _call():
            if self.client is None:
                self.client = make_client(scope="VIDEO")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        try:
            return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
        except Exception:
            if self.strict_online:
                raise
            return (
                f"Cinematic shot in {scene.location} at {scene.time_of_day}. "
                f"Action: {shot.auto_prompt()}. "
                "Detailed environment, dynamic lighting, smooth camera movement, 4k quality."
            )

    def send_generation_request(
        self,
        prompt: str,
        image_paths: list[str],
        video_path: str,
        video_mode: str = "reference",
        first_image_path: str = "",
        last_image_path: str = "",
        reference_image_paths: list[str] | None = None,
        reference_video_paths: list[str] | None = None,
        reference_audio_paths: list[str] | None = None,
        minimax_task: str | None = None,
        duration_sec: float | None = None,
        num_frames: int | None = None,
    ) -> bool:
        def _call():
            provider = video_provider_from_env()
            logger.info(
                "video generate_request provider=%s output=%s mode=%s image_count=%d",
                provider,
                video_path,
                video_mode,
                len(image_paths),
            )
            if provider == "minimax_h3":
                vllm_cfg = minimax_h3_vllm_settings_from_env()
                cfg = minimax_h3_settings_from_env()
                resolved_task = minimax_task or cfg["task"]
                if vllm_cfg["enabled"]:
                    return generate_video_minimax_h3_vllm(
                        prompt=prompt,
                        image_paths=image_paths,
                        output_video_path=video_path,
                        video_mode=video_mode,
                        task=resolved_task,
                        first_image_path=first_image_path,
                        last_image_path=last_image_path,
                        reference_image_paths=reference_image_paths,
                        reference_audio_paths=reference_audio_paths,
                        upload_reference_audio=vllm_cfg["upload_reference_audio"],
                        api_url=vllm_cfg["api_url"],
                        width=vllm_cfg["width"],
                        height=vllm_cfg["height"],
                        num_frames=int(num_frames or vllm_cfg["num_frames"]),
                        fps=vllm_cfg["fps"],
                        steps=vllm_cfg["steps"],
                        flow_shift=vllm_cfg["flow_shift"],
                        seed=vllm_cfg["seed"],
                        duration=None,
                        audio_flow_shift=vllm_cfg["audio_flow_shift"],
                        timeout_sec=vllm_cfg["timeout_sec"],
                        aspect_ratio=vllm_cfg["aspect_ratio"],
                    )
                return generate_video_minimax_h3(
                    prompt=prompt,
                    image_paths=image_paths,
                    output_video_path=video_path,
                    video_mode=video_mode,
                    task=resolved_task,
                    first_image_path=first_image_path,
                    last_image_path=last_image_path,
                    reference_image_paths=reference_image_paths,
                    reference_video_paths=reference_video_paths,
                    reference_audio_paths=reference_audio_paths,
                    model_dir=cfg["model_dir"],
                    conda_env=cfg["conda_env"],
                    script_path=cfg["script_path"],
                    python_bin=cfg["python_bin"],
                    cuda_visible_devices=cfg["cuda_visible_devices"],
                    width=cfg["width"],
                    height=cfg["height"],
                    num_frames=int(num_frames or cfg["num_frames"]),
                    steps=cfg["steps"],
                    seed=cfg["seed"],
                    fps=cfg["fps"],
                    parallel_mode=cfg["parallel_mode"],
                    device_map=cfg["device_map"],
                    max_memory=cfg["max_memory"],
                    text_encoder_placement=cfg["text_encoder_placement"],
                    text_encoder_use_stream=cfg["text_encoder_use_stream"],
                    offload_folder=cfg["offload_folder"],
                    runtime_device_index=cfg["runtime_device_index"],
                    vae_parallel_mode=cfg["vae_parallel_mode"],
                    vae_placement=cfg["vae_placement"],
                    vae_use_tiling=cfg["vae_use_tiling"],
                    vae_use_slicing=cfg["vae_use_slicing"],
                    reference_resize_width=cfg["reference_resize_width"],
                    reference_resize_height=cfg["reference_resize_height"],
                    pytorch_cuda_alloc_conf=cfg["pytorch_cuda_alloc_conf"],
                    timeout_sec=cfg["timeout_sec"],
                )
            if provider == "wan_r2v":
                cfg = wan_r2v_settings_from_env()
                if not cfg["api_key"]:
                    raise RuntimeError("WAN_R2V_API_KEY is required when VIDEO_GEN_PROVIDER=wan_r2v")
                return generate_video_wan_r2v(
                    prompt=prompt,
                    image_paths=image_paths,
                    output_video_path=video_path,
                    api_key=cfg["api_key"],
                    base_http_api_url=cfg["base_http_api_url"],
                    model_name=cfg["model_name"],
                    resolution=cfg["resolution"],
                    duration=cfg["duration"],
                    prompt_extend=cfg["prompt_extend"],
                    watermark=cfg["watermark"],
                    negative_prompt=cfg["negative_prompt"],
                    seed=cfg["seed"],
                )

            cfg = magref_settings_from_env()
            return generate_video_magref(
                prompt=prompt,
                image_paths=image_paths,
                output_video_path=video_path,
                api_url=cfg["api_url"],
            )
        try:
            return call_with_retries(_call, attempts=self.retries, delay_sec=1.5)
        except Exception:
            logger.exception("video generate_request failed output=%s", video_path)
            return False

    def _collect_images(self, shot: Shot, scene: Scene, characters: dict[str, Character]) -> list[str]:
        images: list[str] = []
        images.extend(scene.reference_images)

        matched_characters = self._match_characters_for_shot(shot, characters)
        if matched_characters:
            for c in matched_characters:
                if c.reference_image:
                    images.append(c.reference_image)
                images.extend(c.reference_images)
        elif not scene.reference_images:
            # Fallback: if no scene image and no matched characters, include all character refs.
            for c in characters.values():
                if c.reference_image:
                    images.append(c.reference_image)
                images.extend(c.reference_images)

        uniq = []
        seen = set()
        for p in images:
            if p and p not in seen and Path(p).exists():
                seen.add(p)
                uniq.append(p)
        return uniq

    def process_shots(
        self,
        storyboard: Storyboard,
        output_dir: Path,
        dry_run: bool = False,
        save_task_json: bool = True,
        video_mode: str = "reference",  # reference|keyframe|text|image|first_last|last_frame
    ) -> Storyboard:
        logger.info(
            "video process_shots status=start scenes=%d dry_run=%s mode=%s",
            len(storyboard.scenes),
            dry_run,
            video_mode,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        mode = (video_mode or "reference").strip().lower()
        if mode not in {"reference", "keyframe", "text", "image", "first_last", "last_frame"}:
            mode = "reference"

        ordered_shots = storyboard.ordered_shots_path_first(unique_by_scene_sequence=True)
        prev_scene_id = ""
        for shot in ordered_shots:
            scene = storyboard.scenes.get(shot.scene_id)
            if scene is None:
                scene = Scene(scene_id=shot.scene_id, location="", time_of_day="", plot_description="")
            if scene.scene_id != prev_scene_id:
                logger.info("video scene status=start scene_id=%s", scene.scene_id)
                prev_scene_id = scene.scene_id

            matched = self._match_characters_for_shot(shot, storyboard.screenplay.characters)
            prompt_characters: dict[str, Character]
            if matched:
                prompt_characters = {c.name: c for c in matched}
            else:
                prompt_characters = storyboard.screenplay.characters

            prompt = self.generate_llm_video_prompt(shot, scene, prompt_characters)
            image_paths = self._collect_images(shot, scene, storyboard.screenplay.characters)
            if mode == "text":
                image_paths = []
            elif mode in {"image", "last_frame"}:
                image_paths = image_paths[:1]
            elif mode == "first_last":
                image_paths = image_paths[:2]
            video_path = str(output_dir / f"video_{shot.shot_id}.mp4")
            if mode == "keyframe":
                char_images: list[str] = []
                for c in storyboard.screenplay.characters.values():
                    if c.reference_image and Path(c.reference_image).exists():
                        char_images.append(c.reference_image)
                    char_images.extend(
                        [p for p in c.reference_images if p and Path(p).exists()]
                    )
                char_images = list(dict.fromkeys(char_images))
                keyframe_path = str(output_dir / "keyframes" / f"keyframe_{shot.shot_id}.png")
                tasks.append(
                    {
                        "shot_id": shot.shot_id,
                        "mode": "keyframe",
                        "prompt": prompt,
                        "scene_images": scene.reference_images,
                        "character_images": char_images,
                        "keyframe_path": keyframe_path,
                        "output_video_path": video_path,
                        "path_sequence": int(getattr(shot, "path_sequence", 0) or 0),
                    }
                )
                if dry_run:
                    shot.video_path = video_path
                    logger.info("video shot status=dry_run shot_id=%s output=%s", shot.shot_id, video_path)
                    continue

                # Pseudocode path: keyframe generation + keyframe I2V.
                kf = generate_keyframe_pseudocode(
                    prompt=prompt,
                    scene_images=scene.reference_images,
                    character_images=char_images,
                    output_keyframe_path=keyframe_path,
                )
                pseudo = generate_i2v_from_keyframe_pseudocode(
                    keyframe_path=kf,
                    prompt=prompt,
                    output_video_path=video_path,
                )
                shot.video_path = pseudo
                logger.info("video shot status=done shot_id=%s mode=keyframe output=%s", shot.shot_id, pseudo)
                continue

            tasks.append(
                {
                    "shot_id": shot.shot_id,
                    "mode": mode,
                    "prompt": prompt,
                    "images": image_paths,
                    "first_image": image_paths[0] if image_paths else "",
                    "last_image": image_paths[-1] if len(image_paths) >= 2 else "",
                    "output_video_path": video_path,
                    "path_sequence": int(getattr(shot, "path_sequence", 0) or 0),
                }
            )

            if dry_run:
                shot.video_path = video_path
                logger.info("video shot status=dry_run shot_id=%s output=%s", shot.shot_id, video_path)
                continue

            ok = self.send_generation_request(
                prompt,
                image_paths,
                video_path,
                video_mode=mode,
                first_image_path=image_paths[0] if image_paths and mode in {"image", "first_last"} else "",
                last_image_path=image_paths[-1] if image_paths and mode in {"last_frame", "first_last"} else "",
                reference_image_paths=image_paths if mode == "reference" else None,
            )
            if ok:
                shot.video_path = video_path
                logger.info("video shot status=done shot_id=%s mode=%s output=%s", shot.shot_id, mode, video_path)
            else:
                logger.info("video shot status=failed shot_id=%s mode=%s output=%s", shot.shot_id, mode, video_path)

        if save_task_json:
            (output_dir / "video_info.json").write_text(
                json.dumps(tasks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("video tasks saved path=%s count=%d", str(output_dir / "video_info.json"), len(tasks))
        logger.info("video process_shots status=done")
        return storyboard

    @staticmethod
    def _auxiliary_location_reference_images(node: dict[str, Any]) -> list[str]:
        assets = node.get("visual_assets", {}) if isinstance(node.get("visual_assets"), dict) else {}
        return VideoAgent._dedupe_existing([str(x) for x in assets.get("location_images", []) or []], limit=1)

    @staticmethod
    def _auxiliary_transition_reference_images(node: dict[str, Any], *, max_reference_images: int = 4) -> list[str]:
        assets = node.get("visual_assets", {}) if isinstance(node.get("visual_assets"), dict) else {}
        refs: list[str] = []
        refs.extend(str(x) for x in node.get("world_update_images", []) or [] if str(x).strip())
        refs.extend(str(x) for x in assets.get("world_update_images", []) or [] if str(x).strip())
        return VideoAgent._dedupe_existing(refs, limit=max_reference_images)

    @staticmethod
    def _build_auxiliary_video_task(
        *,
        task_rank: int,
        task_type: str,
        node: dict[str, Any],
        node_id: str,
        refs: list[str],
        output_dir: Path,
        prompt_style: str,
    ) -> dict[str, Any]:
        location = ", ".join(str(x) for x in node.get("locations", []) or [] if str(x).strip()) or "the current location"
        if task_type == "establishing":
            prompt = (
                f"Cinematic establishing shot of {location}. Show the world-space layout before the character event begins. "
                "No characters, no pedestrians, no crowds, no unrelated people, no extra props. Smooth camera motion and stable lighting."
            )
            output_name = f"aux_video_{task_rank:03d}_establishing_{node_id}.mp4"
        else:
            prompt = (
                f"Cinematic world-state transition shot for {location}. Show only the visible location or item change that has just occurred. "
                "No characters, no pedestrians, no crowds, no unrelated people, no extra props. The shot exists to make the world/item physical change readable."
            )
            output_name = f"aux_video_{task_rank:03d}_world_transition_{node_id}.mp4"
        return {
            "rank": task_rank,
            "task_type": task_type,
            "node_id": f"AUX_{task_rank:03d}_{task_type}_{node_id}",
            "event_id": node.get("event_id", ""),
            "mode": "reference",
            "prompt": prompt,
            "skill_name": "builtin-auxiliary-video-prompt",
            "skill_kind": "prompt",
            "skill_source": "builtin",
            "prompt_style": prompt_style,
            "images": refs,
            "reference_images": refs,
            "output_video_path": str(output_dir / output_name),
            "source_steps": node.get("source_steps", []),
            "source_node_ids": node.get("source_node_ids", []),
            "characters": [],
            "locations": node.get("locations", []),
            "items": node.get("items", []),
            "video_description": prompt,
            "story_description": prompt,
            "insert_before_node_id": node_id,
            "reference_mode": "semantic_assets",
            "reference_selection": {
                "reference_mode": "semantic_assets",
                "reference_image_paths": refs,
                "reference_selection_reason": f"{task_type} task uses only location/world-update references",
            },
                "duration_sec": MINIMAX_EVENT_DURATION_SEC,
                "num_frames": MINIMAX_EVENT_NUM_FRAMES,
            "event_shot_plan": [
                {
                    "shot_index": 1,
                    "start_timecode": "00:00.000",
                    "duration_sec": MINIMAX_EVENT_DURATION_SEC,
                    "video_description": prompt,
                    "cinematography": "slow controlled establishing camera movement" if task_type == "establishing" else "slow controlled detail movement over the changed world state",
                    "dialogue": "none",
                    "bgm": "subtle atmospheric underscore",
                }
            ],
            "shot_video_tasks": [],
        }

    def process_orchestrated_events(
        self,
        *,
        narrative_nodes_json: Path,
        path_plan_json: Path,
        output_dir: Path,
        dry_run: bool = False,
        save_task_json: bool = True,
        max_reference_images: int = 2,
        include_character_keyframes: bool = True,
        include_world_images: bool = True,
        start_rank: int = 1,
        end_rank: int | None = None,
        use_previous_tail_frame: bool = False,
        use_previous_audio: bool = False,
        upload_previous_audio: bool = False,
        multi_shot_events: bool = False,
        event_shot_count: int = 4,
        event_duration_sec: float | None = None,
        dialogue_backfill: bool | None = None,
        story_backfill: bool = True,
        insert_auxiliary_videos: bool = False,
        video_reference_mode: str | None = None,
    ) -> dict[str, Any]:
        """Generate event-level videos from NarrativeNodeSelector + PathOrchestrator outputs."""
        nodes_payload = self._load_json(narrative_nodes_json)
        plan_payload = self._load_json(path_plan_json)
        selected_nodes = [
            node for node in nodes_payload.get("selected_nodes", []) or [] if isinstance(node, dict)
        ]
        node_map = {
            str(node.get("node_id") or node.get("event_id")): node
            for node in selected_nodes
            if str(node.get("node_id") or node.get("event_id") or "").strip()
        }
        ordered_ids = [
            str(x)
            for x in plan_payload.get("ordered_node_ids", []) or []
            if str(x).strip()
        ]
        if not ordered_ids:
            ordered_ids = list(node_map.keys())

        output_dir.mkdir(parents=True, exist_ok=True)
        storyline = str(plan_payload.get("storyline") or nodes_payload.get("storyline") or "")
        prompt_style = self._event_prompt_style()
        tasks: list[dict[str, Any]] = []
        generated: list[dict[str, Any]] = []
        success = 0
        failed = 0
        skipped = 0
        previous_tail_frame_path = ""
        previous_audio_path = ""
        continuity_dir = output_dir / "continuity_refs"
        previous_event_node: dict[str, Any] | None = None
        seen_establishing_locations: set[str] = set()
        completed_event_count = 0

        def _append_auxiliary_task(aux_task: dict[str, Any]) -> None:
            nonlocal success, failed, skipped
            timeline_path = write_video_timeline(aux_task)
            aux_task["timeline_path"] = str(timeline_path)
            tasks.append(aux_task)
            aux_refs = list(aux_task.get("reference_images", []) or [])
            if not aux_refs:
                skipped += 1
                generated.append({**aux_task, "status": "skipped", "reason": "no_reference_images"})
                return
            if dry_run:
                success += 1
                generated.append({**aux_task, "status": "dry_run"})
                return
            existing_output = Path(str(aux_task.get("output_video_path") or ""))
            if existing_output.exists() and existing_output.stat().st_size > 0:
                success += 1
                generated.append({**aux_task, "status": "done", "resume_existing": True})
                return
            ok = self.send_generation_request(
                prompt=str(aux_task.get("prompt") or ""),
                image_paths=aux_refs,
                video_path=str(aux_task.get("output_video_path") or ""),
                video_mode="reference",
                reference_image_paths=aux_refs,
                minimax_task="ref2va",
                duration_sec=MINIMAX_EVENT_DURATION_SEC,
                num_frames=MINIMAX_EVENT_NUM_FRAMES,
            )
            if ok:
                success += 1
                generated.append({**aux_task, "status": "done"})
            else:
                failed += 1
                generated.append({**aux_task, "status": "failed"})

        logger.info(
            "video process_orchestrated_events status=start nodes=%d ordered=%d dry_run=%s",
            len(selected_nodes),
            len(ordered_ids),
            dry_run,
        )
        for rank, node_id in enumerate(ordered_ids, start=1):
            if rank < max(1, int(start_rank)):
                continue
            if end_rank is not None and rank > int(end_rank):
                continue
            node = node_map.get(node_id)
            if node is None:
                failed += 1
                logger.info("video event status=failed node_id=%s reason=node_missing", node_id)
                continue
            ref_node = dict(node)
            if video_reference_mode and not ref_node.get("video_reference_mode") and not ref_node.get("reference_mode"):
                ref_node["video_reference_mode"] = video_reference_mode
            if insert_auxiliary_videos:
                for location in [str(x).strip() for x in node.get("locations", []) or [] if str(x).strip()][:1]:
                    loc_key = location.lower()
                    if loc_key not in seen_establishing_locations:
                        est_refs = self._auxiliary_location_reference_images(node)
                        if est_refs:
                            _append_auxiliary_task(
                                self._build_auxiliary_video_task(
                                    task_rank=len(tasks) + 1,
                                    task_type="establishing",
                                    node=node,
                                    node_id=node_id,
                                    refs=est_refs,
                                    output_dir=output_dir,
                                    prompt_style=prompt_style,
                                )
                            )
                        seen_establishing_locations.add(loc_key)
                transition_refs = self._auxiliary_transition_reference_images(
                    node,
                    max_reference_images=max(1, int(max_reference_images)),
                )
                if completed_event_count > 0 and transition_refs:
                    _append_auxiliary_task(
                        self._build_auxiliary_video_task(
                            task_rank=len(tasks) + 1,
                            task_type="world_transition",
                            node=node,
                            node_id=node_id,
                            refs=transition_refs,
                            output_dir=output_dir,
                            prompt_style=prompt_style,
                        )
                    )

            reference_bundle = self._node_reference_bundle(
                ref_node,
                include_character_keyframes=include_character_keyframes,
                include_world_images=include_world_images,
                max_reference_images=max(1, int(max_reference_images)),
            )
            refs = reference_bundle["reference_image_paths"]
            continuity = {
                "used_previous_tail_frame": False,
                "used_previous_audio": False,
                "uploaded_previous_audio": False,
                "previous_tail_frame_path": previous_tail_frame_path,
                "previous_audio_path": previous_audio_path,
                "generated_tail_frame_path": "",
                "generated_audio_path": "",
                "continuity_decision": False,
                "continuity_score": 0.0,
                "continuity_reason": (
                    "Previous tail-frame reference is disabled by default."
                    if not use_previous_tail_frame
                    else "No previous event tail frame is available for continuity judgment."
                ),
            }
            if (
                use_previous_tail_frame
                and previous_event_node is not None
                and previous_tail_frame_path
                and Path(previous_tail_frame_path).exists()
            ):
                decision = self._judge_previous_tail_frame(previous_event_node, node)
                continuity["continuity_decision"] = bool(decision.get("use_previous_tail_frame") is True)
                continuity["continuity_score"] = float(decision.get("continuity_score", 0.0) or 0.0)
                continuity["continuity_reason"] = str(decision.get("reason") or "")
                if continuity["continuity_decision"]:
                    refs = self._dedupe_existing(
                        [previous_tail_frame_path, *refs],
                        limit=max(1, int(max_reference_images)),
                    )
                    continuity["used_previous_tail_frame"] = previous_tail_frame_path in refs
            elif use_previous_tail_frame and previous_event_node is not None:
                continuity["continuity_reason"] = (
                    "Previous event exists, but its tail frame is unavailable; tail frame is not used."
                )
            reference_audios: list[str] = []
            if use_previous_audio and previous_audio_path and Path(previous_audio_path).exists():
                continuity["used_previous_audio"] = True
                if upload_previous_audio:
                    reference_audios = [previous_audio_path]
                    continuity["uploaded_previous_audio"] = True
            output_video_path = str(output_dir / f"event_video_{rank:03d}_{node_id}.mp4")
            prompt_node, story_backfill_result = self._backfill_event_story(
                node,
                storyline=storyline,
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                enabled=story_backfill,
            )
            prompt_node, dialogue_backfill_result = self._backfill_event_dialogue(
                prompt_node,
                storyline=storyline,
                output_dir=output_dir,
                rank=rank,
                node_id=node_id,
                enabled=dialogue_backfill,
            )
            prompt_artifact = self._event_prompt_artifact(
                prompt_node,
                storyline=storyline,
                prompt_style=prompt_style,
            )
            prompt = str(prompt_artifact["prompt"])
            shot_plan: list[dict[str, Any]] = []
            shot_tasks: list[dict[str, Any]] = []
            mode = "multi_shot_reference" if multi_shot_events else "reference"
            task = {
                "rank": rank,
                "node_id": node_id,
                "event_id": node.get("event_id", ""),
                "mode": mode,
                "prompt": prompt,
                "skill_name": prompt_artifact["skill_name"],
                "skill_kind": prompt_artifact["skill_kind"],
                "skill_source": prompt_artifact["skill_source"],
                "prompt_style": prompt_artifact["prompt_style"],
                "images": refs,
                "reference_images": refs,
                "reference_mode": reference_bundle["reference_mode"],
                "reference_selection": reference_bundle,
                "task_type": "event",
                "output_video_path": output_video_path,
                "source_steps": node.get("source_steps", []),
                "source_node_ids": node.get("source_node_ids", []),
                "characters": node.get("characters", []),
                "locations": node.get("locations", []),
                "items": node.get("items", []),
                "video_description": node.get("video_description", node.get("event_summary", "")),
                "story_description": prompt_node.get("story_description", ""),
                "dialogue": prompt_node.get("dialogue", node.get("dialogue", node.get("dialogue_text", ""))),
                "continuity_reference": continuity,
                "story_backfill": story_backfill_result,
                "dialogue_backfill": dialogue_backfill_result,
                "event_shot_plan": shot_plan,
                "shot_video_tasks": shot_tasks,
            }
            for audio_field in ("narration", "voiceover", "narration_text", "voiceover_text"):
                if audio_field in prompt_node:
                    task[audio_field] = prompt_node[audio_field]
            if multi_shot_events:
                video_plan = self._build_event_video_plan(
                    prompt_node,
                    storyline=storyline,
                    output_dir=output_dir,
                    rank=rank,
                    node_id=node_id,
                    event_shot_count=event_shot_count,
                    event_duration_sec=event_duration_sec,
                )
                shot_plan = list(video_plan.get("shots", []) or [])
                task["duration_sec"] = MINIMAX_EVENT_DURATION_SEC
                task["num_frames"] = MINIMAX_EVENT_NUM_FRAMES
                task["shot_planning"] = {k: v for k, v in video_plan.items() if k != "shots"}
                shot_tasks = []
                for shot in shot_plan:
                    shot_artifact = self._multi_shot_prompt_artifact(
                        prompt_node,
                        shot,
                        storyline=storyline,
                        prompt_style=prompt_style,
                    )
                    shot_tasks.append(
                        {
                            "rank": rank,
                            "node_id": node_id,
                            "event_id": node.get("event_id", ""),
                            "shot_index": shot.get("shot_index"),
                            "start_timecode": shot.get("start_timecode"),
                            "duration_sec": shot.get("duration_sec"),
                            "num_frames": task["num_frames"],
                            "mode": "reference",
                            "prompt": shot_artifact["prompt"],
                            "skill_name": shot_artifact["skill_name"],
                            "skill_kind": shot_artifact["skill_kind"],
                            "skill_source": shot_artifact["skill_source"],
                            "prompt_style": shot_artifact["prompt_style"],
                            "images": refs,
                            "reference_images": refs,
                            "output_video_path": "",
                            "planned_only": True,
                        }
                    )
                prompt = self._multi_shot_event_prompt_from_plan(
                    prompt_node,
                    shot_plan,
                    storyline=storyline,
                    prompt_style=prompt_style,
                )
                task["prompt"] = prompt
                task["event_shot_plan"] = shot_plan
                task["shot_video_tasks"] = shot_tasks
            task["duration_sec"] = MINIMAX_EVENT_DURATION_SEC
            task["num_frames"] = MINIMAX_EVENT_NUM_FRAMES
            task["timeline_path"] = str(write_video_timeline(task))
            tasks.append(task)
            existing_output = Path(output_video_path)
            if not dry_run and existing_output.exists() and existing_output.stat().st_size > 0:
                if use_previous_tail_frame or use_previous_audio:
                    extracted = self._extract_video_continuity_assets(
                        output_video_path,
                        str(continuity_dir / f"event_video_{rank:03d}_{node_id}_tail.jpg"),
                        str(continuity_dir / f"event_video_{rank:03d}_{node_id}_audio.wav"),
                    )
                    continuity["generated_tail_frame_path"] = extracted.get("tail_frame_path", "")
                    continuity["generated_audio_path"] = extracted.get("audio_path", "")
                    previous_tail_frame_path = continuity["generated_tail_frame_path"]
                    previous_audio_path = continuity["generated_audio_path"]
                    if previous_tail_frame_path:
                        previous_event_node = node
                success += 1
                generated.append({**task, "status": "done", "resume_existing": True})
                logger.info(
                    "video event status=skip_existing rank=%d node_id=%s output=%s",
                    rank,
                    node_id,
                    output_video_path,
                )
                completed_event_count += 1
                continue
            if not refs:
                skipped += 1
                logger.info("video event status=skip node_id=%s reason=no_reference_images", node_id)
                continue
            if dry_run:
                success += 1
                generated.append({**task, "status": "dry_run"})
                logger.info(
                    "video event status=dry_run rank=%d node_id=%s refs=%d output=%s",
                    rank,
                    node_id,
                    len(refs),
                    output_video_path,
                )
                completed_event_count += 1
                continue

            if multi_shot_events:
                ok = self.send_generation_request(
                    prompt=prompt,
                    image_paths=refs,
                    video_path=output_video_path,
                    video_mode="reference",
                    reference_image_paths=refs,
                    reference_audio_paths=reference_audios,
                    minimax_task="ref2va",
                    duration_sec=MINIMAX_EVENT_DURATION_SEC,
                    num_frames=MINIMAX_EVENT_NUM_FRAMES,
                )
            else:
                ok = self.send_generation_request(
                    prompt=prompt,
                    image_paths=refs,
                    video_path=output_video_path,
                    video_mode="reference",
                    reference_image_paths=refs,
                    reference_audio_paths=reference_audios,
                    minimax_task="ref2va",
                    duration_sec=MINIMAX_EVENT_DURATION_SEC,
                    num_frames=MINIMAX_EVENT_NUM_FRAMES,
                )
            if ok:
                extracted = self._extract_video_continuity_assets(
                    output_video_path,
                    str(continuity_dir / f"event_video_{rank:03d}_{node_id}_tail.jpg"),
                    str(continuity_dir / f"event_video_{rank:03d}_{node_id}_audio.wav"),
                )
                continuity["generated_tail_frame_path"] = extracted.get("tail_frame_path", "")
                continuity["generated_audio_path"] = extracted.get("audio_path", "")
                previous_tail_frame_path = continuity["generated_tail_frame_path"]
                previous_audio_path = continuity["generated_audio_path"]
                if previous_tail_frame_path:
                    previous_event_node = node
                success += 1
                generated.append({**task, "status": "done"})
                completed_event_count += 1
                logger.info(
                    "video event status=done rank=%d node_id=%s refs=%d output=%s",
                    rank,
                    node_id,
                    len(refs),
                    output_video_path,
                )
            else:
                failed += 1
                generated.append({**task, "status": "failed"})
                logger.info(
                    "video event status=failed rank=%d node_id=%s refs=%d output=%s",
                    rank,
                    node_id,
                    len(refs),
                    output_video_path,
                )

        if save_task_json:
            (output_dir / "event_video_tasks.json").write_text(
                json.dumps(tasks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary = {
            "status": "done" if failed == 0 else "partial",
            "narrative_nodes_json": str(narrative_nodes_json),
            "path_plan_json": str(path_plan_json),
            "output_dir": str(output_dir),
            "total": len(tasks),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "generated": generated,
            "task_json": str(output_dir / "event_video_tasks.json") if save_task_json else "",
        }
        (output_dir / "event_video_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "video process_orchestrated_events status=done total=%d success=%d failed=%d skipped=%d",
            len(tasks),
            success,
            failed,
            skipped,
        )
        return summary

    def process_task_file(self, task_json_path: Path, strict_images: bool = False) -> dict[str, int]:
        if not task_json_path.exists():
            raise FileNotFoundError(f"Task JSON not found: {task_json_path}")

        tasks = json.loads(task_json_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise RuntimeError("Task JSON must be a list")

        success = 0
        failed = 0
        skipped = 0

        for task in tasks:
            if not isinstance(task, dict):
                failed += 1
                continue

            prompt = self._safe_text(task.get("prompt", "")).strip()
            mode = self._safe_text(task.get("video_mode", task.get("mode", "reference"))).strip().lower() or "reference"
            minimax_task = self._safe_text(task.get("minimax_task", task.get("task", ""))).strip() or None
            task_num_frames = task.get("num_frames")
            try:
                task_num_frames = int(task_num_frames) if task_num_frames else None
            except Exception:
                task_num_frames = None
            images = task.get("images", task.get("image_paths", []))
            out = self._safe_text(task.get("output_video_path", "")).strip()
            if not prompt or not out:
                failed += 1
                continue

            if not isinstance(images, list):
                images = [self._safe_text(images)]
            image_paths = [self._safe_text(p) for p in images if self._safe_text(p)]
            first_image = self._safe_text(
                task.get("first_image", task.get("image", task.get("first_image_path", "")))
            ).strip()
            last_image = self._safe_text(task.get("last_image", task.get("last_image_path", ""))).strip()
            ref_images = task.get("reference_images", task.get("ref_images", task.get("ref_image", [])))
            ref_videos = task.get("reference_videos", task.get("ref_videos", task.get("ref_video", [])))
            ref_audios = task.get("reference_audios", task.get("ref_audios", task.get("ref_audio", [])))
            if not isinstance(ref_images, list):
                ref_images = [self._safe_text(ref_images)]
            if not isinstance(ref_videos, list):
                ref_videos = [self._safe_text(ref_videos)]
            if not isinstance(ref_audios, list):
                ref_audios = [self._safe_text(ref_audios)]

            existing = [p for p in image_paths if Path(p).exists()]
            existing_refs = [self._safe_text(p) for p in ref_images if self._safe_text(p) and Path(self._safe_text(p)).exists()]
            existing_ref_videos = [self._safe_text(p) for p in ref_videos if self._safe_text(p) and Path(self._safe_text(p)).exists()]
            existing_ref_audios = [self._safe_text(p) for p in ref_audios if self._safe_text(p) and Path(self._safe_text(p)).exists()]
            if strict_images and mode != "text" and not (existing or existing_refs or first_image or last_image):
                skipped += 1
                continue

            ok = self.send_generation_request(
                prompt=prompt,
                image_paths=existing,
                video_path=out,
                video_mode=mode,
                first_image_path=first_image,
                last_image_path=last_image,
                reference_image_paths=existing_refs,
                reference_video_paths=existing_ref_videos,
                reference_audio_paths=existing_ref_audios,
                minimax_task=minimax_task,
                num_frames=task_num_frames,
            )
            if ok:
                success += 1
                logger.info("video process_task status=done output=%s", out)
            else:
                failed += 1
                logger.info("video process_task status=failed output=%s", out)

        logger.info("video process_task_file status=done total=%d success=%d failed=%d skipped=%d", len(tasks), success, failed, skipped)
        return {"total": len(tasks), "success": success, "failed": failed, "skipped": skipped}

    def concat_videos(self, video_paths: list[str], output_path: str | Path) -> bool:
        output_path = Path(output_path).resolve()
        mp4_list = [str(Path(p).resolve()) for p in video_paths if p and Path(p).exists()]
        if not mp4_list:
            logger.info("video concat_videos status=skip reason=no_inputs")
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(mp4_list) == 1:
            shutil.copyfile(mp4_list[0], output_path)
            logger.info("video concat_videos status=done output=%s inputs=1 mode=copy", str(output_path))
            return True

        list_file = output_path.parent / "concat_list.txt"
        list_file.write_text("".join(f"file '{p}'\n" for p in mp4_list), encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("video concat_videos status=done output=%s inputs=%d", str(output_path), len(mp4_list))
            return True
        except Exception:
            logger.exception("video concat_videos status=failed output=%s", str(output_path))
            return False
        finally:
            if list_file.exists():
                list_file.unlink()
