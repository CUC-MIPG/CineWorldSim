from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .client import make_client, pick_model
from .prompts import load_prompt_library_query_dict, build_shot_creation_system_prompt
from .retry import call_with_retries
from .schemas import Shot, Storyboard

logger = logging.getLogger(__name__)


class ShotAgent:
    """Generate shot-level storyboard using loop strategy from CameraAtist."""

    def __init__(
        self,
        prompt_library: Path,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ):
        self.prompt_library = prompt_library
        self.query_dict = load_prompt_library_query_dict(prompt_library)
        self.client = make_client(scope="SHOT")
        self.model = model or pick_model("gpt-5.5", scope="SHOT")
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
    def _json_load_maybe(v: Any) -> Any:
        if isinstance(v, (dict, list)):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v

    def _normalize_shots_payload(self, shot_obj: dict[str, Any], scene_id: str) -> dict[str, Any]:
        shots = self._json_load_maybe(shot_obj.get("Shots", {}))

        # CameraAtist loop prompt usually returns a single shot payload in "Shots".
        # Some models may wrap it into Shot1/Shot2-like keys; pick the first valid one.
        if isinstance(shots, dict):
            if "content" not in shots:
                picked = None
                for key in sorted(shots.keys()):
                    if key.lower().startswith("shot"):
                        picked = self._json_load_maybe(shots[key])
                        break
                if picked is not None:
                    shots = picked
                else:
                    shots = {"content": self._safe_text(shots)}
        elif isinstance(shots, list):
            shots = {"content": " ".join(self._safe_text(x) for x in shots)}
        else:
            shots = {"content": self._safe_text(shots)}

        if not isinstance(shots, dict):
            shots = {"content": self._safe_text(shots)}

        shots["scene"] = scene_id
        if not shots.get("Shot_Location"):
            shots["Shot_Location"] = "Scene midpoint"

        return {"Shots": shots}

    @staticmethod
    def _infer_shot_type(text: str) -> str:
        t = text.lower()
        rules = [
            (("establishing", "全景", "远景"), "Establishing Shot"),
            (("close-up", "close up", "特写"), "Close-up"),
            (("medium shot", "中景"), "Medium Shot"),
            (("wide shot", "long shot", "大全景"), "Wide Shot"),
            (("over-the-shoulder", "肩", "越肩"), "Over-the-Shoulder"),
            (("pov", "第一视角"), "POV"),
            (("tracking", "跟拍"), "Tracking Shot"),
        ]
        for keywords, label in rules:
            if any(k in t for k in keywords):
                return label
        return "Narrative Shot"

    @staticmethod
    def _infer_camera_angle(text: str) -> str:
        t = text.lower()
        rules = [
            (("low angle", "仰拍"), "Low Angle"),
            (("high angle", "俯拍"), "High Angle"),
            (("bird's-eye", "birds-eye", "鸟瞰"), "Bird's-eye View"),
            (("eye-level", "平视"), "Eye-level"),
            (("dutch", "倾斜"), "Dutch Angle"),
        ]
        for keywords, label in rules:
            if any(k in t for k in keywords):
                return label
        return "Eye-level"

    @staticmethod
    def _infer_camera_movement(text: str) -> str:
        t = text.lower()
        rules = [
            (("pan", "横移"), "Pan"),
            (("tilt", "俯仰"), "Tilt"),
            (("dolly", "推拉"), "Dolly"),
            (("zoom", "变焦"), "Zoom"),
            (("tracking", "跟拍"), "Tracking"),
            (("handheld", "手持"), "Handheld"),
            (("crane", "摇臂"), "Crane"),
            (("static", "固定机位"), "Static"),
        ]
        for keywords, label in rules:
            if any(k in t for k in keywords):
                return label
        return "Smooth Follow"

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip())

    def _extract_character_names(self, text: str, known_names: list[str]) -> list[str]:
        if not known_names:
            return []

        found: list[str] = []
        text_lc = text.lower()
        for name in known_names:
            n = self._normalize_name(name)
            if not n:
                continue
            n_lc = n.lower()
            # Full-name match first; then token match for multi-token names.
            if n_lc in text_lc:
                found.append(name)
                continue
            tokens = [x for x in re.split(r"\s+", n_lc) if len(x) > 2]
            if tokens and any(tok in text_lc for tok in tokens):
                found.append(name)

        dedup: list[str] = []
        seen: set[str] = set()
        for n in found:
            if n not in seen:
                seen.add(n)
                dedup.append(n)
        return dedup

    def _validate_and_repair_shot(
        self,
        shot: Shot,
        scene_text: str,
        known_character_names: list[str],
    ) -> Shot:
        base_text = " ".join(
            [
                self._safe_text(shot.prompt),
                self._safe_text(shot.scene_elements),
                self._safe_text(scene_text),
            ]
        )

        if not shot.shot_type:
            shot.shot_type = self._infer_shot_type(base_text)
        if not shot.camera_angle:
            shot.camera_angle = self._infer_camera_angle(base_text)
        if not shot.camera_movement:
            shot.camera_movement = self._infer_camera_movement(base_text)

        if isinstance(shot.scene_elements, str):
            parsed = self._json_load_maybe(shot.scene_elements)
            if isinstance(parsed, dict):
                shot.scene_elements = parsed
            else:
                shot.scene_elements = {"content": self._safe_text(parsed)}
        elif not isinstance(shot.scene_elements, dict):
            shot.scene_elements = {"content": self._safe_text(shot.scene_elements)}

        if "content" not in shot.scene_elements and shot.prompt:
            shot.scene_elements["content"] = shot.prompt
        if "scene_context" not in shot.scene_elements:
            shot.scene_elements["scene_context"] = scene_text

        if not shot.prompt:
            shot.prompt = self._safe_text(shot.scene_elements.get("content", "")).strip()
        if not shot.prompt:
            shot.prompt = base_text.strip()

        if not shot.character_names:
            shot.character_names = self._extract_character_names(base_text, known_character_names)

        if not shot.shot_location:
            shot.shot_location = "Scene midpoint"

        return shot

    def slow_shot_cot(
        self,
        story: str,
        scene_text: str,
        before_shots: str,
        scene_id: str,
    ) -> dict[str, Any]:
        query_prompt = (
            f"**StoryLine**:{story},\n"
            f"**scene**:{scene_text},\n"
            f"**Before_Shots**:{before_shots}"
        )

        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_shot_creation_system_prompt(self.query_dict)},
                    {"role": "user", "content": query_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        try:
            shot_obj = call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
        except Exception:
            if self.strict_online:
                raise
            # Offline fallback to keep the pipeline runnable.
            fallback_loc = "Scene End Point" if before_shots != "None" else "Scene Start Point"
            shot_obj = {
                "Shots": {
                    "content": f"{scene_text}. Camera follows the key action with cinematic clarity.",
                    "scene": scene_id,
                    "Shot_Location": fallback_loc,
                }
            }
        return self._normalize_shots_payload(shot_obj, scene_id)

    def _convert_to_shot(
        self,
        scene_id: str,
        sequence: int,
        scene_text: str,
        known_character_names: list[str],
        shot_obj: dict[str, Any],
    ) -> Shot:
        shots = shot_obj.get("Shots", {})
        content = self._safe_text(shots.get("content", ""))
        shot_location = self._safe_text(shots.get("Shot_Location", "Scene midpoint"))
        character_names = self._json_load_maybe(shots.get("character_names", []))
        if not isinstance(character_names, list):
            character_names = [self._safe_text(character_names)]

        scene_elements = self._json_load_maybe(shots.get("scene_elements", {}))
        if not isinstance(scene_elements, dict):
            scene_elements = {"content": self._safe_text(scene_elements)}
        if content and "content" not in scene_elements:
            scene_elements["content"] = content

        shot = Shot(
            shot_id=f"{scene_id}_SH{sequence:03d}",
            scene_id=scene_id,
            sequence=sequence,
            shot_type=self._safe_text(shots.get("shot_type", "")),
            camera_angle=self._safe_text(shots.get("camera_angle", "")),
            scene_elements=scene_elements,
            camera_movement=self._safe_text(shots.get("camera_movement", "")),
            character_names=[self._safe_text(x) for x in character_names if self._safe_text(x)],
            prompt=content,
            shot_location=shot_location,
        )
        return self._validate_and_repair_shot(
            shot=shot,
            scene_text=scene_text,
            known_character_names=known_character_names,
        )

    def loop_storyboard_generation(
        self,
        storyboard: Storyboard,
        max_shots_per_scene: int = 10,
    ) -> Storyboard:
        logger.info("shot loop_storyboard_generation status=start max_shots_per_scene=%d", max_shots_per_scene)
        story_text = "\n".join(p.description for p in storyboard.screenplay.plot_points) or storyboard.title
        known_character_names = list(storyboard.screenplay.characters.keys())

        for scene_id in sorted(storyboard.scenes.keys()):
            scene = storyboard.scenes[scene_id]
            scene_text = scene.plot_description or f"{scene.location} at {scene.time_of_day}"
            logger.info("shot scene status=start scene_id=%s", scene_id)
            shots_buffer: list[dict[str, Any]] = []
            end_flag = False
            idx = 0

            while (not end_flag) and idx < max_shots_per_scene:
                before_shot = "None" if not shots_buffer else ",".join(json.dumps(x, ensure_ascii=False) for x in shots_buffer)
                one = self.slow_shot_cot(
                    story=story_text,
                    scene_text=scene_text,
                    before_shots=before_shot,
                    scene_id=scene_id,
                )
                shots_buffer.append(one)
                idx += 1
                logger.info("shot scene_id=%s generated_shot_idx=%d", scene_id, idx)

                tail_loc = self._safe_text(one.get("Shots", {}).get("Shot_Location", ""))
                tail_content = self._safe_text(one.get("Shots", {}).get("content", ""))
                if "scene end point" in tail_loc.lower() or "scene end point" in tail_content.lower():
                    end_flag = True

            for seq, one in enumerate(shots_buffer, start=1):
                shot = self._convert_to_shot(
                    scene_id=scene_id,
                    sequence=seq,
                    scene_text=scene_text,
                    known_character_names=known_character_names,
                    shot_obj=one,
                )
                storyboard.shots[shot.shot_id] = shot
            logger.info("shot scene status=done scene_id=%s shots=%d", scene_id, len(shots_buffer))

        logger.info("shot loop_storyboard_generation status=done total_shots=%d", len(storyboard.shots))
        return storyboard
