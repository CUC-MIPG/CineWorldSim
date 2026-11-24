from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.shot_agent import ShotAgent
from agents.schemas import Character, Shot, Storyboard

logger = logging.getLogger(__name__)


class ShotAgentDynamic:
    """Shot agent enhanced with Co-Director style creative direction and asset bindings."""

    def __init__(
        self,
        prompt_library: Path,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ) -> None:
        self.base_shot_agent = ShotAgent(
            prompt_library=prompt_library,
            model=model,
            strict_online=strict_online,
            retries=retries,
        )

    @staticmethod
    def _match_characters(shot: Shot, characters: dict[str, Character]) -> list[Character]:
        out: list[Character] = []
        for n in shot.character_names:
            if n in characters:
                out.append(characters[n])
        return out

    def _collect_asset_bindings(self, shot: Shot, storyboard: Storyboard) -> list[str]:
        refs: list[str] = []
        scene = storyboard.scenes.get(shot.scene_id)
        if scene:
            refs.extend([p for p in scene.reference_images if p and Path(p).exists()])

        matched = self._match_characters(shot, storyboard.screenplay.characters)
        for c in matched:
            if c.reference_image and Path(c.reference_image).exists():
                refs.append(c.reference_image)
            refs.extend([p for p in c.reference_images if p and Path(p).exists()])

        uniq: list[str] = []
        seen: set[str] = set()
        for p in refs:
            r = str(Path(p).resolve())
            if r in seen:
                continue
            seen.add(r)
            uniq.append(r)
        return uniq

    def loop_storyboard_generation(
        self,
        storyboard: Storyboard,
        max_shots_per_scene: int = 8,
        creative_direction: dict[str, Any] | None = None,
    ) -> Storyboard:
        storyboard = self.base_shot_agent.loop_storyboard_generation(
            storyboard,
            max_shots_per_scene=max_shots_per_scene,
        )

        direction_text = ""
        if isinstance(creative_direction, dict):
            direction_text = " ".join(
                [
                    str(creative_direction.get("shot_instruction", "")).strip(),
                    str(creative_direction.get("visual_instruction", "")).strip(),
                ]
            ).strip()

        for shot in storyboard.shots.values():
            bindings = self._collect_asset_bindings(shot, storyboard)
            shot.scene_elements = shot.scene_elements if isinstance(shot.scene_elements, dict) else {}
            shot.scene_elements["asset_bindings"] = bindings
            if creative_direction:
                shot.scene_elements["creative_direction"] = json.loads(json.dumps(creative_direction, ensure_ascii=False))
            if direction_text:
                base_prompt = shot.auto_prompt()
                shot.prompt = f"{base_prompt}\nCreative Direction: {direction_text}".strip()
        logger.info("dynamic_shot status=done shots=%d", len(storyboard.shots))
        return storyboard
