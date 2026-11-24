from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import re


@dataclass
class Character:
    name: str
    appearance: str = ""
    traits: list[str] = field(default_factory=list)
    gender: str = "unknown"
    voice_profile: str = ""
    reference_image: str | None = None
    reference_images: list[str] = field(default_factory=list)
    reference_audio: str | None = None
    reference_audios: list[str] = field(default_factory=list)
    reference_audio_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        if not isinstance(data, dict):
            data = json.loads(data)
        ref_images = list(data.get("reference_images", []))
        ref_image = data.get("reference_image")
        if ref_image and ref_image not in ref_images:
            ref_images.insert(0, ref_image)
        ref_audios = list(data.get("reference_audios", []))
        ref_audio = data.get("reference_audio")
        if ref_audio and ref_audio not in ref_audios:
            ref_audios.insert(0, ref_audio)
        gender = str(data.get("gender", data.get("sex", "unknown")) or "unknown").strip().lower()
        if gender not in {"male", "female", "unknown"}:
            gender = "unknown"
        return cls(
            name=data.get("name", ""),
            appearance=data.get("appearance", ""),
            traits=list(data.get("traits", [])),
            gender=gender,
            voice_profile=str(data.get("voice_profile", "") or "").strip(),
            reference_image=ref_image,
            reference_images=ref_images,
            reference_audio=ref_audio,
            reference_audios=ref_audios,
            reference_audio_text=str(data.get("reference_audio_text", "") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlotPoint:
    description: str
    impact: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotPoint":
        if not isinstance(data, dict):
            data = json.loads(data)
        return cls(
            description=data.get("description", ""),
            impact=data.get("impact", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Screenplay:
    title: str
    genre: str = ""
    plot_points: list[PlotPoint] = field(default_factory=list)
    characters: dict[str, Character] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Screenplay":
        sp = cls(
            title=data.get("title", "Untitled"),
            genre=data.get("genre", ""),
        )
        for p in data.get("plot_points", []):
            sp.plot_points.append(PlotPoint.from_dict(p))
        for c in data.get("characters", []):
            ch = Character.from_dict(c)
            sp.characters[ch.name] = ch
        return sp

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "genre": self.genre,
            "plot_points": [p.to_dict() for p in self.plot_points],
            "characters": [c.to_dict() for c in self.characters.values()],
        }


@dataclass
class Scene:
    scene_id: str
    location: str = ""
    time_of_day: str = ""
    plot_description: str = ""
    reference_images: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            scene_id=str(data.get("scene_id", "")),
            location=data.get("location", ""),
            time_of_day=data.get("time_of_day", ""),
            plot_description=data.get("plot_description", ""),
            reference_images=list(data.get("reference_images", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Shot:
    shot_id: str
    scene_id: str
    sequence: int
    shot_type: str = ""
    camera_angle: str = ""
    scene_elements: dict[str, Any] = field(default_factory=dict)
    camera_movement: str = ""
    character_names: list[str] = field(default_factory=list)
    prompt: str = ""
    shot_location: str = ""
    video_path: str | None = None
    path_sequence: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shot":
        if not isinstance(data, dict):
            data = json.loads(data)
        scene_elements = data.get("scene_elements", {})
        if isinstance(scene_elements, str):
            try:
                scene_elements = json.loads(scene_elements)
            except Exception:
                scene_elements = {"raw": scene_elements}

        return cls(
            shot_id=str(data.get("shot_id", "")),
            scene_id=str(data.get("scene_id", "")),
            sequence=int(data.get("sequence", 0)),
            shot_type=data.get("shot_type", ""),
            camera_angle=data.get("camera_angle", ""),
            scene_elements=scene_elements,
            camera_movement=data.get("camera_movement", ""),
            character_names=list(data.get("character_names", [])),
            prompt=data.get("prompt", ""),
            shot_location=data.get("Shot_Location", data.get("shot_location", "")),
            video_path=data.get("video_path"),
            path_sequence=int(data.get("path_sequence", data.get("global_sequence", 0)) or 0),
        )

    def auto_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        elements_desc = ", ".join(f"{k}: {v}" for k, v in self.scene_elements.items())
        self.prompt = f"{self.shot_type}, {self.camera_angle}, elements: {elements_desc}, movement: {self.camera_movement}"
        return self.prompt

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["Shot_Location"] = d.pop("shot_location")
        return d


@dataclass
class Storyboard:
    title: str
    screenplay: Screenplay
    scenes: dict[str, Scene] = field(default_factory=dict)
    shots: dict[str, Shot] = field(default_factory=dict)
    path_sequence: list[str] = field(default_factory=list)
    path_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Storyboard":
        screenplay = Screenplay.from_dict(data.get("screenplay", {}))
        sb = cls(title=data.get("title", screenplay.title), screenplay=screenplay)

        for s in data.get("scenes", []):
            scene = Scene.from_dict(s)
            sb.scenes[scene.scene_id] = scene

        for sh in data.get("shots", []):
            shot = Shot.from_dict(sh)
            sb.shots[shot.shot_id] = shot

        path_sequence = data.get("path_sequence", [])
        if isinstance(path_sequence, list):
            sb.path_sequence = [str(x) for x in path_sequence if str(x).strip()]
        path_metadata = data.get("path_metadata", {})
        if isinstance(path_metadata, dict):
            sb.path_metadata = path_metadata

        return sb

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "screenplay": self.screenplay.to_dict(),
            "scenes": [self.scenes[k].to_dict() for k in sorted(self.scenes.keys())],
            "shots": [
                s.to_dict()
                for s in self.ordered_shots_scene_first()
            ],
            "path_sequence": list(self.path_sequence),
            "path_metadata": dict(self.path_metadata),
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def get_scene_shots(self, scene_id: str) -> list[Shot]:
        return sorted(
            [s for s in self.shots.values() if s.scene_id == scene_id],
            key=lambda x: x.sequence,
        )
    @staticmethod
    def _is_canonical_scene_shot_id(shot_id: str, scene_id: str) -> bool:
        return str(shot_id or "").upper().startswith(f"{str(scene_id or '').upper()}_SH")

    @staticmethod
    def _scene_sort_key(scene_id: str) -> tuple[int, str]:
        text = str(scene_id or "")
        m = re.search(r"(\d+)", text)
        if m:
            return int(m.group(1)), text
        return 10**9, text

    def ordered_shots_scene_first(self, unique_by_scene_sequence: bool = False) -> list[Shot]:
        ordered = sorted(
            self.shots.values(),
            key=lambda s: (
                self._scene_sort_key(s.scene_id),
                int(s.sequence),
                0 if self._is_canonical_scene_shot_id(s.shot_id, s.scene_id) else 1,
                str(s.shot_id),
            ),
        )
        if not unique_by_scene_sequence:
            return ordered

        deduped: list[Shot] = []
        seen: set[tuple[str, int]] = set()
        for shot in ordered:
            key = (str(shot.scene_id), int(shot.sequence))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(shot)
        return deduped

    def ordered_shots_path_first(self, unique_by_scene_sequence: bool = False) -> list[Shot]:
        has_path = any(int(getattr(s, "path_sequence", 0) or 0) > 0 for s in self.shots.values())
        if not has_path:
            return self.ordered_shots_scene_first(unique_by_scene_sequence=unique_by_scene_sequence)

        ordered = sorted(
            self.shots.values(),
            key=lambda s: (
                int(s.path_sequence) if int(s.path_sequence or 0) > 0 else 10**9,
                self._scene_sort_key(s.scene_id),
                int(s.sequence),
                0 if self._is_canonical_scene_shot_id(s.shot_id, s.scene_id) else 1,
                str(s.shot_id),
            ),
        )
        if not unique_by_scene_sequence:
            return ordered

        deduped: list[Shot] = []
        seen: set[tuple[str, int]] = set()
        for shot in ordered:
            key = (str(shot.scene_id), int(shot.sequence))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(shot)
        return deduped
