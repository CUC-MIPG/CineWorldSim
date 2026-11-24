from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .client import make_client, pick_model
from .generation_backends import (
    image_provider_from_env,
    flux_settings_from_env,
    nano_banana_settings_from_env,
    generate_image_flux,
    generate_image_nano_banana,
)
from .retry import call_with_retries
from .prompts import (
    load_prompt_library_query_dict,
    build_scene_creation_system_prompt,
    build_story_design_system_prompt,
)
from .schemas import Character, PlotPoint, Scene, Screenplay, Storyboard

logger = logging.getLogger(__name__)


class DirectorAgent:
    """Generate story + scene plan, then generate scene reference images."""

    def __init__(
        self,
        prompt_library: Path,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ):
        self.prompt_library = prompt_library
        self.query_dict = load_prompt_library_query_dict(prompt_library)
        self.client = make_client(scope="DIRECTOR")
        self.model = model or pick_model("gpt-5.5", scope="DIRECTOR")
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
    def _json_load_maybe(text_or_obj: Any) -> Any:
        if isinstance(text_or_obj, (dict, list)):
            return text_or_obj

    @staticmethod
    def _normalize_gender(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"male", "m", "man", "boy", "男", "男性"}:
            return "male"
        if raw in {"female", "f", "woman", "girl", "女", "女性"}:
            return "female"
        return "unknown"

    def _character_from_fast(self, name: str, raw_value: Any) -> Character:
        data = self._json_load_maybe(raw_value)
        def _infer_gender_from_text(txt: str) -> str:
            t = (txt or "").lower()
            if any(x in t for x in ["female", "woman", "girl", "her ", " she "]):
                return "female"
            if any(x in t for x in ["male", "man", "boy", "his ", " he "]):
                return "male"
            if any(x in txt for x in ["女性", "女孩", "女人", "她"]):
                return "female"
            if any(x in txt for x in ["男性", "男孩", "男人", "他"]):
                return "male"
            return "unknown"
        if isinstance(data, dict):
            appearance = self._safe_text(data.get("appearance", data.get("description", ""))).strip()
            traits = data.get("traits", [])
            if not isinstance(traits, list):
                traits = [self._safe_text(traits)] if self._safe_text(traits) else []
            gender = self._normalize_gender(data.get("gender", data.get("sex", "unknown")))
            if gender == "unknown":
                gender = _infer_gender_from_text(appearance or self._safe_text(raw_value))
            voice_profile = self._safe_text(data.get("voice_profile", "")).strip()
            return Character(
                name=name,
                appearance=appearance or self._safe_text(raw_value),
                traits=[self._safe_text(x) for x in traits if self._safe_text(x)],
                gender=gender,
                voice_profile=voice_profile,
            )

        appearance_text = self._safe_text(raw_value)
        return Character(
            name=name,
            appearance=appearance_text,
            traits=[],
            gender=_infer_gender_from_text(appearance_text),
        )
        if not isinstance(text_or_obj, str):
            return text_or_obj
        try:
            return json.loads(text_or_obj)
        except Exception:
            return text_or_obj

    def _extract_fast_story_parts(self, storyline: str, fast_obj: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        story = self._safe_text(fast_obj.get("Story", storyline)).strip() or storyline
        scenes = self._json_load_maybe(fast_obj.get("Scenes", {}))
        chars = self._json_load_maybe(fast_obj.get("Characters", {}))
        if not isinstance(scenes, dict):
            scenes = {}
        if not isinstance(chars, dict):
            chars = {}
        return story, scenes, chars

    def _merge_storyboard(
        self,
        storyline: str,
        fast_obj: dict[str, Any],
        structured: dict[str, Any] | None,
    ) -> Storyboard:
        story, fast_scenes, fast_chars = self._extract_fast_story_parts(storyline, fast_obj)
        merged = Storyboard(
            title="Untitled Story",
            screenplay=Screenplay(title="Untitled Story", genre="cinematic"),
        )

        # 1) Start from structured output when available.
        if structured:
            try:
                parsed = Storyboard.from_dict(structured)
                merged = parsed
            except Exception:
                pass

        # 2) Patch screenplay basics.
        merged.title = merged.title or merged.screenplay.title or "Untitled Story"
        merged.screenplay.title = merged.screenplay.title or merged.title
        if not merged.screenplay.genre:
            merged.screenplay.genre = "cinematic"
        if not merged.screenplay.plot_points:
            merged.screenplay.plot_points.append(PlotPoint(description=story[:500], impact="storyline"))

        # 3) Merge characters from fast stage if missing.
        for name, desc in fast_chars.items():
            if not name:
                continue
            if name not in merged.screenplay.characters:
                merged.screenplay.characters[name] = self._character_from_fast(name, desc)
            else:
                c = merged.screenplay.characters[name]
                inc = self._character_from_fast(name, desc)
                if not c.appearance and inc.appearance:
                    c.appearance = inc.appearance
                if (c.gender or "unknown") == "unknown" and inc.gender != "unknown":
                    c.gender = inc.gender
                if not c.voice_profile and inc.voice_profile:
                    c.voice_profile = inc.voice_profile
                if inc.traits:
                    merged_traits = list(c.traits)
                    for t in inc.traits:
                        if t not in merged_traits:
                            merged_traits.append(t)
                    c.traits = merged_traits

        # 4) Merge scenes from fast stage to guarantee scene coverage.
        if isinstance(fast_scenes, dict):
            next_idx = len(merged.scenes) + 1
            existing_desc = {
                (s.plot_description or "").strip().lower(): s.scene_id
                for s in merged.scenes.values()
            }
            for _, raw in sorted(fast_scenes.items()):
                s_obj = self._json_load_maybe(raw)
                if isinstance(s_obj, dict):
                    name = self._safe_text(s_obj.get("Name", f"Scene {next_idx}")).strip() or f"Scene {next_idx}"
                    desc = self._safe_text(s_obj.get("Description", s_obj)).strip()
                    fn = self._safe_text(s_obj.get("Function", "")).strip()
                    if fn and fn.lower() not in desc.lower():
                        desc = f"{desc}\nFunction: {fn}".strip()
                else:
                    name = f"Scene {next_idx}"
                    desc = self._safe_text(s_obj)

                key = desc.lower()
                if key and key in existing_desc:
                    continue

                scene_id = f"SC{next_idx:03d}"
                while scene_id in merged.scenes:
                    next_idx += 1
                    scene_id = f"SC{next_idx:03d}"

                merged.scenes[scene_id] = Scene(
                    scene_id=scene_id,
                    location=name,
                    time_of_day="unknown",
                    plot_description=desc,
                )
                existing_desc[key] = scene_id
                next_idx += 1

        # 5) Final hard fallback: still no scenes.
        if not merged.scenes:
            merged = self._fallback_storyboard(storyline, fast_obj)

        return merged

    def fast_story_cot(self, storyline: str) -> dict[str, Any]:
        system_prompt = build_scene_creation_system_prompt(self.query_dict)
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": storyline},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            obj = json.loads(raw)
            obj["Scenes"] = self._json_load_maybe(obj.get("Scenes", {}))
            obj["Characters"] = self._json_load_maybe(obj.get("Characters", {}))
            return obj
        try:
            return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
        except Exception:
            if self.strict_online:
                raise
            return {
                "Story": storyline,
                "Characters": {
                    "Protagonist": "A determined central character.",
                    "Support": "A key ally in the mission.",
                },
                "Scenes": {
                    "scene1": {
                        "Name": "Opening Situation",
                        "Description": "Set up the world, motivation, and first conflict.",
                    },
                    "scene2": {
                        "Name": "Midpoint Escalation",
                        "Description": "Pressure rises and stakes become personal.",
                    },
                    "scene3": {
                        "Name": "Climax and Resolution",
                        "Description": "Final confrontation and emotional payoff.",
                    },
                },
            }

    def story_design_structured(self, screenplay_text: str) -> dict[str, Any]:
        system_prompt = build_story_design_system_prompt(self.query_dict)
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": screenplay_text},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)

    def _fallback_storyboard(self, storyline: str, fast_obj: dict[str, Any]) -> Storyboard:
        title = "Untitled Story"
        story = fast_obj.get("Story", storyline)
        scenes = fast_obj.get("Scenes", {})
        characters = fast_obj.get("Characters", {})

        screenplay = Screenplay(title=title, genre="cinematic")
        screenplay.plot_points.append(PlotPoint(description=story[:220], impact="storyline"))

        if isinstance(characters, dict):
            for name, desc in characters.items():
                screenplay.characters[name] = self._character_from_fast(name, desc)

        sb = Storyboard(title=title, screenplay=screenplay)
        if isinstance(scenes, dict):
            for idx, key in enumerate(sorted(scenes.keys()), start=1):
                s = scenes[key]
                s_obj = self._json_load_maybe(s)
                if isinstance(s_obj, dict):
                    name = s_obj.get("Name", f"Scene {idx}")
                    desc = s_obj.get("Description", str(s_obj))
                else:
                    name = f"Scene {idx}"
                    desc = str(s_obj)
                scene_id = f"SC{idx:03d}"
                sb.scenes[scene_id] = Scene(
                    scene_id=scene_id,
                    location=name,
                    time_of_day="unknown",
                    plot_description=desc,
                )
        return sb

    def build_storyboard(self, storyline: str) -> Storyboard:
        logger.info("director build_storyboard status=start")
        fast_obj = self.fast_story_cot(storyline)
        screenplay_text = json.dumps(fast_obj, ensure_ascii=False)

        try:
            structured = self.story_design_structured(screenplay_text)
            storyboard = self._merge_storyboard(storyline, fast_obj, structured)
        except Exception:
            storyboard = self._merge_storyboard(storyline, fast_obj, None)
        logger.info(
            "director build_storyboard status=done scenes=%d characters=%d",
            len(storyboard.scenes),
            len(storyboard.screenplay.characters),
        )
        return storyboard

    def _flux_api_url(self) -> str:
        return os.getenv("FLUX_API_URL", "http://127.0.0.1:5010/flux")

    def _generate_scene_image(self, scene: Scene, genre: str, out_path: Path) -> str | None:
        prompt = (
            f"A scene from {scene.location} at {scene.time_of_day}, "
            f"plot: {scene.plot_description}, in the style of {genre}, "
            "highly detailed environment, atmospheric lighting, wide shot"
        )

        try:
            provider = image_provider_from_env()
            if provider == "nano_banana":
                cfg = nano_banana_settings_from_env()
                if not cfg["api_key"]:
                    raise RuntimeError("NANO_BANANA_API_KEY is required when IMAGE_GEN_PROVIDER=nano_banana")
                return generate_image_nano_banana(
                    prompt=prompt,
                    output_path=out_path,
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    model_name=cfg["model_name"],
                    timeout_sec=cfg["timeout_sec"],
                )

            cfg = flux_settings_from_env()
            return generate_image_flux(
                prompt=prompt,
                output_path=out_path,
                api_url=cfg["api_url"],
                width=cfg["width"],
                height=cfg["height"],
                timeout_sec=cfg["timeout_sec"],
            )
        except Exception:
            return None

    def _generate_character_image(self, character: Character, genre: str, out_path: Path) -> str | None:
        traits = ", ".join(character.traits) if character.traits else "cinematic presence"
        prompt = (
            f"Character portrait of {character.name}. "
            f"Appearance: {character.appearance}. Traits: {traits}. "
            f"Style: {genre}, high-detail concept art, clean readable silhouette, studio lighting."
        )
        try:
            provider = image_provider_from_env()
            if provider == "nano_banana":
                cfg = nano_banana_settings_from_env()
                if not cfg["api_key"]:
                    raise RuntimeError("NANO_BANANA_API_KEY is required when IMAGE_GEN_PROVIDER=nano_banana")
                return generate_image_nano_banana(
                    prompt=prompt,
                    output_path=out_path,
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    model_name=cfg["model_name"],
                    timeout_sec=cfg["timeout_sec"],
                )

            cfg = flux_settings_from_env()
            return generate_image_flux(
                prompt=prompt,
                output_path=out_path,
                api_url=cfg["api_url"],
                width=cfg["width"],
                height=cfg["height"],
                timeout_sec=cfg["timeout_sec"],
            )
        except Exception:
            return None

    def generate_scene_images(self, storyboard: Storyboard, image_dir: Path) -> Storyboard:
        for scene_id, scene in sorted(storyboard.scenes.items()):
            logger.info("director scene_image scene_id=%s status=start", scene_id)
            out_path = image_dir / f"scene_{scene_id}.png"
            img = self._generate_scene_image(scene, storyboard.screenplay.genre or "cinematic", out_path)
            if img and img not in scene.reference_images:
                scene.reference_images.append(img)
                logger.info("director scene_image scene_id=%s status=ok path=%s", scene_id, img)
            else:
                logger.info("director scene_image scene_id=%s status=skip_or_fail", scene_id)
        return storyboard

    @staticmethod
    def _normalize_token(value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    @classmethod
    def _extract_path_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        rows: list[str] = []
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return []
            # Accept legacy comma-separated string or JSON string payload.
            if txt.startswith(("[", "{")):
                try:
                    return cls._extract_path_list(json.loads(txt))
                except Exception:
                    pass
            rows = [x.strip() for x in txt.split(",") if x.strip()]
        elif isinstance(value, dict):
            # Accept object-as-set style: {"path1": true, "path2": true}
            rows = [str(k).strip() for k in value.keys() if str(k).strip()]
        elif isinstance(value, (list, tuple, set)):
            rows = [str(x).strip() for x in value if str(x).strip()]
        else:
            txt = str(value).strip()
            rows = [txt] if txt else []

        uniq: list[str] = []
        seen: set[str] = set()
        for p in rows:
            norm = p.strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            uniq.append(norm)
        return uniq

    @staticmethod
    def _resolve_path(raw_path: str, base_dir: Path | None = None) -> str:
        p = Path(raw_path)
        if p.is_absolute() or base_dir is None:
            return str(p)
        return str((base_dir / p).resolve())

    @classmethod
    def attach_character_reference_map(
        cls,
        storyboard: Storyboard,
        character_map: dict[str, Any] | None = None,
        base_dir: Path | None = None,
    ) -> Storyboard:
        if not isinstance(character_map, dict):
            return storyboard

        for name, c in storyboard.screenplay.characters.items():
            target_key = cls._normalize_token(name)
            matched_paths: list[str] = []
            matched_audio_paths: list[str] = []
            matched_gender: str = ""
            matched_voice_profile: str = ""
            matched_ref_audio_text: str = ""
            for key, value in character_map.items():
                key_norm = cls._normalize_token(str(key))
                if key_norm == target_key or key_norm in target_key or target_key in key_norm:
                    if isinstance(value, dict):
                        matched_paths.extend(
                            cls._extract_path_list(
                                value.get("reference_images", value.get("images", value.get("image", [])))
                            )
                        )
                        matched_audio_paths.extend(
                            cls._extract_path_list(
                                value.get("reference_audios", value.get("audios", value.get("audio", [])))
                            )
                        )
                        if not matched_gender:
                            raw_gender = str(value.get("gender", value.get("sex", "")) or "").strip()
                            if raw_gender:
                                matched_gender = raw_gender
                        if not matched_voice_profile:
                            raw_profile = str(value.get("voice_profile", "") or "").strip()
                            if raw_profile:
                                matched_voice_profile = raw_profile
                        if not matched_ref_audio_text:
                            raw_ref_text = str(value.get("reference_audio_text", value.get("ref_text", "")) or "").strip()
                            if raw_ref_text:
                                matched_ref_audio_text = raw_ref_text
                    else:
                        matched_paths.extend(cls._extract_path_list(value))

            existing: list[str] = []
            seen: set[str] = set()
            for p in matched_paths:
                resolved = cls._resolve_path(p, base_dir=base_dir)
                if resolved in seen:
                    continue
                seen.add(resolved)
                if Path(resolved).exists():
                    existing.append(resolved)

            if not existing:
                existing = []
            for p in existing:
                if p not in c.reference_images:
                    c.reference_images.append(p)
            if c.reference_images and (not c.reference_image or not Path(c.reference_image).exists()):
                c.reference_image = c.reference_images[0]

            audio_existing: list[str] = []
            seen_audio: set[str] = set()
            for p in matched_audio_paths:
                resolved = cls._resolve_path(p, base_dir=base_dir)
                if resolved in seen_audio:
                    continue
                seen_audio.add(resolved)
                if Path(resolved).exists():
                    audio_existing.append(resolved)
            for p in audio_existing:
                if p not in c.reference_audios:
                    c.reference_audios.append(p)
            if c.reference_audios and (not c.reference_audio or not Path(c.reference_audio).exists()):
                c.reference_audio = c.reference_audios[0]

            norm_gender = cls._normalize_gender(matched_gender) if matched_gender else "unknown"
            if norm_gender != "unknown":
                c.gender = norm_gender
            if matched_voice_profile:
                c.voice_profile = matched_voice_profile
            if matched_ref_audio_text:
                c.reference_audio_text = matched_ref_audio_text
        return storyboard

    @classmethod
    def _scene_key_to_id(cls, key: str, storyboard: Storyboard) -> str | None:
        key_norm = cls._normalize_token(key)
        if key_norm in {cls._normalize_token(sid) for sid in storyboard.scenes.keys()}:
            for sid in storyboard.scenes.keys():
                if cls._normalize_token(sid) == key_norm:
                    return sid

        # Scene1 / scene_1 / SC001 style to sorted index.
        digits = "".join(ch for ch in key if ch.isdigit())
        if digits:
            idx = int(digits)
            ordered = sorted(storyboard.scenes.keys())
            if 1 <= idx <= len(ordered):
                return ordered[idx - 1]

        # Fuzzy match by location text.
        for sid, scene in storyboard.scenes.items():
            loc_norm = cls._normalize_token(scene.location or "")
            if not loc_norm:
                continue
            if key_norm == loc_norm or key_norm in loc_norm or loc_norm in key_norm:
                return sid
        return None

    @classmethod
    def attach_scene_reference_map(
        cls,
        storyboard: Storyboard,
        scene_map: dict[str, Any] | None = None,
        base_dir: Path | None = None,
    ) -> Storyboard:
        if not isinstance(scene_map, dict):
            return storyboard

        for key, value in scene_map.items():
            scene_id = cls._scene_key_to_id(str(key), storyboard)
            if not scene_id:
                continue
            scene = storyboard.scenes[scene_id]
            for p in cls._extract_path_list(value):
                resolved = cls._resolve_path(p, base_dir=base_dir)
                if Path(resolved).exists() and resolved not in scene.reference_images:
                    scene.reference_images.append(resolved)
        return storyboard

    @classmethod
    def attach_reference_assets_json(
        cls,
        storyboard: Storyboard,
        reference_assets_json: Path | None = None,
    ) -> Storyboard:
        if reference_assets_json is None or not reference_assets_json.exists():
            return storyboard
        logger.info("director attach_reference_assets_json path=%s", str(reference_assets_json))

        try:
            obj = json.loads(reference_assets_json.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid reference assets JSON: {reference_assets_json}") from exc
        if not isinstance(obj, dict):
            return storyboard

        character_map = obj.get("character")
        if not isinstance(character_map, dict):
            character_map = obj.get("characters")
        scene_map = obj.get("scene")
        if not isinstance(scene_map, dict):
            scene_map = obj.get("scenes")

        base_dir = reference_assets_json.parent
        storyboard = cls.attach_character_reference_map(
            storyboard,
            character_map=character_map,
            base_dir=base_dir,
        )
        storyboard = cls.attach_scene_reference_map(
            storyboard,
            scene_map=scene_map,
            base_dir=base_dir,
        )
        char_ref_count = sum(len(c.reference_images) for c in storyboard.screenplay.characters.values())
        char_audio_ref_count = sum(len(c.reference_audios) for c in storyboard.screenplay.characters.values())
        scene_ref_count = sum(len(s.reference_images) for s in storyboard.scenes.values())
        logger.info(
            "director attach_reference_assets_json status=done character_refs=%d character_audio_refs=%d scene_refs=%d",
            char_ref_count,
            char_audio_ref_count,
            scene_ref_count,
        )
        return storyboard

    @classmethod
    def attach_character_reference_images(
        storyboard: Storyboard,
        character_image_dir: Path | None = None,
    ) -> Storyboard:
        if character_image_dir is None or not character_image_dir.exists():
            return storyboard

        files = [p for p in character_image_dir.iterdir() if p.is_file()]
        character_map: dict[str, list[str]] = {}
        for f in files:
            character_map.setdefault(f.stem, []).append(str(f))
        return cls.attach_character_reference_map(storyboard, character_map=character_map)

    @classmethod
    def attach_scene_reference_images(
        storyboard: Storyboard,
        scene_image_dir: Path | None = None,
    ) -> Storyboard:
        if scene_image_dir is None or not scene_image_dir.exists():
            return storyboard

        files = [p for p in scene_image_dir.iterdir() if p.is_file()]
        scene_map: dict[str, list[str]] = {}
        ordered = sorted(files)
        for f in ordered:
            stem = f.stem
            scene_map.setdefault(stem, []).append(str(f))
        # Also provide index aliases: scene1 -> first file, scene2 -> second file...
        for idx, f in enumerate(ordered, start=1):
            scene_map.setdefault(f"scene{idx}", []).append(str(f))
        return cls.attach_scene_reference_map(storyboard, scene_map=scene_map)

    def ensure_scene_images(
        self,
        storyboard: Storyboard,
        image_dir: Path,
        image_policy: str = "auto",  # auto|always|never
    ) -> Storyboard:
        policy = image_policy.lower().strip()
        if policy not in {"auto", "always", "never"}:
            policy = "auto"

        if policy == "never":
            logger.info("director ensure_scene_images policy=never")
            return storyboard

        if policy == "auto":
            # Generate only missing scene images.
            for scene_id, scene in sorted(storyboard.scenes.items()):
                if any(Path(p).exists() for p in scene.reference_images):
                    logger.info("director ensure_scene_images scene_id=%s status=exists_skip", scene_id)
                    continue
                out_path = image_dir / f"scene_{scene_id}.png"
                img = self._generate_scene_image(scene, storyboard.screenplay.genre or "cinematic", out_path)
                if img:
                    scene.reference_images.append(img)
                    logger.info("director ensure_scene_images scene_id=%s status=generated path=%s", scene_id, img)
                else:
                    logger.info("director ensure_scene_images scene_id=%s status=failed", scene_id)
            return storyboard

        # always
        logger.info("director ensure_scene_images policy=always")
        return self.generate_scene_images(storyboard, image_dir=image_dir)

    def ensure_character_images(
        self,
        storyboard: Storyboard,
        image_dir: Path,
        image_policy: str = "auto",  # auto|always|never
    ) -> Storyboard:
        policy = image_policy.lower().strip()
        if policy not in {"auto", "always", "never"}:
            policy = "auto"

        if policy == "never":
            logger.info("director ensure_character_images policy=never")
            return storyboard

        for name, c in storyboard.screenplay.characters.items():
            has_ref = any(Path(p).exists() for p in c.reference_images)
            has_main_ref = bool(c.reference_image and Path(c.reference_image).exists())
            if policy == "auto" and (has_main_ref or has_ref):
                logger.info("director ensure_character_images character=%s status=exists_skip", name)
                continue
            out_path = image_dir / f"character_{name.strip().replace(' ', '_')}.png"
            img = self._generate_character_image(
                c,
                storyboard.screenplay.genre or "cinematic",
                out_path,
            )
            if img:
                c.reference_image = img
                if img not in c.reference_images:
                    c.reference_images.append(img)
                logger.info("director ensure_character_images character=%s status=generated path=%s", name, img)
            else:
                logger.info("director ensure_character_images character=%s status=failed", name)
        return storyboard
