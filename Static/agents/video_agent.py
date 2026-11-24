from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
import re

from .client import make_client, pick_model
from .generation_backends import (
    video_provider_from_env,
    magref_settings_from_env,
    wan_r2v_settings_from_env,
    generate_video_magref,
    generate_video_wan_r2v,
    generate_keyframe_pseudocode,
    generate_i2v_from_keyframe_pseudocode,
)
from .retry import call_with_retries
from .schemas import Character, Scene, Shot, Storyboard

logger = logging.getLogger(__name__)


class VideoAgent:
    """Generate shot videos and concatenate final movie."""

    def __init__(
        self,
        model: str | None = None,
        strict_online: bool = True,
        retries: int = 3,
    ):
        self.client = make_client(scope="VIDEO")
        self.model = model or pick_model("gpt-5.5", scope="VIDEO")
        self.strict_online = strict_online
        self.retries = retries

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
            "Generate one high-quality English image-to-video prompt paragraph."
        )
        character_info = "\n".join(f"{c.name}: {c.appearance}" for c in characters.values())
        elements_info = "\n".join(f"{k}: {v}" for k, v in shot.scene_elements.items())

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

Return only one polished English paragraph for image-to-video generation.
"""
        def _call():
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

    def send_generation_request(self, prompt: str, image_paths: list[str], video_path: str) -> bool:
        def _call():
            provider = video_provider_from_env()
            logger.info(
                "video generate_request provider=%s output=%s image_count=%d",
                provider,
                video_path,
                len(image_paths),
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
        video_mode: str = "reference",  # reference|keyframe
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
        if mode not in {"reference", "keyframe"}:
            mode = "reference"

        unique_scene_shots = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
        scene_shot_map: dict[str, list[Shot]] = {}
        for sh in unique_scene_shots:
            scene_shot_map.setdefault(str(sh.scene_id), []).append(sh)

        for scene_id in sorted(storyboard.scenes.keys()):
            scene = storyboard.scenes[scene_id]
            scene_shots = scene_shot_map.get(str(scene_id), [])
            logger.info("video scene status=start scene_id=%s shots=%d", scene_id, len(scene_shots))
            for shot in scene_shots:
                matched = self._match_characters_for_shot(shot, storyboard.screenplay.characters)
                prompt_characters: dict[str, Character]
                if matched:
                    prompt_characters = {c.name: c for c in matched}
                else:
                    prompt_characters = storyboard.screenplay.characters

                prompt = self.generate_llm_video_prompt(shot, scene, prompt_characters)
                image_paths = self._collect_images(shot, scene, storyboard.screenplay.characters)
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
                        "mode": "reference",
                        "prompt": prompt,
                        "images": image_paths,
                        "output_video_path": video_path,
                    }
                )

                if dry_run:
                    shot.video_path = video_path
                    logger.info("video shot status=dry_run shot_id=%s output=%s", shot.shot_id, video_path)
                    continue

                ok = self.send_generation_request(prompt, image_paths, video_path)
                if ok:
                    shot.video_path = video_path
                    logger.info("video shot status=done shot_id=%s mode=reference output=%s", shot.shot_id, video_path)
                else:
                    logger.info("video shot status=failed shot_id=%s mode=reference output=%s", shot.shot_id, video_path)
            logger.info("video scene status=done scene_id=%s", scene_id)

        if save_task_json:
            (output_dir / "video_info.json").write_text(
                json.dumps(tasks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("video tasks saved path=%s count=%d", str(output_dir / "video_info.json"), len(tasks))
        logger.info("video process_shots status=done")
        return storyboard

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
            images = task.get("images", [])
            out = self._safe_text(task.get("output_video_path", "")).strip()
            if not prompt or not out:
                failed += 1
                continue

            if not isinstance(images, list):
                images = [self._safe_text(images)]
            image_paths = [self._safe_text(p) for p in images if self._safe_text(p)]

            existing = [p for p in image_paths if Path(p).exists()]
            if strict_images and not existing:
                skipped += 1
                continue

            ok = self.send_generation_request(prompt=prompt, image_paths=existing, video_path=out)
            if ok:
                success += 1
                logger.info("video process_task status=done output=%s", out)
            else:
                failed += 1
                logger.info("video process_task status=failed output=%s", out)

        logger.info("video process_task_file status=done total=%d success=%d failed=%d skipped=%d", len(tasks), success, failed, skipped)
        return {"total": len(tasks), "success": success, "failed": failed, "skipped": skipped}

    def concat_videos(self, video_paths: list[str], output_path: Path) -> bool:
        mp4_list = [str(Path(p)) for p in video_paths if p and Path(p).exists()]
        if not mp4_list:
            logger.info("video concat_videos status=skip reason=no_inputs")
            return False

        list_file = output_path.parent / "concat_list.txt"
        list_file.parent.mkdir(parents=True, exist_ok=True)
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
