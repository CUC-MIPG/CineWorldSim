from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from .post_production_tools import (
    edit_master_video_pseudocode,
    generate_bgm_pseudocode,
    generate_dubbing_pseudocode,
    generate_narration_pseudocode,
)
from .schemas import Character, Scene, Shot, Storyboard

logger = logging.getLogger(__name__)


class PostProductionAgent:
    """Post-production orchestrator with shot-level dubbing and scene-level bgm/narration."""

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
    def _norm_token(text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip().lower())

    @classmethod
    def _normalize_gender(cls, value: str) -> str:
        raw = cls._norm_token(value)
        if raw in {"male", "m", "man", "boy", "男", "男性"}:
            return "male"
        if raw in {"female", "f", "woman", "girl", "女", "女性"}:
            return "female"
        return "unknown"

    @classmethod
    def _voice_profile_from_gender(cls, gender: str) -> str:
        g = cls._normalize_gender(gender)
        if g == "male":
            return "男，中音调"
        if g == "female":
            return "女，中音调"
        return "青年，中音调"

    @classmethod
    def _extract_scene_plan(cls, supervision_plan: dict[str, Any] | None, scene_id: str) -> dict[str, Any] | None:
        if not isinstance(supervision_plan, dict):
            return None
        rows = supervision_plan.get("scene_plans", [])
        if not isinstance(rows, list):
            return None
        for row in rows:
            if isinstance(row, dict) and str(row.get("scene_id", "")) == str(scene_id):
                return row
        return None

    def _build_scene_narration(self, scene: Scene, shots: list[Shot]) -> str:
        beat_count = len(shots)
        return (
            f"旁白草案：在{scene.location or '未知地点'}的{scene.time_of_day or '未知时间'}，"
            f"故事围绕“{scene.plot_description or '该场景核心事件'}”展开，"
            f"并通过{beat_count}个镜头推进情绪与信息。"
        )

    def _build_scene_bgm_prompt(self, scene: Scene) -> str:
        return (
            f"Cinematic background score, scene={scene.scene_id}, "
            f"location={scene.location or 'unknown'}, time={scene.time_of_day or 'unknown'}, "
            "adaptive dynamics, emotional continuity, no vocals."
        )

    @staticmethod
    def _ffmpeg_trim_or_pad_audio(src: Path, dst: Path, target_sec: float) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-af",
            f"apad=pad_dur={target_sec:.3f},atrim=0:{target_sec:.3f}",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0

    @staticmethod
    def _ffmpeg_make_silence(dst: Path, duration_sec: float) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            f"{duration_sec:.3f}",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0

    @classmethod
    def _ffmpeg_concat_audio_to_duration(cls, src_paths: list[Path], dst: Path, duration_sec: float) -> bool:
        work = dst.parent / "_shot_audio_work"
        work.mkdir(parents=True, exist_ok=True)
        if not src_paths:
            return cls._ffmpeg_make_silence(dst, duration_sec)

        concat_list = work / f"{dst.stem}_concat.txt"
        concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in src_paths), encoding="utf-8")
        merged = work / f"{dst.stem}_merged.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(merged),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not merged.exists() or merged.stat().st_size <= 0:
            return False
        return cls._ffmpeg_trim_or_pad_audio(merged, dst, duration_sec)

    @staticmethod
    def _allocate_line_durations(rows: list[dict[str, Any]], budget_sec: float) -> list[float]:
        count = len(rows)
        if count <= 0:
            return []
        budget = max(0.6, float(budget_sec))
        max_voice = max(0.5, budget * 0.98)
        default = max(0.5, min(3.0, max_voice / count))
        seed: list[float] = []
        for row in rows:
            d = row.get("duration_sec")
            if isinstance(d, (int, float)) and float(d) > 0:
                seed.append(float(d))
            else:
                seed.append(default)
        total = sum(seed)
        if total <= 0:
            seed = [default for _ in rows]
            total = sum(seed)
        if total > max_voice:
            scale = max_voice / total
            seed = [max(0.4, x * scale) for x in seed]
        out = [round(x, 2) for x in seed]
        fix = round(max_voice - sum(out), 2)
        if abs(fix) >= 0.01:
            idx = len(out) - 1
            out[idx] = round(max(0.4, out[idx] + fix), 2)
        return out

    @classmethod
    def _resolve_character(
        cls,
        speaker: str,
        shot: Shot,
        storyboard: Storyboard,
    ) -> tuple[str, Character | None]:
        chars = storyboard.screenplay.characters
        if not chars:
            return speaker or "角色", None

        speaker_norm = cls._norm_token(speaker)
        if speaker_norm:
            for name, c in chars.items():
                if cls._norm_token(name) == speaker_norm:
                    return name, c
            for name, c in chars.items():
                n = cls._norm_token(name)
                if speaker_norm in n or n in speaker_norm:
                    return name, c

        for n in shot.character_names:
            if n in chars:
                return n, chars[n]
        first_name = sorted(chars.keys())[0]
        return first_name, chars[first_name]

    @classmethod
    def _build_scene_shot_plans(
        cls,
        scene_id: str,
        shots: list[Shot],
        scene_plan: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], float]:
        seq_map = {int(s.sequence): s for s in shots}
        default_each = 3.0 if shots else 0.0
        if scene_plan:
            est = scene_plan.get("estimated_scene_duration_sec")
            if isinstance(est, (int, float)) and shots:
                default_each = max(0.8, float(est) / len(shots))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw = scene_plan.get("shot_plans", []) if isinstance(scene_plan, dict) else []
        if isinstance(raw, list):
            for one in raw:
                if not isinstance(one, dict):
                    continue
                raw_id = cls._safe_text(one.get("shot_id", ""))
                seq = one.get("sequence")
                shot: Shot | None = None
                if raw_id:
                    for s in shots:
                        if s.shot_id == raw_id:
                            shot = s
                            break
                if shot is None and isinstance(seq, (int, float)):
                    shot = seq_map.get(int(seq))
                if shot is None or shot.shot_id in seen:
                    continue
                dur = one.get("duration_sec")
                dur_val = float(dur) if isinstance(dur, (int, float)) and float(dur) > 0 else default_each
                rows.append(
                    {
                        "shot_id": shot.shot_id,
                        "sequence": int(shot.sequence),
                        "duration_sec": max(0.8, dur_val),
                    }
                )
                seen.add(shot.shot_id)
        for shot in shots:
            if shot.shot_id not in seen:
                rows.append({"shot_id": shot.shot_id, "sequence": int(shot.sequence), "duration_sec": max(0.8, default_each)})
        rows = sorted(rows, key=lambda x: int(x["sequence"]))
        total = round(sum(float(x["duration_sec"]) for x in rows), 2)
        return rows, total

    @classmethod
    def _build_scene_dialogue_rows(
        cls,
        scene: Scene,
        shots: list[Shot],
        scene_plan: dict[str, Any] | None,
        shot_durations: dict[str, float],
    ) -> dict[str, list[dict[str, Any]]]:
        by_shot: dict[str, list[dict[str, Any]]] = {s.shot_id: [] for s in shots}
        raw_rows = scene_plan.get("dialogue", []) if isinstance(scene_plan, dict) else []
        rows = [r for r in raw_rows if isinstance(r, dict)] if isinstance(raw_rows, list) else []
        shot_ids = [s.shot_id for s in shots]
        if shot_ids:
            idx = 0
            for row in rows:
                explicit = cls._safe_text(row.get("shot_id", ""))
                target = explicit if explicit in by_shot else shot_ids[idx % len(shot_ids)]
                by_shot[target].append(
                    {
                        "speaker": cls._safe_text(row.get("speaker", "角色")) or "角色",
                        "text": cls._safe_text(row.get("text", "")),
                        "duration_sec": float(row["duration_sec"]) if isinstance(row.get("duration_sec"), (int, float)) else None,
                    }
                )
                idx += 1

        for shot in shots:
            if by_shot[shot.shot_id]:
                continue
            speaker = shot.character_names[0] if shot.character_names else "角色"
            prompt = cls._safe_text(shot.auto_prompt())
            fallback_text = f"{speaker}：{(prompt[:42] + '...') if len(prompt) > 42 else prompt}"
            by_shot[shot.shot_id].append(
                {
                    "speaker": speaker,
                    "text": fallback_text or f"{speaker}：继续推进当前情节。",
                    "duration_sec": max(0.6, min(2.5, float(shot_durations.get(shot.shot_id, 2.5)) * 0.45)),
                }
            )
        return by_shot

    def _ensure_character_reference_audios(self, storyboard: Storyboard, output_dir: Path) -> None:
        voice_ref_dir = output_dir / "voice_refs"
        voice_ref_dir.mkdir(parents=True, exist_ok=True)
        for name, c in storyboard.screenplay.characters.items():
            existing = [p for p in c.reference_audios if p and Path(p).exists()]
            if c.reference_audio and Path(c.reference_audio).exists() and c.reference_audio not in existing:
                existing.insert(0, c.reference_audio)
            if existing:
                c.reference_audios = existing
                c.reference_audio = existing[0]
                continue

            profile = c.voice_profile or self._voice_profile_from_gender(c.gender)
            ref_text = c.reference_audio_text or f"你好，我是{name}。"
            out_path = voice_ref_dir / f"{self._norm_token(name) or 'character'}_ref.wav"
            res = generate_dubbing_pseudocode(
                script_text=ref_text,
                voice_profile=profile,
                output_audio_path=str(out_path),
                language="zh-CN",
            )
            p = Path(str(res.get("output_audio_path", "")))
            if not bool(res.get("ok")) or not p.exists() or p.stat().st_size <= 0:
                raise RuntimeError(
                    f"Character reference audio generation failed character={name} "
                    f"mode={res.get('mode','')} error={res.get('error','')} output={p}"
                )
            c.reference_audio = str(p)
            if str(p) not in c.reference_audios:
                c.reference_audios.append(str(p))
            c.reference_audio_text = ref_text

    def process(
        self,
        storyboard: Storyboard,
        output_dir: Path,
        final_video_path: str = "",
        supervision_plan: dict[str, Any] | None = None,
        include_narration: bool = True,
    ) -> dict[str, Any]:
        logger.info(
            "post_production process status=start scenes=%d include_narration=%s",
            len(storyboard.scenes),
            include_narration,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_character_reference_audios(storyboard, output_dir)

        dubbing_shot_dir = output_dir / "dubbing_shots"
        narration_dir = output_dir / "narration"
        bgm_dir = output_dir / "music"
        master_dir = output_dir / "master"
        dubbing_shot_dir.mkdir(parents=True, exist_ok=True)
        narration_dir.mkdir(parents=True, exist_ok=True)
        bgm_dir.mkdir(parents=True, exist_ok=True)
        master_dir.mkdir(parents=True, exist_ok=True)

        ordered_shots = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
        ordered_by_scene: dict[str, list[Shot]] = {}
        for sh in ordered_shots:
            ordered_by_scene.setdefault(sh.scene_id, []).append(sh)

        shot_dubbing_paths: list[str] = []
        shot_segment_durations: list[float] = []
        narration_paths: list[str] = []
        bgm_paths: list[str] = []
        scene_segment_durations: list[float] = []
        scene_outputs: list[dict[str, Any]] = []

        for scene_id in sorted(storyboard.scenes.keys()):
            scene = storyboard.scenes[scene_id]
            shots = ordered_by_scene.get(scene_id, [])
            logger.info("post_production scene status=start scene_id=%s shots=%d", scene_id, len(shots))
            scene_plan = self._extract_scene_plan(supervision_plan, scene_id)
            shot_plan_rows, scene_duration_sec = self._build_scene_shot_plans(scene_id, shots, scene_plan)
            shot_duration_map = {str(x["shot_id"]): float(x["duration_sec"]) for x in shot_plan_rows}
            scene_segment_durations.append(scene_duration_sec)

            shot_dialogues = self._build_scene_dialogue_rows(
                scene=scene,
                shots=shots,
                scene_plan=scene_plan,
                shot_durations=shot_duration_map,
            )
            shot_outputs: list[dict[str, Any]] = []
            for shot in shots:
                shot_id = shot.shot_id
                shot_duration = max(0.8, float(shot_duration_map.get(shot_id, 2.5)))
                rows = shot_dialogues.get(shot_id, [])
                allocated = self._allocate_line_durations(rows, budget_sec=shot_duration)
                line_norm_paths: list[Path] = []
                line_rows: list[dict[str, Any]] = []
                shot_line_dir = dubbing_shot_dir / shot_id
                shot_line_dir.mkdir(parents=True, exist_ok=True)

                for idx, row in enumerate(rows, start=1):
                    text = self._safe_text(row.get("text", ""))
                    if not text:
                        continue
                    speaker = self._safe_text(row.get("speaker", "")) or "角色"
                    resolved_name, char_obj = self._resolve_character(speaker=speaker, shot=shot, storyboard=storyboard)
                    gender = char_obj.gender if char_obj else "unknown"
                    profile = (char_obj.voice_profile if char_obj else "") or self._voice_profile_from_gender(gender)
                    ref_audio = char_obj.reference_audio if char_obj and char_obj.reference_audio else ""
                    ref_text = char_obj.reference_audio_text if char_obj and char_obj.reference_audio_text else (f"你好，我是{resolved_name}。")
                    target_line_sec = allocated[idx - 1] if idx - 1 < len(allocated) else max(0.4, shot_duration * 0.2)

                    raw_out = shot_line_dir / f"line_{idx:03d}_{self._norm_token(resolved_name) or 'speaker'}.wav"
                    res = generate_dubbing_pseudocode(
                        script_text=text,
                        voice_profile=profile,
                        output_audio_path=str(raw_out),
                        language="zh-CN",
                        reference_audio_path=ref_audio,
                        reference_text=ref_text,
                    )
                    p = Path(str(res.get("output_audio_path", "")))
                    if not bool(res.get("ok")) or not p.exists() or p.stat().st_size <= 0:
                        raise RuntimeError(
                            f"PostProduction dubbing failed scene={scene_id} shot={shot_id} line={idx} "
                            f"speaker={resolved_name} mode={res.get('mode','')} error={res.get('error','')}"
                        )
                    norm_out = shot_line_dir / f"line_{idx:03d}_{self._norm_token(resolved_name) or 'speaker'}_norm.wav"
                    if not self._ffmpeg_trim_or_pad_audio(p, norm_out, target_line_sec):
                        raise RuntimeError(
                            f"PostProduction dubbing duration normalize failed scene={scene_id} shot={shot_id} line={idx}"
                        )
                    line_norm_paths.append(norm_out)
                    line_rows.append(
                        {
                            "line_index": idx,
                            "speaker": resolved_name,
                            "gender": gender,
                            "text": text,
                            "duration_sec": target_line_sec,
                            "raw_audio_path": str(p),
                            "normalized_audio_path": str(norm_out),
                        }
                    )

                shot_mix_path = dubbing_shot_dir / f"{shot_id}_dubbing.wav"
                if not self._ffmpeg_concat_audio_to_duration(line_norm_paths, shot_mix_path, shot_duration):
                    raise RuntimeError(f"PostProduction shot dubbing concat failed scene={scene_id} shot={shot_id}")

                shot_dubbing_paths.append(str(shot_mix_path))
                shot_segment_durations.append(shot_duration)
                shot_outputs.append(
                    {
                        "shot_id": shot_id,
                        "sequence": int(shot.sequence),
                        "duration_sec": shot_duration,
                        "dubbing_audio_path": str(shot_mix_path),
                        "dialogue_lines": line_rows,
                    }
                )

            if include_narration:
                narration_text = self._safe_text(
                    (scene_plan.get("narration") or {}).get("text", "")
                    if isinstance(scene_plan, dict) and isinstance(scene_plan.get("narration"), dict)
                    else ""
                )
                if not narration_text:
                    narration_text = self._build_scene_narration(scene, shots)
                narration_out = narration_dir / f"{scene_id}_narration.wav"
                narration_res = generate_narration_pseudocode(
                    narration_text=narration_text,
                    narrator_profile="女，中音调",
                    output_audio_path=str(narration_out),
                    language="zh-CN",
                )
                np = Path(str(narration_res.get("output_audio_path", "")))
                if not bool(narration_res.get("ok")) or not np.exists() or np.stat().st_size <= 0:
                    raise RuntimeError(
                        f"PostProduction narration failed scene={scene_id} mode={narration_res.get('mode','')}"
                    )
                narration_paths.append(str(np))
            else:
                narration_res = {
                    "ok": True,
                    "tool": "narration",
                    "mode": "skipped",
                    "output_audio_path": "",
                    "todo_file": "",
                }

            music = scene_plan.get("music") if isinstance(scene_plan, dict) and isinstance(scene_plan.get("music"), dict) else {}
            bgm_prompt = self._safe_text(music.get("description", "")) or self._build_scene_bgm_prompt(scene)
            bgm_out = bgm_dir / f"{scene_id}_bgm.wav"
            bgm_res = generate_bgm_pseudocode(
                style_prompt=bgm_prompt,
                duration_sec=scene_duration_sec,
                output_audio_path=str(bgm_out),
            )
            bp = Path(str(bgm_res.get("output_audio_path", "")))
            if not bool(bgm_res.get("ok")) or not bp.exists() or bp.stat().st_size <= 0:
                raise RuntimeError(
                    f"PostProduction bgm failed scene={scene_id} mode={bgm_res.get('mode','')} error={bgm_res.get('error','')}"
                )
            bgm_paths.append(str(bp))

            scene_outputs.append(
                {
                    "scene_id": scene_id,
                    "source_scene_plan_used": bool(scene_plan),
                    "scene_duration_sec": scene_duration_sec,
                    "shot_plans": shot_plan_rows,
                    "shot_outputs": shot_outputs,
                    "narration": narration_res,
                    "bgm": bgm_res,
                }
            )
            logger.info("post_production scene status=done scene_id=%s duration_sec=%.2f", scene_id, scene_duration_sec)

        shot_video_paths = [s.video_path for s in ordered_shots if s.video_path]
        input_videos = [p for p in shot_video_paths if p]
        if final_video_path and Path(final_video_path).exists():
            input_videos = [final_video_path]

        master_out = master_dir / "final_master_post.mp4"
        edit_res = edit_master_video_pseudocode(
            input_video_paths=input_videos,
            dubbing_paths=shot_dubbing_paths,
            narration_paths=narration_paths,
            bgm_paths=bgm_paths,
            output_video_path=str(master_out),
            edit_instructions=(
                "Mix shot-level dubbing, scene-level narration and bgm with ducking; "
                "align dubbing to shot starts and bgm to scene boundaries."
            ),
            dubbing_segment_durations=shot_segment_durations,
            narration_segment_durations=scene_segment_durations if include_narration else [],
            bgm_segment_durations=scene_segment_durations,
            include_narration=include_narration,
        )
        logger.info(
            "post_production master_edit status=%s output=%s",
            "ok" if bool(edit_res.get("ok")) else "failed",
            str(edit_res.get("output_video_path", "")),
        )

        storyboard_out = output_dir / "storyboard_post_audio.json"
        storyboard_out.write_text(
            json.dumps(storyboard.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "agent": "PostProductionAgent",
            "movie_title": storyboard.title,
            "scene_count": len(scene_outputs),
            "scene_outputs": scene_outputs,
            "dubbing_shot_count": len(shot_dubbing_paths),
            "dubbing_shot_paths": shot_dubbing_paths,
            "dubbing_shot_durations": shot_segment_durations,
            "bgm_scene_paths": bgm_paths,
            "bgm_scene_durations": scene_segment_durations,
            "master_edit": edit_res,
            "include_narration": include_narration,
            "storyboard_post_audio_json": str(storyboard_out),
        }
        (output_dir / "post_production_info.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("post_production process status=done output=%s", str(output_dir / "post_production_info.json"))
        return summary
