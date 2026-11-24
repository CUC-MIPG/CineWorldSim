from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.post_production_agent import PostProductionAgent
from agents.post_production_tools import (
    edit_master_video_pseudocode,
    generate_dubbing_pseudocode,
    generate_narration_pseudocode,
)
from agents.schemas import Storyboard

logger = logging.getLogger(__name__)


def _todo_to_target(todo_path: Path) -> Path:
    raw = str(todo_path)
    if raw.endswith(".todo.txt"):
        return Path(raw[: -len(".todo.txt")])
    return todo_path.with_suffix("")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_post_agent(
    *,
    storyboard: Storyboard,
    output_dir: Path,
    final_video_path: str,
    supervision_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    agent = PostProductionAgent()
    return agent.process(
        storyboard=storyboard,
        output_dir=output_dir,
        final_video_path=final_video_path,
        supervision_plan=supervision_plan,
    )


def _run_omnivoice_from_todos(
    *,
    output_dir: Path,
    dubbing_voice_profile: str,
    narration_voice_profile: str,
    language: str,
) -> dict[str, Any]:
    # Dubbing and narration both support local OmniVoice via provider env vars.
    os.environ["POST_DUBBING_PROVIDER"] = "omnivoice"
    os.environ["POST_NARRATION_PROVIDER"] = "omnivoice"

    dubbing_dir = output_dir / "dubbing"
    narration_dir = output_dir / "narration"
    results: dict[str, Any] = {"dubbing": [], "narration": []}

    for todo in sorted(dubbing_dir.glob("*.todo.txt")):
        payload = _load_json(todo).get("payload", {})
        target = _todo_to_target(todo)
        text = str(payload.get("script_text", "")).strip()
        if not text:
            continue
        res = generate_dubbing_pseudocode(
            script_text=text,
            voice_profile=dubbing_voice_profile,
            output_audio_path=str(target),
            language=language,
        )
        if target.exists() and target.stat().st_size > 0:
            todo.unlink(missing_ok=True)
        results["dubbing"].append(
            {"todo": str(todo), "target": str(target), "mode": res.get("mode", ""), "ok": bool(res.get("ok"))}
        )

    for todo in sorted(narration_dir.glob("*.todo.txt")):
        payload = _load_json(todo).get("payload", {})
        target = _todo_to_target(todo)
        text = str(payload.get("narration_text", "")).strip()
        if not text:
            continue
        res = generate_narration_pseudocode(
            narration_text=text,
            narrator_profile=narration_voice_profile,
            output_audio_path=str(target),
            language=language,
        )
        if target.exists() and target.stat().st_size > 0:
            todo.unlink(missing_ok=True)
        results["narration"].append(
            {"todo": str(todo), "target": str(target), "mode": res.get("mode", ""), "ok": bool(res.get("ok"))}
        )

    return results


def _run_acestep_from_todos(
    *,
    output_dir: Path,
    ace_step_root: Path,
    acestep_script: Path,
    uv_cache_dir: Path,
    acestep_device: str,
    acestep_with_lm: bool,
    acestep_thinking: bool,
) -> dict[str, Any]:
    music_dir = output_dir / "music"
    tmp_root = music_dir / "_acestep_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for todo in sorted(music_dir.glob("*.todo.txt")):
        payload = _load_json(todo).get("payload", {})
        style_prompt = str(payload.get("style_prompt", "")).strip()
        duration_sec = float(payload.get("duration_sec", 12.0) or 12.0)
        duration_sec = max(10.0, min(600.0, duration_sec))

        target = _todo_to_target(todo)
        scene_name = target.stem.replace("_bgm", "")
        scene_tmp = tmp_root / scene_name
        scene_tmp.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(acestep_script),
            "--prompt",
            style_prompt or f"Cinematic background score for {scene_name}",
            "--duration",
            str(duration_sec),
            "--audio-format",
            "wav",
            "--device",
            acestep_device,
            "--output-dir",
            str(scene_tmp),
        ]
        if acestep_with_lm:
            cmd.append("--with-lm")
            if acestep_thinking:
                cmd.append("--thinking")

        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(uv_cache_dir)
        logger.info("acestep run scene=%s duration=%.2f", scene_name, duration_sec)
        proc = subprocess.run(
            cmd,
            cwd=str(ace_step_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            rows.append(
                {
                    "todo": str(todo),
                    "target": str(target),
                    "ok": False,
                    "error": "acestep_failed",
                    "stderr_tail": proc.stderr[-1000:],
                }
            )
            continue

        wavs = sorted(scene_tmp.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wavs:
            rows.append(
                {
                    "todo": str(todo),
                    "target": str(target),
                    "ok": False,
                    "error": "no_wav_generated",
                }
            )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wavs[0], target)
        if target.exists() and target.stat().st_size > 0:
            todo.unlink(missing_ok=True)
        rows.append(
            {
                "todo": str(todo),
                "target": str(target),
                "ok": True,
                "source_wav": str(wavs[0]),
            }
        )

    return {"music": rows}


def _build_scene_audio_lists(
    storyboard: Storyboard,
    output_dir: Path,
    include_narration: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    scene_ids = sorted(storyboard.scenes.keys())
    dubbing_paths: list[str] = []
    narration_paths: list[str] = []
    bgm_paths: list[str] = []

    for sid in scene_ids:
        d = output_dir / "dubbing" / f"{sid}_dubbing.wav"
        n = output_dir / "narration" / f"{sid}_narration.wav"
        m = output_dir / "music" / f"{sid}_bgm.wav"
        if d.exists() and d.stat().st_size > 0:
            dubbing_paths.append(str(d))
        if include_narration and n.exists() and n.stat().st_size > 0:
            narration_paths.append(str(n))
        if m.exists() and m.stat().st_size > 0:
            bgm_paths.append(str(m))
    return dubbing_paths, narration_paths, bgm_paths


def _build_edit_audio_inputs(
    *,
    storyboard: Storyboard,
    output_dir: Path,
    include_narration: bool,
    supervision_plan: dict[str, Any] | None,
) -> tuple[list[str], list[float], list[str], list[str], list[float]]:
    info_path = output_dir / "post_production_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    else:
        info = {}

    shot_dubbing = info.get("dubbing_shot_paths", [])
    shot_durations = info.get("dubbing_shot_durations", [])
    bgm_paths = info.get("bgm_scene_paths", [])
    bgm_durations = info.get("bgm_scene_durations", [])
    narration_paths = []

    if include_narration:
        for sid in sorted(storyboard.scenes.keys()):
            p = output_dir / "narration" / f"{sid}_narration.wav"
            if p.exists() and p.stat().st_size > 0:
                narration_paths.append(str(p))

    if (
        isinstance(shot_dubbing, list)
        and shot_dubbing
        and isinstance(shot_durations, list)
        and len(shot_dubbing) == len(shot_durations)
    ):
        if not isinstance(bgm_paths, list) or not bgm_paths:
            _, _, bgm_paths = _build_scene_audio_lists(
                storyboard=storyboard,
                output_dir=output_dir,
                include_narration=include_narration,
            )
        if not isinstance(bgm_durations, list) or len(bgm_durations) != len(bgm_paths):
            bgm_durations = _build_scene_segment_durations(storyboard=storyboard, supervision_plan=supervision_plan)
        return (
            [str(x) for x in shot_dubbing if x],
            [float(x) for x in shot_durations],
            narration_paths,
            [str(x) for x in bgm_paths if x],
            [float(x) for x in bgm_durations],
        )

    dubbing_paths, _, bgm_paths2 = _build_scene_audio_lists(
        storyboard=storyboard,
        output_dir=output_dir,
        include_narration=include_narration,
    )
    scene_durations = _build_scene_segment_durations(storyboard=storyboard, supervision_plan=supervision_plan)
    return dubbing_paths, scene_durations, narration_paths, bgm_paths2, scene_durations


def _build_scene_segment_durations(
    storyboard: Storyboard,
    supervision_plan: dict[str, Any] | None,
) -> list[float]:
    plan_by_scene: dict[str, dict[str, Any]] = {}
    if isinstance(supervision_plan, dict):
        rows = supervision_plan.get("scene_plans")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("scene_id", "")).strip()
                if sid:
                    plan_by_scene[sid] = row

    durations: list[float] = []
    for sid in sorted(storyboard.scenes.keys()):
        default_dur = max(3.0, len(storyboard.get_scene_shots(sid)) * 3.0)
        plan = plan_by_scene.get(sid, {})
        dur = None
        if isinstance(plan, dict):
            music = plan.get("music")
            if isinstance(music, dict):
                m_dur = music.get("duration_sec")
                if isinstance(m_dur, (int, float)) and float(m_dur) > 0:
                    dur = float(m_dur)
            if dur is None:
                est = plan.get("estimated_scene_duration_sec")
                if isinstance(est, (int, float)) and float(est) > 0:
                    dur = float(est)
        durations.append(float(dur if dur is not None else default_dur))
    return durations


def _build_input_videos(storyboard: Storyboard, final_video_path: str) -> list[str]:
    if final_video_path and Path(final_video_path).exists():
        return [final_video_path]

    ordered = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
    paths = [s.video_path for s in ordered if s.video_path and Path(s.video_path).exists()]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated post-production tool test: OmniVoice(dubbing+narration) + ACE-Step(bgm) + edit."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/data1/AgentG/hhb/CameraMaster/outputs/e2e_text_only_full_run"),
    )
    parser.add_argument("--storyboard-json", type=Path, default=None)
    parser.add_argument("--story-supervisor-plan", type=Path, default=None)
    parser.add_argument("--final-video-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--language", type=str, default="zh-CN")
    parser.add_argument("--dubbing-voice-profile", type=str, default="男，中音调")
    parser.add_argument("--narration-voice-profile", type=str, default="女，中音调")
    parser.add_argument(
        "--omnivoice-model-dir",
        type=Path,
        default=Path("/data1/AgentG/hhb/CameraMaster/tool_hub/models/dubbing/OmniVoice"),
    )
    parser.add_argument("--omnivoice-device", type=str, default="cuda:0")
    parser.add_argument("--omnivoice-dtype", type=str, default="float16")
    parser.add_argument("--ace-step-root", type=Path, default=Path("/data1/AgentG/hhb/ACE-Step-1.5"))
    parser.add_argument(
        "--acestep-script",
        type=Path,
        default=Path(
            "/data1/AgentG/hhb/CameraMaster/tool_hub/code/post_production/providers/acestep_local_infer.py"
        ),
    )
    parser.add_argument("--uv-cache-dir", type=Path, default=Path("/tmp/uv-cache"))
    parser.add_argument("--acestep-device", type=str, default="auto")
    parser.add_argument("--acestep-with-lm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--acestep-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-post-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--include-narration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include narration in final edit mix.",
    )
    parser.add_argument(
        "--edit-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip post-agent/omnivoice/acestep generation and only run final edit using existing audio files.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    run_dir = args.run_dir.resolve()
    storyboard_json = args.storyboard_json.resolve() if args.storyboard_json else (run_dir / "storyboard.json")
    plan_json = args.story_supervisor_plan.resolve() if args.story_supervisor_plan else (
        run_dir / "story_supervisor" / "story_supervisor_plan.json"
    )
    final_video_path = (
        str(args.final_video_path.resolve())
        if args.final_video_path
        else str((run_dir / "final_concat.mp4").resolve())
    )
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "post_production_omni_ace_integrated")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not storyboard_json.exists():
        raise SystemExit(f"storyboard-json not found: {storyboard_json}")

    storyboard = Storyboard.from_dict(json.loads(storyboard_json.read_text(encoding="utf-8")))
    supervision_plan = _load_json(plan_json) if plan_json.exists() else None

    # OmniVoice runtime env
    os.environ["OMNIVOICE_MODEL_DIR"] = str(args.omnivoice_model_dir.resolve())
    os.environ["OMNIVOICE_DEVICE"] = args.omnivoice_device
    os.environ["OMNIVOICE_DTYPE"] = args.omnivoice_dtype

    integrated_report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "storyboard_json": str(storyboard_json),
        "plan_json": str(plan_json) if plan_json.exists() else "",
        "output_dir": str(output_dir),
    }

    if not args.skip_post_agent and not args.edit_only:
        logger.info("stage=post_agent start")
        summary = _run_post_agent(
            storyboard=storyboard,
            output_dir=output_dir,
            final_video_path=final_video_path,
            supervision_plan=supervision_plan,
            include_narration=args.include_narration,
        )
        integrated_report["post_agent"] = {
            "scene_count": summary.get("scene_count", 0),
            "movie_title": summary.get("movie_title", ""),
        }
        logger.info("stage=post_agent done")
    else:
        logger.info("stage=post_agent skipped")

    if args.edit_only:
        logger.info("stage=omnivoice skipped(edit_only=true)")
        integrated_report["omnivoice"] = {"skipped": True, "reason": "edit_only"}
    else:
        logger.info("stage=omnivoice start")
        omni_res = _run_omnivoice_from_todos(
            output_dir=output_dir,
            dubbing_voice_profile=args.dubbing_voice_profile,
            narration_voice_profile=args.narration_voice_profile,
            language=args.language,
        )
        integrated_report["omnivoice"] = omni_res
        logger.info("stage=omnivoice done")

    if args.edit_only:
        logger.info("stage=acestep skipped(edit_only=true)")
        integrated_report["acestep"] = {"skipped": True, "reason": "edit_only"}
    else:
        logger.info("stage=acestep start")
        ace_res = _run_acestep_from_todos(
            output_dir=output_dir,
            ace_step_root=args.ace_step_root.resolve(),
            acestep_script=args.acestep_script.resolve(),
            uv_cache_dir=args.uv_cache_dir.resolve(),
            acestep_device=args.acestep_device,
            acestep_with_lm=args.acestep_with_lm,
            acestep_thinking=args.acestep_thinking,
        )
        integrated_report["acestep"] = ace_res
        logger.info("stage=acestep done")

    logger.info("stage=edit start")
    input_videos = _build_input_videos(storyboard=storyboard, final_video_path=final_video_path)
    dubbing_paths, dubbing_segment_durations, narration_paths, bgm_paths, bgm_segment_durations = _build_edit_audio_inputs(
        storyboard=storyboard,
        output_dir=output_dir,
        include_narration=args.include_narration,
        supervision_plan=supervision_plan,
    )
    master_out = output_dir / "master" / "final_master_post_omni_ace.mp4"
    edit_res = edit_master_video_pseudocode(
        input_video_paths=input_videos,
        dubbing_paths=dubbing_paths,
        narration_paths=narration_paths,
        bgm_paths=bgm_paths,
        output_video_path=str(master_out),
        edit_instructions="Scene-first shot order; mix dubbing+narration+bgm; keep dialogue intelligible.",
        dubbing_segment_durations=dubbing_segment_durations,
        narration_segment_durations=bgm_segment_durations if args.include_narration else [],
        bgm_segment_durations=bgm_segment_durations,
        include_narration=args.include_narration,
    )
    integrated_report["edit"] = edit_res
    integrated_report["counts"] = {
        "input_videos": len(input_videos),
        "dubbing_wavs": len(dubbing_paths),
        "narration_wavs": len(narration_paths),
        "bgm_wavs": len(bgm_paths),
    }
    logger.info("stage=edit done")

    report_path = output_dir / "integrated_test_report.json"
    report_path.write_text(json.dumps(integrated_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Integrated Post-Production Tool Test Completed ===")
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Master: {master_out}")


if __name__ == "__main__":
    main()
