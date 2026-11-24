#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is available in the target env.
    load_dotenv = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/keyframe_only_event_video_pipeline"
DEFAULT_MINIMAX_API_URL = "http://127.0.0.1:9001/v1/videos/sync"
CHILD_LEVEL_RE = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|")


def _bool_flag(enabled: bool, flag: str) -> list[str]:
    return [flag] if enabled else [f"--no-{flag[2:]}"]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_narrative_world_video_pipeline.py"),
    ]
    if args.story_file:
        cmd.extend(["--story-file", str(args.story_file)])
    elif str(args.storyline or "").strip():
        cmd.extend(["--storyline", str(args.storyline).strip()])
    else:
        raise ValueError("Either --story-file or --storyline is required.")

    cmd.extend(
        [
            "--characters",
            str(args.characters),
            "--output-dir",
            str(args.output_dir),
            "--backend-mode",
            getattr(args, "backend_mode", "all_banana"),
            "--path-mode",
            args.path_mode,
            "--max-steps",
            str(max(1, int(args.max_steps))),
            *_bool_flag(bool(args.stop_on_complete), "--stop-on-complete"),
            "--enable-narrative-node-selector",
            "--node-selection-mode",
            args.node_selection_mode,
            "--node-min-nodes",
            str(max(1, int(args.node_min_nodes))),
            "--node-max-nodes",
            str(max(1, int(args.node_max_nodes))),
            *_bool_flag(bool(args.node_llm_scoring), "--node-llm-scoring"),
            *_bool_flag(bool(args.strict_online), "--strict-online"),
            "--retries",
            str(max(1, int(args.retries))),
            *_bool_flag(bool(args.video_dry_run), "--video-dry-run"),
            "--max-reference-images",
            str(max(1, int(args.max_reference_images))),
            "--multi-shot-events",
            "--event-shot-count",
            str(max(1, int(args.event_shot_count))),
            *_bool_flag(bool(args.use_previous_tail_frame), "--use-previous-tail-frame"),
            *_bool_flag(bool(args.use_previous_audio), "--use-previous-audio"),
            *_bool_flag(bool(args.upload_previous_audio), "--upload-previous-audio"),
            *_bool_flag(bool(args.dialogue_backfill), "--dialogue-backfill"),
            *_bool_flag(bool(args.insert_auxiliary_videos), "--insert-auxiliary-videos"),
            "--video-reference-mode",
            "event_keyframe_only",
            "--log-level",
            args.log_level,
            "--mcts-budget-iters",
            str(max(1, int(args.mcts_budget_iters))),
            "--mcts-topk-actions",
            str(max(1, int(args.mcts_topk_actions))),
            "--mcts-cpuct",
            str(float(args.mcts_cpuct)),
            "--mcts-rollout-horizon",
            str(max(1, int(args.mcts_rollout_horizon))),
            "--mcts-rollout-topk",
            str(max(1, int(args.mcts_rollout_topk))),
            "--mcts-rollout-epsilon",
            str(float(args.mcts_rollout_epsilon)),
            "--mcts-pw-base",
            str(float(args.mcts_pw_base)),
            "--mcts-pw-alpha",
            str(float(args.mcts_pw_alpha)),
            "--pov-switch-penalty",
            str(float(args.pov_switch_penalty)),
            "--over-focus-threshold",
            str(max(1, int(args.over_focus_threshold))),
            "--over-focus-penalty",
            str(float(args.over_focus_penalty)),
            "--w-narrative-causality",
            str(float(args.w_narrative_causality)),
            "--w-visual-transition",
            str(float(args.w_visual_transition)),
            "--w-rhythmic-pacing",
            str(float(args.w_rhythmic_pacing)),
            "--random-seed",
            str(int(args.random_seed)),
        ]
    )
    if bool(getattr(args, "dry_run", False)):
        cmd.append("--dry-run")
    if args.event_duration_sec is not None:
        cmd.extend(["--event-duration-sec", str(float(args.event_duration_sec))])
    if args.mcts_model:
        cmd.extend(["--mcts-model", str(args.mcts_model)])
    return cmd


def _make_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("VIDEO_GEN_PROVIDER", "minimax_h3")
    env.setdefault("MINIMAX_H3_PROMPT_STYLE", args.prompt_style)
    env.setdefault("MINIMAX_H3_VLLM_ENABLED", "true" if args.vllm_enabled else "false")
    env.setdefault("MINIMAX_H3_VLLM_API_URL", args.minimax_api_url)
    env.setdefault("MINIMAX_H3_CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("MINIMAX_H3_VLLM_TIMEOUT_SEC", str(int(args.timeout_sec)))
    env["MINIMAX_H3_NUM_FRAMES"] = "243"
    env["CAMERAMASTER_RUN_LOG_DIR"] = str(args.log_dir)
    env["CAMERAMASTER_CALL_LOG_JSONL"] = str(args.log_dir / "keyframe_only_event_video_model_tool_calls.jsonl")
    return env


def _configure_logging(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("keyframe_only_event_video_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _stream_process(cmd: list[str], *, env: dict[str, str], logger: logging.Logger, log_file: Path) -> None:
    logger.info("CMD %s", " ".join(str(x) for x in cmd))
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _line_level(line: str, fallback: int) -> int:
        match = CHILD_LEVEL_RE.search(line)
        if not match:
            return fallback
        return getattr(logging, match.group(1), fallback)

    def pump(stream: Any, prefix: str, fallback: int) -> None:
        if stream is None:
            return
        with log_file.open("a", encoding="utf-8") as fh:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                rendered = f"[full_pipeline]{prefix}{line}"
                fh.write(rendered + "\n")
                fh.flush()
                logger.log(_line_level(line, fallback), rendered)
            stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, "[stdout] ", logging.INFO), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, "[stderr] ", logging.INFO), daemon=True),
    ]
    for thread in threads:
        thread.start()
    rc = proc.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    logger.info("DONE rc=%d elapsed_sec=%.1f", rc, time.time() - started)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summarize_outputs(output_dir: Path, backend_mode: str = "all_banana") -> dict[str, Any]:
    run_dir = output_dir / backend_mode
    path_dir = run_dir / "path_orchestration"
    video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
    path_manifest = _load_json(path_dir / "pipeline_manifest.json")
    nodes_payload = _load_json(path_dir / "narrative_nodes.json")
    tasks_payload = _load_json(video_dir / "event_video_tasks.json")
    summary_payload = _load_json(video_dir / "event_video_summary.json")
    if not isinstance(path_manifest, dict):
        path_manifest = {}
    if not isinstance(nodes_payload, dict):
        nodes_payload = {}
    if not isinstance(tasks_payload, list):
        tasks_payload = []
    if not isinstance(summary_payload, dict):
        summary_payload = {}

    event_tasks = [task for task in tasks_payload if isinstance(task, dict) and task.get("task_type", "event") == "event"]
    auxiliary_tasks = [task for task in tasks_payload if isinstance(task, dict) and task.get("task_type", "event") != "event"]
    unexpected = [
        {
            "node_id": str(task.get("node_id", "")),
            "reference_mode": str(task.get("reference_mode", "")),
        }
        for task in event_tasks
        if str(task.get("reference_mode", "")) != "event_keyframe_only"
    ]
    final_video = video_dir / "final_orchestrated_events.mp4"
    artifacts = path_manifest.get("artifacts", {}) if isinstance(path_manifest.get("artifacts"), dict) else {}
    path_plan_json = (
        str(artifacts.get("llm_path_json") or artifacts.get("mcts_path_json") or "").strip()
        or str(path_dir / "path_llm_direct" / "narrative_path_plan.json")
    )
    return {
        "output_dir": str(output_dir),
        "pipeline_result_json": str(run_dir / "pipeline_result.json"),
        "narrative_nodes_json": str(path_dir / "narrative_nodes.json"),
        "path_plan_json": path_plan_json,
        "video_task_json": str(video_dir / "event_video_tasks.json"),
        "video_summary_json": str(video_dir / "event_video_summary.json"),
        "final_video_path": str(final_video),
        "final_video_exists": final_video.exists() and final_video.stat().st_size > 0,
        "selected_node_count": len(nodes_payload.get("selected_nodes", []) or []),
        "event_task_count": len(event_tasks),
        "auxiliary_task_count": len(auxiliary_tasks),
        "event_reference_mode_ok": not unexpected,
        "unexpected_event_reference_modes": unexpected,
        "concat_ok": bool(summary_payload.get("concat_ok", False)),
        "video_success": int(summary_payload.get("success", 0) or 0),
        "video_failed": int(summary_payload.get("failed", 0) or 0),
        "video_skipped": int(summary_payload.get("skipped", 0) or 0),
    }


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Run CineWorldSim with Banana world/keyframe generation, up to 45 narrative nodes, "
            "event videos using event keyframes only, and auxiliary establishing/transition videos retained."
        )
    )
    parser.add_argument("--story-file", type=Path, default=None)
    parser.add_argument("--storyline", type=str, default="")
    parser.add_argument("--characters", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backend-mode",
        choices=["all_banana", "all_flux2_klein"],
        default="all_banana",
        help="Image backend for world assets and event keyframes.",
    )
    parser.add_argument("--path-mode", choices=["llm", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--stop-on-complete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--node-selection-mode", choices=["hybrid", "threshold", "top_k"], default="top_k")
    parser.add_argument("--node-min-nodes", type=int, default=1)
    parser.add_argument("--node-max-nodes", type=int, default=45)
    parser.add_argument("--node-llm-scoring", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=None)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--use-previous-tail-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dialogue-backfill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--prompt-style", type=str, default="official_ref")
    parser.add_argument("--num-frames", type=int, default=243)
    parser.add_argument("--vllm-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimax-api-url", type=str, default=DEFAULT_MINIMAX_API_URL)
    parser.add_argument("--cuda-visible-devices", type=str, default="2,3")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--mcts-model", type=str, default=None)
    parser.add_argument("--mcts-budget-iters", type=int, default=1)
    parser.add_argument("--mcts-topk-actions", type=int, default=8)
    parser.add_argument("--mcts-cpuct", type=float, default=1.2)
    parser.add_argument("--mcts-rollout-horizon", type=int, default=32)
    parser.add_argument("--mcts-rollout-topk", type=int, default=5)
    parser.add_argument("--mcts-rollout-epsilon", type=float, default=0.2)
    parser.add_argument("--mcts-pw-base", type=float, default=1.5)
    parser.add_argument("--mcts-pw-alpha", type=float, default=0.55)
    parser.add_argument("--pov-switch-penalty", type=float, default=0.05)
    parser.add_argument("--over-focus-threshold", type=int, default=4)
    parser.add_argument("--over-focus-penalty", type=float, default=0.08)
    parser.add_argument("--w-narrative-causality", type=float, default=0.4)
    parser.add_argument("--w-visual-transition", type=float, default=0.3)
    parser.add_argument("--w-rhythmic-pacing", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.log_dir = args.output_dir / "logs"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(args.log_dir / "keyframe_only_event_video_pipeline.log", args.log_level)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Event video reference mode: event_keyframe_only")
    logger.info("Auxiliary videos enabled: %s", bool(args.insert_auxiliary_videos))
    logger.info("Node cap: min=%d max=%d strict_45=false", args.node_min_nodes, args.node_max_nodes)

    cmd = _build_cmd(args)
    _stream_process(cmd, env=_make_env(args), logger=logger, log_file=args.log_dir / "full_pipeline_stream.log")
    summary = _summarize_outputs(args.output_dir, args.backend_mode)
    manifest_path = _write_json(args.output_dir / "keyframe_only_event_video_pipeline_manifest.json", summary)
    logger.info("Manifest written: %s", manifest_path)
    if not summary["event_reference_mode_ok"]:
        raise SystemExit(
            "Event video reference mode check failed: "
            + json.dumps(summary["unexpected_event_reference_modes"], ensure_ascii=False)
        )
    print(manifest_path)


if __name__ == "__main__":
    main()
