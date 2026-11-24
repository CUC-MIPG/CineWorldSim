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

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/four_genre_demos_15nodes_20260816"
DEFAULT_MINIMAX_API_URL = "http://127.0.0.1:9001/v1/videos/sync"
CHILD_LEVEL_RE = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|")

DEFAULT_DEMOS = [
    {
        "label": "01_demo1_shared_space",
        "story_file": PROJECT_ROOT / "inputs/demo1_adventure_same_space.txt",
        "characters": "Mara Venn,Ilya Trent,Nora Hale",
    },
    {
        "label": "02_demo2_three_spaces",
        "story_file": PROJECT_ROOT / "inputs/demo2_adventure_three_spaces.txt",
        "characters": "Leona,Darius,Mira",
    },
]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _bool_flag(enabled: bool, flag: str) -> list[str]:
    return [flag] if enabled else [f"--no-{flag[2:]}"]


def _make_env(args: argparse.Namespace, *, demo_label: str, log_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MINIMAX_H3_PROMPT_STYLE", args.prompt_style)
    env["MINIMAX_H3_NUM_FRAMES"] = "243"
    env.setdefault("MINIMAX_H3_VLLM_ENABLED", "true" if args.vllm_enabled else "false")
    env.setdefault("MINIMAX_H3_VLLM_API_URL", args.minimax_api_url)
    env.setdefault("MINIMAX_H3_CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("MINIMAX_H3_VLLM_TIMEOUT_SEC", str(args.timeout_sec))
    env["CAMERAMASTER_RUN_LOG_DIR"] = str(log_dir)
    env["CAMERAMASTER_CALL_LOG_JSONL"] = str(log_dir / f"{demo_label}_model_tool_calls.jsonl")
    return env


def _preflight_env(backend_mode: str = "all_banana") -> None:
    if backend_mode == "all_flux2_klein":
        return
    missing = [key for key in ["NANO_BANANA_API_KEY"] if not os.getenv(key, "").strip()]
    if missing:
        raise SystemExit(
            "Missing required Nano Banana env vars: "
            + ", ".join(missing)
            + ". Set them before running the real pipeline."
        )

    base_url = os.getenv("NANO_BANANA_BASE_URL", "https://api.xi-ai.cn/v1").strip()
    if not base_url:
        raise SystemExit("NANO_BANANA_BASE_URL is empty.")
    if base_url.startswith("https://api.xi-ai.cn") and not os.getenv("NANO_BANANA_API_KEY", "").strip():
        raise SystemExit(
            "NANO_BANANA_BASE_URL points to api.xi-ai.cn but NANO_BANANA_API_KEY is missing; "
            "the image backend cannot start."
        )


def _build_cmd(
    args: argparse.Namespace,
    story_file: Path,
    output_dir: Path,
    characters: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_narrative_world_video_pipeline.py"),
        "--story-file",
        str(story_file),
        "--characters",
        characters,
        "--output-dir",
        str(output_dir),
        "--backend-mode",
        args.backend_mode,
        "--path-mode",
        args.path_mode,
        "--max-steps",
        str(args.max_steps),
        "--stop-on-complete",
        "--enable-narrative-node-selector",
        "--node-selection-mode",
        args.node_selection_mode,
        "--node-min-nodes",
        str(args.node_min_nodes),
        "--node-max-nodes",
        str(args.node_max_nodes),
        "--node-llm-scoring",
        "--strict-online",
        "--multi-shot-events",
        "--event-shot-count",
        str(args.event_shot_count),
        "--max-reference-images",
        str(args.max_reference_images),
        "--no-use-previous-audio",
        "--no-upload-previous-audio",
        "--log-level",
        args.log_level,
    ]
    if args.event_duration_sec is not None:
        cmd.extend(["--event-duration-sec", str(args.event_duration_sec)])
    cmd.extend(_bool_flag(args.insert_auxiliary_videos, "--insert-auxiliary-videos"))
    if args.video_reference_mode:
        cmd.extend(["--video-reference-mode", args.video_reference_mode])
    cmd.extend(_bool_flag(args.use_previous_tail_frame, "--use-previous-tail-frame"))
    cmd.extend(_bool_flag(args.dialogue_backfill, "--dialogue-backfill"))
    return cmd


def _configure_logging(log_file: Path, log_level: str) -> logging.Logger:
    logger = logging.getLogger("two_demo_runner")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _stream_process(
    proc: subprocess.Popen[str],
    *,
    logger: logging.Logger,
    label: str,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _line_level(line: str, fallback_level: int) -> int:
        match = CHILD_LEVEL_RE.search(line)
        if not match:
            return fallback_level
        return getattr(logging, match.group(1), fallback_level)

    def pump(stream: Any, fallback_level: int, prefix: str) -> None:
        if stream is None:
            return
        with log_path.open("a", encoding="utf-8") as fh:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                rendered = f"[{label}]{prefix}{line}"
                fh.write(rendered + "\n")
                fh.flush()
                logger.log(_line_level(line, fallback_level), rendered)
            stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, logging.INFO, "[stdout] "), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, logging.ERROR, "[stderr] "), daemon=True),
    ]
    for thread in threads:
        thread.start()
    rc = proc.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)


def run_demo(args: argparse.Namespace, *, demo: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    story_file = Path(demo["story_file"])
    if not story_file.exists():
        raise FileNotFoundError(f"Story file not found: {story_file}")

    demo_root = args.output_root / str(demo["label"])
    pipeline_root = demo_root / "pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / f"{demo['label']}.log"

    env = _make_env(args, demo_label=str(demo["label"]), log_dir=log_dir)
    cmd = _build_cmd(args, story_file, pipeline_root, str(demo["characters"]))
    start = time.time()
    logger.info("START demo=%s story=%s output=%s", demo["label"], story_file, pipeline_root)
    logger.info("CMD demo=%s %s", demo["label"], " ".join(str(x) for x in cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        _stream_process(proc, logger=logger, label=str(demo["label"]), log_path=run_log)
    finally:
        if proc.poll() is None:
            proc.kill()

    elapsed = time.time() - start
    manifest = {
        "demo_label": demo["label"],
        "story_file": str(story_file),
        "characters": demo["characters"],
        "output_dir": str(pipeline_root),
        "log_file": str(run_log),
        "command": cmd,
        "elapsed_sec": round(elapsed, 2),
        "returncode": 0,
    }
    logger.info("DONE demo=%s elapsed_sec=%.1f", demo["label"], elapsed)
    return manifest


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Run both adventure demos with real-time logging.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--story1-file", type=Path, default=DEFAULT_DEMOS[0]["story_file"])
    parser.add_argument("--story2-file", type=Path, default=DEFAULT_DEMOS[1]["story_file"])
    parser.add_argument(
        "--backend-mode",
        choices=["all_flux", "all_banana", "all_flux2_klein"],
        default="all_banana",
    )
    parser.add_argument("--path-mode", choices=["llm", "mcts", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--node-selection-mode", choices=["hybrid", "threshold", "top_k"], default="top_k")
    parser.add_argument("--node-min-nodes", type=int, default=15)
    parser.add_argument("--node-max-nodes", type=int, default=15)
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=None)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--prompt-style", type=str, default="official_ref")
    parser.add_argument("--num-frames", type=int, default=243)
    parser.add_argument("--cuda-visible-devices", type=str, default="2,3")
    parser.add_argument("--minimax-api-url", type=str, default=DEFAULT_MINIMAX_API_URL)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--use-previous-tail-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dialogue-backfill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-reference-mode",
        choices=["event_keyframe_only", "semantic_assets", "all_allowed"],
        default="all_allowed",
    )
    parser.add_argument("--vllm-enabled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _preflight_env(args.backend_mode)
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(args.output_root / "logs" / "two_demo_runner.log", args.log_level)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output root: %s", args.output_root)
    logger.info("MiniMax API: %s", args.minimax_api_url)
    logger.info("Tail frame enabled: %s", args.use_previous_tail_frame)
    logger.info("Dialogue backfill: %s", args.dialogue_backfill)
    logger.info("Auxiliary videos: %s", args.insert_auxiliary_videos)
    logger.info("Video reference mode: %s", args.video_reference_mode)

    demos = [
        {"label": "01_demo1_shared_space", "story_file": args.story1_file, "characters": DEFAULT_DEMOS[0]["characters"]},
        {"label": "02_demo2_three_spaces", "story_file": args.story2_file, "characters": DEFAULT_DEMOS[1]["characters"]},
    ]
    manifests: list[dict[str, Any]] = []
    for demo in demos:
        manifests.append(run_demo(args, demo=demo, logger=logger))

    manifest_path = _write_json(
        args.output_root / "two_demo_runner_manifest.json",
        {
            "output_root": str(args.output_root),
            "demos": manifests,
        },
    )
    logger.info("Manifest written: %s", manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
