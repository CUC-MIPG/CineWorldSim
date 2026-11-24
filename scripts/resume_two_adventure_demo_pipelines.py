#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/four_genre_demos_15nodes_20260816"


def _configure_logging(log_file: Path, log_level: str) -> logging.Logger:
    logger = logging.getLogger("two_demo_resume")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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


def _run_streamed(cmd: list[str], *, label: str, env: dict[str, str], logger: logging.Logger, log_file: Path) -> None:
    logger.info("START %s", label)
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

    def pump(stream: Any, prefix: str) -> None:
        if stream is None:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                rendered = f"[{label}]{prefix}{line}"
                fh.write(rendered + "\n")
                fh.flush()
                # Child Python logging writes INFO to stderr; keep it as INFO unless the line declares ERROR/CRITICAL.
                level = logging.ERROR if " | ERROR | " in line or "Traceback" in line else logging.INFO
                logger.log(level, rendered)
            stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, "[stdout] "), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, "[stderr] "), daemon=True),
    ]
    for thread in threads:
        thread.start()
    rc = proc.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    elapsed = time.time() - started
    logger.info("DONE %s rc=%d elapsed_sec=%.1f", label, rc, elapsed)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _env(args: argparse.Namespace, log_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    env.setdefault("MINIMAX_H3_NUM_FRAMES", "243")
    env.setdefault("MINIMAX_H3_VLLM_ENABLED", "true")
    env.setdefault("MINIMAX_H3_VLLM_API_URL", args.minimax_api_url)
    env.setdefault("MINIMAX_H3_CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("MINIMAX_H3_VLLM_TIMEOUT_SEC", str(args.timeout_sec))
    env["CAMERAMASTER_RUN_LOG_DIR"] = str(log_dir)
    env["CAMERAMASTER_CALL_LOG_JSONL"] = str(log_dir / "resume_model_tool_calls.jsonl")
    return env


def _demo1_video_cmd(args: argparse.Namespace) -> list[str]:
    run_dir = args.output_root / "01_demo1_shared_space" / "pipeline" / "all_banana"
    video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_orchestrated_event_video_generation.py"),
        "--pipeline-run-dir",
        str(run_dir),
        "--output-dir",
        str(video_dir),
        "--max-reference-images",
        str(args.max_reference_images),
        "--start-rank",
        "1",
        "--multi-shot-events",
        "--event-shot-count",
        str(args.event_shot_count),
        "--no-use-previous-tail-frame",
        "--no-use-previous-audio",
        "--no-upload-previous-audio",
        "--dialogue-backfill",
        "--no-dry-run",
        "--strict-online",
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--log-level",
        args.log_level,
    ]
    if args.event_duration_sec is not None:
        cmd.extend(["--event-duration-sec", str(args.event_duration_sec)])
    cmd.extend(["--insert-auxiliary-videos" if args.insert_auxiliary_videos else "--no-insert-auxiliary-videos"])
    if args.video_reference_mode:
        cmd.extend(["--video-reference-mode", args.video_reference_mode])
    return cmd


def _demo2_full_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_narrative_world_video_pipeline.py"),
        "--story-file",
        str(PROJECT_ROOT / "inputs" / "demo2_adventure_three_spaces.txt"),
        "--characters",
        "Leona,Darius,Mira",
        "--output-dir",
        str(args.output_root / "02_demo2_three_spaces" / "pipeline"),
        "--backend-mode",
        "all_banana",
        "--path-mode",
        "both",
        "--max-steps",
        "15",
        "--stop-on-complete",
        "--enable-narrative-node-selector",
        "--node-selection-mode",
        "top_k",
        "--node-min-nodes",
        "15",
        "--node-max-nodes",
        "15",
        "--node-llm-scoring",
        "--strict-online",
        "--multi-shot-events",
        "--event-shot-count",
        str(args.event_shot_count),
        "--max-reference-images",
        str(args.max_reference_images),
        "--no-use-previous-tail-frame",
        "--no-use-previous-audio",
        "--no-upload-previous-audio",
        "--dialogue-backfill",
        "--log-level",
        args.log_level,
    ]
    if args.event_duration_sec is not None:
        cmd.extend(["--event-duration-sec", str(args.event_duration_sec)])
    cmd.extend(["--insert-auxiliary-videos" if args.insert_auxiliary_videos else "--no-insert-auxiliary-videos"])
    if args.video_reference_mode:
        cmd.extend(["--video-reference-mode", args.video_reference_mode])
    return cmd


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"output_root": str(args.output_root), "demos": {}}
    for demo in ["01_demo1_shared_space", "02_demo2_three_spaces"]:
        root = args.output_root / demo / "pipeline" / "all_banana"
        video_dir = root / "event_video_pipeline_minimax_h3_vllm_multishot"
        item: dict[str, Any] = {
            "pipeline_result": str(root / "pipeline_result.json"),
            "pipeline_result_exists": (root / "pipeline_result.json").exists(),
            "narrative_nodes_exists": (root / "path_orchestration" / "narrative_nodes.json").exists(),
            "video_count": len(list(video_dir.glob("event_video_*.mp4"))) if video_dir.exists() else 0,
            "event_video_summary_exists": (video_dir / "event_video_summary.json").exists(),
            "final_video_exists": (video_dir / "final_orchestrated_events.mp4").exists(),
        }
        if item["narrative_nodes_exists"]:
            data = json.loads((root / "path_orchestration" / "narrative_nodes.json").read_text(encoding="utf-8"))
            item["selected_nodes"] = len(data.get("selected_nodes") or [])
        out["demos"][demo] = item
    return out


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Resume the interrupted two adventure demo run.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cuda-visible-devices", type=str, default="2,3")
    parser.add_argument("--minimax-api-url", type=str, default="http://127.0.0.1:9001/v1/videos/sync")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=None)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-reference-mode",
        choices=["event_keyframe_only", "semantic_assets", "all_allowed"],
        default="all_allowed",
    )
    parser.add_argument("--skip-demo1-video", action="store_true")
    parser.add_argument("--skip-demo2", action="store_true")
    args = parser.parse_args()

    log_dir = args.output_root / "logs"
    logger = _configure_logging(log_dir / "resume_two_demo_runner.log", args.log_level)
    env = _env(args, log_dir)

    if not args.skip_demo1_video:
        _run_streamed(
            _demo1_video_cmd(args),
            label="resume_demo1_video",
            env=env,
            logger=logger,
            log_file=log_dir / "resume_demo1_video.log",
        )
    if not args.skip_demo2:
        _run_streamed(
            _demo2_full_cmd(args),
            label="run_demo2_full",
            env=env,
            logger=logger,
            log_file=log_dir / "resume_demo2_full.log",
        )

    summary_path = args.output_root / "resume_two_demo_summary.json"
    summary_path.write_text(json.dumps(_summarize(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("SUMMARY %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
