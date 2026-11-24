#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_existing_video_paths(summary_json: Path, tasks_json: Path) -> list[str]:
    summary = _load_json(summary_json) if summary_json.exists() else {}
    tasks = _load_json(tasks_json) if tasks_json.exists() else []
    ordered_tasks = tasks if isinstance(tasks, list) else []
    done_paths: set[str] = set()
    for row in summary.get("generated", []) if isinstance(summary, dict) else []:
        if not isinstance(row, dict) or row.get("status") != "done":
            continue
        path = str(row.get("output_video_path") or "").strip()
        if path and Path(path).exists():
            done_paths.add(str(Path(path).resolve()))
    ordered: list[str] = []
    for row in ordered_tasks:
        if not isinstance(row, dict):
            continue
        path = str(row.get("output_video_path") or "").strip()
        if not path:
            continue
        resolved = str(Path(path).resolve())
        if resolved in done_paths and resolved not in ordered:
            ordered.append(resolved)
    if not ordered:
        for path in done_paths:
            ordered.append(path)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Concat already generated event videos without calling generation APIs.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--concat-output", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    video_dir = args.video_dir
    summary_json = video_dir / "event_video_summary.json"
    tasks_json = video_dir / "event_video_tasks.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"event video summary not found: {summary_json}")
    if not tasks_json.exists():
        raise FileNotFoundError(f"event video tasks not found: {tasks_json}")

    concat_output = args.concat_output or (video_dir / "final_orchestrated_events.mp4")
    video_paths = collect_existing_video_paths(summary_json, tasks_json)
    agent = VideoAgent(strict_online=True, retries=1)
    concat_ok = agent.concat_videos(video_paths, concat_output)

    summary = _load_json(summary_json)
    if not isinstance(summary, dict):
        summary = {}
    summary["concat_output"] = str(concat_output)
    summary["concat_inputs"] = video_paths
    summary["concat_ok"] = bool(concat_ok)
    _write_json(summary_json, summary)

    print(json.dumps(
        {
            "concat_ok": bool(concat_ok),
            "concat_output": str(concat_output),
            "concat_inputs": video_paths,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
