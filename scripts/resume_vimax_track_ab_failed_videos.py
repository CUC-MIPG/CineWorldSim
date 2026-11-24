#!/usr/bin/env python3
"""Resume only missing event videos from a CameraMaster Track A/B run.

The original ``event_video_tasks.json`` is treated as immutable input: its
stored prompts and reference-image selections are reused verbatim.  This
avoids rerunning the narrative simulation or regenerating videos that already
exist.  The script updates the accompanying summary and concatenates the
final video only if every task has a non-empty MP4 output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_dotenv_if_unset(path: Path) -> None:
    """Load the simple KEY=VALUE entries needed by generation backends.

    This keeps the recovery utility usable before an interactive conda shell is
    activated. Existing exported variables keep priority, matching the project
    pipeline's dotenv configuration.
    """

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _has_video(task: dict[str, Any]) -> bool:
    output = str(task.get("output_video_path") or "").strip()
    if not output:
        return False
    path = Path(output)
    return path.is_file() and path.stat().st_size > 0


def _check_service(url: str, timeout_sec: float) -> None:
    """Confirm the configured API endpoint accepts connections.

    The sync endpoint normally rejects a GET request with HTTP 405.  That is
    still a successful reachability check; connection, DNS, and timeout
    failures are not.
    """

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec):
            return
    except urllib.error.HTTPError:
        return
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"MiniMax service is not reachable at {url}: {exc}") from exc


def _refresh_summary(summary: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Mark only outputs proven present on disk as done, without replacing prompts."""

    rows = summary.get("generated")
    if not isinstance(rows, list) or not rows:
        rows = [dict(task) for task in tasks]
        summary["generated"] = rows

    task_outputs = {str(task.get("output_video_path") or "") for task in tasks}
    seen_outputs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = str(row.get("output_video_path") or "")
        if output not in task_outputs:
            continue
        seen_outputs.add(output)
        if _has_video(row):
            row["status"] = "done"
            row.pop("resume_existing", None)
        elif row.get("status") == "done":
            row["status"] = "failed"

    # Preserve a complete per-task accounting even if a historic summary was
    # truncated by a previous interrupted run.
    for task in tasks:
        output = str(task.get("output_video_path") or "")
        if output in seen_outputs:
            continue
        rows.append({**task, "status": "done" if _has_video(task) else "failed"})

    done = sum(isinstance(row, dict) and row.get("status") == "done" for row in rows)
    failed = sum(isinstance(row, dict) and row.get("status") == "failed" for row in rows)
    skipped = sum(isinstance(row, dict) and row.get("status") == "skipped" for row in rows)
    summary.update(
        status="done" if failed == 0 and done == len(tasks) else "partial",
        total=len(tasks),
        success=done,
        failed=failed,
        skipped=skipped,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume only missing MP4s from an existing CameraMaster event-video run."
    )
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing event_video_tasks.json")
    parser.add_argument("--resume-task-json", type=Path, default=None, help="Where to save the generated missing-task list")
    parser.add_argument("--concat-output", type=Path, default=None)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument("--skip-service-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write and report the missing-task list")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    _load_dotenv_if_unset(ROOT / ".env")

    video_dir = args.video_dir.resolve()
    tasks_path = video_dir / "event_video_tasks.json"
    summary_path = video_dir / "event_video_summary.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Task list not found: {tasks_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Video summary not found: {summary_path}")

    tasks_payload = _load_json(tasks_path)
    if not isinstance(tasks_payload, list) or not all(isinstance(row, dict) for row in tasks_payload):
        raise RuntimeError(f"Invalid task list: {tasks_path}")
    tasks: list[dict[str, Any]] = tasks_payload
    pending = [task for task in tasks if not _has_video(task)]
    resume_path = (args.resume_task_json or (video_dir / "resume_missing_tasks.json")).resolve()
    _write_json(resume_path, pending)

    print(f"Tasks total: {len(tasks)}")
    print(f"Videos already present: {len(tasks) - len(pending)}")
    print(f"Videos pending: {len(pending)}")
    print(f"Resume task list: {resume_path}")
    for task in pending:
        print(f"PENDING: {task.get('output_video_path', '')}")

    if args.dry_run:
        return

    if pending:
        endpoint = os.getenv("MINIMAX_H3_VLLM_API_URL", "http://127.0.0.1:9001/v1/videos/sync")
        if not args.skip_service_check:
            _check_service(endpoint, args.service_timeout_sec)
            print(f"MiniMax service reachable: {endpoint}")

        agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)
        result = agent.process_task_file(resume_path, strict_images=False)
        print(
            "Resume generation: "
            f"total={result['total']} success={result['success']} "
            f"failed={result['failed']} skipped={result['skipped']}"
        )

    summary_payload = _load_json(summary_path)
    summary = summary_payload if isinstance(summary_payload, dict) else {}
    summary = _refresh_summary(summary, tasks)
    _write_json(summary_path, summary)
    print(
        "Updated summary: "
        f"status={summary['status']} success={summary['success']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )

    if summary["status"] != "done":
        raise SystemExit("Some videos are still missing; final concatenation was intentionally not replaced.")

    concat_output = (args.concat_output or (video_dir / "final_orchestrated_events.mp4")).resolve()
    ordered_paths = [str(Path(str(task["output_video_path"])).resolve()) for task in tasks]
    agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)
    if not agent.concat_videos(ordered_paths, concat_output):
        raise SystemExit(f"Failed to concatenate complete video set: {concat_output}")

    summary["concat_output"] = str(concat_output)
    summary["concat_inputs"] = ordered_paths
    summary["concat_ok"] = True
    _write_json(summary_path, summary)
    print(f"Completed video: {concat_output}")


if __name__ == "__main__":
    main()
