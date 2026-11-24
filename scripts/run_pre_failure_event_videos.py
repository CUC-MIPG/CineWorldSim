#!/usr/bin/env python3
"""Generate event videos before a known failed simulation step.

The script keeps the original pipeline untouched, filters event nodes by their
source step, and deliberately does not concatenate the generated clips.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_test_credentials(path: Path, model: str) -> None:
    """Map the legacy API_KEY/BASE_URL file to the project's env names."""
    if path.exists():
        values: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        if values.get("API_KEY"):
            os.environ["OPENAI_API_KEY"] = values["API_KEY"]
            os.environ["VIDEO_API_KEY"] = values["API_KEY"]
        if values.get("BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = values["BASE_URL"]
            os.environ["VIDEO_BASE_URL"] = values["BASE_URL"]
    os.environ["OPENAI_MODEL"] = model
    os.environ["VIDEO_MODEL_NAME"] = model


def _node_step(node: dict[str, Any]) -> int | None:
    steps = node.get("source_steps")
    if isinstance(steps, list):
        parsed = []
        for value in steps:
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                continue
        if parsed:
            return min(parsed)
    for key in ("step", "step_id"):
        try:
            if node.get(key) is not None:
                return int(node[key])
        except (TypeError, ValueError):
            pass
    return None


def _filter_inputs(
    nodes_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    failed_step: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    selected = [
        node
        for node in nodes_payload.get("selected_nodes", []) or []
        if isinstance(node, dict) and (_node_step(node) is not None)
        and _node_step(node) < failed_step
    ]
    allowed_ids = {
        str(node.get("node_id") or node.get("event_id"))
        for node in selected
    }
    filtered_nodes = dict(nodes_payload)
    filtered_nodes["selected_nodes"] = selected
    filtered_nodes["selected_count"] = len(selected)

    ordered = [
        str(node_id)
        for node_id in plan_payload.get("ordered_node_ids", []) or []
        if str(node_id) in allowed_ids
    ]
    filtered_plan = dict(plan_payload)
    filtered_plan["ordered_node_ids"] = ordered
    filtered_plan["node_rank_map"] = {
        node_id: rank for rank, node_id in enumerate(ordered, start=1)
    }
    filtered_plan["candidate_count"] = len(ordered)
    return filtered_nodes, filtered_plan, len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate separate event videos before a known failed step."
    )
    parser.add_argument("--pipeline-run-dir", type=Path, required=True)
    parser.add_argument("--failed-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-env-file", type=Path, default=ROOT / "test.txt")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--video-reference-mode", choices=["event_keyframe_only", "semantic_assets", "all_allowed"], default="all_allowed")
    parser.add_argument("--multi-shot-events", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=None)
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=False,
                        help="Concatenate generated clips. Disabled by default.")
    parser.add_argument("--dialogue-backfill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-previous-tail-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    pipeline_dir = args.pipeline_run_dir.resolve()
    nodes_path = pipeline_dir / "path_orchestration" / "narrative_nodes.json"
    plan_path = pipeline_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"
    for path in (nodes_path, plan_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.failed_step <= 1:
        raise ValueError("--failed-step must be greater than 1")

    load_dotenv(override=False)
    _load_test_credentials(args.api_env_file.resolve(), args.model)
    configure_logging(args.log_level)

    nodes, plan, count = _filter_inputs(_read_json(nodes_path), _read_json(plan_path), args.failed_step)
    if count == 0:
        raise RuntimeError(f"No selected event nodes found before step {args.failed_step}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "filtered_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    filtered_nodes_path = input_dir / "narrative_nodes.json"
    filtered_plan_path = input_dir / "narrative_path_plan.json"
    filtered_nodes_path.write_text(json.dumps(nodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    filtered_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=filtered_nodes_path,
        path_plan_json=filtered_plan_path,
        output_dir=output_dir,
        dry_run=args.dry_run,
        max_reference_images=args.max_reference_images,
        include_character_keyframes=True,
        include_world_images=True,
        use_previous_tail_frame=args.use_previous_tail_frame,
        use_previous_audio=args.use_previous_audio,
        upload_previous_audio=args.upload_previous_audio,
        multi_shot_events=args.multi_shot_events,
        event_shot_count=args.event_shot_count,
        event_duration_sec=args.event_duration_sec,
        dialogue_backfill=args.dialogue_backfill,
        insert_auxiliary_videos=args.insert_auxiliary_videos,
        video_reference_mode=args.video_reference_mode,
    )
    summary.update({
        "concat_ok": False,
        "concat_skipped": not args.concat,
        "failed_step_boundary": args.failed_step,
        "source_pipeline_dir": str(pipeline_dir),
        "filtered_node_count": count,
    })
    if args.concat and not args.dry_run:
        video_paths = [
            str(row.get("output_video_path", ""))
            for row in summary.get("generated", [])
            if isinstance(row, dict) and row.get("status") == "done"
        ]
        concat_output = output_dir / "final_orchestrated_events.mp4"
        summary["concat_ok"] = bool(agent.concat_videos(video_paths, concat_output))
        summary["concat_output"] = str(concat_output)
    (output_dir / "event_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {summary.get('success', 0)}/{summary.get('total', 0)} event videos")
    print(f"Filtered nodes: {count} (source step < {args.failed_step})")
    print(f"Output: {output_dir}")
    print(f"Concat: {'enabled' if args.concat else 'skipped'}")
    return 0 if not summary.get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
