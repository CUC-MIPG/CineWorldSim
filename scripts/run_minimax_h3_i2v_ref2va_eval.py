#!/usr/bin/env python3
"""Compare MiniMax-H3 event-keyframe reference video against semantic multi-reference video.

Both groups consume the same narrative nodes and prompt. The event-keyframe
group always uses ref2va, regardless of whether the event has one or multiple
keyframes. The semantic group excludes event keyframes and uses only character,
location, and explicitly mentioned item assets.
"""

from __future__ import annotations

import argparse
import json
import logging
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


LOGGER = logging.getLogger(__name__)
EVENT_KEYFRAME_GROUP = "fl2va_event_keyframe_only"
SEMANTIC_GROUP = "ref2va_semantic_assets"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _node_map(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("selected_nodes") or payload.get("nodes") or payload.get("narrative_nodes") or []
    else:
        rows = []
    return {
        str(row.get("node_id")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("node_id") or "").strip()
    }


def _ordered_nodes(nodes_path: Path, plan_path: Path) -> tuple[list[dict[str, Any]], str]:
    nodes_payload = _read_json(nodes_path)
    plan_payload = _read_json(plan_path)
    node_map = _node_map(nodes_payload)
    if isinstance(plan_payload, dict):
        ordered_ids = [str(value) for value in plan_payload.get("ordered_node_ids", []) or [] if str(value).strip()]
        storyline = str(plan_payload.get("storyline") or nodes_payload.get("storyline", ""))
    else:
        ordered_ids = []
        storyline = str(nodes_payload.get("storyline", "")) if isinstance(nodes_payload, dict) else ""
    if not ordered_ids:
        ordered_ids = list(node_map)
    nodes = [node_map[node_id] for node_id in ordered_ids if node_id in node_map]
    return nodes, storyline


def _find_source_task_json(pipeline_dir: Path) -> Path | None:
    candidates = sorted(pipeline_dir.rglob("event_video_tasks.json"))
    return candidates[0] if candidates else None


def _load_source_prompts(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    rows = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("node_id")): row
        for row in rows
        if isinstance(row, dict)
        and row.get("task_type", "event") == "event"
        and str(row.get("node_id") or "").strip()
        and str(row.get("prompt") or "").strip()
    }


def _prompt_for_node(
    agent: VideoAgent,
    node: dict[str, Any],
    storyline: str,
    source_prompts: dict[str, dict[str, Any]],
    prompt_style: str | None,
) -> tuple[str, str]:
    source = source_prompts.get(str(node.get("node_id")))
    if source is not None:
        return str(source["prompt"]), "source_event_video_tasks"
    artifact = agent._event_prompt_artifact(node, storyline=storyline, prompt_style=prompt_style)
    return str(artifact["prompt"]), "generated_from_narrative_node"


def _build_group_task(
    *,
    group: str,
    rank: int,
    node: dict[str, Any],
    prompt: str,
    prompt_source: str,
    refs: list[str],
    reference_bundle: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    node_id = str(node.get("node_id"))
    keyframe_group = group == EVENT_KEYFRAME_GROUP
    minimax_task = "ref2va"
    output_path = output_root / group / f"event_video_{rank:03d}_{node_id}.mp4"
    return {
        "rank": rank,
        "node_id": node_id,
        "event_id": node.get("event_id", ""),
        "task_type": "event",
        "comparison_group": group,
        "comparison_definition": (
            "event keyframe only; single and multiple keyframes both use ref2va"
            if keyframe_group
            else "semantic assets only; character, location, and explicitly mentioned item references"
        ),
        "reference_mode": "event_keyframe_only" if keyframe_group else "semantic_assets",
        "minimax_task": minimax_task,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "reference_images": refs,
        "reference_selection": reference_bundle,
        "characters": node.get("characters", []),
        "locations": node.get("locations", []),
        "items": node.get("items", []),
        "event_summary": node.get("event_summary", ""),
        "story_description": node.get("story_description", ""),
        "output_video_path": str(output_path),
        "duration_sec": 10,
        "num_frames": 243,
        "status": "planned",
    }


def build_comparison_tasks(
    nodes_path: Path,
    plan_path: Path,
    output_root: Path,
    *,
    max_reference_images: int = 4,
    start_rank: int = 1,
    end_rank: int | None = None,
    source_task_json: Path | None = None,
    prompt_style: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    nodes, storyline = _ordered_nodes(nodes_path, plan_path)
    agent = VideoAgent(strict_online=False, retries=0)
    source_prompts = _load_source_prompts(source_task_json)
    groups = {EVENT_KEYFRAME_GROUP: [], SEMANTIC_GROUP: []}

    for rank, node in enumerate(nodes, start=1):
        if rank < max(1, int(start_rank)) or (end_rank is not None and rank > int(end_rank)):
            continue
        prompt, prompt_source = _prompt_for_node(agent, node, storyline, source_prompts, prompt_style)

        assets = node.get("visual_assets", {}) if isinstance(node.get("visual_assets"), dict) else {}
        keyframe_candidates = [
            str(path)
            for path in [*(assets.get("event_keyframes", []) or []), *(assets.get("character_keyframes", []) or [])]
            if str(path).strip()
        ]
        keyframe_refs = agent._dedupe_existing(
            keyframe_candidates,
            limit=max(1, int(max_reference_images)),
        )
        keyframe_bundle = {
            "reference_mode": "event_keyframe_only",
            "reference_image_paths": keyframe_refs,
            "event_keyframe_paths": keyframe_refs,
            "character_reference_paths": [],
            "location_reference_paths": [],
            "item_reference_paths": [],
            "excluded_reference_paths": [],
            "mentioned_items": [],
            "multi_person_reference_allowed": False,
            "reference_selection_reason": "all available event keyframes retained for this comparison group",
        }
        groups[EVENT_KEYFRAME_GROUP].append(
            _build_group_task(
                group=EVENT_KEYFRAME_GROUP,
                rank=rank,
                node=node,
                prompt=prompt,
                prompt_source=prompt_source,
                refs=keyframe_bundle["reference_image_paths"],
                reference_bundle=keyframe_bundle,
                output_root=output_root,
            )
        )

        semantic_node = {**node, "video_reference_mode": "semantic_assets"}
        semantic_bundle = agent._node_reference_bundle(
            semantic_node,
            include_character_keyframes=True,
            include_world_images=True,
            max_reference_images=max(1, int(max_reference_images)),
        )
        groups[SEMANTIC_GROUP].append(
            _build_group_task(
                group=SEMANTIC_GROUP,
                rank=rank,
                node=node,
                prompt=prompt,
                prompt_source=prompt_source,
                refs=semantic_bundle["reference_image_paths"],
                reference_bundle=semantic_bundle,
                output_root=output_root,
            )
        )
    return groups


def _write_group_tasks(group_dir: Path, tasks: list[dict[str, Any]]) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / "event_video_tasks.json"
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _run_group(
    agent: VideoAgent,
    group: str,
    tasks: list[dict[str, Any]],
    *,
    dry_run: bool,
    concat: bool,
    output_root: Path,
) -> dict[str, Any]:
    group_dir = output_root / group
    task_json = _write_group_tasks(group_dir, tasks)
    generated: list[dict[str, Any]] = []
    success = failed = skipped = 0
    for task in tasks:
        output = Path(task["output_video_path"])
        refs = list(task.get("reference_images", []) or [])
        if not refs:
            task["status"] = "skipped"
            task["error"] = "no_reference_images"
            skipped += 1
            generated.append(dict(task))
            continue
        if output.exists() and output.stat().st_size > 0:
            task["status"] = "done"
            task["resume_existing"] = True
            success += 1
            generated.append(dict(task))
            continue
        if dry_run:
            task["status"] = "dry_run"
            success += 1
            generated.append(dict(task))
            continue
        try:
            ok = agent.send_generation_request(
                prompt=str(task["prompt"]),
                image_paths=refs,
                video_path=str(output),
                video_mode="reference",
                first_image_path="",
                reference_image_paths=refs,
                minimax_task=str(task["minimax_task"]),
                duration_sec=10.0,
                num_frames=243,
            )
            task["status"] = "done" if ok else "failed"
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            LOGGER.exception("comparison generation failed group=%s node_id=%s", group, task["node_id"])
        generated.append(dict(task))

    task_json.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "group": group,
        "task_json": str(task_json),
        "total": len(tasks),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "generated": generated,
    }
    if concat and not dry_run:
        paths = [row["output_video_path"] for row in generated if row.get("status") == "done"]
        concat_path = group_dir / "final_comparison.mp4"
        result["concat_output"] = str(concat_path)
        result["concat_ok"] = bool(paths and agent.concat_videos(paths, str(concat_path)))
    return result


def run_comparison(
    *,
    nodes_path: Path,
    plan_path: Path,
    output_root: Path,
    dry_run: bool = True,
    max_reference_images: int = 4,
    start_rank: int = 1,
    end_rank: int | None = None,
    source_task_json: Path | None = None,
    prompt_style: str | None = None,
    concat: bool = False,
    strict_online: bool = True,
    retries: int = 3,
) -> dict[str, Any]:
    groups = build_comparison_tasks(
        nodes_path,
        plan_path,
        output_root,
        max_reference_images=max_reference_images,
        start_rank=start_rank,
        end_rank=end_rank,
        source_task_json=source_task_json,
        prompt_style=prompt_style,
    )
    agent = VideoAgent(strict_online=strict_online, retries=retries)
    group_results = {
        group: _run_group(agent, group, tasks, dry_run=dry_run, concat=concat, output_root=output_root)
        for group, tasks in groups.items()
    }
    summary = {
        "nodes_json": str(nodes_path),
        "path_plan_json": str(plan_path),
        "output_root": str(output_root),
        "dry_run": dry_run,
        "max_reference_images": max_reference_images,
        "source_task_json": str(source_task_json) if source_task_json else "",
        "video_provider": os.getenv("VIDEO_GEN_PROVIDER", ""),
        "minimax_task_semantics": {
            EVENT_KEYFRAME_GROUP: "single and multiple event keyframes -> ref2va",
            SEMANTIC_GROUP: "character/location/mentioned-item assets -> ref2va",
        },
        "groups": group_results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MiniMax-H3 event-keyframe I2V with semantic multi-reference video.")
    parser.add_argument("--pipeline-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--narrative-nodes-json", type=Path, default=None)
    parser.add_argument("--path-plan-json", type=Path, default=None)
    parser.add_argument("--source-task-json", type=Path, default=None)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--end-rank", type=int, default=None)
    parser.add_argument("--prompt-style", default=None)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    load_dotenv(override=False)
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    pipeline_dir = args.pipeline_run_dir.resolve()
    nodes_path = (args.narrative_nodes_json or pipeline_dir / "path_orchestration" / "narrative_nodes.json").resolve()
    plan_path = (
        args.path_plan_json or pipeline_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"
    ).resolve()
    if not nodes_path.exists():
        raise FileNotFoundError(f"narrative nodes JSON not found: {nodes_path}")
    if not plan_path.exists():
        raise FileNotFoundError(f"path plan JSON not found: {plan_path}")
    source_task_json = args.source_task_json.resolve() if args.source_task_json else _find_source_task_json(pipeline_dir)
    summary = run_comparison(
        nodes_path=nodes_path,
        plan_path=plan_path,
        output_root=args.output_root.resolve(),
        dry_run=args.dry_run,
        max_reference_images=args.max_reference_images,
        start_rank=args.start_rank,
        end_rank=args.end_rank,
        source_task_json=source_task_json,
        prompt_style=args.prompt_style,
        concat=args.concat,
        strict_online=args.strict_online,
        retries=args.retries,
    )
    print(json.dumps({"summary": str(args.output_root.resolve() / "comparison_summary.json"), "groups": {
        key: {field: value.get(field) for field in ("total", "success", "failed", "skipped")}
        for key, value in summary["groups"].items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
