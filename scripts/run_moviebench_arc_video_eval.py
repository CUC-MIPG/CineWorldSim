#!/usr/bin/env python3
"""Run a paired MiniMax-H3 video-reference pilot for one MovieBench L4 arc."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.call_logger import call_context
from agents.generation_backends import generate_image_nano_banana, nano_banana_settings_from_env
from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent
from narrative.visual_prompt_constraints import event_needs_multi_person_references
from scripts.run_minimax_h3_i2v_ref2va_eval import run_comparison

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")[:96] or "asset"


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        value = payload.get("result", {}).get(key, [])
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _l3_candidates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _resolve_l3_contexts(
    source_ids: list[str],
    source_l2_ids: list[str],
    l3_scenes: dict[str, Any],
    event_index: int,
    source_l1_by_l2: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], str, set[str]]:
    contexts: list[dict[str, Any]] = []
    source_l2_set = set(source_l2_ids)
    source_l1_shots = {
        shot_id
        for l2_id in source_l2_ids
        for shot_id in (source_l1_by_l2 or {}).get(l2_id, [])
    }
    unresolved: list[str] = []
    for source_id in source_ids:
        candidates = _l3_candidates(l3_scenes.get(source_id))
        if not candidates:
            unresolved.append(source_id)
            continue
        if len(candidates) == 1:
            chosen = candidates[0]
            trace_ids = set(_as_text_list((chosen.get("evidence_trace") or {}).get("source_l2_ids")))
            if source_l2_set and trace_ids and not source_l2_set.intersection(trace_ids):
                unresolved.append(source_id)
                continue
        else:
            matches = []
            for candidate in candidates:
                trace_ids = set(_as_text_list((candidate.get("evidence_trace") or {}).get("source_l2_ids")))
                if source_l2_set.intersection(trace_ids):
                    matches.append(candidate)
            if len(matches) != 1:
                unresolved.append(source_id)
                continue
            chosen = matches[0]
        contexts.append(chosen)
    if not unresolved:
        return contexts, "l3_id_l2_evidence", source_l1_shots
    if unresolved:
        if not source_l1_shots:
            raise ValueError(
                f"L4 event {event_index} cannot resolve L3 references {unresolved}: "
                "no source L1 shot evidence is available"
            )
        evidence_matches: dict[tuple[str, int, int, tuple[str, ...]], dict[str, Any]] = {}
        for candidate_rows in l3_scenes.values():
            for candidate in _l3_candidates(candidate_rows):
                candidate_shots = set(_as_text_list((candidate.get("evidence_trace") or {}).get("source_l1_ids")))
                if not candidate_shots.intersection(source_l1_shots):
                    continue
                time_range = candidate.get("time_range") or {}
                key = (
                    str(candidate.get("l3_scene_id") or ""),
                    int(time_range.get("start_ms", 0) or 0),
                    int(time_range.get("end_ms", 0) or 0),
                    tuple(sorted(candidate_shots)),
                )
                evidence_matches[key] = candidate
        if not evidence_matches:
            raise ValueError(
                f"L4 event {event_index} cannot resolve L3 references {unresolved}: "
                "no L3 scene shares source L1 shots with the referenced L2 events"
            )
        contexts = list(evidence_matches.values())
        return contexts, "shared_l1_shot_evidence", source_l1_shots
    return contexts, "l3_id_l2_evidence", set()


def build_arc_nodes(
    arc: dict[str, Any],
    l3_scenes: dict[str, Any],
    source_l1_by_l2: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Expand an L4 arc into chronologically ordered, evidence-resolved L3 scene nodes."""
    arc_id = str(arc.get("l4_story_arc_id") or "unknown_arc")
    events = arc.get("essential_events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"L4 arc has no essential_events: {arc_id}")

    scene_records: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or not str(event.get("event") or "").strip():
            raise ValueError(f"L4 arc event {index} has no event text: {arc_id}")
        source_ids = _as_text_list(event.get("source_l3_ids"))
        source_l2_ids = _as_text_list(event.get("source_l2_ids"))
        contexts, resolution_method, source_l1_shots = _resolve_l3_contexts(
            source_ids, source_l2_ids, l3_scenes, index, source_l1_by_l2
        )
        for scene in contexts:
            trace_ids = _as_text_list((scene.get("evidence_trace") or {}).get("source_l2_ids"))
            occurrence_key = tuple([str(scene.get("l3_scene_id") or ""), *trace_ids])
            record = scene_records.setdefault(
                occurrence_key,
                {"scene": scene, "l4_event_ids": [], "l3_resolution": {}},
            )
            l4_event_id = f"{arc_id}_event_{index:03d}"
            if l4_event_id not in record["l4_event_ids"]:
                record["l4_event_ids"].append(l4_event_id)
            scene_l1_shots = set(_as_text_list((scene.get("evidence_trace") or {}).get("source_l1_ids")))
            record["l3_resolution"][l4_event_id] = {
                "method": resolution_method,
                "source_l4_l3_references": source_ids,
                "shared_l1_shot_ids": sorted(scene_l1_shots.intersection(source_l1_shots)),
            }

    ordered_records = sorted(
        scene_records.values(),
        key=lambda record: (
            int((record["scene"].get("time_range") or {}).get("start_ms", 0) or 0),
            str(record["scene"].get("l3_scene_id") or ""),
        ),
    )
    arc_items = _as_text_list(arc.get("key_objects"))
    nodes: list[dict[str, Any]] = []
    for index, record in enumerate(ordered_records, start=1):
        scene = record["scene"]
        scene_narrative = str(scene.get("scene_narrative") or "").strip()
        scene_purpose = str(scene.get("scene_purpose") or "").strip()
        progression = _as_text_list(scene.get("event_progression"))
        characters = _as_text_list(scene.get("active_characters"))
        location = str(scene.get("scene_setting") or "").strip()
        locations = [location] if location else []
        items = [item for item in arc_items if VideoAgent._mentions_name(scene_narrative.lower(), item)]
        trace_ids = _as_text_list((scene.get("evidence_trace") or {}).get("source_l2_ids"))
        nodes.append({
            "node_id": f"{_safe_name(arc_id)}_scene_{index:03d}",
            "event_id": f"{arc_id}_scene_{index:03d}",
            "event_summary": scene_narrative or record["l4_event_ids"][0],
            "story_description": " ".join([scene_purpose, *progression]).strip(),
            "characters": characters,
            "locations": locations,
            "items": items,
            "source_l3_ids": [str(scene.get("l3_scene_id") or "")],
            "l3_resolution": list(record["l3_resolution"].values()),
            "source_l2_ids": trace_ids,
            "source_l4_event_ids": record["l4_event_ids"],
            "visual_assets": {
                "event_keyframes": [],
                "character_reference_images": [],
                "location_images": [],
                "item_images": [],
            },
        })
    storyline = str(arc.get("short_storyline_L") or arc.get("full_story_S") or "").strip()
    return nodes, storyline


def _arc_from_file(path: Path, arc_id: str) -> dict[str, Any]:
    payload = _read_json(path)
    arcs = _rows(payload, "story_arcs")
    for arc in arcs:
        if str(arc.get("l4_story_arc_id") or "") == arc_id:
            return arc
    raise ValueError(f"L4 arc not found: {arc_id} in {path}")


def _load_l3_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    scenes = _rows(_read_json(path), "l3_scenes")
    result: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        scene_id = str(scene.get("l3_scene_id") or "").strip()
        if scene_id:
            result.setdefault(scene_id, []).append(scene)
    return result


def _load_l1_shots_by_l2(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for unit_path in sorted(path.glob("*.json")):
        for event in _rows(_read_json(unit_path), "l2_events"):
            event_id = str(event.get("l2_event_id") or "").strip()
            shot_ids = _as_text_list(event.get("source_shot_ids"))
            if event_id and shot_ids:
                result[event_id] = shot_ids
    return result


def _asset_requests(nodes: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for node in nodes:
        for kind, values in (
            ("character", node["characters"]),
            ("location", node["locations"]),
            ("item", node["items"]),
        ):
            for value in values:
                if kind == "character" and "unidentified" in value.lower():
                    continue
                key = (kind, value)
                if key in seen:
                    continue
                seen.add(key)
                if kind == "character":
                    prompt = (
                        f"Create a clean initial character reference portrait of {value}. "
                        f"Visual context from the source scene: {node['story_description']}. "
                        "Keep the character large and clearly visible; minimize background and extra elements. "
                        "Do not add other people or unsupported story details."
                    )
                elif kind == "location":
                    prompt = (
                        f"Create a clear cinematic location reference image of {value}. "
                        "Show the space without people, with a readable layout and restrained incidental objects."
                    )
                else:
                    prompt = (
                        f"Create an isolated visual reference image for this story object/state: {value}. "
                        "Make the object large and clearly visible; minimize background and unrelated elements."
                    )
                requests.append({
                    "kind": kind,
                    "label": value,
                    "prompt": prompt,
                    "output_path": str(output_dir / "assets" / f"{kind}s" / f"{_safe_name(value)}.png"),
                })
    return requests


def _generate_image(prompt: str, output_path: Path, refs: list[str], *, retries: int, dry_run: bool) -> str:
    if dry_run:
        return str(output_path)
    settings = nano_banana_settings_from_env()
    if not settings.get("api_key"):
        raise RuntimeError("NANO_BANANA_API_KEY is required for --run")
    if output_path.exists() and output_path.suffix.lower() in IMAGE_SUFFIXES and output_path.stat().st_size:
        return str(output_path)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with call_context(
                provider="xi-api",
                tool="nano_banana",
                attempt=attempt,
                model=str(settings["model_name"]),
                output_path=str(output_path),
            ):
                result = generate_image_nano_banana(
                    prompt=prompt,
                    output_path=output_path,
                    api_key=str(settings["api_key"]),
                    base_url=str(settings["base_url"]),
                    model_name=str(settings["model_name"]),
                    timeout_sec=int(settings["timeout_sec"]),
                    reference_image_paths=refs,
                )
            if result and Path(result).is_file():
                return str(Path(result).resolve())
            last_error = "Banana returned no image file"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("image generation failed attempt=%d/%d path=%s error=%s", attempt, retries, output_path, last_error)
        if attempt < retries:
            time.sleep(min(attempt * 2, 10))
    raise RuntimeError(f"image generation failed: {output_path}: {last_error}")


def _attach_asset_paths(nodes: list[dict[str, Any]], asset_map: dict[tuple[str, str], str]) -> None:
    for node in nodes:
        assets = node["visual_assets"]
        assets["character_reference_images"] = [asset_map[("character", name)] for name in node["characters"] if ("character", name) in asset_map]
        assets["location_images"] = [asset_map[("location", name)] for name in node["locations"] if ("location", name) in asset_map]
        assets["item_images"] = [asset_map[("item", name)] for name in node["items"] if ("item", name) in asset_map]


def _event_keyframe_prompt(node: dict[str, Any]) -> str:
    return (
        f"Cinematic event keyframe. Event: {node['event_summary']} "
        f"Scene context: {node['story_description']} "
        f"Location: {', '.join(node['locations']) or 'unspecified'}. "
        f"Characters: {', '.join(node['characters']) or 'none visible'}. "
        "Include only people, setting details, and objects relevant to this described event. "
        "16:9 film still, clear readable action, no text or collage."
    )


def _write_inputs(nodes: list[dict[str, Any]], storyline: str, output_dir: Path) -> tuple[Path, Path]:
    nodes_path = output_dir / "narrative_nodes.json"
    plan_path = output_dir / "narrative_path_plan.json"
    _write_json(nodes_path, {"storyline": storyline, "selected_nodes": nodes})
    _write_json(plan_path, {"storyline": storyline, "ordered_node_ids": [node["node_id"] for node in nodes]})
    return nodes_path, plan_path


def _comparison_plan(
    nodes: list[dict[str, Any]], storyline: str, max_refs: int, prompt_style: str | None
) -> list[dict[str, Any]]:
    agent = VideoAgent(strict_online=False, retries=0)
    rows: list[dict[str, Any]] = []
    for rank, node in enumerate(nodes, start=1):
        prompt = str(agent._event_prompt_artifact(node, storyline=storyline, prompt_style=prompt_style)["prompt"])
        assets = node["visual_assets"]
        event_refs = list(assets["event_keyframes"])
        item_names = VideoAgent.referenced_event_items(node)
        allow_multi = bool(assets.get("multi_person_reference_allowed")) or event_needs_multi_person_references(node)
        character_refs = assets["character_reference_images"] if allow_multi else assets["character_reference_images"][:1]
        semantic_refs = [
            *character_refs,
            *(assets["location_images"][:1]),
            *(asset_map_path for item in item_names for asset_map_path in assets["item_images"]
              if _safe_name(item).lower() in Path(asset_map_path).stem.lower()),
        ][:max(1, max_refs)]
        rows.append({
            "rank": rank,
            "node_id": node["node_id"],
            "prompt": prompt,
            "duration_sec": 10,
            "num_frames": 243,
            "groups": {
                "fl2va_event_keyframe_only": {
                    "minimax_task": "fl2va" if len(event_refs) == 1 else "ref2va",
                    "reference_images": event_refs,
                    "reference_mode": "event_keyframe_only",
                },
                "ref2va_semantic_assets": {
                    "minimax_task": "ref2va",
                    "reference_images": semantic_refs,
                    "reference_mode": "semantic_assets",
                    "mentioned_items": item_names,
                },
            },
            "same_prompt_for_both_groups": True,
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CAMERAMASTER_CALL_LOG_JSONL"] = str(output_dir / "model_tool_calls.jsonl")
    arc = _arc_from_file(args.l4_arcs, args.arc_id)
    l3_map = _load_l3_map(args.l3_scenes)
    l1_shots_by_l2 = _load_l1_shots_by_l2(args.l2_events)
    nodes, storyline = build_arc_nodes(arc, l3_map, l1_shots_by_l2)
    image_requests = _asset_requests(nodes, output_dir)
    asset_map: dict[tuple[str, str], str] = {}

    for request in image_requests:
        path = _generate_image(
            request["prompt"], Path(request["output_path"]), [], retries=args.retries, dry_run=not args.run
        )
        asset_map[(request["kind"], request["label"])] = path
        request["status"] = "planned" if not args.run else "done"
    _attach_asset_paths(nodes, asset_map)

    keyframe_requests: list[dict[str, Any]] = []
    for node in nodes:
        refs = [*node["visual_assets"]["character_reference_images"], *node["visual_assets"]["location_images"]]
        mentioned_items = [
            item for item in node["items"]
            if any(token.lower() in node["event_summary"].lower() for token in re.findall(r"[A-Za-z]{4,}", item))
        ]
        refs.extend(asset_map[("item", item)] for item in mentioned_items if ("item", item) in asset_map)
        output = output_dir / "event_keyframes" / f"{node['node_id']}.png"
        request = {"node_id": node["node_id"], "prompt": _event_keyframe_prompt(node), "reference_images": refs, "output_path": str(output)}
        if args.run:
            request["output_path"] = _generate_image(request["prompt"], output, refs, retries=args.retries, dry_run=False)
            request["status"] = "done"
        else:
            request["status"] = "planned"
        node["visual_assets"]["event_keyframes"] = [request["output_path"]]
        keyframe_requests.append(request)

    _write_json(output_dir / "banana_image_generation_plan.json", {"assets": image_requests, "event_keyframes": keyframe_requests})
    nodes_path, plan_path = _write_inputs(nodes, storyline, output_dir)
    comparison_plan = _comparison_plan(nodes, storyline, args.max_reference_images, args.prompt_style)
    _write_json(output_dir / "video_comparison_plan.json", comparison_plan)
    if not args.run:
        planned = {
            "status": "dry_run",
            "movie_id": args.movie_id,
            "arc_id": args.arc_id,
            "event_count": len(nodes),
            "asset_image_count": len(image_requests),
            "event_keyframe_count": len(keyframe_requests),
            "planned_video_jobs": len(nodes) * 2,
            "nodes_json": str(nodes_path),
            "path_plan_json": str(plan_path),
            "image_plan": str(output_dir / "banana_image_generation_plan.json"),
            "comparison_plan": str(output_dir / "video_comparison_plan.json"),
            "note": "No Banana or video API calls were made. Pass --run to generate images and videos.",
        }
        _write_json(output_dir / "moviebench_arc_video_eval_summary.json", planned)
        return planned

    comparison = run_comparison(
        nodes_path=nodes_path,
        plan_path=plan_path,
        output_root=output_dir / "video_comparison",
        dry_run=False,
        max_reference_images=args.max_reference_images,
        source_task_json=None,
        prompt_style=args.prompt_style,
        concat=args.concat,
        strict_online=args.strict_online,
        retries=args.retries,
    )
    result = {
        "status": "done" if all(group["failed"] == 0 and group["skipped"] == 0 for group in comparison["groups"].values()) else "partial",
        "movie_id": args.movie_id,
        "arc_id": args.arc_id,
        "event_count": len(nodes),
        "asset_image_count": len(image_requests),
        "event_keyframe_count": len(keyframe_requests),
        "nodes_json": str(nodes_path),
        "path_plan_json": str(plan_path),
        "image_plan": str(output_dir / "banana_image_generation_plan.json"),
        "comparison_plan": str(output_dir / "video_comparison_plan.json"),
        "comparison_summary": str(output_dir / "video_comparison" / "comparison_summary.json"),
        "groups": {
            name: {key: group[key] for key in ("total", "success", "failed", "skipped", "concat_ok") if key in group}
            for name, group in comparison["groups"].items()
        },
    }
    _write_json(output_dir / "moviebench_arc_video_eval_summary.json", result)
    return result


def run_all_arcs(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.l4_arcs)
    arcs = _rows(payload, "story_arcs")
    if not arcs:
        raise ValueError(f"No L4 story arcs found in {args.l4_arcs}")
    output_root = (getattr(args, "all_arcs_output_dir", None) or (args.output_dir.parent / "all_arcs")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    arc_results: list[dict[str, Any]] = []
    for arc in arcs:
        arc_id = str(arc.get("l4_story_arc_id") or "").strip()
        if not arc_id:
            arc_results.append({"arc_id": "", "status": "failed", "error": "missing l4_story_arc_id"})
            continue
        child_args = argparse.Namespace(**vars(args))
        child_args.arc_id = arc_id
        child_args.output_dir = output_root / _safe_name(arc_id)
        try:
            result = run(child_args)
            arc_results.append({"arc_id": arc_id, **result})
        except Exception as exc:  # Continue so the aggregate identifies every problematic arc.
            LOGGER.exception("MovieBench arc failed arc_id=%s", arc_id)
            arc_results.append({
                "arc_id": arc_id,
                "status": "failed",
                "output_dir": str(child_args.output_dir),
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [row for row in arc_results if row.get("status") in {"dry_run", "done", "partial"}]
    summary = {
        "status": "dry_run" if not args.run and len(successful) == len(arcs) else (
            "done" if args.run and len(successful) == len(arcs) and all(row.get("status") == "done" for row in successful) else "partial"
        ),
        "movie_id": args.movie_id,
        "mode": "all_l4_arcs",
        "arc_count": len(arcs),
        "arc_success_count": len(successful),
        "arc_failed_count": len(arcs) - len(successful),
        "event_count": sum(int(row.get("event_count", 0) or 0) for row in successful),
        "asset_image_count": sum(int(row.get("asset_image_count", 0) or 0) for row in successful),
        "event_keyframe_count": sum(int(row.get("event_keyframe_count", 0) or 0) for row in successful),
        "planned_video_jobs": sum(int(row.get("planned_video_jobs", int(row.get("event_count", 0) or 0) * 2) or 0) for row in successful),
        "output_root": str(output_root),
        "arcs": arc_results,
    }
    _write_json(output_root / "moviebench_all_arcs_eval_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs" / "moviebench_cineworldsim" / "0001_American_Beauty"
    parser.add_argument("--movie-id", default="0001_American_Beauty")
    parser.add_argument("--arc-id", default="0001_American_Beauty_arc_01")
    parser.add_argument("--l4-arcs", type=Path, default=base / "l4_story" / "story_arcs.json")
    parser.add_argument("--l3-scenes", type=Path, default=base / "l3_scenes" / "final.json")
    parser.add_argument("--l2-events", type=Path, default=base / "l2_events" / "units", help="Directory of L2 event unit JSON files used to recover source L1 shot IDs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "moviebench_video_eval" / "0001_American_Beauty" / "arc_01")
    parser.add_argument("--all-arcs", action="store_true", help="Run every L4 arc from the input story_arcs.json")
    parser.add_argument("--all-arcs-output-dir", type=Path, default=None, help="Output root for --all-arcs; defaults to <output-dir parent>/all_arcs")
    parser.add_argument("--max-reference-images", type=int, default=6)
    parser.add_argument("--prompt-style", default=None)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="Call Banana and video APIs; omitted by default for a no-call dry-run.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    load_dotenv(override=False)
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    if not args.l4_arcs.is_file():
        raise FileNotFoundError(args.l4_arcs)
    if not args.l3_scenes.is_file():
        raise FileNotFoundError(args.l3_scenes)
    if not args.l2_events.is_dir():
        raise FileNotFoundError(args.l2_events)
    result = run_all_arcs(args) if args.all_arcs else run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
