#!/usr/bin/env python3
"""Run CameraMaster Track A/B experiments on ViMax benchmark cases.

Track A uses each ViMax ``first_frame`` text as a Banana prompt, then uses the
generated image as the sole event-keyframe reference for the shot video.
Track B only consumes ``story_overview`` plus explicitly supplied character
constraints and lets the narrative/world pipeline create its own nodes/assets.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from agents.llm_tool import LLMJsonTool
from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig
from agents.path_orchestrator_agent import PathOrchestratorAgent
from agents.video_agent import VideoAgent
from narrative.world_renderer import WorldRenderer
from scripts.run_dual_backend_narrative_world_pipeline import _run_one_backend

LOGGER = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def split_character_names(value: str) -> list[str]:
    """Parse comma-separated names while preserving first occurrence order."""
    result: list[str] = []
    seen: set[str] = set()
    for item in str(value or "").split(","):
        name = item.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _character_evidence(case: dict[str, Any]) -> str:
    """Build cast-identification evidence without making it simulation input."""
    sections = ["STORY OVERVIEW:\n" + str(case.get("story_overview") or "").strip()]
    for scene_index, scene in enumerate(case.get("scenes", []) or [], start=1):
        if not isinstance(scene, dict):
            continue
        scene_num = scene.get("scene_num", scene_index)
        for shot_index, shot in enumerate(scene.get("shots", []) or [], start=1):
            if not isinstance(shot, dict):
                continue
            first_frame = str(shot.get("first_frame") or "").strip()
            video_prompt = str(shot.get("video_prompt") or "").strip()
            if not first_frame and not video_prompt:
                continue
            shot_id = shot.get("shot_id", shot_index)
            sections.append(
                f"SCENE {scene_num} SHOT {shot_id}:\n"
                f"first_frame: {first_frame}\n"
                f"video_prompt: {video_prompt}"
            )
    return "\n\n".join(sections)


def _normalize_extracted_character_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    values = payload.get("characters")
    if values is None:
        values = payload.get("character_names")
    if not isinstance(values, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        name = str(value or "").strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def resolve_track_b_characters(
    case: dict[str, Any],
    requested: str | None,
    *,
    tool: Any | None = None,
    retries: int = 3,
) -> tuple[list[str], str]:
    """Resolve Track B's cast, keeping shot text out of narrative simulation."""
    if requested is not None:
        names = split_character_names(requested)
        if not names:
            raise ValueError("--track-b-characters was provided but contains no character names")
        return names, "cli_override"

    extractor = tool or LLMJsonTool(scope="OPENAI", retries=retries)
    system_prompt = (
        "Identify the named or clearly identifiable human/character entities in the supplied "
        "story evidence. Return JSON only in the form {\"characters\": [\"Name\", ...]}. "
        "Use concise stable names suitable for initializing narrative agents. Do not include "
        "locations, objects, generic groups, camera terms, or unnamed extras. Deduplicate names."
    )
    user_prompt = (
        "Extract the complete relevant cast from this evidence. Shot descriptions are used only "
        "for cast identification; they must not be treated as Track B anchors or simulation events.\n\n"
        + _character_evidence(case)
    )
    payload = extractor.run_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    names = _normalize_extracted_character_names(payload)
    if not names:
        raise ValueError("LLM character extraction returned no character names")
    return names, "llm_extracted"


def load_vimax_case(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"ViMax case has no scenes: {path}")
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("shots"), list):
            raise ValueError(f"ViMax case contains an invalid scene: {path}")
    payload["_case_file"] = str(path.resolve())
    return payload


def resolve_case_file(benchmark_root: Path, case: str) -> Path:
    requested = str(case or "").strip()
    if not requested:
        raise ValueError("--case is required")
    direct = benchmark_root / requested
    if direct.is_file():
        return direct
    if not requested.endswith(".json"):
        themed = benchmark_root / f"{requested}.json"
        if themed.is_file():
            return themed
    index_path = benchmark_root / "benchmark_index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        for row in index.get("stories", []) or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("id")) == requested or str(row.get("theme")) == requested:
                candidate = benchmark_root / str(row.get("file") or "")
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f"ViMax case not found under {benchmark_root}: {case}")


def _safe_name(value: Any, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return text or fallback


def build_track_a_nodes(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Map ViMax scene/shot records to Track A L3-anchor/L1-shot nodes."""
    nodes: list[dict[str, Any]] = []
    for scene in sorted(case.get("scenes", []), key=lambda row: int(row.get("scene_num", 0) or 0)):
        scene_num = int(scene.get("scene_num", 0) or 0)
        scene_id = f"scene_{scene_num:03d}"
        shots = scene.get("shots", []) or []
        for shot in sorted(shots, key=lambda row: int(row.get("shot_id", 0) or 0)):
            shot_id = int(shot.get("shot_id", 0) or 0)
            first_frame = str(shot.get("first_frame") or "").strip()
            video_prompt = str(shot.get("video_prompt") or "").strip()
            if not first_frame:
                raise ValueError(f"Missing first_frame: scene={scene_num} shot={shot_id}")
            if not video_prompt:
                raise ValueError(f"Missing video_prompt: scene={scene_num} shot={shot_id}")
            nodes.append(
                {
                    "node_id": f"{scene_id}_shot_{shot_id:03d}",
                    "event_id": f"{scene_id}_shot_{shot_id:03d}",
                    "anchor_l3_id": scene_id,
                    "anchor_l3_title": f"ViMax scene {scene_num}",
                    "source_scene_num": scene_num,
                    "source_shot_id": shot_id,
                    "source_l1": shot,
                    "first_frame_prompt": first_frame,
                    "event_summary": first_frame,
                    "story_description": first_frame,
                    "video_description": video_prompt,
                    "characters": [],
                    "locations": [],
                    "items": [],
                    "dialogue": "",
                    "visual_assets": {"event_keyframes": []},
                }
            )
    return nodes


def _track_a_keyframe_prompt(node: dict[str, Any]) -> str:
    return (
        f"{node['first_frame_prompt']}\n\n"
        "Render this description as a cinematic 16:9 still frame for a video benchmark. "
        "Preserve every identity, wardrobe, prop, environment, composition, and lighting detail. "
        "Do not add captions, logos, borders, or written text."
    )


def render_track_a_keyframes(
    nodes: list[dict[str, Any]], *, output_dir: Path, backend: str, dry_run: bool,
) -> dict[str, Any]:
    renderer = WorldRenderer(backend_mode=backend, dry_run=dry_run)
    rows: list[dict[str, Any]] = []
    for node in nodes:
        scene_dir = output_dir / "keyframes" / str(node["anchor_l3_id"])
        output_path = scene_dir / f"shot_{int(node['source_shot_id']):03d}.png"
        prompt = _track_a_keyframe_prompt(node)
        result = renderer.render_world_asset(prompt=prompt, output_path=output_path)
        node["keyframe_generation"] = {**result, "prompt": prompt}
        path = str(result.get("output_path") or "") if result.get("ok") else ""
        node["visual_assets"]["event_keyframes"] = [path] if path else []
        rows.append({
            "node_id": node["node_id"],
            "first_frame_prompt": node["first_frame_prompt"],
            "prompt": prompt,
            "result": result,
        })
        LOGGER.info(
            "vimax track_a keyframe scene=%s shot=%s ok=%s output=%s",
            node["anchor_l3_id"], node["source_shot_id"], bool(result.get("ok")), path,
        )
    return {"total": len(rows), "success": sum(bool(row["result"].get("ok")) for row in rows), "items": rows}


def _arrange_track_b_nodes(
    nodes: list[dict[str, Any]], *, storyline: str, output_dir: Path,
    strict_online: bool, retries: int, dry_run: bool,
) -> dict[str, Any]:
    candidates = [
        {
            "node_id": str(node.get("node_id") or node.get("event_id") or ""),
            "text": " ".join(str(node.get(key) or "") for key in ("event_summary", "story_description", "video_description")),
        }
        for node in nodes
    ]
    if dry_run:
        ordered = [row["node_id"] for row in candidates]
        return {"mode": "dry_run_source_order", "storyline": storyline, "ordered_node_ids": ordered}
    plan = PathOrchestratorAgent(strict_online=strict_online, retries=retries).arrange_candidate_path(
        storyline=storyline, candidates=candidates, output_dir=output_dir,
    )
    ordered = [str(value) for value in plan.get("ordered_node_ids", []) if str(value) in {row["node_id"] for row in candidates}]
    return {**plan, "ordered_node_ids": ordered or [row["node_id"] for row in candidates]}


def _concat_video_summary(video_agent: VideoAgent, summary: dict[str, Any], output_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    output_path = output_dir / "final_orchestrated_events.mp4"
    paths = [
        str(row.get("output_video_path") or "")
        for row in summary.get("generated", []) or []
        if isinstance(row, dict) and row.get("status") == "done" and Path(str(row.get("output_video_path") or "")).exists()
    ]
    ok = False if dry_run else video_agent.concat_videos(paths, output_path)
    summary["concat_output"] = str(output_path)
    summary["concat_inputs"] = paths
    summary["concat_ok"] = bool(ok)
    if dry_run:
        summary["concat_skip_reason"] = "video_dry_run"
    return summary


def _require_complete_video_summary(summary: dict[str, Any], *, dry_run: bool, track: str) -> None:
    if dry_run:
        return
    total = int(summary.get("total", 0) or 0)
    success = int(summary.get("success", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    if total < 1 or success != total or failed or not summary.get("concat_ok"):
        raise RuntimeError(
            f"Track {track} video output is incomplete: total={total}, success={success}, "
            f"failed={failed}, concat_ok={summary.get('concat_ok')}. "
            "Use scripts/resume_vimax_track_ab_failed_videos.py for missing videos, "
            "then rerun the experiment script."
        )


def run_track_a(case: dict[str, Any], *, output_dir: Path, backend: str, video_dry_run: bool, dry_run: bool,
                max_reference_images: int, strict_online: bool, retries: int) -> dict[str, Any]:
    nodes = build_track_a_nodes(case)
    keyframe_summary = render_track_a_keyframes(nodes, output_dir=output_dir, backend=backend, dry_run=dry_run)
    storyline = str(case.get("story_overview") or "")
    plan = {"mode": "source_scene_shot_order", "storyline": storyline, "ordered_node_ids": [node["node_id"] for node in nodes]}
    nodes_path = _write_json(output_dir / "narrative_nodes.json", {"storyline": storyline, "selected_nodes": nodes, "track": "A"})
    plan_path = _write_json(output_dir / "path_plan.json", plan)
    agent = VideoAgent(strict_online=strict_online, retries=retries)
    effective_video_dry_run = bool(video_dry_run or dry_run)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_path, path_plan_json=plan_path, output_dir=output_dir / "video",
        dry_run=effective_video_dry_run, max_reference_images=max_reference_images,
        include_character_keyframes=True, include_world_images=False,
        multi_shot_events=False, insert_auxiliary_videos=False,
        video_reference_mode="event_keyframe_only", dialogue_backfill=False, story_backfill=False,
    )
    summary = _concat_video_summary(agent, summary, output_dir / "video", dry_run=effective_video_dry_run)
    _write_json(output_dir / "keyframe_summary.json", keyframe_summary)
    _write_json(output_dir / "video" / "event_video_summary.json", summary)
    _require_complete_video_summary(summary, dry_run=effective_video_dry_run, track="A")
    return {"track": "A", "nodes": len(nodes), "keyframes": keyframe_summary, "video": summary}


def run_track_b(case: dict[str, Any], *, output_dir: Path, backend: str, characters: list[str], max_steps: int,
                max_nodes: int, video_dry_run: bool, dry_run: bool, max_reference_images: int,
                strict_online: bool, retries: int, story_control_enabled: bool = True,
                require_full_coverage: bool = True) -> dict[str, Any]:
    storyline = str(case.get("story_overview") or "").strip()
    simulation_dir = output_dir / "narrative_simulation"
    run_dir = _run_one_backend(
        story=storyline, characters=characters, backend_mode=backend, max_steps=max_steps,
        # A verified terminal beat ends the story even when fewer than
        # max_nodes candidates exist; max_nodes is an upper bound.
        output_dir=simulation_dir, dry_run=dry_run,
        stop_on_complete=story_control_enabled,
        strict_online=strict_online, retries=retries, story_control_enabled=story_control_enabled,
    )
    result_path = run_dir / "pipeline_result.json"
    result = _read_json(result_path)
    selected = NarrativeNodeSelector(NarrativeNodeSelectorConfig(
        selection_mode="top_k", score_threshold=0.0, min_nodes=1, max_nodes=max_nodes, llm_enabled=False,
    )).select(result, source_pipeline_result_json=str(result_path))
    nodes = [dict(row) for row in selected.get("selected_nodes", []) or [] if isinstance(row, dict)]
    if not nodes:
        raise RuntimeError("Track B narrative simulation produced no selected nodes")
    coverage_report = selected.get("coverage_report", {}) if isinstance(selected.get("coverage_report"), dict) else {}
    missing_beats = [str(item) for item in coverage_report.get("missing_beat_ids", []) or [] if str(item).strip()]
    if story_control_enabled:
        _write_json(output_dir / "story_coverage_report.json", {
            "story_control_path": str(run_dir / "story_control.json"),
            "selector_coverage": coverage_report,
            "selected_node_count": len(nodes),
        })
    if missing_beats and require_full_coverage:
        _write_json(output_dir / "story_coverage_failure.json", {
            "missing_beat_ids": missing_beats,
            "coverage_report": coverage_report,
            "message": "Video generation was intentionally blocked: required Track B story beats have no selected event evidence.",
        })
        raise RuntimeError(
            "Track B story coverage is incomplete; video generation was blocked. "
            f"Missing beat IDs: {', '.join(missing_beats)}. See {output_dir / 'story_coverage_failure.json'}"
        )
    plan = _arrange_track_b_nodes(
        nodes, storyline=storyline, output_dir=output_dir / "llm_node_order",
        strict_online=strict_online, retries=retries, dry_run=dry_run,
    )
    if story_control_enabled:
        control_path = run_dir / "story_control.json"
        control = _read_json(control_path) if control_path.exists() else {}
        beats = (control.get("contract") or {}).get("hard_beats", [])
        evidence = control.get("beat_evidence") or {}
        terminal_id = str(beats[-1].get("beat_id", "")) if beats else ""
        terminal_step = (evidence.get(terminal_id) or {}).get("evidence_step")
        terminal_nodes = [str(node.get("node_id")) for node in nodes
                          if terminal_step is not None and terminal_step in (node.get("source_steps") or [])]
        if terminal_nodes:
            ordered = [str(item) for item in plan["ordered_node_ids"] if str(item) not in terminal_nodes]
            plan["ordered_node_ids"] = ordered + terminal_nodes
            plan["terminal_node_ids"] = terminal_nodes
    nodes_path = _write_json(output_dir / "narrative_nodes.json", {
        "storyline": storyline, "selected_nodes": nodes, "track": "B",
        "source_pipeline_result": str(result_path), "characters": characters,
    })
    plan_path = _write_json(output_dir / "path_plan.json", plan)
    agent = VideoAgent(strict_online=strict_online, retries=retries)
    effective_video_dry_run = bool(video_dry_run or dry_run)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_path, path_plan_json=plan_path, output_dir=output_dir / "video",
        dry_run=effective_video_dry_run, max_reference_images=max_reference_images,
        include_character_keyframes=True, include_world_images=True,
        multi_shot_events=False, insert_auxiliary_videos=False,
        video_reference_mode="all_allowed", dialogue_backfill=False,
        story_backfill=not effective_video_dry_run,
    )
    summary = _concat_video_summary(agent, summary, output_dir / "video", dry_run=effective_video_dry_run)
    _write_json(output_dir / "video" / "event_video_summary.json", summary)
    _require_complete_video_summary(summary, dry_run=effective_video_dry_run, track="B")
    return {
        "track": "B", "nodes": len(nodes), "simulation_result": str(result_path),
        "story_control": str(run_dir / "story_control.json") if story_control_enabled else "",
        "coverage_report": coverage_report, "video": summary,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = args.benchmark_root.resolve()
    case_path = resolve_case_file(benchmark_root, args.case)
    case = load_vimax_case(case_path)
    case_name = _safe_name(case_path.stem)
    output_dir = args.output_root.resolve() / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    manifest: dict[str, Any] = {
        "benchmark_root": str(benchmark_root), "case_file": str(case_path.resolve()),
        "case_name": case_name, "story_overview": case.get("story_overview", ""),
        "consistency_type": case.get("consistency_type", ""), "tracks": {},
    }
    if "A" in tracks:
        manifest["tracks"]["A"] = run_track_a(
            case, output_dir=output_dir / "track_A", backend=args.backend,
            video_dry_run=args.video_dry_run, dry_run=args.dry_run,
            max_reference_images=args.max_reference_images,
            strict_online=args.strict_online, retries=args.retries,
        )
    if "B" in tracks:
        characters, character_source = resolve_track_b_characters(
            case, args.track_b_characters, retries=args.retries,
        )
        manifest["tracks"]["B"] = run_track_b(
            case, output_dir=output_dir / "track_B", backend=args.backend,
            characters=characters, max_steps=args.max_steps, max_nodes=args.max_nodes,
            video_dry_run=args.video_dry_run, dry_run=args.dry_run,
            max_reference_images=args.max_reference_images,
            strict_online=args.strict_online, retries=args.retries,
            story_control_enabled=args.track_b_story_control,
            require_full_coverage=args.track_b_require_full_coverage,
        )
        manifest["tracks"]["B"]["characters"] = characters
        manifest["tracks"]["B"]["character_source"] = character_source
    manifest_path = _write_json(output_dir / "vimax_track_ab_manifest.json", manifest)
    print(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--case", required=True, help="Case filename, theme, or benchmark index id")
    parser.add_argument("--track", choices=["A", "B", "both"], default="both")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["all_banana", "all_flux", "all_flux2_klein"], default="all_banana")
    parser.add_argument(
        "--track-b-characters", default=None,
        help="Optional comma-separated roles for Track B; omitted means LLM cast discovery",
    )
    parser.add_argument(
        "--track-b-story-control", action=argparse.BooleanOptionalAction, default=True,
        help="Use hidden Story Contract, terminal-distance control, and world-compiled observations for Track B.",
    )
    parser.add_argument(
        "--track-b-require-full-coverage", action=argparse.BooleanOptionalAction, default=True,
        help="Block Track B video generation if required Story Contract beats lack selected event evidence.",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
