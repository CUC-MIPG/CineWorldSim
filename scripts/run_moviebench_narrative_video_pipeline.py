#!/usr/bin/env python3
"""Run MovieBench L4 narrative simulation, Track A/B node ordering, and video generation."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig
from agents.path_orchestrator_agent import PathOrchestratorAgent
from agents.video_agent import VideoAgent
from narrative.event_keyframe_renderer import EventKeyframeRenderer
from scripts.run_dual_backend_narrative_world_pipeline import _run_one_backend

LOGGER = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _result(payload: Any) -> dict[str, Any]:
    return payload.get("result", payload) if isinstance(payload, dict) else {}


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    rows = _result(payload).get(key, [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _index_units(folder: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        payload = _read_json(path)
        unit_rows = payload.get("l2_events", []) if isinstance(payload, dict) else []
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            unit_rows = payload["result"].get("l2_events", unit_rows)
        if isinstance(unit_rows, list):
            for row in unit_rows:
                if isinstance(row, dict) and row.get("l2_event_id"):
                    result[str(row["l2_event_id"])] = row
        unit = payload.get("result", payload) if isinstance(payload, dict) else {}
        if isinstance(unit, dict) and unit.get("l1_shot_id"):
            result[str(unit["l1_shot_id"])] = unit
    return result


def _asset_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _shared_visual_asset_catalog(pipeline_result: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Index generated character, location, and item images from shared inference."""
    catalog: dict[str, dict[str, str]] = {
        "character_reference_images": {},
        "location_images": {},
        "item_images": {},
    }

    def add(group: str, name: Any, path: Any) -> None:
        key = _asset_key(name)
        image = str(path or "").strip()
        if key and image and Path(image).is_file():
            catalog[group].setdefault(key, image)

    portraits = pipeline_result.get("character_portraits", {})
    if isinstance(portraits, dict):
        for name, path in portraits.items():
            add("character_reference_images", name, path)
    for character in pipeline_result.get("character_configs", []) or []:
        if isinstance(character, dict):
            add("character_reference_images", character.get("name"), character.get("portrait_path"))

    states = pipeline_result.get("world_states", [])
    for state in reversed(states if isinstance(states, list) else []):
        if not isinstance(state, dict):
            continue
        for source_group, target_group in (("locations", "location_images"), ("items", "item_images")):
            for row in state.get(source_group, []) or []:
                if not isinstance(row, dict):
                    continue
                paths = [row.get("last_image_path")]
                views = row.get("views", {})
                if isinstance(views, dict):
                    paths.extend(
                        view.get("last_image_path")
                        for view in views.values()
                        if isinstance(view, dict)
                    )
                image = next((path for path in paths if path and Path(str(path)).is_file()), "")
                add(target_group, row.get("name"), image)
                add(target_group, row.get("id"), image)
    return catalog


def _matching_asset_paths(names: list[str], assets: dict[str, str]) -> list[str]:
    indexed_assets = {_asset_key(name): path for name, path in assets.items()}
    paths: list[str] = []
    for name in names:
        path = indexed_assets.get(_asset_key(name), "")
        if path and path not in paths:
            paths.append(path)
    return paths


def _related_network_assets(
    events: list[dict[str, Any]], characters: list[str], network_nodes: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Collect visual assets from simulation nodes related to this L2 event."""
    event_text = " ".join(str(event.get("event_description") or event.get("event_title") or "") for event in events)
    event_terms = set(re.findall(r"[a-zA-Z0-9]+", event_text.lower()))
    event_characters = {_asset_key(name) for name in characters if _asset_key(name)}
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for node in network_nodes:
        node_text = " ".join(
            str(node.get(key) or "")
            for key in ("event_summary", "story_description", "video_description")
        )
        node_terms = set(re.findall(r"[a-zA-Z0-9]+", node_text.lower()))
        node_characters = {_asset_key(name) for name in node.get("characters", []) or []}
        score = len(event_terms & node_terms) + 3 * len(event_characters & node_characters)
        if score:
            scored.append((score, str(node.get("node_id") or ""), node))
    scored.sort(key=lambda row: (-row[0], row[1]))

    result = {key: [] for key in ("character_reference_images", "location_images", "item_images")}
    for _, _, node in scored[:3]:
        assets = node.get("visual_assets", {})
        if not isinstance(assets, dict):
            continue
        for key in result:
            for path in assets.get(key, []) or []:
                value = str(path or "").strip()
                if value and value not in result[key]:
                    result[key].append(value)
    return result


def _merge_manifest_track_runs(
    previous: dict[str, Any], replacements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace only tracks rerun now, retaining records for untouched tracks."""
    replaced_tracks = {str(row.get("track")) for row in replacements}
    previous_rows = previous.get("track_runs", [])
    preserved = [
        row for row in previous_rows
        if isinstance(row, dict) and str(row.get("track")) not in replaced_tracks
    ] if isinstance(previous_rows, list) else []
    return {**previous, "track_runs": [*preserved, *replacements]}


def build_track_a_l1_nodes(
    arc: dict[str, Any],
    l2_by_id: dict[str, dict[str, Any]],
    l1_by_id: dict[str, dict[str, Any]],
    *,
    ordered_l2_ids: list[str] | None = None,
    shared_assets: dict[str, dict[str, str]] | None = None,
    shared_network_nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand an L3 anchor through its ordered L2 events into L1 generation nodes."""
    l2_ids = [str(value) for value in (ordered_l2_ids or arc.get("l2_event_ids", []))]
    rank = {event_id: index for index, event_id in enumerate(l2_ids)}
    shot_to_events: dict[str, list[str]] = {}
    for event_id in l2_ids:
        event = l2_by_id.get(event_id, {})
        for shot_id in event.get("source_shot_ids", []) or []:
            shot_to_events.setdefault(str(shot_id), []).append(event_id)
    arc_shots = [str(value) for value in arc.get("l1_shot_ids", []) or []]
    arc_shots.sort(
        key=lambda shot_id: (
            min((rank.get(event_id, len(rank)) for event_id in shot_to_events.get(shot_id, [])), default=len(rank)),
            int(l1_by_id.get(shot_id, {}).get("order_index", 0) or 0),
        )
    )
    nodes: list[dict[str, Any]] = []
    for shot_id in arc_shots:
        shot = l1_by_id.get(shot_id)
        if not shot:
            continue
        event_ids = sorted(shot_to_events.get(shot_id, []), key=lambda value: rank.get(value, len(rank)))
        events = [l2_by_id[value] for value in event_ids if value in l2_by_id]
        facts = shot.get("observable_facts", []) or []
        actions = shot.get("actions", []) or []
        characters = [str(row.get("name", "")).strip() for row in shot.get("characters", []) or [] if isinstance(row, dict) and row.get("name")]
        locations = [str(row.get("name", "")).strip() for row in shot.get("location_mentions", []) or [] if isinstance(row, dict) and row.get("name")]
        items = [str(row.get("name", "")).strip() for row in shot.get("item_mentions", []) or [] if isinstance(row, dict) and row.get("name")]
        shared_assets = shared_assets or {}
        inherited_assets = _related_network_assets(events, characters, shared_network_nodes or [])
        nodes.append({
            "node_id": shot_id,
            "event_id": shot_id,
            "anchor_l3_id": str(arc.get("l3_story_arc_id") or ""),
            "anchor_l3_title": str(arc.get("arc_title") or ""),
            "source_l2_ids": event_ids,
            "source_steps": [],
            "event_summary": " ".join(str(event.get("event_description") or "") for event in events).strip() or " ".join(str(item) for item in facts),
            "story_description": "; ".join(str(row.get("event_description", "")) for row in events),
            "video_description": " ".join(str(item) for item in facts),
            "characters": characters,
            "locations": locations,
            "items": items,
            "dialogue": "",
            "visual_assets": {
                "event_keyframes": [],
                "character_reference_images": list(dict.fromkeys([
                    *inherited_assets["character_reference_images"],
                    *_matching_asset_paths(characters, shared_assets.get("character_reference_images", {})),
                ])),
                "location_images": list(dict.fromkeys([
                    *inherited_assets["location_images"],
                    *_matching_asset_paths(locations, shared_assets.get("location_images", {})),
                ])),
                "item_images": list(dict.fromkeys([
                    *inherited_assets["item_images"],
                    *_matching_asset_paths(items, shared_assets.get("item_images", {})),
                ])),
            },
            "source_time_range": shot.get("time_range", {}),
            "source_l1": shot,
        })
    return nodes


def build_track_b_nodes(narrative_nodes_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Track B consumes generated simulation nodes only, never benchmark answer fields."""
    return [dict(row) for row in narrative_nodes_payload.get("selected_nodes", []) or [] if isinstance(row, dict)]


def track_b_generation_context(global_story: str, characters: list[str], known_facts: str) -> dict[str, Any]:
    return {"storyline": global_story, "known_characters": list(characters), "known_facts": known_facts}


def _rank_l2_for_anchor(
    *, arc: dict[str, Any], l2_by_id: dict[str, dict[str, Any]], simulation_nodes: list[dict[str, Any]],
    output_dir: Path, strict_online: bool, retries: int, dry_run: bool = False,
) -> list[str]:
    wanted = set(str(value) for value in arc.get("l2_event_ids", []) or [])
    candidates = []
    for event_id in arc.get("l2_event_ids", []) or []:
        event_id = str(event_id)
        event = l2_by_id.get(event_id)
        if not event:
            continue
        description = str(event.get("event_description") or event.get("event_title") or "")
        event_terms = set(re.findall(r"[a-zA-Z0-9]+", description.lower()))
        event_characters = {str(value).lower() for value in event.get("main_characters", []) or []}
        related: list[tuple[int, dict[str, Any]]] = []
        for node in simulation_nodes:
            node_text = " ".join(str(node.get(key) or "") for key in ("event_summary", "story_description", "video_description"))
            node_terms = set(re.findall(r"[a-zA-Z0-9]+", node_text.lower()))
            node_characters = {str(value).lower() for value in node.get("characters", []) or []}
            score = len(event_terms & node_terms) + 3 * len(event_characters & node_characters)
            if score:
                related.append((score, node))
        related.sort(key=lambda row: (-row[0], str(row[1].get("node_id", ""))))
        related_nodes = [
            {"node_id": row.get("node_id", ""), "summary": row.get("event_summary", ""), "step": row.get("source_steps", [])}
            for _, row in related[:3]
        ]
        candidates.append({"node_id": event_id, "text": description, "related_simulation_nodes": related_nodes})
    if not candidates:
        raise RuntimeError(f"L3 anchor {arc.get('l3_story_arc_id')} has no resolvable L2 events")
    if dry_run:
        ordered = [str(row["node_id"]) for row in candidates]
        _write_json(output_dir / "path_plan.json", {"mode": "dry_run_source_order", "ordered_node_ids": ordered, "candidate_count": len(ordered)})
        return ordered
    agent = PathOrchestratorAgent(strict_online=strict_online, retries=retries)
    plan = agent.arrange_candidate_path(
        storyline=f"L3 anchor: {arc.get('arc_title', '')}. {arc.get('full_story_S', '')}",
        candidates=candidates, id_field="node_id", text_field="text",
        output_list_field="ordered_node_ids", output_dir=output_dir,
    )
    ordered = [str(value) for value in plan.get("ordered_node_ids", []) if str(value) in wanted]
    return ordered or [str(row["node_id"]) for row in candidates]


def _render_l1_keyframes(
    nodes: list[dict[str, Any]], *, run_dir: Path, backend: str, dry_run: bool,
    max_reference_images: int = 4,
) -> None:
    renderer = EventKeyframeRenderer(backend_mode=backend, dry_run=dry_run)
    for index, node in enumerate(nodes, 1):
        shot = node["source_l1"]
        first_action = next(iter(shot.get("actions", []) or []), {})
        location = next(iter(shot.get("location_mentions", []) or []), {})
        role = str(first_action.get("actor") or (node.get("characters") or ["Scene"])[0])
        role_log = {
            "role_name": role,
            "role_output_text": node["video_description"],
            "next_action": {"action": first_action.get("action", ""), "location": location.get("name", "")},
        }
        refs = []
        result = renderer.render_event_keyframe(
            step=index, role_log=role_log,
            output_dir=run_dir / "l1_keyframes",
            reference_image_paths=list(dict.fromkeys(
                [
                    *node["visual_assets"].get("character_reference_images", []),
                    *node["visual_assets"].get("location_images", []),
                    *node["visual_assets"].get("item_images", []),
                ]
            ))[:max(0, max_reference_images)],
            event_context={"story_description": node["video_description"], "locations": node["locations"], "items": node["items"]},
        )
        path = str(result.get("output_path") or "")
        if result.get("ok") and path:
            refs.append(path)
        node["visual_assets"]["event_keyframes"] = refs
        node["keyframe_result"] = result


def _arrange_nodes(nodes: list[dict[str, Any]], storyline: str, *, output_dir: Path, strict_online: bool, retries: int, dry_run: bool = False) -> dict[str, Any]:
    candidates = [{"node_id": str(node.get("node_id") or node.get("event_id")), "text": " ".join(str(node.get(field) or "") for field in ("event_summary", "story_description", "video_description"))} for node in nodes]
    if dry_run:
        ordered = [row["node_id"] for row in candidates]
        return {"mode": "dry_run_source_order", "storyline": storyline, "ordered_node_ids": ordered, "node_rank_map": {value: index + 1 for index, value in enumerate(ordered)}}
    result = PathOrchestratorAgent(strict_online=strict_online, retries=retries).arrange_candidate_path(
        storyline=storyline, candidates=candidates, output_dir=output_dir,
    )
    return {**result, "node_rank_map": {value: index + 1 for index, value in enumerate(result.get("ordered_node_ids", []))}}


def _global_story_and_characters(data_root: Path) -> tuple[str, list[str]]:
    bible = _result(_read_json(data_root / "l4_global" / "movie_bible.json"))
    global_story = bible.get("global_storyline")
    if isinstance(global_story, dict):
        story = str(global_story.get("story") or "").strip()
    else:
        story = str(global_story or "").strip()
    if not story:
        synopsis = bible.get("movie_synopsis")
        if isinstance(synopsis, dict):
            story = str(synopsis.get("short_synopsis") or "").strip()
        else:
            story = str(synopsis or "").strip()
    characters = [
        str(row.get("character") or row.get("name") or row.get("character_name") or "").strip()
        for row in bible.get("global_character_profiles", []) or []
        if isinstance(row, dict)
    ]
    characters = [value for value in characters if value]
    if not story:
        raise ValueError("L4 movie_bible.json has no global_storyline or movie_synopsis")
    return story, characters


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.dataset_root.resolve()
    story, characters = _global_story_and_characters(data_root)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    previous_manifest_path = output_root / "moviebench_pipeline_manifest.json"
    previous_manifest = _read_json(previous_manifest_path) if args.resume_video_only and previous_manifest_path.is_file() else {}
    manifest: dict[str, Any] = {
        **(previous_manifest if isinstance(previous_manifest, dict) else {}),
        "dataset_root": str(data_root),
        "global_story_source": "l4_global/movie_bible.json",
        "track_runs": [],
    }
    l2_by_id = _index_units(data_root / "l2_events" / "units")
    l1_by_id = _index_units(data_root / "l1_shots" / "units")
    arcs = _rows(_read_json(data_root / "l3_arcs" / "story_arcs.json"), "story_arcs")
    selected_arcs = [arc for arc in arcs if not args.sample_id or str(arc.get("l3_story_arc_id")) == args.sample_id or str(arc.get("source_l4_story_arc_id")) == args.sample_id]
    if "A" in tracks and not selected_arcs:
        raise ValueError(f"No L3 arc matches --sample-id {args.sample_id!r}")

    run_dir = output_root / "narrative_simulation" / args.backend
    result_path = run_dir / "pipeline_result.json"
    if args.resume_video_only:
        if not result_path.is_file():
            raise FileNotFoundError(f"Cannot resume: shared narrative result not found: {result_path}")
        LOGGER.info("resume video-only using shared narrative result=%s", result_path)
    else:
        run_dir = _run_one_backend(
            story=story, characters=characters, backend_mode=args.backend,
            max_steps=args.max_steps, output_dir=output_root / "narrative_simulation",
            dry_run=args.dry_run, stop_on_complete=False,
            strict_online=args.strict_online, retries=args.retries,
        )
        result_path = run_dir / "pipeline_result.json"
    network = _read_json(result_path)
    shared_assets = _shared_visual_asset_catalog(network)
    selected = NarrativeNodeSelector(NarrativeNodeSelectorConfig(
        selection_mode="top_k", score_threshold=0.0, min_nodes=1,
        max_nodes=args.max_nodes, llm_enabled=False,
    )).select(network, source_pipeline_result_json=str(result_path))
    network_nodes = build_track_b_nodes(selected)

    for track in tracks:
        # Both tracks consume the same generated network; only their conditioning differs.
        track_dir = output_root / f"track_{track}"
        track_rows: list[dict[str, Any]] = []

        if track == "A":
            for arc in selected_arcs:
                arc_id = str(arc.get("l3_story_arc_id") or arc.get("source_l4_story_arc_id"))
                sample_dir = track_dir / arc_id
                existing_l2_plan = sample_dir / "l2_anchor_order" / "path_plan.json"
                if args.resume_video_only and existing_l2_plan.is_file():
                    saved_plan = _read_json(existing_l2_plan)
                    ordered_l2 = [str(value) for value in saved_plan.get("ordered_node_ids", [])]
                elif args.resume_video_only:
                    raise FileNotFoundError(
                        f"Cannot resume without saved L2 ordering: {existing_l2_plan}"
                    )
                else:
                    ordered_l2 = _rank_l2_for_anchor(
                        arc=arc, l2_by_id=l2_by_id, simulation_nodes=network_nodes,
                        output_dir=sample_dir / "l2_anchor_order", strict_online=args.strict_online, retries=args.retries,
                        dry_run=args.dry_run,
                    )
                nodes = build_track_a_l1_nodes(
                    arc, l2_by_id, l1_by_id, ordered_l2_ids=ordered_l2,
                    shared_assets=shared_assets, shared_network_nodes=network_nodes,
                )
                if args.max_l1_nodes:
                    nodes = nodes[:args.max_l1_nodes]
                nodes_path = sample_dir / "narrative_nodes.json"
                plan_path = sample_dir / "path_plan.json"
                if args.resume_video_only:
                    if not nodes_path.is_file() or not plan_path.is_file():
                        raise FileNotFoundError(
                            f"Cannot resume without saved Track A nodes and path plan: {sample_dir}"
                        )
                    saved_nodes = _read_json(nodes_path)
                    saved_by_id = {
                        str(row.get("node_id")): row
                        for row in saved_nodes.get("selected_nodes", []) or []
                        if isinstance(row, dict)
                    }
                    for node in nodes:
                        saved = saved_by_id.get(str(node["node_id"]), {})
                        old_assets = saved.get("visual_assets", {})
                        if isinstance(old_assets, dict):
                            node["visual_assets"]["event_keyframes"] = list(old_assets.get("event_keyframes", []) or [])
                        if saved.get("keyframe_result"):
                            node["keyframe_result"] = saved["keyframe_result"]
                    saved_plan = _read_json(plan_path)
                    order = [str(value) for value in saved_plan.get("ordered_node_ids", [])]
                    node_rank = {str(node["node_id"]): node for node in nodes}
                    nodes = [node_rank[value] for value in order if value in node_rank]
                    if not nodes:
                        raise ValueError(f"Saved Track A path plan has no matching L1 nodes: {plan_path}")
                    path_plan = saved_plan
                    _write_json(nodes_path, {**saved_nodes, "selected_nodes": nodes})
                else:
                    _render_l1_keyframes(
                        nodes, run_dir=sample_dir, backend=args.backend, dry_run=args.dry_run,
                        max_reference_images=args.max_reference_images,
                    )
                    ordered_l1 = [str(node["node_id"]) for node in nodes]
                    path_plan = {"storyline": str(arc.get("full_story_S") or arc.get("arc_title") or ""), "ordered_node_ids": ordered_l1, "anchor_l3_id": arc_id, "ordered_l2_ids": ordered_l2}
                    _write_json(nodes_path, {"storyline": path_plan["storyline"], "selected_nodes": nodes})
                    _write_json(plan_path, path_plan)
                video_summary = VideoAgent(strict_online=args.strict_online, retries=args.retries).process_orchestrated_events(
                    narrative_nodes_json=nodes_path, path_plan_json=plan_path,
                    output_dir=sample_dir / "video", dry_run=args.video_dry_run,
                    max_reference_images=args.max_reference_images,
                    multi_shot_events=False, include_character_keyframes=True,
                    include_world_images=True, video_reference_mode="all_allowed",
                    dialogue_backfill=False if args.video_dry_run else None,
                    story_backfill=not args.video_dry_run,
                )
                track_rows.append({"l3_anchor_id": arc_id, "l2_order": ordered_l2, "l1_count": len(nodes), "video_summary": video_summary})
        else:
            context = track_b_generation_context(story, characters, "")
            track_dir.mkdir(parents=True, exist_ok=True)
            nodes_path = track_dir / "narrative_nodes.json"
            plan_path = track_dir / "path_plan.json"
            if args.resume_video_only:
                if not nodes_path.is_file() or not plan_path.is_file():
                    raise FileNotFoundError(f"Cannot resume without saved Track B nodes and path plan: {track_dir}")
                saved_nodes = _read_json(nodes_path)
                path_plan = _read_json(plan_path)
                _write_json(nodes_path, saved_nodes)
            else:
                _write_json(nodes_path, {"storyline": story, "selected_nodes": network_nodes, "track_b_context": context})
                path_plan = _arrange_nodes(network_nodes, story, output_dir=track_dir / "llm_node_order", strict_online=args.strict_online, retries=args.retries, dry_run=args.dry_run)
                _write_json(plan_path, path_plan)
            video_summary = VideoAgent(strict_online=args.strict_online, retries=args.retries).process_orchestrated_events(
                narrative_nodes_json=nodes_path, path_plan_json=plan_path,
                output_dir=track_dir / "video", dry_run=args.video_dry_run,
                max_reference_images=args.max_reference_images,
                multi_shot_events=False, video_reference_mode="event_keyframe_only",
                dialogue_backfill=False if args.video_dry_run else None,
                story_backfill=not args.video_dry_run,
            )
            track_rows.append({"l4_global": True, "node_count": len(network_nodes), "video_summary": video_summary})
        manifest["track_runs"].append({"track": track, "simulation_result": str(result_path), "samples": track_rows})
    if args.resume_video_only:
        manifest = _merge_manifest_track_runs(previous_manifest, manifest["track_runs"])
        manifest["dataset_root"] = str(data_root)
        manifest["global_story_source"] = "l4_global/movie_bible.json"
    _write_json(output_root / "moviebench_pipeline_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Restructured MovieBench movie directory containing l4_global/, l3_arcs/, l2_events/, l1_shots/.")
    parser.add_argument("--track", choices=["A", "B", "both"], default="A")
    parser.add_argument("--sample-id", default="", help="Track A L3 arc ID or source arc ID; omit to run every L3 arc.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["all_banana", "all_flux2_klein", "all_flux"], default="all_banana")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=45)
    parser.add_argument("--max-l1-nodes", type=int, default=0, help="Optional Track A cap for pilot runs; 0 means all L1 shots in the selected L3 arc.")
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Skip online narrative and image generation; use for wiring checks.")
    parser.add_argument("--video-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-video-only", action="store_true",
        help="Reuse the existing shared narrative result, saved L2 ordering, and generated L1 keyframes; do not rerun inference or image generation.",
    )
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    manifest = run_pipeline(args)
    print(args.output_dir.resolve() / "moviebench_pipeline_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
