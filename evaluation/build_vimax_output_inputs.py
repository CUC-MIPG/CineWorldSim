#!/usr/bin/env python3
"""Build normalized ViMax evaluator inputs from CameraMaster Track A/B outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _description(node: dict[str, Any]) -> str:
    source = node.get("source_l1") or {}
    return str(
        node.get("video_prompt")
        or node.get("event_summary")
        or source.get("video_prompt")
        or source.get("first_frame")
        or ""
    ).strip()


def build_track(root: Path, track: str, b_scene_source: str) -> dict[str, Any]:
    track_dir = root / f"track_{track}"
    nodes_payload = _load(track_dir / "narrative_nodes.json")
    path_payload = _load(track_dir / "path_plan.json")
    nodes = {
        str(row.get("node_id") or row.get("event_id")): row
        for row in nodes_payload.get("selected_nodes", [])
    }
    ordered = [str(value) for value in path_payload.get("ordered_node_ids", [])]
    video_dir = track_dir / "video"
    shots = []
    scene_proxy = False
    for index, node_id in enumerate(ordered, start=1):
        node = nodes.get(node_id)
        if node is None:
            raise ValueError(f"{track}: path plan node is absent from narrative_nodes: {node_id}")
        if track == "A":
            pattern = f"event_video_{index:03d}_*.mp4"
            matches = sorted(video_dir.glob(pattern))
            scene_match = re.search(r"event_video_\d+_(scene_\d+)_shot_\d+\.mp4$", matches[0].name) if matches else None
            scene_id = scene_match.group(1) if scene_match else None
        else:
            matches = sorted(video_dir.glob(f"event_video_{index:03d}_{node_id}.mp4"))
            scene_id = None
            if b_scene_source == "location":
                locations = node.get("locations") or []
                if locations:
                    scene_id = "location::" + str(locations[0])
                    scene_proxy = True
        if len(matches) != 1:
            raise ValueError(f"{track}: expected one video for {node_id}, found {len(matches)}")
        shots.append({
            "shot_id": node_id,
            "scene_id": scene_id,
            "video_path": str(matches[0].resolve()),
            "storyboard_description": _description(node),
            "source_node_id": node_id,
        })
    return {
        "sample_id": f"artist_extreme_weather_typeA_track_{track}",
        "story_id": "artist_extreme_weather_typeA",
        "track": track,
        "prompt": str(nodes_payload.get("storyline", "")),
        "shots": shots,
        "scene_label_provenance": "official_filename_scene" if track == "A" else ("location_proxy" if scene_proxy else "missing"),
        "source_output_root": str(root.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--track-b-scene-source", choices=("location", "none"), default="location")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    samples = [build_track(root, track, args.track_b_scene_source) for track in args.tracks]
    payload = {
        "schema_version": "cameramaster-vimax-output-adapter-v1",
        "samples": samples,
        "protocol_notes": [
            "Track A scene IDs are parsed from generated filenames and match the source scene grouping.",
            "Track B location scene IDs are custom proxies, not official ViMax scene annotations." if args.track_b_scene_source == "location" else "Track B scene labels are intentionally absent.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(samples), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
