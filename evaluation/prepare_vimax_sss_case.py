#!/usr/bin/env python3
"""Prepare blinded SSS inputs for a CameraMaster run of a ViMax case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_case_inputs(case_json: Path, output_root: Path, tracks: tuple[str, ...] = ("A", "B")) -> dict[str, Any]:
    case = _load(case_json)
    summary = case.get("story_overview")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("benchmark case must contain a non-empty story_overview")

    samples = []
    for track in tracks:
        track_dir = output_root / f"track_{track}"
        plan = _load(track_dir / "path_plan.json")
        ordered_ids = [str(value) for value in plan.get("ordered_node_ids", [])]
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError(f"Track {track} path_plan must contain unique ordered_node_ids")

        clips = []
        for index, node_id in enumerate(ordered_ids, start=1):
            if track == "A":
                matches = list((track_dir / "video").glob(f"event_video_{index:03d}_*_shot_*.mp4"))
            else:
                matches = list((track_dir / "video").glob(f"event_video_{index:03d}_{node_id}.mp4"))
            if len(matches) != 1:
                raise ValueError(
                    f"Track {track}: expected exactly one clip for ordered node {node_id} "
                    f"at index {index}, found {len(matches)}"
                )
            clips.append(str(matches[0].resolve()))

        video = (track_dir / "video" / "final_orchestrated_events.mp4").resolve()
        if not video.is_file():
            raise ValueError(f"Track {track} concatenated video is missing: {video}")
        samples.append({
            "track": track,
            "video_path": str(video),
            "source_clip_paths": clips,
            "ordered_node_ids": ordered_ids,
        })

    return {
        "summary": summary.strip(),
        "summary_provenance": "benchmark_story_overview_proxy",
        "summary_source": str(case_json.resolve()),
        "adapter": {
            "schema_version": "cameramaster-sss-track-adapter-v1",
            "case_id": case_json.stem,
            "samples": samples,
            "protocol_notes": [
                "The benchmark story_overview is used only as the reference for SSS scoring.",
                "Track mapping contains clip paths and order only; benchmark storyboard text is not sent to blind reconstruction.",
                "Track A v3 uses generated clip boundaries; Track B v2 uses automatic shot detection.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-case", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", choices=("A", "B"), default=("A", "B"))
    args = parser.parse_args()

    result = build_case_inputs(args.benchmark_case.resolve(), args.output_root.resolve(), tuple(args.tracks))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_story.txt").write_text(result["summary"] + "\n", encoding="utf-8")
    (args.output_dir / "track_adapter_inputs.json").write_text(
        json.dumps(result["adapter"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "prepared",
        "output_dir": str(args.output_dir.resolve()),
        "summary_provenance": result["summary_provenance"],
        "tracks": [
            {"track": sample["track"], "video_path": sample["video_path"], "clip_count": len(sample["source_clip_paths"])}
            for sample in result["adapter"]["samples"]
        ],
        "note": "No LLM or embedding API was called. Reference text is not included in adapter inputs.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
