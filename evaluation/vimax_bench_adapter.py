"""Adapt CameraMaster/DirectorBench outputs to ViMax-Bench-style shot samples.

The adapter intentionally keeps source clips separate.  A concatenated track is
useful for other benchmarks, but ViMax computes one feature per generated shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}: {exc}") from exc


def load_scene_map(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("scene map must be a JSON object mapping shot/node IDs to scene IDs")
    return {str(key): value for key, value in payload.items()}


def _storyboard_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("shots", payload.get("storyboard", []))
    if not isinstance(payload, list):
        raise ValueError(f"storyboard must be a JSON list: {path}")
    rows = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"storyboard row {index} is not an object")
        rows.append(row)
    return rows


def _description(row: Mapping[str, Any]) -> str:
    for key in ("storyboard_description", "first_frame_prompt", "description", "story_description", "video_description"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _shot_row(row: Mapping[str, Any], video_path: str, index: int, scene_map: Mapping[str, Any]) -> dict[str, Any]:
    shot_id = str(row.get("shot_id") or row.get("node_id") or row.get("id") or f"shot_{index:03d}")
    scene_id = row.get("scene_id")
    if scene_id is None:
        scene_id = scene_map.get(shot_id, scene_map.get(str(row.get("node_id", ""))))
    return {
        "shot_index": index,
        "shot_id": shot_id,
        "scene_id": None if scene_id is None else str(scene_id),
        "video_path": str(Path(video_path).resolve()),
        "first_frame_prompt": str(row.get("first_frame_prompt") or row.get("description") or "").strip(),
        "motion_prompt": str(row.get("motion_prompt") or row.get("video_description") or "").strip(),
        "storyboard_description": _description(row),
        "characters": row.get("characters", []),
        "locations": row.get("locations", []),
        "source_node_id": row.get("node_id"),
        "source_event_id": row.get("event_id"),
        "duration_sec": row.get("duration_sec"),
    }


def _protocol(shots: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(shots)
    shot_count_status = "official_range" if 8 <= count <= 16 else "out_of_protocol"
    scene_complete = bool(shots) and all(shot.get("scene_id") is not None for shot in shots)
    scene_ids = {shot["scene_id"] for shot in shots if shot.get("scene_id") is not None}
    return {
        "shot_count_status": shot_count_status,
        "scene_label_status": "complete" if scene_complete else "missing",
        "scene_count": len(scene_ids),
        "official_vimax_result_eligible": shot_count_status == "official_range" and scene_complete and 2 <= len(scene_ids) <= 4,
        "note": "CameraMaster samples are custom adaptations, not official ViMax-Bench stories.",
    }


def build_vimax_samples(
    inputs_json: str | Path,
    track: str | None = None,
    shots_per_sample: int | None = None,
    scene_map: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one or more ViMax-style samples from DirectorBench inputs.

    ``shots_per_sample`` chunks the original source clip order.  Scene labels
    must be supplied explicitly (or embedded in storyboard rows); they are
    never guessed from clip position.
    """
    payload = _read_json(Path(inputs_json))
    records = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("inputs JSON must contain a samples list")
    if track is not None:
        records = [record for record in records if str(record.get("track")) == str(track)]
    if not records:
        raise ValueError("no matching sample in inputs JSON")
    if shots_per_sample is not None and shots_per_sample <= 0:
        raise ValueError("shots_per_sample must be positive")
    scene_map = dict(scene_map or {})
    output: list[dict[str, Any]] = []
    for record in records:
        direct_shots = record.get("shots")
        if isinstance(direct_shots, list):
            source_shots = []
            for index, row in enumerate(direct_shots, start=1):
                if not isinstance(row, dict) or not row.get("video_path"):
                    raise ValueError(f"sample {record.get('sample_id')} has an invalid normalized shot")
                source_shots.append(_shot_row(row, str(row["video_path"]), index, scene_map))
            chunk = shots_per_sample or len(source_shots)
            for offset in range(0, len(source_shots), chunk):
                shots = source_shots[offset:offset + chunk]
                suffix = f"__part{offset // chunk + 1:02d}" if len(source_shots) > chunk else ""
                output.append({
                    "sample_id": f"{record.get('sample_id', 'sample')}{suffix}",
                    "story_id": record.get("story_id") or record.get("movie_id") or record.get("sample_id"),
                    "track": record.get("track"), "split": record.get("split", "custom"),
                    "consistency_type": record.get("consistency_type", "unknown"),
                    "prompt": record.get("prompt", ""), "shots": shots, "protocol": _protocol(shots),
                    "source": {"inputs_json": str(Path(inputs_json).resolve()), "normalized_input": True},
                })
            continue
        clip_paths = record.get("source_clip_paths") or []
        storyboard_path = record.get("storyboard_path")
        if not clip_paths or not storyboard_path:
            raise ValueError(f"sample {record.get('sample_id')} lacks source_clip_paths/storyboard_path")
        rows = _storyboard_rows(Path(storyboard_path))
        if len(rows) != len(clip_paths):
            raise ValueError(f"sample {record.get('sample_id')} has {len(clip_paths)} clips but {len(rows)} storyboard rows")
        chunk = shots_per_sample or len(clip_paths)
        for offset in range(0, len(clip_paths), chunk):
            pairs = list(zip(clip_paths[offset:offset + chunk], rows[offset:offset + chunk]))
            shots = [_shot_row(row, path, offset + index + 1, scene_map) for index, (path, row) in enumerate(pairs)]
            suffix = f"__part{offset // chunk + 1:02d}" if len(clip_paths) > chunk else ""
            sample = {
                "sample_id": f"{record.get('sample_id', 'sample')}{suffix}",
                "story_id": record.get("movie_id") or record.get("sample_id"),
                "track": record.get("track"),
                "split": "custom",
                "consistency_type": "unknown",
                "prompt": record.get("prompt", ""),
                "shots": shots,
                "protocol": _protocol(shots),
                "source": {
                    "inputs_json": str(Path(inputs_json).resolve()),
                    "storyboard_path": str(Path(storyboard_path).resolve()),
                    "source_clip_count": len(clip_paths),
                },
            }
            output.append(sample)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--track")
    parser.add_argument("--shots-per-sample", type=int)
    parser.add_argument("--scene-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = build_vimax_samples(args.inputs_json, args.track, args.shots_per_sample, load_scene_map(args.scene_map))
    args.output.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(samples), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
