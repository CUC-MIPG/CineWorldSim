"""Prepare ConStory and DirectorBench inputs from a CameraMaster case output."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CASE_ROOT = Path(
    "/data1/AgentG/hhb/CameraMaster/outputs/moviebench_video_pipeline/"
    "0001_American_Beauty/arc_001_real"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "adapted_cases" / "0001_American_Beauty_arc_001_real"
DIRECTORBENCH_ROOT = Path(__file__).parent / "DirectorBench-master"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_artifact(case_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else case_root / path


def _unique_index(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            raise ValueError(f"{label} row is missing {field}")
        key = str(value)
        if key in index:
            raise ValueError(f"duplicate {label} {field}: {key}")
        index[key] = row
    return index


def _storyline(narrative_nodes: dict[str, Any], track: str) -> tuple[str, str]:
    if track == "B":
        context_storyline = narrative_nodes.get("track_b_context", {}).get("storyline")
        if context_storyline:
            return str(context_storyline).strip(), "track_b_context.storyline"
    storyline = narrative_nodes.get("storyline")
    if storyline:
        return str(storyline).strip(), "narrative_nodes.storyline"
    raise ValueError(f"Track {track} has no storyline context")


def build_track_record(
    track: str,
    movie_id: str,
    generated_items: list[dict[str, Any]],
    path_plan: dict[str, Any],
    narrative_nodes: dict[str, Any],
) -> dict[str, Any]:
    """Join nodes and generated outputs in the explicitly planned order."""
    ordered_ids = [str(value) for value in path_plan.get("ordered_node_ids", [])]
    if not ordered_ids:
        raise ValueError(f"Track {track} path plan has no ordered_node_ids")
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"duplicate IDs in Track {track} ordered path plan")

    generated_by_id = _unique_index(generated_items, "node_id", "generated node")
    nodes_by_id = _unique_index(narrative_nodes.get("selected_nodes", []), "node_id", "narrative node")
    missing_generated = [node_id for node_id in ordered_ids if node_id not in generated_by_id]
    if missing_generated:
        raise ValueError(f"Track {track} path plan has no generated record for: {missing_generated}")
    missing_nodes = [node_id for node_id in ordered_ids if node_id not in nodes_by_id]
    if missing_nodes:
        raise ValueError(f"Track {track} path plan has no narrative node for: {missing_nodes}")

    extra_generated = sorted(set(generated_by_id) - set(ordered_ids))
    if extra_generated:
        raise ValueError(f"Track {track} generated records are absent from path plan: {extra_generated}")

    prompt, prompt_source = _storyline(narrative_nodes, track)
    clips: list[dict[str, Any]] = []
    storyboard: list[dict[str, Any]] = []
    story_parts: list[str] = []
    for node_id in ordered_ids:
        generated = generated_by_id[node_id]
        node = nodes_by_id[node_id]
        video_path = generated.get("output_video_path")
        if not video_path:
            raise ValueError(f"Track {track} generated node {node_id} has no output_video_path")
        story_description = str(generated.get("story_description") or "").strip()
        if not story_description:
            raise ValueError(f"Track {track} generated node {node_id} has no story_description")

        dialogue = str(generated.get("dialogue") or "").strip()
        story_text = story_description
        if dialogue and dialogue.lower() not in {"none", "null", "n/a"}:
            story_text += f"\nDialogue: {dialogue}"
        story_parts.append(story_text)

        clips.append({
            "node_id": node_id,
            "event_id": generated.get("event_id") or node.get("event_id"),
            "output_video_path": str(Path(video_path).resolve()),
            "source_steps": generated.get("source_steps", node.get("source_steps", [])),
            "source_l2_ids": node.get("source_l2_ids", []),
            "duration_sec_planned": generated.get("duration_sec"),
        })
        storyboard.append({
            "shot_index": len(storyboard) + 1,
            "node_id": node_id,
            "event_id": generated.get("event_id") or node.get("event_id"),
            "description": generated.get("video_description") or node.get("video_description") or story_description,
            "story_description": story_description,
            "dialogue": dialogue if dialogue.lower() not in {"none", "null", "n/a"} else "",
            "characters": node.get("characters", generated.get("characters", [])),
            "locations": node.get("locations", generated.get("locations", [])),
            "items": node.get("items", generated.get("items", [])),
            "source_l2_ids": node.get("source_l2_ids", []),
            "source_time_range": node.get("source_time_range"),
        })

    return {
        "sample_id": f"{movie_id}__Track{track}",
        "movie_id": movie_id,
        "track": track,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "prompt_is_original_runtime_prompt": False,
        "generated_story": "\n\n".join(story_parts),
        "ordered_node_ids": ordered_ids,
        "clips": clips,
        "storyboard": storyboard,
    }


def _probe_media(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe could not inspect {path}: {exc}") from exc
    streams = payload.get("streams", [])
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise ValueError(f"No video stream in {path}")
    return {"duration_sec": duration, "has_audio": any(stream.get("codec_type") == "audio" for stream in streams)}


def _concat_videos(clip_paths: list[Path], list_path: Path, output_path: Path) -> dict[str, Any]:
    infos = [_probe_media(path) for path in clip_paths]
    audio_flags = {info["has_audio"] for info in infos}
    if len(audio_flags) > 1:
        raise ValueError("Cannot concatenate a track with mixed audio/no-audio clips")
    if any("\n" in str(path) or "\r" in str(path) or "'" in str(path) for path in clip_paths):
        raise ValueError("ffmpeg concat paths cannot contain quotes or newlines")

    list_path.write_text("".join(f"file '{path}'\n" for path in clip_paths), encoding="utf-8")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
        "-safe", "0", "-i", str(list_path), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "stderr", "")
        raise ValueError(f"ffmpeg concatenation failed for {output_path}: {details}") from exc
    return _probe_media(output_path)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_case(
    case_root: Path,
    output_dir: Path,
    profile_id: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    case_root, output_dir = Path(case_root).resolve(), Path(output_dir).resolve()
    manifest = _read_json(case_root / "moviebench_pipeline_manifest.json")
    track_runs = manifest.get("track_runs", [])
    movie_id = case_root.parent.name
    records: dict[str, dict[str, Any]] = {}
    run_metadata: dict[str, dict[str, Any]] = {}
    for track_run in track_runs:
        track = str(track_run.get("track", ""))
        if track not in {"A", "B"}:
            continue
        if track in records:
            raise ValueError(f"Duplicate Track {track} in pipeline manifest")
        samples = track_run.get("samples", [])
        if len(samples) != 1:
            raise ValueError(f"Expected one sample for Track {track}; found {len(samples)}")
        summary = samples[0].get("video_summary", {})
        if summary.get("failed", 0) or summary.get("skipped", 0):
            raise ValueError(f"Track {track} has failed/skipped video tasks")
        generated = summary.get("generated", [])
        narrative_path = _resolve_artifact(case_root, summary["narrative_nodes_json"])
        plan_path = _resolve_artifact(case_root, summary["path_plan_json"])
        narrative_nodes = _read_json(narrative_path)
        path_plan = _read_json(plan_path)
        records[track] = build_track_record(track, movie_id, generated, path_plan, narrative_nodes)
        run_metadata[track] = {
            "narrative_nodes_json": str(narrative_path),
            "path_plan_json": str(plan_path),
            "task_total": summary.get("total"),
            "task_success": summary.get("success"),
        }
    if set(records) != {"A", "B"}:
        raise ValueError(f"Expected Track A and Track B; found {sorted(records)}")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite to replace files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    constory_dir = output_dir / "constory"
    director_dir = output_dir / "directorbench"
    constory_dir.mkdir(parents=True, exist_ok=True)
    director_dir.mkdir(parents=True, exist_ok=True)

    parquet_rows = []
    director_inputs = []
    track_reports = {}
    for track in ("A", "B"):
        record = records[track]
        clips = [Path(clip["output_video_path"]) for clip in record["clips"]]
        for path in clips:
            if not path.is_file():
                raise ValueError(f"Missing Track {track} video clip: {path}")
        video_path = director_dir / f"track_{track}.mp4"
        concat_list = director_dir / f"track_{track}_concat.txt"
        media_info = _concat_videos(clips, concat_list, video_path)

        storyboard_path = director_dir / f"track_{track}_storyboard.json"
        script_path = director_dir / f"track_{track}_script.txt"
        _write_json(storyboard_path, record["storyboard"])
        script_path.write_text(record["generated_story"] + "\n", encoding="utf-8")
        run_output = director_dir / f"run_track_{track}"
        command = [
            "python", "-m", "directorbench.main",
            "--video", str(video_path), "--prompt", record["prompt"],
            "--script", str(script_path), "--storyboard", str(storyboard_path),
            "--profile-id", str(profile_id), "--output-dir", str(run_output),
        ]
        director_inputs.append({
            "sample_id": record["sample_id"],
            "track": track,
            "video_path": str(video_path),
            "prompt": record["prompt"],
            "prompt_source": record["prompt_source"],
            "prompt_is_original_runtime_prompt": False,
            "script_path": str(script_path),
            "storyboard_path": str(storyboard_path),
            "profile_id": profile_id,
            "duration_sec": media_info["duration_sec"],
            "has_audio": media_info["has_audio"],
            "duration_status": "minute_scale_candidate" if 45 <= media_info["duration_sec"] <= 90 else "outside_nominal_minute_range",
            "source_clip_paths": [str(path) for path in clips],
            "ordered_node_ids": record["ordered_node_ids"],
            "command": command,
            "command_shell": " ".join(shlex.quote(value) for value in command),
        })
        parquet_rows.append({
            "id": record["sample_id"],
            "prompt": record["prompt"],
            "generated_story": record["generated_story"],
            "track": track,
            "prompt_source": record["prompt_source"],
            "prompt_is_original_runtime_prompt": False,
        })
        track_reports[track] = {
            "sample_id": record["sample_id"],
            "node_count": len(clips),
            "duration_sec": media_info["duration_sec"],
            "has_audio": media_info["has_audio"],
            "duration_status": director_inputs[-1]["duration_status"],
            "ordered_node_ids": record["ordered_node_ids"],
            "source_clip_paths": [str(path) for path in clips],
            "prompt_source": record["prompt_source"],
            "prompt_is_original_runtime_prompt": False,
            **run_metadata[track],
        }

    parquet_path = constory_dir / "cameramaster_stories.parquet"
    pd.DataFrame(parquet_rows).to_parquet(parquet_path, index=False)
    director_inputs_path = director_dir / "inputs.json"
    _write_json(director_inputs_path, {"samples": director_inputs})
    commands_path = director_dir / "run_commands.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(DIRECTORBENCH_ROOT)) + "\n\n"
        + "\n\n".join(row["command_shell"] for row in director_inputs) + "\n",
        encoding="utf-8",
    )
    commands_path.chmod(0o755)

    case_report = {
        "schema_version": "cameramaster-external-bench-inputs-v1",
        "source_case_root": str(case_root),
        "source_modified": False,
        "evaluation_apis_called": False,
        "tracks": track_reports,
        "outputs": {
            "constory_parquet": str(parquet_path),
            "directorbench_inputs": str(director_inputs_path),
            "directorbench_commands": str(commands_path),
        },
        "warnings": [
            "Track A is an L3 arc sequence; Track B is a global-story sequence, so their scores are not a matched A/B comparison.",
            "Track B duration is outside DirectorBench's nominal minute-long target; treat its result as a long-video diagnostic.",
            "The recorded storyline is a context fallback, not the original runtime user prompt.",
        ],
    }
    _write_json(output_dir / "case_adapter_manifest.json", case_report)
    return case_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_case(args.case_root, args.output_dir, args.profile_id, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
