"""Video-only segmentation and sampling primitives for hierarchical SSS."""

from __future__ import annotations

import json
import re
import subprocess
import base64
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


def sample_timestamps(start_sec: float, end_sec: float, count: int) -> list[float]:
    start, end = float(start_sec), float(end_sec)
    if count <= 0:
        raise ValueError("frame count must be positive")
    if start < 0 or end <= start:
        raise ValueError("segment must have a positive duration and non-negative start")
    duration = end - start
    return [start + duration * (index + 0.5) / count for index in range(count)]


def _read_adapter(adapter_inputs: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(adapter_inputs, Mapping):
        return adapter_inputs
    try:
        payload = json.loads(Path(adapter_inputs).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter inputs {adapter_inputs}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("adapter inputs must be a JSON object")
    return payload


def track_a_segments(
    adapter_inputs: str | Path | Mapping[str, Any],
    track: str,
    video_path: str | Path,
    duration_resolver: Callable[[str], float],
) -> list[dict[str, Any]]:
    """Translate ordered Track A source clips to offsets in its concatenated video.

    Only temporal identity and source paths are returned. Storyboard, prompt,
    and narrative metadata are deliberately not copied to reconstruction input.
    """
    payload = _read_adapter(adapter_inputs)
    samples = payload.get("samples", [])
    matches = [sample for sample in samples if str(sample.get("track", "")).upper() == track.upper()]
    if len(matches) != 1:
        raise ValueError(f"adapter inputs must contain exactly one Track {track} sample; found {len(matches)}")
    sample = matches[0]
    expected_video = sample.get("video_path")
    if not expected_video or Path(expected_video).resolve() != Path(video_path).resolve():
        raise ValueError("selected video path does not match the adapter sample video_path")
    clips = sample.get("source_clip_paths") or []
    node_ids = sample.get("ordered_node_ids") or []
    if not clips or len(clips) != len(node_ids):
        raise ValueError("Track A mapping requires matching non-empty source_clip_paths and ordered_node_ids")
    segments = []
    start = 0.0
    for index, (clip_path, node_id) in enumerate(zip(clips, node_ids)):
        try:
            duration = float(duration_resolver(str(clip_path)))
        except Exception as exc:
            raise ValueError(f"cannot resolve duration for Track A clip {clip_path}: {exc}") from exc
        if duration <= 0:
            raise ValueError(f"Track A clip has invalid duration: {clip_path}")
        end = start + duration
        segments.append({
            "shot_index": index,
            "shot_id": str(node_id),
            "start_sec": start,
            "end_sec": end,
            "source_clip_path": str(clip_path),
        })
        start = end
    return segments


def resolve_segments(
    mode: str,
    track: str,
    video_path: str | Path,
    duration_sec: float,
    adapter_inputs: str | Path | Mapping[str, Any] | None,
    duration_resolver: Callable[[str], float],
    threshold: float = 0.3,
    uniform_fallback_sec: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = mode.lower()
    track = track.upper()
    if mode not in {"v2", "v3"}:
        raise ValueError("hierarchical segmentation requires mode v2 or v3")
    if mode == "v3" and track == "A":
        if adapter_inputs is None:
            raise ValueError("v3 Track A requires --track-adapter-inputs")
        segments = track_a_segments(adapter_inputs, track, video_path, duration_resolver)
        mapped_duration = segments[-1]["end_sec"]
        return segments, {
            "segmentation": "track_a_generated_clip_boundaries",
            "track": track,
            "mapped_duration_sec": mapped_duration,
            "video_duration_delta_sec": round(float(duration_sec) - mapped_duration, 3),
            "source_mapping": str(adapter_inputs) if not isinstance(adapter_inputs, Mapping) else "inline",
        }
    shots, detector_info = detect_shots(video_path, duration_sec, threshold, uniform_fallback_sec)
    for shot in shots:
        shot["shot_id"] = f"shot_{shot['shot_index'] + 1:04d}"
    return shots, {"segmentation": "automatic", "track": track, **detector_info}


def detect_shots(
    video_path: str | Path,
    duration_sec: float,
    threshold: float = 0.3,
    uniform_fallback_sec: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect scene cuts with PySceneDetect or ffmpeg; fallback only by request."""
    duration = float(duration_sec)
    if duration <= 0:
        raise ValueError("video duration must be positive")
    cuts: list[float] = []
    detector = ""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=float(threshold) * 100.0))
        manager.detect_scenes(video)
        cuts = [float(start.get_seconds()) for start, _ in manager.get_scene_list() if 0 < start.get_seconds() < duration]
        detector = "pyscenedetect_content"
    except ImportError:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-i", str(video_path), "-vf",
                    f"select='gt(scene,{float(threshold):.4f})',showinfo", "-an", "-f", "null", "-",
                ],
                check=True, capture_output=True, text=True,
            )
            cuts = [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
            detector = "ffmpeg_scene_filter"
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            if uniform_fallback_sec is None:
                raise RuntimeError(f"scene detection failed: {exc}") from exc
            if uniform_fallback_sec <= 0:
                raise ValueError("uniform fallback segment duration must be positive") from exc
            shots = []
            start = 0.0
            while start < duration:
                end = min(start + uniform_fallback_sec, duration)
                shots.append({"shot_index": len(shots), "start_sec": start, "end_sec": end})
                start = end
            return shots, {"detector": "uniform_fallback", "fallback": True, "segment_sec": uniform_fallback_sec}
    except Exception as exc:
        if uniform_fallback_sec is None:
            raise RuntimeError(f"scene detection failed: {exc}") from exc
        if uniform_fallback_sec <= 0:
            raise ValueError("uniform fallback segment duration must be positive") from exc
        shots = []
        start = 0.0
        while start < duration:
            end = min(start + uniform_fallback_sec, duration)
            shots.append({"shot_index": len(shots), "start_sec": start, "end_sec": end})
            start = end
        return shots, {"detector": "uniform_fallback", "fallback": True, "segment_sec": uniform_fallback_sec}

    unique_cuts = sorted({round(cut, 3) for cut in cuts if 0 < cut < duration})
    boundaries = [0.0, *unique_cuts, duration]
    shots = [
        {"shot_index": index, "start_sec": start, "end_sec": end}
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        if end > start
    ]
    return shots, {"detector": detector, "fallback": False, "threshold": float(threshold), "cut_count": len(unique_cuts)}


def extract_segment_frames(
    video_path: str | Path,
    segment: Mapping[str, Any],
    frame_dir: str | Path,
    count: int = 1,
    max_width: int = 768,
) -> list[dict[str, Any]]:
    """Extract evenly spaced midpoint frames inside one shot segment."""
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    timestamps = sample_timestamps(segment["start_sec"], segment["end_sec"], count)
    target_dir = Path(frame_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    shot_id = str(segment.get("shot_id", segment.get("shot_index", 0)))
    for frame_index, timestamp in enumerate(timestamps, start=1):
        output = target_dir / f"shot_{shot_id}_frame_{frame_index:02d}.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}",
            "-i", str(video_path), "-frames:v", "1", "-vf", f"scale='min({max_width},iw)':-2",
            "-q:v", "4", str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            details = getattr(exc, "stderr", "")
            raise RuntimeError(f"cannot extract shot {shot_id} frame at {timestamp:.3f}s: {details}") from exc
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty frame for shot {shot_id} at {timestamp:.3f}s")
        frames.append({
            "index": frame_index,
            "timestamp_sec": round(timestamp, 3),
            "path": str(output),
        })
    return frames


def build_l1_prompt(segment: Mapping[str, Any]) -> str:
    return (
        "Write an evidence-grounded caption for this video shot. Describe only visible people/entities, "
        "actions, composition, setting, and observable state changes. Do not infer unseen causes, "
        "relationships, intentions, or plot context. Mark uncertainty. Return JSON with keys "
        "caption (string), visible_entities (array), actions (array), state_changes (array), "
        "evidence (array of visible details), and uncertainties (array).\n"
        f"Shot ID: {segment.get('shot_id', segment.get('shot_index', 'shot'))}\n"
        f"Time range: {float(segment['start_sec']):.3f}-{float(segment['end_sec']):.3f} seconds."
    )


def build_l2_prompt(l1_captions: list[Mapping[str, Any]]) -> str:
    return (
        "Group the following ordered shot captions into the smallest set of continuous micro-events. "
        "Merge shots that depict one continuing action; preserve event order and visible state changes. "
        "Do not add causes, motives, or events absent from captions. Return JSON with key events, an array "
        "of objects containing event_id, shot_ids, description, state_changes, and uncertainties.\n\n"
        + json.dumps(l1_captions, ensure_ascii=False)
    )


def build_l3_prompt(l2_events: list[Mapping[str, Any]]) -> str:
    return (
        "Group the ordered micro-events into scene paragraphs. A scene is a continuous local setting/time "
        "unit; do not invent a scene boundary unless the events support it. Preserve character interactions, "
        "conflict, and state progression. Return JSON with key scenes, an array of objects containing scene_id, "
        "event_ids, description, transition_to_next, and uncertainties.\n\n"
        + json.dumps(l2_events, ensure_ascii=False)
    )


def build_l4_prompt(l3_scenes: list[Mapping[str, Any]], max_words: int = 180) -> str:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    return (
        "Write a concise, coherent global story synopsis grounded only in these reconstructed scene paragraphs. "
        "Preserve the main visible characters, conflict, key event progression, relationships only when evidenced, "
        "and ending only when shown. Do not add unseen motivation or resolve ambiguity. Use a consistent synopsis "
        f"style and no more than {max_words} words. Return JSON with keys story, characters, conflict, "
        "event_progression, ending, and uncertainties.\n\n"
        + json.dumps(l3_scenes, ensure_ascii=False)
    )


def build_l4_normalization_prompt(raw_story: str, max_words: int = 180) -> str:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    return (
        "Normalize the following video-only reconstructed story into a concise global synopsis. Preserve only its supported "
        "characters, conflict, event progression, and shown ending; do not add facts or use outside knowledge. This is a "
        "format/length normalization only. Do not expose the reference or infer any reference story. Use no more than "
        f"{max_words} words for the story field. Return JSON with keys story, characters, conflict, event_progression, ending, "
        f"and uncertainties.\n\nVIDEO-ONLY STORY TO NORMALIZE:\n{raw_story}"
    )


def _required_text(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {field} must be non-empty text")
    return value.strip()


def validate_l1_payload(payload: Mapping[str, Any], shot_id: str) -> dict[str, Any]:
    return {
        "shot_id": shot_id,
        "caption": _required_text(payload, "caption", "L1"),
        "visible_entities": payload.get("visible_entities", []),
        "actions": payload.get("actions", []),
        "state_changes": payload.get("state_changes", []),
        "evidence": payload.get("evidence", []),
        "uncertainties": payload.get("uncertainties", []),
    }


def _validate_rows(payload: Mapping[str, Any], field: str, id_field: str, label: str) -> list[dict[str, Any]]:
    rows = payload.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain a non-empty {field} array")
    normalized = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} {field} entries must be objects")
        normalized_row = dict(row)
        normalized_row[id_field] = str(row.get(id_field) or f"{id_field.removesuffix('_id')}_{index:03d}")
        normalized_row["description"] = _required_text(row, "description", label)
        normalized.append(normalized_row)
    return normalized


def validate_l2_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _validate_rows(payload, "events", "event_id", "L2")


def validate_l3_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _validate_rows(payload, "scenes", "scene_id", "L3")


def validate_l4_payload(payload: Mapping[str, Any], max_words: int = 180) -> dict[str, Any]:
    story = _required_text(payload, "story", "L4")
    word_count = len(story.split())
    if word_count > max_words:
        raise ValueError(f"L4 story exceeds word limit ({word_count} > {max_words})")
    return {
        "story": story,
        "characters": payload.get("characters", []),
        "conflict": payload.get("conflict", ""),
        "event_progression": payload.get("event_progression", []),
        "ending": payload.get("ending", ""),
        "uncertainties": payload.get("uncertainties", []),
        "word_count": word_count,
    }


def _multimodal_l1_content(prompt: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Video frame at {frame['timestamp_sec']:.3f} seconds:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"}})
    return content


def _validated_reconstruction_json(
    chat_call: Callable[[list[dict[str, Any]], str], str],
    messages: list[dict[str, Any]],
    context: str,
    raw_path: Path,
    validator: Callable[[Mapping[str, Any]], Any],
    output_dir: Path,
    stage: str,
) -> Any:
    """Retry once when a hierarchical reconstruction response violates its JSON schema."""

    from sss_smoke_test import extract_json_object

    raw = chat_call(messages, context)
    raw_path.write_text(raw, encoding="utf-8")
    try:
        return validator(extract_json_object(raw))
    except (ValueError, json.JSONDecodeError, TypeError) as first_error:
        original_content = messages[0].get("content", "") if messages else ""
        retry_instruction = (
            "Your previous response failed schema validation: "
            f"{first_error}. Return only one valid JSON object that exactly satisfies "
            "the requested keys and structure. Do not include Markdown or commentary."
        )
        if isinstance(original_content, list):
            retry_content: str | list[dict[str, Any]] = [
                {"type": "text", "text": retry_instruction},
                *original_content,
            ]
        else:
            retry_content = f"{original_content}\n\n{retry_instruction}"
        retry_raw = chat_call([{"role": "user", "content": retry_content}], f"{context}.json_retry_1")
        retry_path = raw_path.with_name(f"{raw_path.stem}.json_retry_1{raw_path.suffix}")
        retry_path.write_text(retry_raw, encoding="utf-8")
        try:
            value = validator(extract_json_object(retry_raw))
        except (ValueError, json.JSONDecodeError, TypeError) as retry_error:
            details_path = output_dir / f"{stage}_schema_failure.json"
            details_path.write_text(json.dumps({
                "stage": stage,
                "context": context,
                "first_error": str(first_error),
                "retry_error": str(retry_error),
                "first_response_path": str(raw_path),
                "retry_response_path": str(retry_path),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                "sss_hierarchical: reconstruction JSON retry failed "
                f"stage={stage} first_error={first_error!s} retry_error={retry_error!s} "
                f"details={details_path} first_response={raw_path} retry_response={retry_path}",
                file=sys.stderr,
            )
            raise ValueError(
                f"{stage} response remained invalid after one JSON retry; details={details_path}"
            ) from retry_error
        print(
            "sss_hierarchical: reconstruction JSON retry succeeded "
            f"stage={stage} first_error={first_error!s} "
            f"first_response={raw_path} retry_response={retry_path}",
            file=sys.stderr,
        )
        return value


def run_hierarchical_reconstruction(
    segments: list[Mapping[str, Any]],
    video_path: str | Path,
    frame_dir: str | Path,
    output_dir: str | Path,
    chat_call: Callable[[list[dict[str, Any]], str], str],
    frames_per_shot: int = 1,
    max_l4_words: int = 180,
) -> dict[str, Any]:
    """Run blind L1-L4 reconstruction using only video-derived evidence."""
    if not segments:
        raise ValueError("at least one shot segment is required")
    output = Path(output_dir)
    raw_dir = output / "raw_responses"
    prompt_dir = output / "prompts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    l1_captions = []
    frame_manifest = []
    for index, segment in enumerate(segments, start=1):
        shot_id = str(segment.get("shot_id", segment.get("shot_index", index - 1)))
        frames = extract_segment_frames(video_path, segment, frame_dir, frames_per_shot)
        frame_manifest.append({"shot_id": shot_id, "frames": frames})
        prompt = build_l1_prompt(segment)
        (prompt_dir / f"l1_{index:04d}.txt").write_text(prompt, encoding="utf-8")
        caption = _validated_reconstruction_json(
            chat_call,
            [{"role": "user", "content": _multimodal_l1_content(prompt, frames)}],
            f"reconstruction.L1.shot_{index:04d}:{shot_id}",
            raw_dir / f"l1_{index:04d}.txt",
            lambda payload: validate_l1_payload(payload, shot_id),
            output, f"L1_shot_{index:04d}",
        )
        caption.update({"start_sec": segment["start_sec"], "end_sec": segment["end_sec"]})
        l1_captions.append(caption)
    l2_prompt = build_l2_prompt(l1_captions)
    (prompt_dir / "l2.txt").write_text(l2_prompt, encoding="utf-8")
    l2_events = _validated_reconstruction_json(
        chat_call, [{"role": "user", "content": l2_prompt}], "reconstruction.L2",
        raw_dir / "l2.txt", validate_l2_payload, output, "L2",
    )
    l3_prompt = build_l3_prompt(l2_events)
    (prompt_dir / "l3.txt").write_text(l3_prompt, encoding="utf-8")
    l3_scenes = _validated_reconstruction_json(
        chat_call, [{"role": "user", "content": l3_prompt}], "reconstruction.L3",
        raw_dir / "l3.txt", validate_l3_payload, output, "L3",
    )
    l4_prompt = build_l4_prompt(l3_scenes, max_l4_words)
    (prompt_dir / "l4.txt").write_text(l4_prompt, encoding="utf-8")
    l4 = _validated_reconstruction_json(
        chat_call, [{"role": "user", "content": l4_prompt}], "reconstruction.L4",
        raw_dir / "l4.txt", lambda payload: validate_l4_payload(payload, max_l4_words), output, "L4",
    )
    result = {"L1": l1_captions, "L2": l2_events, "L3": l3_scenes, "L4": l4}
    (output / "frame_manifest.json").write_text(json.dumps({"shots": frame_manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "hierarchical_reconstruction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
