from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_NONE_VALUES = {"", "none", "null", "n/a", "no dialogue", "no narration"}


def _text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_empty(value: Any) -> bool:
    return _text(value).lower() in _NONE_VALUES


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timecode(value: Any) -> float:
    raw = _text(value)
    if not raw:
        return 0.0
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(raw)
    except ValueError:
        return 0.0


def _timecode(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _speaker_text(value: Any, *, default_speaker: str = "", default_slot: str = "") -> list[dict[str, str]]:
    if isinstance(value, dict):
        value = value.get("text") or value.get("dialogue") or value.get("line") or ""
    if _is_empty(value):
        return []
    raw = _text(value)
    match = re.match(r"^(.+?):\s*(.+)$", raw)
    if match:
        speaker, raw = match.groups()
        return [{"speaker": speaker.strip(), "character_slot": default_slot, "text": raw.strip()}]
    says_match = re.search(r"\((Character\d+)\).*?says,\s*(.+)$", raw, flags=re.IGNORECASE)
    if says_match:
        return [{"speaker": default_speaker, "character_slot": says_match.group(1), "text": says_match.group(2).strip()}]
    return [{"speaker": default_speaker, "character_slot": default_slot, "text": raw}]


def _dialogue_entries(value: Any, *, slot_mapping: dict[str, str] | None = None) -> Iterable[dict[str, str]]:
    mapping = slot_mapping or {}
    if isinstance(value, dict):
        for slot, line in value.items():
            slot_name = str(slot)
            speaker = str(mapping.get(slot_name) or "")
            for entry in _speaker_text(line, default_speaker=speaker, default_slot=slot_name):
                yield entry
        return
    yield from _speaker_text(value)


def _narration_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("narration") or value.get("voiceover") or ""
    return "" if _is_empty(value) else _text(value)


def _shots_for_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    raw_shots = task.get("event_shot_plan") or []
    shots = [shot for shot in raw_shots if isinstance(shot, dict)]
    if shots:
        return shots, True
    duration = _number(task.get("duration_sec"), 5.0)
    return [{"shot_index": 1, "start_sec": 0.0, "duration_sec": duration}], False


def build_video_timeline(task: dict[str, Any]) -> dict[str, Any]:
    shots, has_shot_plan = _shots_for_task(task)
    dialogue_backfill = task.get("dialogue_backfill")
    slot_mapping = dialogue_backfill.get("slot_mapping", {}) if isinstance(dialogue_backfill, dict) else {}
    segments: list[dict[str, Any]] = []

    for shot in shots:
        start = _number(shot.get("start_sec"), _parse_timecode(shot.get("start_timecode")))
        duration = max(0.0, _number(shot.get("duration_sec"), _number(task.get("duration_sec"), 5.0)))
        end = start + duration
        shot_index = int(shot.get("shot_index", 1) or 1)
        source = "shot_plan" if has_shot_plan else "dialogue_backfill"

        dialogue_value = shot.get("dialogue") if has_shot_plan else None
        if not has_shot_plan:
            if isinstance(dialogue_backfill, dict) and isinstance(dialogue_backfill.get("dialogue"), dict):
                dialogue_value = dialogue_backfill["dialogue"]
            else:
                dialogue_value = task.get("dialogue") or task.get("dialogue_text")
        for entry in _dialogue_entries(dialogue_value, slot_mapping=slot_mapping):
            if not entry.get("text"):
                continue
            segments.append(
                {
                    "index": len(segments) + 1,
                    "type": "dialogue",
                    "speaker": entry.get("speaker", ""),
                    "character_slot": entry.get("character_slot", ""),
                    "text": entry["text"],
                    "shot_index": shot_index,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "start_timecode": _timecode(start),
                    "end_timecode": _timecode(end),
                    "source": source,
                }
            )

        narration_value = None
        for key in ("narration", "voiceover", "narration_text", "voiceover_text"):
            if shot.get(key) is not None:
                narration_value = shot.get(key)
                break
        if narration_value is None:
            for key in ("narration", "voiceover", "narration_text", "voiceover_text"):
                if task.get(key) is not None:
                    narration_value = task.get(key)
                    break
        narration = _narration_text(narration_value)
        if narration:
            segments.append(
                {
                    "index": len(segments) + 1,
                    "type": "narration",
                    "speaker": "",
                    "character_slot": "",
                    "text": narration,
                    "shot_index": shot_index,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "start_timecode": _timecode(start),
                    "end_timecode": _timecode(end),
                    "source": source,
                }
            )

    video_path = str(task.get("output_video_path") or "")
    video_file = Path(video_path).name
    return {
        "schema_version": "1.0",
        "video_path": video_path,
        "video_filename": video_file,
        "rank": task.get("rank"),
        "node_id": task.get("node_id", ""),
        "event_id": task.get("event_id", ""),
        "task_type": task.get("task_type", "event"),
        "timing_basis": "shot_interval",
        "duration_sec": _number(task.get("duration_sec"), 5.0),
        "num_frames": task.get("num_frames"),
        "source_artifacts": {
            "dialogue": (dialogue_backfill or {}).get("artifact_path", "") if isinstance(dialogue_backfill, dict) else "",
            "story": (task.get("story_backfill") or {}).get("artifact_path", "") if isinstance(task.get("story_backfill"), dict) else "",
            "shot_plan": (task.get("shot_planning") or {}).get("artifact_path", "") if isinstance(task.get("shot_planning"), dict) else "",
        },
        "segments": segments,
    }


def timeline_path_for_video(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".timeline.json")


def write_video_timeline(task: dict[str, Any]) -> Path:
    video_path = str(task.get("output_video_path") or "")
    if not video_path:
        raise ValueError("task output_video_path is required for a timeline")
    path = timeline_path_for_video(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_video_timeline(task)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
