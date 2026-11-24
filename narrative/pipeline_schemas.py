from __future__ import annotations

from typing import Any

LOCATION_VIEW_NAMES = ("extreme_close", "close", "medium", "wide")
LOCATION_VIEW_SET = set(LOCATION_VIEW_NAMES)
DURATION_FRAME_MAP = {5: 124, 10: 243}


def normalize_location_view(value: Any, default: str = "wide") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "establishing": "wide",
        "wide_establishing": "wide",
        "mid": "medium",
        "medium_view": "medium",
        "closeup": "close",
        "close_up": "close",
        "detail": "extreme_close",
        "extreme_close_up": "extreme_close",
        "macro": "extreme_close",
    }
    text = aliases.get(text, text)
    if text in LOCATION_VIEW_SET:
        return text
    fallback = str(default or "wide").strip().lower()
    return fallback if fallback in LOCATION_VIEW_SET else "wide"


def frames_for_duration_sec(duration_sec: Any) -> int:
    duration = int(round(float(duration_sec)))
    if duration not in DURATION_FRAME_MAP:
        raise ValueError("MiniMax H3 video duration must be exactly 5 or 10 seconds")
    return DURATION_FRAME_MAP[duration]


def normalize_duration_frame_plan(payload: dict[str, Any] | None, *, default_duration_sec: int = 5) -> dict[str, Any]:
    data = dict(payload or {})
    duration = int(round(float(data.get("duration_sec", default_duration_sec) or default_duration_sec)))
    if duration not in DURATION_FRAME_MAP:
        duration = 10 if duration >= 8 else 5
    data["duration_sec"] = duration
    data["num_frames"] = DURATION_FRAME_MAP[duration]
    shots = data.get("shots", [])
    data["shots"] = shots if isinstance(shots, list) else []
    return data
