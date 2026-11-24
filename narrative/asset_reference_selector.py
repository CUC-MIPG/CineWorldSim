from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from narrative.pipeline_schemas import LOCATION_VIEW_NAMES, normalize_location_view
from narrative.visual_prompt_constraints import event_needs_multi_person_references

_REFERENCE_NAME_STOPWORDS = {
    "the",
    "and",
    "old",
    "new",
    "dark",
    "light",
    "small",
    "large",
    "true",
    "hidden",
    "sealed",
    "broken",
    "cracked",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value).strip()


def _text_blob(event: dict[str, Any]) -> str:
    fields = [
        "story_description",
        "video_description",
        "event_summary",
        "dialogue",
        "rendered_dialogue",
        "event_type",
        "story_phase",
    ]
    return " ".join(_safe_text(event.get(field)) for field in fields).lower()


def _name_parts(name: str) -> list[str]:
    text = re.sub(r"[_\-]+", " ", _safe_text(name)).lower()
    return [
        part
        for part in re.findall(r"[a-z0-9]+", text)
        if len(part) >= 4 and part not in _REFERENCE_NAME_STOPWORDS
    ]


def _mentions(text: str, name: str) -> bool:
    normalized = re.sub(r"[_\-]+", " ", _safe_text(name)).lower().strip()
    if not normalized:
        return False
    if normalized in text:
        return True
    parts = _name_parts(normalized)
    return bool(parts) and any(part in text for part in parts)


def _matches_asset(row: dict[str, Any], name: str) -> bool:
    wanted = re.sub(r"[_\-]+", " ", _safe_text(name)).lower()
    rid = re.sub(r"[_\-]+", " ", _safe_text(row.get("id"))).lower()
    rname = re.sub(r"[_\-]+", " ", _safe_text(row.get("name"))).lower()
    return bool(wanted and (wanted == rid or wanted == rname or wanted in {rid, rname}))


def _existing(path: str) -> str:
    value = _safe_text(path)
    if not value:
        return ""
    p = Path(value)
    if p.exists() and p.stat().st_size > 0:
        return value
    return value if not p.is_absolute() else ""


def _dedupe(paths: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = _existing(path)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def _location_path(row: dict[str, Any], preferred_view: str = "medium") -> str:
    views = row.get("views", {}) if isinstance(row.get("views"), dict) else {}
    ordered_views = [normalize_location_view(preferred_view, default="medium"), "medium", "wide", "close", "extreme_close"]
    for view in ordered_views:
        payload = views.get(view)
        if isinstance(payload, dict):
            path = _safe_text(payload.get("last_image_path") or payload.get("output_path"))
            if path:
                return path
    return _safe_text(row.get("last_image_path"))


def _item_path(row: dict[str, Any]) -> str:
    views = row.get("views", {}) if isinstance(row.get("views"), dict) else {}
    isolated = views.get("isolated") if isinstance(views.get("isolated"), dict) else {}
    return _safe_text(
        isolated.get("last_image_path")
        or isolated.get("output_path")
        or row.get("last_image_path")
        or row.get("initial_image_path")
    )


def select_event_reference_images(
    *,
    event: dict[str, Any],
    current_state: dict[str, Any],
    character_portraits: dict[str, str] | None = None,
    max_reference_images: int = 4,
) -> dict[str, Any]:
    text = _text_blob(event)
    characters = [_safe_text(x) for x in event.get("characters", []) or [] if _safe_text(x)]
    locations = [_safe_text(x) for x in event.get("locations", []) or [] if _safe_text(x)]
    items = [_safe_text(x) for x in event.get("items", []) or [] if _safe_text(x)]
    allow_multi = event_needs_multi_person_references(event)
    cap = max(1, int(max_reference_images or 1))

    portraits = character_portraits or {}
    selected_chars: list[str] = []
    for name in (characters if allow_multi else characters[:1]):
        path = _safe_text(portraits.get(name))
        if path:
            selected_chars.append(path)

    selected_locations: list[str] = []
    preferred_view = _safe_text(event.get("preferred_location_view") or event.get("location_view") or "medium")
    for wanted in locations[:1]:
        for row in current_state.get("locations", []) or []:
            if isinstance(row, dict) and _matches_asset(row, wanted):
                path = _location_path(row, preferred_view=preferred_view)
                if path:
                    selected_locations.append(path)
                break

    selected_items: list[str] = []
    mentioned_items = [item for item in items if _mentions(text, item)]
    for wanted in mentioned_items:
        for row in current_state.get("items", []) or []:
            if isinstance(row, dict) and _matches_asset(row, wanted):
                path = _item_path(row)
                if path:
                    selected_items.append(path)
                break

    selected = _dedupe([*selected_chars, *selected_locations, *selected_items], limit=cap)
    excluded: list[str] = []
    all_character_paths = [_safe_text(x) for x in portraits.values() if _safe_text(x)]
    all_location_paths = [_location_path(row, preferred_view=preferred_view) for row in current_state.get("locations", []) or [] if isinstance(row, dict)]
    all_item_paths = [_item_path(row) for row in current_state.get("items", []) or [] if isinstance(row, dict)]
    for path in _dedupe([*all_character_paths, *all_location_paths, *all_item_paths]):
        if path not in selected:
            excluded.append(path)

    return {
        "reference_image_paths": selected,
        "character_reference_paths": [p for p in _dedupe(selected_chars) if p in selected],
        "location_reference_paths": [p for p in _dedupe(selected_locations) if p in selected],
        "item_reference_paths": [p for p in _dedupe(selected_items) if p in selected],
        "excluded_assets": excluded,
        "mentioned_items": mentioned_items,
        "multi_person_reference_allowed": allow_multi,
        "reference_selection_reason": "selected only characters, location view, and items explicitly required by the event text",
    }
