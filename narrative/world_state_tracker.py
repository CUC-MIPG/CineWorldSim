from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from narrative.pipeline_schemas import LOCATION_VIEW_NAMES, normalize_location_view


def _image_for_backend(images: list[dict[str, Any]], backend_tool: str) -> str:
    for image in images or []:
        if image.get("tool") == backend_tool and image.get("ok"):
            return str(image.get("output_path", "") or "")
    for image in images or []:
        if image.get("ok"):
            return str(image.get("output_path", "") or "")
    return ""


def _view_image_path(view_payload: dict[str, Any], backend_tool: str) -> str:
    if not isinstance(view_payload, dict):
        return ""
    if view_payload.get("output_path"):
        return str(view_payload.get("output_path", "") or "")
    if isinstance(view_payload.get("images"), list):
        return _image_for_backend(view_payload.get("images", []), backend_tool)
    return ""


def _history_entry(step: int, path: str, reason: str) -> dict[str, Any]:
    return {"step": int(step), "path": path, "reason": reason}


def _location_asset_row(asset: dict[str, Any], backend_tool: str) -> dict[str, Any]:
    description = str(asset.get("description", "") or "")
    raw_views = asset.get("views", {}) if isinstance(asset.get("views"), dict) else {}
    fallback_path = str(asset.get("last_image_path") or asset.get("initial_image_path") or "")
    if not fallback_path:
        fallback_path = _image_for_backend(asset.get("images", []), backend_tool)
    views: dict[str, dict[str, Any]] = {}
    for view in LOCATION_VIEW_NAMES:
        view_payload = raw_views.get(view, {}) if isinstance(raw_views, dict) else {}
        path = _view_image_path(view_payload, backend_tool) or fallback_path
        views[view] = {
            "current_state_text": str(asset.get("current_state_text") or description),
            "last_image_path": path,
            "image_history": [_history_entry(0, path, f"initial_location_{view}")],
        }
    image_path = views["wide"]["last_image_path"] or fallback_path
    return {
        "id": str(asset.get("asset_id") or asset.get("id") or asset.get("name") or ""),
        "name": str(asset.get("name", "") or ""),
        "description": description,
        "current_state_text": str(asset.get("current_state_text") or description),
        "last_image_path": image_path,
        "changed_this_step": False,
        "changed_views_this_step": [],
        "change_reason": "",
        "image_history": [_history_entry(0, image_path, "initial_world_model")],
        "views": views,
    }


def _item_asset_row(asset: dict[str, Any], backend_tool: str) -> dict[str, Any]:
    description = str(asset.get("description", "") or "")
    image_path = str(asset.get("initial_image_path") or asset.get("last_image_path") or "")
    if not image_path:
        image_path = _image_for_backend(asset.get("images", []), backend_tool)
    view = {
        "current_state_text": str(asset.get("current_state_text") or description),
        "last_image_path": image_path,
        "image_history": [_history_entry(0, image_path, "initial_item_model")],
    }
    return {
        "id": str(asset.get("asset_id") or asset.get("id") or asset.get("name") or ""),
        "name": str(asset.get("name", "") or ""),
        "description": description,
        "current_state_text": str(asset.get("current_state_text") or description),
        "last_image_path": image_path,
        "changed_this_step": False,
        "changed_views_this_step": [],
        "change_reason": "",
        "image_history": [_history_entry(0, image_path, "initial_world_model")],
        "views": {"isolated": view},
    }


def _character_asset_row(asset: dict[str, Any]) -> dict[str, Any]:
    views = asset.get("views", {}) if isinstance(asset.get("views"), dict) else {}
    normalized_views = {}
    for name in ("front_full", "side", "portrait"):
        payload = views.get(name, {}) if isinstance(views.get(name), dict) else {}
        normalized_views[name] = {
            "last_image_path": str(payload.get("output_path", "") or ""),
            "image_history": [_history_entry(0, str(payload.get("output_path", "") or ""), f"initial_character_{name}")],
        }
    front = normalized_views["front_full"]["last_image_path"]
    return {
        "id": str(asset.get("name") or asset.get("id") or ""),
        "name": str(asset.get("name") or ""),
        "current_state_text": str(asset.get("role_profile") or asset.get("name") or ""),
        "last_image_path": front,
        "changed_this_step": False,
        "change_reason": "",
        "views": normalized_views,
        "appearance_updates": [],
    }
def build_initial_world_state(*, world_model: dict[str, Any], backend_tool: str) -> dict[str, Any]:
    return {
        "step": 0,
        "summary": str(world_model.get("summary", "") or ""),
        "locations": [_location_asset_row(asset, backend_tool) for asset in world_model.get("locations", [])],
        "items": [_item_asset_row(asset, backend_tool) for asset in world_model.get("items", [])],
        "characters": [_character_asset_row(asset) for asset in world_model.get("characters", [])],
    }


def _apply_updates(rows: list[dict[str, Any]], updates: list[dict[str, Any]], *, is_location: bool) -> list[dict[str, Any]]:
    by_id = {str(update.get("id", "")): update for update in updates or []}
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = deepcopy(row)
        new_row["changed_this_step"] = False
        new_row["changed_views_this_step"] = []
        new_row["change_reason"] = ""
        update = by_id.get(str(row.get("id", "")))
        if update:
            new_row["current_state_text"] = str(update.get("new_state_text") or row.get("current_state_text") or "")
            new_row["changed_this_step"] = bool(update.get("image_refresh_required", False))
            new_row["change_reason"] = str(update.get("reason", "") or "")
            view_name = normalize_location_view(update.get("view_to_update"), default="wide") if is_location else "isolated"
            views = new_row.get("views", {}) if isinstance(new_row.get("views"), dict) else {}
            if view_name not in views:
                views[view_name] = {
                    "current_state_text": str(row.get("current_state_text") or ""),
                    "last_image_path": str(row.get("last_image_path") or ""),
                    "image_history": list(row.get("image_history", [])),
                }
            views[view_name]["current_state_text"] = new_row["current_state_text"]
            if update.get("new_image_path"):
                new_row["last_image_path"] = str(update["new_image_path"])
                history = list(new_row.get("image_history", []))
                history.append(
                    {
                        "step": int(update.get("step", 0) or 0),
                        "path": str(update["new_image_path"]),
                        "reason": str(update.get("reason", "") or "world_delta_refresh"),
                    }
                )
                new_row["image_history"] = history
                view_history = list(views[view_name].get("image_history", []))
                view_history.append(
                    {
                        "step": int(update.get("step", 0) or 0),
                        "path": str(update["new_image_path"]),
                        "reason": str(update.get("reason", "") or "world_delta_refresh"),
                    }
                )
                views[view_name]["last_image_path"] = str(update["new_image_path"])
                views[view_name]["image_history"] = view_history
            new_row["views"] = views
            if update.get("image_refresh_required"):
                new_row["changed_views_this_step"] = [view_name]
        updated_rows.append(new_row)
    return updated_rows


def update_world_state_from_delta(*, previous_state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(previous_state)
    state["step"] = int(delta.get("step", previous_state.get("step", 0)))
    state["summary"] = str(delta.get("summary") or previous_state.get("summary") or "")
    state["locations"] = _apply_updates(state.get("locations", []), delta.get("location_updates", []), is_location=True)
    state["items"] = _apply_updates(state.get("items", []), delta.get("item_updates", []), is_location=False)
    state["characters"] = _apply_character_updates(state.get("characters", []), delta.get("character_updates", []))
    return state


def _apply_character_updates(rows: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(update.get("id", "")): update for update in updates or []}
    result = []
    for row in rows:
        current = deepcopy(row)
        current["changed_this_step"] = False
        update = by_id.get(str(row.get("id", "")))
        if update:
            path = str(update.get("new_image_path", "") or "")
            current["current_state_text"] = str(update.get("new_state_text") or current.get("current_state_text", ""))
            current["changed_this_step"] = bool(update.get("image_refresh_required"))
            current["change_reason"] = str(update.get("reason", "") or "")
            if path:
                current["last_image_path"] = path
                front = current.setdefault("views", {}).setdefault("front_full", {})
                front["last_image_path"] = path
                front.setdefault("image_history", []).append({"step": int(update.get("step", 0)), "path": path, "reason": current["change_reason"]})
            current.setdefault("appearance_updates", []).append(deepcopy(update))
        result.append(current)
    return result


def write_world_state(path: Path, state: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
