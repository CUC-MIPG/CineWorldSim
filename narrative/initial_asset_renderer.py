from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from narrative.pipeline_schemas import LOCATION_VIEW_NAMES
from narrative.visual_prompt_constraints import build_people_control_clause


def _safe_name(value: Any, fallback: str = "asset") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("_")
    return text or fallback


def _field(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_character_initial_prompt(character: Any, *, storyline: str) -> str:
    name = str(_field(character, "name", "Character") or "Character").strip()
    profile = str(_field(character, "role_profile", "") or "").strip()
    goal = str(_field(character, "final_goal", "") or "").strip()
    relations = _field(character, "known_relations", []) or []
    if not isinstance(relations, list):
        relations = [str(relations)]
    return (
        f"Initial character reference portrait for {name}. "
        f"Identity/profile: {profile or name}. Final goal: {goal or 'unknown'}. "
        f"Known relations: {', '.join(str(x) for x in relations if str(x).strip()) or 'none'}. "
        f"Story context: {storyline}. "
        "Frame the character as the dominant subject with maximum subject occupancy, clear face/body/costume readability, "
        "minimal background, minimal props, and no extra narrative action. "
        "This is an identity reference image, not an event keyframe. "
        f"\n{build_people_control_clause([name], allow_multi_people=False)}"
    )


def _character_view_prompt(character: Any, *, storyline: str, view: str) -> str:
    name = str(_field(character, "name", "Character") or "Character").strip()
    profile = str(_field(character, "role_profile", "") or name).strip()
    framing = {
        "front_full": "A straight-on, full-body front view, head to toe, standing neutrally.",
        "side": "A clean full-body side profile view, standing neutrally.",
        "portrait": "A head-and-shoulders portrait with a clear, readable face.",
    }[view]
    return (
        f"Character identity reference for {name}. Appearance: {profile}. Story context: {storyline}. "
        f"{framing} Preserve the same face, hairstyle, clothing, colors, and body proportions. "
        "Plain minimal background, no other people, no narrative action. "
        f"\n{build_people_control_clause([name], allow_multi_people=False)}"
    )


def build_item_initial_prompt(item: dict[str, Any], *, storyline: str) -> str:
    name = str(item.get("name") or item.get("asset_id") or "item").strip()
    description = str(item.get("description") or item.get("image_prompt") or name).strip()
    return (
        f"Initial isolated item reference image for {name}. Description: {description}. Story context: {storyline}. "
        "Make the item occupy the largest possible share of the frame with maximum subject occupancy, sharp silhouette, "
        "clear material details, minimal background, no hands, no people, no unrelated objects, and no extra scene elements."
    )


def build_location_view_prompts(location: dict[str, Any], *, storyline: str) -> dict[str, str]:
    name = str(location.get("name") or location.get("asset_id") or "location").strip()
    description = str(location.get("description") or location.get("image_prompt") or name).strip()
    base = (
        f"Initial world-space reference image for location {name}. Description: {description}. Story context: {storyline}. "
        "No characters, no people, no pedestrians, no crowds, no unrelated props beyond the described location. "
        "Preserve coherent film lighting and stable spatial/material identity."
    )
    return {
        "wide": base + " Create a wide establishing view showing the full spatial layout, entrances, exits, and navigable geometry.",
        "medium": base + " Create a medium view centered on the main actionable area where characters can stage interactions.",
        "close": base + " Create a close view of the most narratively useful surface, doorway, mechanism, or interaction region.",
        "extreme_close": base + " Create an extreme close detail view emphasizing texture, inscriptions, damage marks, material clues, or tactile detail.",
    }


def render_character_initial_assets(
    *,
    story: str,
    character_configs: list[Any],
    renderer: Any,
    run_dir: Path,
) -> dict[str, str]:
    portraits: dict[str, str] = {}
    for character in character_configs:
        name = str(_field(character, "name", "Character") or "Character").strip()
        asset_id = _safe_name(name, "Character")
        views: dict[str, dict[str, Any]] = {}
        for view in ("front_full", "side", "portrait"):
            output_path = run_dir / "narrative_artifacts" / "portraits" / f"{asset_id}_{view}.png"
            prompt = _character_view_prompt(character, storyline=story, view=view)
            result = renderer.render_world_asset(prompt=prompt, output_path=output_path)
            views[view] = {**result, "prompt": prompt, "view": view}
        portrait_path = str(views["portrait"].get("output_path", "")) if views["portrait"].get("ok") else ""
        if portrait_path:
            portraits[name] = portrait_path
        values = {"portrait_path": portrait_path, "views": views}
        if isinstance(character, dict):
            character.update(values)
        else:
            for key, value in values.items():
                try:
                    setattr(character, key, value)
                except Exception:
                    pass
    return portraits


def render_initial_assets(
    *,
    story: str,
    character_configs: list[Any],
    world_model: dict[str, Any],
    renderer: Any,
    run_dir: Path,
) -> dict[str, Any]:
    updated_world = dict(world_model or {})
    character_portraits = render_character_initial_assets(
        story=story,
        character_configs=character_configs,
        renderer=renderer,
        run_dir=run_dir,
    )

    locations: list[dict[str, Any]] = []
    for location in list(updated_world.get("locations", []) or []):
        if not isinstance(location, dict):
            continue
        row = dict(location)
        asset_id = str(row.get("asset_id") or _safe_name(row.get("name"), "location"))
        row["asset_id"] = asset_id
        prompts = build_location_view_prompts(row, storyline=story)
        views: dict[str, dict[str, Any]] = {}
        previous_path = ""
        for view in LOCATION_VIEW_NAMES:
            output_path = run_dir / "narrative_artifacts" / "world_model" / "locations" / asset_id / f"{view}_{renderer.tool_name}.png"
            refs = [previous_path] if previous_path else None
            result = renderer.render_world_asset(
                prompt=prompts[view], output_path=output_path, reference_image_paths=refs
            )
            views[view] = {**result, "prompt": prompts[view], "view": view}
            if result.get("ok"):
                previous_path = str(result.get("output_path", ""))
        row["views"] = views
        row["images"] = [views["wide"]]
        row["last_image_path"] = str(views["wide"].get("output_path", ""))
        locations.append(row)

    items: list[dict[str, Any]] = []
    for item in list(updated_world.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        asset_id = str(row.get("asset_id") or _safe_name(row.get("name"), "item"))
        row["asset_id"] = asset_id
        prompt = build_item_initial_prompt(row, storyline=story)
        output_path = run_dir / "narrative_artifacts" / "world_model" / "items" / asset_id / f"initial_{renderer.tool_name}.png"
        result = renderer.render_world_asset(prompt=prompt, output_path=output_path)
        row["initial_image_path"] = str(result.get("output_path", "")) if result.get("ok") else ""
        row["last_image_path"] = row["initial_image_path"]
        row["images"] = [result]
        row["views"] = {"isolated": {**result, "prompt": prompt, "view": "isolated"}}
        items.append(row)

    updated_world["locations"] = locations
    updated_world["items"] = items
    updated_world["character_portraits"] = character_portraits
    updated_world["characters"] = [dict(c) for c in character_configs if isinstance(c, dict)]
    manifest = {
        "story": story,
        "character_portraits": character_portraits,
        "characters": updated_world["characters"],
        "locations": locations,
        "items": items,
        "renderer_tool": getattr(renderer, "tool_name", ""),
    }
    manifest_path = run_dir / "narrative_artifacts" / "initial_assets" / "initial_assets.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"character_portraits": character_portraits, "world_model": updated_world, "manifest_path": str(manifest_path)}
