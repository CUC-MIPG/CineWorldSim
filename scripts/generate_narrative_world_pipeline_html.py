#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "pipeline_result.json").read_text(encoding="utf-8"))


def _rel(path_value: str, output_html: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        return path.relative_to(output_html.parent.resolve()).as_posix()
    except Exception:
        return os.path.relpath(path, output_html.parent.resolve()).replace(os.sep, "/")


def _image_tag(path_value: str, output_html: Path, alt: str) -> str:
    rel = _rel(path_value, output_html)
    if not rel:
        return "<div class='missing'>missing</div>"
    return f'<img src="{html.escape(rel)}" alt="{html.escape(alt)}">'


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _view_path(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("last_image_path") or value.get("output_path") or "")
    return str(value or "")


def _initial_asset_path(asset: dict[str, Any]) -> str:
    if asset.get("initial_image_path"):
        return str(asset.get("initial_image_path") or "")
    views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
    for key in ["isolated", "wide", "medium", "close", "extreme_close"]:
        path = _view_path(views.get(key))
        if path:
            return path
    if asset.get("last_image_path"):
        return str(asset.get("last_image_path") or "")
    for row in _as_list(asset.get("images")):
        if isinstance(row, dict) and row.get("output_path"):
            return str(row.get("output_path") or "")
    return ""


def _render_reference_selection(selection: Any) -> str:
    if not isinstance(selection, dict) or not selection:
        return ""
    return (
        "<details><summary>Reference Selection</summary>"
        f"<pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre>"
        "</details>"
    )


def _render_initial_assets(payload: dict[str, Any], output_html: Path) -> str:
    portrait_cards = []
    portraits = payload.get("character_portraits") if isinstance(payload.get("character_portraits"), dict) else {}
    for name, path in portraits.items():
        portrait_cards.append(
            "<article class='thumb'>"
            f"<h4>{html.escape(str(name))}</h4>"
            f"{_image_tag(str(path), output_html, str(name))}"
            "</article>"
        )
    world_model = payload.get("world_model") if isinstance(payload.get("world_model"), dict) else {}
    asset_cards = []
    for loc in _as_list(world_model.get("locations") if isinstance(world_model, dict) else []):
        if not isinstance(loc, dict):
            continue
        views = loc.get("views") if isinstance(loc.get("views"), dict) else {}
        view_html = "".join(
            "<article class='thumb small'>"
            f"<h4>{html.escape(str(loc.get('name') or loc.get('asset_id') or 'location'))} / {html.escape(str(view))}</h4>"
            f"{_image_tag(_view_path(view_payload), output_html, str(view))}"
            "</article>"
            for view, view_payload in views.items()
        )
        asset_cards.append(view_html)
    for item in _as_list(world_model.get("items") if isinstance(world_model, dict) else []):
        if not isinstance(item, dict):
            continue
        asset_cards.append(
            "<article class='thumb small'>"
            f"<h4>{html.escape(str(item.get('name') or item.get('asset_id') or 'item'))}</h4>"
            f"{_image_tag(_initial_asset_path(item), output_html, str(item.get('name') or 'item'))}"
            "</article>"
        )
    blocks = []
    if portrait_cards:
        blocks.append("<h3>Initial Character Portraits</h3><div class='thumb-grid'>" + "".join(portrait_cards) + "</div>")
    if asset_cards:
        blocks.append("<h3>Location Views And Initial Items</h3><div class='thumb-grid'>" + "".join(asset_cards) + "</div>")
    return "".join(blocks)


def _render_keyframes(payload: dict[str, Any], output_html: Path) -> str:
    cards = []
    for kf in payload.get("event_keyframes", []) or []:
        cards.append(
            "<article class='thumb'>"
            f"<h4>Step {html.escape(str(kf.get('step', '')))} / {html.escape(str(kf.get('role_name', '')))}</h4>"
            f"{_image_tag(str(kf.get('output_path', '')), output_html, str(kf.get('role_name', 'keyframe')))}"
            f"<p>{html.escape(str(kf.get('prompt', ''))[:220])}</p>"
            f"{_render_reference_selection(kf.get('reference_selection'))}"
            "</article>"
        )
    return "".join(cards) or "<p class='muted'>No event keyframes recorded.</p>"


def _render_world_state(state: dict[str, Any], output_html: Path) -> str:
    rows = []
    for group_name in ["locations", "items"]:
        for asset in state.get(group_name, []) or []:
            changed = "changed" if asset.get("changed_this_step") else "stable"
            views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
            if views:
                media = "<div class='view-grid'>" + "".join(
                    "<div>"
                    f"<div class='tag'>{html.escape(str(view_name))}</div>"
                    f"{_image_tag(_view_path(view_payload), output_html, str(view_name))}"
                    "</div>"
                    for view_name, view_payload in views.items()
                ) + "</div>"
            else:
                media = _image_tag(str(asset.get("last_image_path", "")), output_html, str(asset.get("name", "asset")))
            rows.append(
                "<article class='world-asset'>"
                f"<div class='tag'>{html.escape(group_name[:-1])} / {html.escape(changed)}</div>"
                f"<h4>{html.escape(str(asset.get('name', '') or asset.get('id', '')))}</h4>"
                f"<small>changed views: {html.escape(', '.join(str(x) for x in _as_list(asset.get('changed_views_this_step'))) or 'none')}</small>"
                f"{media}"
                f"<p>{html.escape(str(asset.get('current_state_text', '')))}</p>"
                f"<small>{html.escape(str(asset.get('change_reason', '')))}</small>"
                "</article>"
            )
    return (
        f"<section class='state'><h3>World Step {html.escape(str(state.get('step', '')))}</h3>"
        f"<p>{html.escape(str(state.get('summary', '')))}</p>"
        "<div class='world-grid'>"
        + "".join(rows)
        + "</div></section>"
    )


def _render_run(payload: dict[str, Any], output_html: Path) -> str:
    backend = html.escape(str(payload.get("backend_mode", "")))
    story = html.escape(str(payload.get("story", "")))
    characters = ", ".join(str(x) for x in payload.get("characters", []) or [])
    step_rows = []
    for step in payload.get("step_logs", []) or []:
        role_bits = []
        for role in step.get("role_logs", []) or []:
            action = role.get("next_action", {}) or {}
            role_bits.append(
                "<li>"
                f"<strong>{html.escape(str(role.get('role_name', '')))}</strong>: "
                f"{html.escape(str(action.get('action') or action.get('action_type') or role.get('role_output_text') or ''))}"
                "</li>"
            )
        step_rows.append(
            "<article class='row'>"
            f"<h3>Step {html.escape(str(step.get('step', '')))}</h3>"
            f"<p>{html.escape(str(step.get('sam_summary', '') or ''))}</p>"
            f"<ul>{''.join(role_bits)}</ul>"
            "</article>"
        )
    state_rows = "".join(_render_world_state(state, output_html) for state in payload.get("world_states", []) or [])
    deltas = payload.get("world_deltas", []) or []
    delta_text = html.escape(json.dumps(deltas, ensure_ascii=False, indent=2))
    return (
        f"<section class='run'><div class='run-head'><h2>{backend}</h2><span>{html.escape(characters)}</span></div>"
        f"<p class='story'>{story}</p>"
        + _render_initial_assets(payload, output_html)
        + "<h3>Narrative Time</h3>"
        + "".join(step_rows)
        + "<h3>Event Keyframes</h3><div class='thumb-grid'>"
        + _render_keyframes(payload, output_html)
        + "</div><h3>World Space</h3>"
        + state_rows
        + f"<details><summary>World Delta Decisions</summary><pre>{delta_text}</pre></details>"
        + "</section>"
    )


def generate_pipeline_html(*, run_dirs: list[Path], output_html: Path) -> Path:
    runs = [_load_run(path) for path in run_dirs]
    body = "".join(_render_run(payload, output_html) for payload in runs)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Narrative World Pipeline</title>"
        "<style>"
        "body{font-family:Georgia,'Times New Roman',serif;margin:0;background:#f5f6f1;color:#1f2421}"
        "main{width:min(1440px,calc(100vw - 32px));margin:28px auto 64px}"
        "h1{font-size:42px;margin:0 0 18px}h2{font-size:30px;margin:0}h3{margin:24px 0 10px}"
        ".run{border:1px solid #ccd2c8;padding:22px;margin:18px 0;background:#fff;border-radius:8px}"
        ".run-head{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #dde2d8;padding-bottom:12px}"
        ".story{line-height:1.65}.row{border-top:1px solid #e1e4dd;padding:12px 0}.row ul{margin:8px 0 0}"
        ".world-grid,.thumb-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}"
        ".view-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:8px}"
        ".world-asset,.thumb{border:1px solid #d7dccf;padding:12px;border-radius:8px;background:#fbfcf8}"
        ".thumb.small h4{font-size:14px}"
        ".tag{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#52695b}"
        "img{width:100%;display:block;border-radius:6px;border:1px solid #d7dccf;background:#eef1ea}"
        ".missing{min-height:120px;display:grid;place-items:center;background:#eef1ea;color:#8a4d2b;border-radius:6px}"
        ".muted{color:#6d766e}pre{white-space:pre-wrap;background:#f0f2ec;padding:12px;border-radius:6px;overflow:auto}"
        "</style></head><body><main><h1>Backend Comparison</h1>"
        + body
        + "</main></body></html>",
        encoding="utf-8",
    )
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate timeline and world-space HTML for narrative world runs.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    print(generate_pipeline_html(run_dirs=args.run_dir, output_html=args.output_html))


if __name__ == "__main__":
    main()
