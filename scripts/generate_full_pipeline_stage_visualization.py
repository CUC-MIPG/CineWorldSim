#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def short(text: Any, limit: int = 360) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_path(path_value: Any, cwd: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        resolved = path
    else:
        resolved = (cwd / path).resolve()
    if resolved.exists():
        return resolved
    if resolved.suffix.lower() in IMAGE_EXTS:
        for suffix in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            candidate = resolved.with_suffix(suffix)
            if candidate.exists():
                return candidate
    return resolved


def rel_url(path_value: Any, output_html: Path, cwd: Path) -> str:
    path = normalize_path(path_value, cwd)
    if path is None:
        return ""
    try:
        return path.relative_to(output_html.parent.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path, output_html.parent.resolve()).replace(os.sep, "/")


def tag_list(items: list[Any], class_name: str = "tag") -> str:
    return "".join(f"<span class='{class_name}'>{html.escape(str(x))}</span>" for x in items if str(x))


def img_tag(path_value: Any, output_html: Path, cwd: Path, alt: str = "image") -> str:
    src = rel_url(path_value, output_html, cwd)
    if not src:
        return "<div class='media missing'>missing image</div>"
    return (
        "<figure class='media image-media'>"
        f"<img loading='lazy' src='{html.escape(src)}' alt='{html.escape(alt)}'>"
        f"<figcaption>{html.escape(Path(str(path_value)).name)}</figcaption>"
        "</figure>"
    )


def video_tag(path_value: Any, output_html: Path, cwd: Path, title: str = "video") -> str:
    src = rel_url(path_value, output_html, cwd)
    if not src:
        return "<div class='media missing'>missing video</div>"
    return (
        "<figure class='media video-media'>"
        f"<video controls preload='metadata' src='{html.escape(src)}'></video>"
        f"<figcaption>{html.escape(title)} · {html.escape(Path(str(path_value)).name)}</figcaption>"
        "</figure>"
    )


def first_ok_image(asset: dict[str, Any]) -> str:
    for item in as_list(asset.get("images")):
        if isinstance(item, dict) and item.get("ok") and item.get("output_path"):
            return str(item["output_path"])
    for item in as_list(asset.get("images")):
        if isinstance(item, dict) and item.get("output_path"):
            return str(item["output_path"])
    return ""


def view_image_path(view_payload: Any) -> str:
    if isinstance(view_payload, dict):
        return str(view_payload.get("last_image_path") or view_payload.get("output_path") or "")
    return str(view_payload or "")


def asset_initial_image(asset: dict[str, Any], backend_tool: str = "") -> str:
    if asset.get("initial_image_path"):
        return str(asset.get("initial_image_path") or "")
    views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
    if isinstance(views, dict):
        if "isolated" in views:
            path = view_image_path(views.get("isolated"))
            if path:
                return path
        for view in ["wide", "medium", "close", "extreme_close"]:
            path = view_image_path(views.get(view))
            if path:
                return path
    if backend_tool:
        for item in as_list(asset.get("images")):
            if isinstance(item, dict) and item.get("tool") == backend_tool and item.get("output_path"):
                return str(item["output_path"])
    return first_ok_image(asset)


def render_reference_selection(selection: Any) -> str:
    if not isinstance(selection, dict) or not selection:
        return ""
    return (
        "<details class='audit'><summary>Reference Selection</summary>"
        f"<pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre>"
        "</details>"
    )


def build_keyframe_map(event_keyframes: list[dict[str, Any]]) -> dict[tuple[int, str], list[str]]:
    result: dict[tuple[int, str], list[str]] = {}
    for kf in event_keyframes:
        if not isinstance(kf, dict) or not kf.get("ok", True):
            continue
        step = int(kf.get("step") or 0)
        role = str(kf.get("role_name") or "")
        path = str(kf.get("output_path") or "")
        if step and role and path:
            result.setdefault((step, role), []).append(path)
    return result


def render_stage_header(index: str, title: str, desc: str, section_id: str = "") -> str:
    id_attr = f" id='{html.escape(section_id)}'" if section_id else ""
    return (
        f"<section class='stage'{id_attr}>"
        "<div class='stage-title'>"
        f"<span class='stage-index'>{html.escape(index)}</span>"
        f"<div><h2>{html.escape(title)}</h2><p>{html.escape(desc)}</p></div>"
        "</div>"
    )


def render_story_stage(pipeline: dict[str, Any], output_html: Path, cwd: Path) -> str:
    chars = as_list(pipeline.get("characters"))
    configs = as_list(pipeline.get("character_configs"))
    portraits = pipeline.get("character_portraits") if isinstance(pipeline.get("character_portraits"), dict) else {}
    cards = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        cards.append(
            "<article class='card character-card'>"
            f"<h3>{html.escape(str(cfg.get('name', 'Character')))}</h3>"
            f"<p><b>Goal</b> {html.escape(short(cfg.get('final_goal'), 260))}</p>"
            f"<p><b>Profile</b> {html.escape(short(cfg.get('role_profile'), 320))}</p>"
            f"{tag_list(as_list(cfg.get('known_relations')), 'tag faint')}"
            "</article>"
        )
    portrait_cards = []
    for name, path in portraits.items():
        portrait_cards.append(
            "<article class='card portrait-card'>"
            f"<h3>{html.escape(str(name))}</h3>"
            f"<p class='eyebrow'>Initial Character Portraits</p>"
            + img_tag(path, output_html, cwd, f"{name} portrait")
            + "</article>"
        )
    return (
        render_stage_header(
            "00",
            "Story And Agents",
            "Input story, role decomposition, and character-agent configuration produced by OrchestratorAgent.",
            "stage-00",
        )
        + "<div class='hero-grid'>"
        + "<article class='story-panel'><h3>Storyline</h3>"
        + f"<p>{html.escape(str(pipeline.get('story', '')))}</p>"
        + f"<div class='tag-row'>{tag_list(chars)}</div>"
        + "</article>"
        + "<article class='metric-panel'>"
        + f"<div><b>Backend</b><span>{html.escape(str(pipeline.get('backend_mode', '')))}</span></div>"
        + f"<div><b>Image Tool</b><span>{html.escape(str(pipeline.get('backend_tool', '')))}</span></div>"
        + f"<div><b>Characters</b><span>{len(chars)}</span></div>"
        + f"<div><b>Dry Run</b><span>{html.escape(str(pipeline.get('dry_run', '')))}</span></div>"
        + "</article></div>"
        + "<div class='card-grid three'>"
        + "".join(cards)
        + "</div>"
        + ("<h3>Initial Character Portraits</h3><div class='card-grid assets'>" + "".join(portrait_cards) + "</div>" if portrait_cards else "")
        + "</section>"
    )


def render_world_model_stage(pipeline: dict[str, Any], output_html: Path, cwd: Path) -> str:
    world = pipeline.get("world_model") or {}
    backend_tool = str(pipeline.get("backend_tool") or "")
    cards = []
    for group, label in [("locations", "Location"), ("items", "Item")]:
        for asset in as_list(world.get(group)):
            if not isinstance(asset, dict):
                continue
            views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
            media_html = ""
            if group == "locations" and views:
                view_cards = []
                for view_name in ["wide", "medium", "close", "extreme_close"]:
                    if view_name not in views:
                        continue
                    view_cards.append(
                        "<div class='view-card'>"
                        f"<b>{html.escape(view_name)}</b>"
                        + img_tag(view_image_path(views.get(view_name)), output_html, cwd, f"{asset.get('name', 'location')} {view_name}")
                        + "</div>"
                    )
                media_html = "<h4>Location Views</h4><div class='view-grid'>" + "".join(view_cards) + "</div>"
            else:
                media_html = img_tag(asset_initial_image(asset, backend_tool), output_html, cwd, str(asset.get("name", "asset")))
            cards.append(
                "<article class='card asset-card'>"
                f"<div class='eyebrow'>{label}</div>"
                f"<h3>{html.escape(str(asset.get('name', asset.get('asset_id', 'Asset'))))}</h3>"
                + media_html
                + f"<p>{html.escape(short(asset.get('description'), 300))}</p>"
                + f"<small>{html.escape(short(asset.get('evidence'), 180))}</small>"
                + "</article>"
            )
    return (
        render_stage_header(
            "01A",
            "World Model Assets",
            "WorldModelAgent extracts spatial locations and important props, then renders initial visual assets.",
            "stage-01",
        )
        + f"<p class='summary'>{html.escape(short(world.get('summary'), 900))}</p>"
        + "<div class='card-grid assets'>"
        + "".join(cards)
        + "</div></section>"
    )


def render_simulation_stage(
    pipeline: dict[str, Any], step_logs: list[dict[str, Any]], output_html: Path, cwd: Path
) -> str:
    keyframes = build_keyframe_map(as_list(pipeline.get("event_keyframes")))
    rows = []
    for step in step_logs:
        if not isinstance(step, dict):
            continue
        step_no = int(step.get("step") or 0)
        role_cards = []
        for role in as_list(step.get("role_logs")):
            if not isinstance(role, dict):
                continue
            role_name = str(role.get("role_name") or "")
            action = role.get("next_action") or {}
            kf_html = "".join(
                img_tag(path, output_html, cwd, f"{role_name} step {step_no}") for path in keyframes.get((step_no, role_name), [])[:1]
            )
            role_cards.append(
                "<article class='role-event'>"
                f"<h4>{html.escape(role_name)}</h4>"
                + kf_html
                + "<dl>"
                + f"<dt>action</dt><dd>{html.escape(str(action.get('action_type') or action.get('action') or ''))}</dd>"
                + f"<dt>target</dt><dd>{html.escape(str(action.get('target_character') or 'none'))}</dd>"
                + f"<dt>location</dt><dd>{html.escape(str(action.get('location') or ''))}</dd>"
                + f"<dt>reason</dt><dd>{html.escape(short(action.get('reason'), 260))}</dd>"
                + "</dl>"
                + f"<p class='quote'>{html.escape(short(action.get('monologue'), 260))}</p>"
                + "</article>"
            )
        completion = step.get("storyline_completion") or {}
        rows.append(
            "<article class='step-card'>"
            f"<div class='step-head'><h3>Step {step_no:02d}</h3>"
            + f"<span>score {html.escape(str(completion.get('completion_score', '')))}</span>"
            + f"<span>{'complete' if completion.get('complete') else 'in progress'}</span></div>"
            + f"<p>{html.escape(short(step.get('sam_summary'), 520))}</p>"
            + f"<p class='gap'><b>Remaining gap</b> {html.escape(short(step.get('sam_remaining_gap_text'), 520))}</p>"
            + f"<p class='phase'><b>Phase</b> {html.escape(short(step.get('sam_story_phase'), 360))}</p>"
            + "<div class='role-grid'>"
            + "".join(role_cards)
            + "</div></article>"
        )
    return (
        render_stage_header(
            "01B",
            "Spatiotemporal Co-simulation",
            "StateAnalysisMachine runs synchronous character actions, updates narrative state, and records keyframes per role.",
            "stage-01b",
        )
        + "<div class='timeline'>"
        + "".join(rows)
        + "</div></section>"
    )


def world_state_path_for_step(run_dir: Path, step: int) -> Path:
    return run_dir / f"world_state_step{step}.json"


def render_world_evolution_stage(pipeline: dict[str, Any], output_html: Path, cwd: Path) -> str:
    states = as_list(pipeline.get("world_states"))
    rows = []
    for state in states:
        if not isinstance(state, dict):
            continue
        step = state.get("step")
        assets = []
        for group in ["locations", "items"]:
            for asset in as_list(state.get(group)):
                if not isinstance(asset, dict):
                    continue
                view_names = as_list(asset.get("changed_views_this_step"))
                if not view_names and asset.get("changed_view"):
                    view_names = [asset.get("changed_view")]
                views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
                if views:
                    view_html = "<div class='view-grid compact'>" + "".join(
                        "<div class='view-card'>"
                        f"<b>{html.escape(str(view_name))}</b>"
                        + img_tag(view_image_path(view_payload), output_html, cwd, f"{asset.get('name', 'asset')} {view_name}")
                        + "</div>"
                        for view_name, view_payload in views.items()
                    ) + "</div>"
                else:
                    view_html = img_tag(asset.get("last_image_path"), output_html, cwd, str(asset.get("name", "asset")))
                assets.append(
                    "<article class='world-mini'>"
                    f"<b>{html.escape(str(asset.get('name', asset.get('id', 'Asset'))))}</b>"
                    + f"<div class='tag-row'>{tag_list([f'changed view: {x}' for x in view_names], 'tag item')}</div>"
                    + view_html
                    + f"<small>{html.escape(short(asset.get('change_reason'), 120))}</small>"
                    + "</article>"
                )
        rows.append(
            "<details class='world-step'>"
            f"<summary>World State Step {html.escape(str(step))}: {html.escape(short(state.get('summary'), 180))}</summary>"
            + "<div class='world-mini-grid'>"
            + "".join(assets)
            + "</div></details>"
        )
    delta_details = (
        "<details class='world-step'><summary>World Delta Decisions</summary>"
        f"<pre>{html.escape(json.dumps(as_list(pipeline.get('world_deltas')), ensure_ascii=False, indent=2))}</pre>"
        "</details>"
    )
    return (
        render_stage_header(
            "01C",
            "World State Evolution",
            "WorldStateTracker keeps spatial assets evolving with LLM-gated physical-change decisions across time.",
            "stage-01c",
        )
        + "".join(rows)
        + delta_details
        + "</section>"
    )


def render_keyframe_reference_stage(pipeline: dict[str, Any], output_html: Path, cwd: Path) -> str:
    cards = []
    for kf in as_list(pipeline.get("event_keyframes")):
        if not isinstance(kf, dict):
            continue
        cards.append(
            "<article class='node-card'>"
            f"<div class='node-head'><h3>Step {html.escape(str(kf.get('step', '')))} · {html.escape(str(kf.get('role_name', '')))}</h3><span>keyframe</span></div>"
            + img_tag(kf.get("output_path"), output_html, cwd, str(kf.get("role_name", "keyframe")))
            + f"<p>{html.escape(short(kf.get('prompt'), 420))}</p>"
            + render_reference_selection(kf.get("reference_selection") or kf.get("event_context", {}).get("reference_selection"))
            + "</article>"
        )
    if not cards:
        return ""
    return (
        render_stage_header(
            "01D",
            "Event Keyframe References",
            "Audit of content-driven references used for event keyframe generation.",
            "stage-01d",
        )
        + "<div class='node-grid'>"
        + "".join(cards)
        + "</div></section>"
    )


def render_node_selector_stage(nodes_payload: dict[str, Any], output_html: Path, cwd: Path) -> str:
    config = nodes_payload.get("selector_config") or {}
    selected = as_list(nodes_payload.get("selected_nodes"))
    cards = []
    for node in selected:
        if not isinstance(node, dict):
            continue
        assets = node.get("visual_assets") or {}
        ref_paths = (
            as_list(assets.get("event_keyframes"))[:3]
            + as_list(assets.get("location_images"))[:1]
            + as_list(assets.get("item_images"))[:1]
        )
        refs = "".join(img_tag(path, output_html, cwd, str(node.get("node_id", "node"))) for path in ref_paths)
        score = node.get("score", node.get("completion_score", ""))
        cards.append(
            "<article class='node-card'>"
            f"<div class='node-head'><h3>{html.escape(str(node.get('node_id', 'NN')))} · {html.escape(str(node.get('event_id', 'EVT')))}</h3>"
            + f"<span>{html.escape(str(score))}</span></div>"
            + f"<p>{html.escape(short(node.get('event_summary'), 560))}</p>"
            + f"<div class='tag-row'>{tag_list(as_list(node.get('characters')))}{tag_list(as_list(node.get('locations')), 'tag location')}{tag_list(as_list(node.get('items')), 'tag item')}</div>"
            + f"<p class='phase'><b>Type</b> {html.escape(str(node.get('event_type', '')))} · <b>Phase</b> {html.escape(short(node.get('story_phase'), 260))}</p>"
            + f"<p class='gap'><b>Selection reason</b> {html.escape(short(node.get('selection_reason'), 360))}</p>"
            + "<div class='mini-media-strip'>"
            + refs
            + "</div></article>"
        )
    rejected_count = len(as_list(nodes_payload.get("rejected_nodes")))
    return (
        render_stage_header(
            "02",
            "Narrative Node Selection",
            "NarrativeNodeSelector scores event candidates and keeps the most useful narrative nodes for video planning.",
            "stage-02",
        )
        + "<div class='metrics'>"
        + f"<span>mode {html.escape(str(config.get('selection_mode', '')))}</span>"
        + f"<span>LLM {html.escape(str(config.get('llm_enabled', '')))}</span>"
        + f"<span>candidates {html.escape(str(nodes_payload.get('candidate_count', '')))}</span>"
        + f"<span>selected {html.escape(str(nodes_payload.get('selected_count', len(selected))))}</span>"
        + f"<span>rejected {rejected_count}</span>"
        + "</div><div class='node-grid'>"
        + "".join(cards)
        + "</div></section>"
    )


def render_path_stage(path_plan: dict[str, Any], nodes_payload: dict[str, Any]) -> str:
    nodes = {
        str(node.get("node_id")): node
        for node in as_list(nodes_payload.get("selected_nodes"))
        if isinstance(node, dict) and node.get("node_id")
    }
    order = as_list(path_plan.get("ordered_node_ids"))
    cards = []
    for idx, node_id in enumerate(order, start=1):
        node = nodes.get(str(node_id), {})
        cards.append(
            "<article class='path-node'>"
            f"<span class='rank'>{idx:02d}</span>"
            + f"<h3>{html.escape(str(node_id))}</h3>"
            + f"<p>{html.escape(short(node.get('event_summary'), 260))}</p>"
            + f"<small>{html.escape(', '.join(str(x) for x in as_list(node.get('source_steps'))))}</small>"
            + "</article>"
        )
    checks = path_plan.get("quality_checks") or {}
    return (
        render_stage_header(
            "03",
            "Path Orchestration",
            "PathOrchestratorAgent converts selected narrative nodes into an ordered event sequence for video generation.",
            "stage-03",
        )
        + "<div class='path-summary'>"
        + f"<p>{html.escape(short(path_plan.get('storyline_fit_summary'), 900))}</p>"
        + f"<div class='metrics'><span>mode {html.escape(str(path_plan.get('mode', '')))}</span><span>nodes {len(order)}</span><span>agent {html.escape(str(path_plan.get('agent', '')))}</span></div>"
        + "<details><summary>Quality checks</summary>"
        + f"<pre>{html.escape(json.dumps(checks, ensure_ascii=False, indent=2))}</pre>"
        + "</details></div>"
        + "<div class='path-rail'>"
        + "".join(cards)
        + "</div></section>"
    )


def render_video_stage(video_summary: dict[str, Any], video_tasks: list[dict[str, Any]], output_html: Path, cwd: Path) -> str:
    generated = {
        str(item.get("node_id")): item
        for item in as_list(video_summary.get("generated"))
        if isinstance(item, dict) and item.get("node_id")
    }
    cards = []
    for task in video_tasks:
        if not isinstance(task, dict):
            continue
        node_id = str(task.get("node_id", ""))
        task_type = str(task.get("task_type") or "event")
        reference_mode = str(task.get("reference_mode") or task.get("reference_selection", {}).get("mode") or "")
        duration = task.get("duration_sec", "")
        frames = task.get("num_frames", "")
        status = str((generated.get(node_id) or {}).get("status") or "unknown")
        ref_paths = as_list(task.get("reference_images")) or as_list(task.get("reference_image_paths"))
        refs = "".join(
            img_tag(path, output_html, cwd, f"{node_id} reference") for path in ref_paths[:4]
        )
        cards.append(
            "<article class='video-card'>"
            f"<div class='node-head'><h3>Rank {html.escape(str(task.get('rank', '')))} · {html.escape(node_id)}</h3><span>{html.escape(status)}</span></div>"
            + f"<p class='phase'><b>Task</b> {html.escape(task_type)} · <b>Mode</b> {html.escape(reference_mode or 'default')} · <b>duration {html.escape(str(duration))}s</b> · <b>frames {html.escape(str(frames))}</b></p>"
            + video_tag(task.get("output_video_path"), output_html, cwd, f"{node_id} event video")
            + f"<div class='tag-row'>{tag_list(as_list(task.get('characters')))}{tag_list(as_list(task.get('locations')), 'tag location')}{tag_list(as_list(task.get('items')), 'tag item')}</div>"
            + "<div class='mini-media-strip'>"
            + refs
            + "</div>"
            + render_reference_selection(task.get("reference_selection"))
            + "<details><summary>Shot Planning</summary>"
            + f"<pre>{html.escape(json.dumps(task.get('shot_planning') or task.get('event_shot_plan') or {}, ensure_ascii=False, indent=2))}</pre>"
            + "</details>"
            + f"<details><summary>MiniMax-H3 prompt</summary><pre>{html.escape(str(task.get('prompt', '')))}</pre></details>"
            + "</article>"
        )
    concat = video_summary.get("concat_output")
    return (
        render_stage_header(
            "04",
            "Event Video Generation",
            "VideoAgent uses MiniMax-H3 vLLM reference-video generation to render every orchestrated event, then concatenates the final film.",
            "stage-04",
        )
        + "<div class='final-video'>"
        + "<div><h3>Final Orchestrated Video</h3>"
        + f"<p>Status {html.escape(str(video_summary.get('status', '')))} · success {html.escape(str(video_summary.get('success', '')))} / total {html.escape(str(video_summary.get('total', '')))} · concat {html.escape(str(video_summary.get('concat_ok', '')))}</p></div>"
        + video_tag(concat, output_html, cwd, "final_orchestrated_events")
        + "</div><div class='video-grid'>"
        + "".join(cards)
        + "</div></section>"
    )


def collect_refs_from_html(output_html: Path) -> tuple[int, list[str]]:
    text = output_html.read_text(encoding="utf-8")
    refs = re.findall(r"(?:src)='([^']+)'", text)
    missing = []
    for ref in refs:
        if ref.startswith(("http://", "https://", "data:")):
            continue
        path = (output_html.parent / ref).resolve()
        if not path.exists():
            missing.append(ref)
    return len(refs), missing


def generate_html(run_dir: Path, video_dir: Path, output_html: Path, cwd: Path) -> Path:
    pipeline = load_json(run_dir / "pipeline_result.json", {})
    step_logs = load_json(run_dir / "step_logs.json", pipeline.get("step_logs") or [])
    nodes_payload = load_json(run_dir / "path_orchestration" / "narrative_nodes.json", {})
    path_plan = load_json(
        run_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json", {}
    )
    video_summary = load_json(video_dir / "event_video_summary.json", {})
    video_tasks = load_json(video_dir / "event_video_tasks.json", [])

    output_html.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            render_story_stage(pipeline, output_html, cwd),
            render_world_model_stage(pipeline, output_html, cwd),
            render_simulation_stage(pipeline, step_logs, output_html, cwd),
            render_world_evolution_stage(pipeline, output_html, cwd),
            render_keyframe_reference_stage(pipeline, output_html, cwd),
            render_node_selector_stage(nodes_payload, output_html, cwd),
            render_path_stage(path_plan, nodes_payload),
            render_video_stage(video_summary, video_tasks, output_html, cwd),
        ]
    )
    output_html.write_text(render_document(body, run_dir, video_dir), encoding="utf-8")
    return output_html


def render_document(body: str, run_dir: Path, video_dir: Path) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>CineWorldSim Full Pipeline Visualization</title>"
        "<style>"
        ":root{--ink:#111812;--muted:#647067;--paper:#f4efe4;--panel:#fffaf0;--line:#d9cab1;--accent:#9b3f1f;--accent2:#1f6f63;--gold:#c5973e;--shadow:0 16px 48px rgba(33,24,12,.14)}"
        "*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0%,#fff6dc 0,#f4efe4 28%,#e7ddcb 100%);color:var(--ink);font-family:Palatino,Georgia,'Times New Roman',serif}"
        "body:before{content:'';position:fixed;inset:0;pointer-events:none;background:linear-gradient(90deg,rgba(70,45,20,.045) 1px,transparent 1px),linear-gradient(rgba(70,45,20,.035) 1px,transparent 1px);background-size:34px 34px;mix-blend-mode:multiply}"
        "main{width:min(1680px,calc(100vw - 36px));margin:28px auto 80px}.top{padding:30px;border:1px solid var(--line);background:rgba(255,250,240,.82);box-shadow:var(--shadow);border-radius:26px;position:relative;overflow:hidden}"
        ".top:after{content:'';position:absolute;width:360px;height:360px;right:-120px;top:-180px;border-radius:50%;background:rgba(155,63,31,.14)}"
        "h1{font-size:clamp(36px,5vw,76px);line-height:.95;margin:0 0 16px;letter-spacing:-.04em}.top p{max-width:900px;font-size:18px;line-height:1.6;color:#384139}.source{font-size:13px;color:var(--muted);word-break:break-all}"
        ".nav{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 0}.nav a,.tag,.metrics span{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:#fff7e8;color:#342719;text-decoration:none;font-size:13px}.tag.location{background:#e9f4ef}.tag.item{background:#fff0ce}.tag.faint{opacity:.75}"
        ".stage{margin:30px 0;padding:24px;border:1px solid var(--line);background:rgba(255,250,240,.9);border-radius:24px;box-shadow:0 12px 32px rgba(33,24,12,.08)}.stage-title{display:flex;gap:16px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:20px}.stage-index{font-family:monospace;background:var(--ink);color:#fff;padding:8px 10px;border-radius:12px}.stage h2{font-size:32px;margin:0 0 6px}.stage-title p,.summary{margin:0;color:var(--muted);line-height:1.55}"
        ".hero-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,.8fr);gap:18px}.story-panel,.metric-panel,.card,.step-card,.node-card,.path-summary,.video-card,.final-video{border:1px solid var(--line);background:var(--panel);border-radius:18px;padding:16px}.story-panel p{font-size:20px;line-height:1.65}.metric-panel{display:grid;gap:12px}.metric-panel div{display:flex;justify-content:space-between;gap:16px;border-bottom:1px dotted var(--line);padding-bottom:10px}.metric-panel span{color:var(--accent2);font-weight:bold}"
        ".card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.card-grid.assets{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.card h3,.step-card h3,.node-card h3,.video-card h3,.path-node h3{margin:0 0 10px}.card p,.step-card p,.node-card p,.video-card p,.path-node p{line-height:1.55;color:#374038}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}"
        ".media{margin:0 0 10px}.media img,.media video{width:100%;display:block;border-radius:14px;border:1px solid #cfc0a7;background:#21180c;aspect-ratio:16/9;object-fit:cover}.media video{object-fit:contain}.media figcaption{font-size:11px;color:var(--muted);margin-top:6px;word-break:break-all}.missing{min-height:160px;display:grid;place-items:center;color:#8b2d1d;background:#f3ddcf;border:1px solid #e0b9a7;border-radius:14px}"
        ".timeline{display:grid;gap:18px}.step-head,.node-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.step-head span,.node-head span{background:#efe0c4;border-radius:999px;padding:6px 10px;color:#5a4321;font-size:13px}.role-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:12px}.role-event{border:1px solid #dfd0b6;border-radius:16px;padding:12px;background:#fffdf7}.role-event h4{margin:0 0 8px;font-size:20px}dl{display:grid;grid-template-columns:74px 1fr;gap:6px;margin:8px 0}dt{color:var(--accent);font-weight:bold}dd{margin:0;color:#303830}.quote{font-style:italic;color:#5c665f}.gap{border-left:4px solid var(--gold);padding-left:10px}.phase{color:#46504a}"
        ".world-step{border:1px solid var(--line);border-radius:16px;padding:12px;background:#fffaf0;margin:12px 0}.world-step summary{cursor:pointer;font-weight:bold}.world-mini-grid,.mini-media-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:12px}.world-mini{border:1px solid #dfd0b6;border-radius:14px;padding:10px;background:#fffdf7}.mini-media-strip .media img{border-radius:12px}.node-grid,.video-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px}.metrics{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}.path-rail{display:flex;gap:14px;overflow-x:auto;padding:8px 2px 18px}.path-node{min-width:280px;position:relative;border:1px solid var(--line);background:#fffdf7;border-radius:18px;padding:14px}.path-node:after{content:'>';position:absolute;right:-13px;top:42%;color:var(--accent);font-weight:bold}.path-node:last-child:after{content:''}.rank{display:inline-grid;place-items:center;width:36px;height:36px;border-radius:50%;background:var(--accent2);color:#fff;font-family:monospace;margin-bottom:10px}"
        ".final-video{display:grid;grid-template-columns:minmax(280px,.7fr) minmax(420px,1.3fr);gap:18px;align-items:center;margin-bottom:18px;background:#111812;color:#fff}.final-video p{color:#d9cab1}.final-video .media figcaption{color:#d9cab1}.video-card details,.path-summary details{margin-top:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow:auto;background:#1b201c;color:#f7ecd7;border-radius:14px;padding:14px;font-size:12px;line-height:1.45;max-height:420px}"
        "@media(max-width:820px){main{width:min(100vw - 20px,1680px)}.hero-grid,.final-video{grid-template-columns:1fr}.node-grid,.video-grid{grid-template-columns:1fr}.stage{padding:16px}.role-grid{grid-template-columns:1fr}h1{font-size:40px}}"
        "</style></head><body><main>"
        "<header class='top'><h1>CineWorldSim Full Pipeline</h1>"
        "<p>A single-page audit view for the completed story-to-world-to-events-to-video pipeline. It keeps every major artifact visible: agent setup, world assets, narrative simulation, state evolution, selected narrative nodes, path orchestration, per-event videos, references, and the final concatenated film.</p>"
        f"<p class='source'><b>Run dir</b> {html.escape(str(run_dir))}<br><b>Video dir</b> {html.escape(str(video_dir))}</p>"
        "<nav class='nav'><a href='#'>Top</a><a href='#stage-01'>World</a><a href='#stage-02'>Nodes</a><a href='#stage-03'>Path</a><a href='#stage-04'>Videos</a></nav>"
        "</header>"
        + body
        + "</main></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a full-stage CineWorldSim pipeline visualization HTML.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/current_full_built_pipeline_demo_dual/all_banana"),
        help="Completed pipeline run directory.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(
            "outputs/current_full_built_pipeline_demo_dual/all_banana/event_video_pipeline_minimax_h3_vllm_all_events"
        ),
        help="Completed event video output directory.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path(
            "outputs/current_full_built_pipeline_demo_dual/all_banana/visualizations/full_stage_pipeline_visualization.html"
        ),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    run_dir = normalize_path(args.run_dir, cwd)
    video_dir = normalize_path(args.video_dir, cwd)
    output_html = normalize_path(args.output_html, cwd)
    if run_dir is None or video_dir is None or output_html is None:
        raise SystemExit("invalid path")

    output = generate_html(run_dir, video_dir, output_html, cwd)
    ref_count, missing = collect_refs_from_html(output)
    print(output)
    print(f"media_refs={ref_count}")
    print(f"missing_refs={len(missing)}")
    for item in missing[:50]:
        print(f"missing: {item}")


if __name__ == "__main__":
    main()
