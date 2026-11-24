#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from narrative.event_keyframe_renderer import EventKeyframeRenderer
from narrative.asset_reference_selector import select_event_reference_images
from narrative.initial_asset_renderer import (
    build_item_initial_prompt,
    build_location_view_prompts,
    render_character_initial_assets,
)
from narrative.pipeline_schemas import LOCATION_VIEW_NAMES, normalize_location_view
from narrative.simulation_scaffold import OrchestratorAgent, StateAnalysisMachine, WorldModelAgent, _safe_name
from narrative.story_control import StoryController
from narrative.visual_prompt_constraints import build_people_control_clause
from narrative.world_delta_judge import WorldDeltaJudge
from narrative.world_renderer import WorldRenderer, backend_mode_to_tool
from narrative.world_state_tracker import (
    build_initial_world_state,
    update_world_state_from_delta,
    write_world_state,
)
from scripts.generate_narrative_world_pipeline_html import generate_pipeline_html

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _read_story(args: argparse.Namespace) -> str:
    if args.story_file:
        return Path(args.story_file).read_text(encoding="utf-8").strip()
    if args.storyline:
        return str(args.storyline).strip()
    return (
        "On a storm island, a journalist enters an abandoned lighthouse, discovers a cracked compass, "
        "and forces the hidden witness to reveal what happened in the flooded archive."
    )


def _parse_characters(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _serialize_character_config(cfg: Any) -> dict[str, Any]:
    model_cfg = getattr(cfg, "model_config", None)
    return {
        "name": getattr(cfg, "name", ""),
        "final_goal": getattr(cfg, "final_goal", ""),
        "role_profile": getattr(cfg, "role_profile", ""),
        "known_relations": list(getattr(cfg, "known_relations", []) or []),
        "current_environment": getattr(cfg, "current_environment", ""),
        "prompt_arm": getattr(cfg, "prompt_arm", ""),
        "model_name": getattr(model_cfg, "model_name", "") if model_cfg else "",
        "base_url": getattr(model_cfg, "base_url", "") if model_cfg else "",
        "llm_enabled": bool(getattr(model_cfg, "llm_enabled", False)) if model_cfg else False,
        "system_prompt_path": getattr(model_cfg, "system_prompt_path", "") if model_cfg else "",
        "dynamic_system_prompt": getattr(cfg, "dynamic_system_prompt", ""),
        "portrait_path": getattr(cfg, "portrait_path", ""),
        "views": getattr(cfg, "views", {}),
    }


def _asset_prompt(asset: dict[str, Any], state_text: str = "") -> str:
    name = str(asset.get("name") or asset.get("id") or "world asset")
    description = str(asset.get("description") or "")
    current = str(state_text or asset.get("current_state_text") or description)
    base_prompt = str(asset.get("image_prompt") or description or name).strip()
    return (
        f"{base_prompt}. "
        f"Cinematic 16:9 world asset update for {name}. "
        f"Base description: {description}. Current state: {current}. "
        "Describe only the updated physical world state of this asset. "
        "No characters, no people, no pedestrians, no crowds, no unrelated objects, and no extra scene elements. "
        "Wide horizontal frame, clear spatial/object readability, coherent film lighting."
        f"\n{build_people_control_clause([], allow_multi_people=False)}"
    )


def _render_blueprint_records(
    *,
    records: list[dict[str, Any]],
    group_name: str,
    renderer: WorldRenderer,
    run_dir: Path,
    story: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        asset_id = _safe_name(row.get("name", group_name))
        row["asset_id"] = asset_id
        out_dir = run_dir / "narrative_artifacts" / "world_model" / group_name / asset_id
        if group_name == "locations":
            prompts = build_location_view_prompts(row, storyline=story)
            views: dict[str, dict[str, Any]] = {}
            for view in LOCATION_VIEW_NAMES:
                out_path = out_dir / f"{view}_{renderer.tool_name}.png"
                result = renderer.render_world_asset(
                    prompt=prompts[view],
                    output_path=out_path,
                )
                views[view] = {**result, "view": view, "prompt": prompts[view]}
            row["views"] = views
            row["images"] = [views["wide"]]
            row["last_image_path"] = str(views["wide"].get("output_path", ""))
            ok = all(bool(views[v].get("ok")) for v in LOCATION_VIEW_NAMES)
        else:
            out_path = out_dir / f"initial_{renderer.tool_name}.png"
            prompt = build_item_initial_prompt(row, storyline=story)
            result = renderer.render_world_asset(
                prompt=prompt,
                output_path=out_path,
            )
            row["initial_image_path"] = str(result.get("output_path", "")) if result.get("ok") else ""
            row["last_image_path"] = row["initial_image_path"]
            row["images"] = [result]
            row["views"] = {"isolated": {**result, "view": "isolated", "prompt": prompt}}
            ok = bool(result.get("ok"))
        rows.append(row)
        LOGGER.info(
            "pipeline world asset rendered backend=%s group=%s asset=%s ok=%s",
            renderer.backend_mode,
            group_name,
            asset_id,
            ok,
        )
    return rows


def _dry_world_blueprint(story: str) -> dict[str, Any]:
    return {
        "summary": f"Dry-run world extracted from story: {story[:160]}",
        "locations": [
            {
                "name": "Abandoned Lighthouse",
                "description": "A storm-battered lighthouse with a locked stairwell and a hidden archive entrance.",
                "evidence": "lighthouse",
                "mentions_estimate": 1,
                "image_prompt": "Cinematic 16:9 abandoned lighthouse on a storm island.",
            }
        ],
        "items": [
            {
                "name": "Cracked Compass",
                "description": "A cracked brass compass used as the main clue.",
                "evidence": "compass",
                "mentions_estimate": 1,
                "image_prompt": "Cinematic 16:9 cracked brass compass evidence object.",
            }
        ],
    }


def _current_world_reference_images(state: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for group in ["locations", "items"]:
        for row in state.get(group, []) or []:
            if not isinstance(row, dict):
                continue
            path = str(row.get("last_image_path", "") or "").strip()
            if path:
                paths.append(path)
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _items_from_step_text(state: dict[str, Any], text: str) -> list[str]:
    lower = str(text or "").lower()
    items: list[str] = []
    for row in state.get("items", []) or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("id") or "").strip()
        rid = str(row.get("id") or "").replace("_", " ").strip()
        if (name and name.lower() in lower) or (rid and rid.lower() in lower):
            items.append(name or rid)
    return items


def _event_context_for_role(
    *,
    role_log: dict[str, Any],
    step_log: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    action = role_log.get("next_action", {}) if isinstance(role_log.get("next_action"), dict) else {}
    role_name = str(role_log.get("role_name", "") or "").strip()
    location = str(action.get("location") or "").strip()
    text = " ".join(
        str(x or "")
        for x in [
            role_log.get("story_description", ""),
            role_log.get("role_output_text", ""),
            action.get("action", ""),
            action.get("action_type", ""),
            action.get("reason", ""),
            action.get("monologue", ""),
            step_log.get("sam_summary", ""),
        ]
    )
    story_description = str(role_log.get("story_description") or role_log.get("role_output_text") or step_log.get("sam_summary") or "").strip()
    return {
        "characters": [role_name] if role_name else [],
        "locations": [location] if location else [],
        "items": _items_from_step_text(current_state, text),
        "story_description": story_description,
        "video_description": str(role_log.get("role_output_text") or ""),
        "event_summary": str(step_log.get("sam_summary") or ""),
        "story_phase": str(step_log.get("sam_story_phase") or role_log.get("story_phase_text") or ""),
    }


def _build_world_model(
    *,
    story: str,
    renderer: WorldRenderer,
    run_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    blueprint = _dry_world_blueprint(story) if dry_run else WorldModelAgent().extract_world_blueprint(story)
    locations = _render_blueprint_records(
        records=list(blueprint.get("locations", []) or []),
        group_name="locations",
        renderer=renderer,
        run_dir=run_dir,
        story=story,
    )
    items = _render_blueprint_records(
        records=list(blueprint.get("items", []) or []),
        group_name="items",
        renderer=renderer,
        run_dir=run_dir,
        story=story,
    )
    bundle = {
        "summary": str(blueprint.get("summary", "")),
        "story_prompt": story,
        "image_tools": [renderer.tool_name],
        "locations": locations,
        "items": items,
        "output_dir": str(run_dir / "narrative_artifacts" / "world_model"),
    }
    bundle["context_text"] = WorldModelAgent.build_context_text(bundle)
    return bundle


def _dry_step_logs(story: str, characters: list[str], max_steps: int) -> list[dict[str, Any]]:
    rows = []
    names = characters or ["Journalist", "HiddenWitness"]
    for step in range(1, max_steps + 1):
        role_logs = []
        for name in names:
            role_logs.append(
                {
                    "role_name": name,
                    "environment_description": {
                        "world_summary": story,
                        "current_environment": story,
                        "story_anchor": f"Dry step {step}",
                        "public_events": [],
                    },
                    "role_output_text": f"{name} advances the investigation at step {step}.",
                    "action_keyframe_path": "",
                    "portrait_path": "",
                    "next_action": {
                        "action": "opens the lighthouse door and takes the compass" if step == 1 else "studies the new evidence",
                        "action_type": "investigate",
                        "target_character": "",
                        "reason": "to expose the hidden truth",
                        "monologue": "This clue changes the shape of the case.",
                        "location": "Abandoned Lighthouse",
                        "confidence": 0.8,
                    },
                    "sam_environment_update_text": story,
                    "gap_analysis_text": "",
                    "person_progress_analysis_text": "",
                    "event_summary_analysis_text": "",
                    "story_phase_text": "development",
                    "remaining_gap_text": "",
                }
            )
        rows.append(
            {
                "step": step,
                "role_logs": role_logs,
                "sam_story_anchor": f"Dry step {step}",
                "sam_summary": "The lighthouse door is opened and the compass is taken."
                if step == 1
                else "The characters use the new clue to push the story forward.",
                "sam_current_environment": story,
                "sam_story_phase": "development",
                "sam_remaining_gap_text": "",
                "storyline_completion": {"is_story_complete": step >= max_steps, "completion_score": min(1.0, step / max_steps)},
            }
        )
    return rows


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _character_configs_from_checkpoint(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in checkpoint.get("agents", []) or []:
        if not isinstance(item, dict):
            continue
        state = item.get("state", {}) if isinstance(item.get("state"), dict) else {}
        model_cfg = item.get("model_config", {}) if isinstance(item.get("model_config"), dict) else {}
        rows.append(
            {
                "name": str(state.get("name", "")),
                "final_goal": str(state.get("final_goal", "")),
                "role_profile": str(state.get("role_profile", "")),
                "known_relations": [str(x) for x in state.get("known_relations", []) or []],
                "current_environment": str(state.get("current_environment", "")),
                "prompt_arm": "",
                "model_name": str(model_cfg.get("model_name", "")),
                "base_url": str(model_cfg.get("base_url", "")),
                "llm_enabled": bool(model_cfg.get("llm_enabled", False)),
                "system_prompt_path": str(model_cfg.get("system_prompt_path", "")),
                "dynamic_system_prompt": "",
                "portrait_path": str(item.get("portrait_path", "")),
            }
        )
    return rows


def _merge_step_logs(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for item in [*existing, *new]:
        if not isinstance(item, dict):
            continue
        step = int(item.get("step", 0) or 0)
        if step > 0:
            by_step[step] = item
    return [by_step[step] for step in sorted(by_step)]


def _real_step_logs(
    story: str,
    characters: list[str],
    world_model: dict[str, Any],
    max_steps: int,
    *,
    stop_on_complete: bool = True,
    checkpoint_path: Path | None = None,
    renderer: WorldRenderer | None = None,
    run_dir: Path | None = None,
    story_control_enabled: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = _read_json_if_exists(checkpoint_path)
    checkpoint_step_logs = [
        item for item in checkpoint.get("step_logs", []) or [] if isinstance(item, dict)
    ]
    completed_steps = int(checkpoint.get("completed_steps", 0) or 0)
    if story_control_enabled and checkpoint_step_logs:
        saved_control = checkpoint.get("story_control")
        if (
            not isinstance(saved_control, dict)
            or int(saved_control.get("version", 0) or 0) != StoryController.VERSION
        ):
            raise RuntimeError(
                "This Track B checkpoint predates the current Story Control evidence rules and cannot be reused safely. "
                "Start the revised pipeline with a new --output-root (recommended), or explicitly run "
                "--no-track-b-story-control for legacy behavior."
            )
    if checkpoint_step_logs and completed_steps >= max_steps:
        checkpoint_configs = _character_configs_from_checkpoint(checkpoint)
        if renderer is not None and run_dir is not None:
            portraits = render_character_initial_assets(
                story=story,
                character_configs=checkpoint_configs,
                renderer=renderer,
                run_dir=run_dir,
            )
            world_model["character_portraits"] = portraits
            world_model["characters"] = checkpoint_configs
        LOGGER.info("narrative resume status=complete checkpoint=%s steps=%d", checkpoint_path, len(checkpoint_step_logs))
        return checkpoint_step_logs, checkpoint_configs

    orchestrator = OrchestratorAgent()
    plan = orchestrator.understand_input(story, characters)
    plan["character_configs"] = WorldModelAgent.enrich_character_configs(plan["character_configs"], world_model)
    if renderer is not None and run_dir is not None:
        portraits = render_character_initial_assets(
            story=story,
            character_configs=plan["character_configs"],
            renderer=renderer,
            run_dir=run_dir,
        )
        world_model["character_portraits"] = portraits
        world_model["characters"] = plan["character_configs"]
    modeled_prompt = WorldModelAgent.enrich_story_prompt(story, world_model)
    agents = orchestrator.build_character_agents(plan["character_configs"])
    sam = StateAnalysisMachine()
    # No output_dir here: event keyframes are generated by this pipeline's selected backend.
    result = sam.run_narrative_deduction(
        prompt=modeled_prompt,
        agents=agents,
        max_steps=max_steps,
        output_dir=None,
        checkpoint_path=checkpoint_path,
        resume_state=checkpoint or None,
        stop_on_complete=stop_on_complete,
        story_control_enabled=story_control_enabled,
    )
    if story_control_enabled and run_dir is not None:
        _write_json(run_dir / "story_control.json", result.get("story_control", {}))
    step_logs = _merge_step_logs(checkpoint_step_logs, list(result.get("step_logs", []) or []))
    return step_logs, [_serialize_character_config(cfg) for cfg in plan.get("character_configs", [])]


def _render_world_refreshes(
    *,
    delta: dict[str, Any],
    previous_state: dict[str, Any],
    renderer: WorldRenderer,
    run_dir: Path,
) -> dict[str, Any]:
    by_group = {
        "location_updates": ("locations", previous_state.get("locations", []) or []),
        "item_updates": ("items", previous_state.get("items", []) or []),
    }
    for update_key, (group_name, assets) in by_group.items():
        assets_by_id = {str(asset.get("id", "")): asset for asset in assets if isinstance(asset, dict)}
        for update in delta.get(update_key, []) or []:
            if not update.get("image_refresh_required"):
                continue
            asset = assets_by_id.get(str(update.get("id", "")), {})
            out_dir = run_dir / "narrative_artifacts" / "world_updates" / group_name / str(update.get("id", "asset"))
            view_name = normalize_location_view(update.get("view_to_update"), default="wide") if group_name == "locations" else "isolated"
            suffix = f"_{view_name}" if group_name == "locations" else ""
            out_path = out_dir / f"step_{int(delta.get('step', 0)):03d}{suffix}_{renderer.tool_name}.png"
            previous_ref = ""
            views = asset.get("views", {}) if isinstance(asset.get("views"), dict) else {}
            if isinstance(views.get(view_name), dict):
                previous_ref = str(views[view_name].get("last_image_path", "") or "")
            if not previous_ref:
                previous_ref = str(asset.get("last_image_path", "") or "")
            result = renderer.render_world_asset(
                prompt=_asset_prompt(asset, str(update.get("new_state_text", ""))),
                output_path=out_path,
                reference_image_paths=[previous_ref] if previous_ref else None,
            )
            if result.get("ok"):
                update["new_image_path"] = result["output_path"]
                update["reference_image_paths"] = [previous_ref] if previous_ref else []
            update["render_result"] = result
    characters_by_id = {str(row.get("id", "")): row for row in previous_state.get("characters", []) or []}
    for update in delta.get("character_updates", []) or []:
        if not update.get("image_refresh_required"):
            continue
        character = characters_by_id.get(str(update.get("id", "")), {})
        views = character.get("views", {}) if isinstance(character.get("views"), dict) else {}
        previous_ref = str((views.get("front_full") or {}).get("last_image_path", "") or character.get("last_image_path", ""))
        output_path = run_dir / "narrative_artifacts" / "character_updates" / _safe_name(str(update.get("id", "character"))) / f"step_{int(delta.get('step', 0)):03d}_front_full_{renderer.tool_name}.png"
        result = renderer.render_world_asset(
            prompt=(f"Update the front full-body character reference for {character.get('name', update.get('id', 'character'))}. "
                    f"Maintain identity and unchanged traits. Apply only this appearance change: {update.get('physical_change', '')}. "
                    f"Resulting appearance: {update.get('new_state_text', '')}. Straight-on full-body view, plain background."),
            output_path=output_path,
            reference_image_paths=[previous_ref] if previous_ref else None,
        )
        update["view_to_update"] = "front_full"
        update["reference_image_paths"] = [previous_ref] if previous_ref else []
        update["render_result"] = result
        if result.get("ok"):
            update["new_image_path"] = str(result.get("output_path", ""))
    return delta


def _run_one_backend(
    *,
    story: str,
    characters: list[str],
    backend_mode: str,
    max_steps: int,
    output_dir: Path,
    dry_run: bool,
    stop_on_complete: bool = True,
    strict_online: bool = False,
    retries: int = 3,
    story_control_enabled: bool = False,
) -> Path:
    run_dir = output_dir / backend_mode
    run_dir.mkdir(parents=True, exist_ok=True)
    renderer = WorldRenderer(backend_mode=backend_mode, dry_run=dry_run)
    keyframe_renderer = EventKeyframeRenderer(backend_mode=backend_mode, dry_run=dry_run)
    backend_tool = backend_mode_to_tool(backend_mode)

    LOGGER.info(
        "pipeline run status=start backend=%s max_steps=%d dry_run=%s story_control=%s",
        backend_mode, max_steps, dry_run, story_control_enabled,
    )
    world_model = _build_world_model(story=story, renderer=renderer, run_dir=run_dir, dry_run=dry_run)
    character_configs: list[dict[str, Any]] = [
        {"name": name, "final_goal": "Advance the dry-run story.", "role_profile": f"{name} in the story."}
        for name in characters
    ]
    if dry_run:
        portraits = render_character_initial_assets(
            story=story,
            character_configs=character_configs,
            renderer=renderer,
            run_dir=run_dir,
        )
        world_model["character_portraits"] = portraits
        step_logs = _dry_step_logs(story, characters, max_steps)
    else:
        step_logs, character_configs = _real_step_logs(
            story,
            characters,
            world_model,
            max_steps,
            stop_on_complete=stop_on_complete,
            checkpoint_path=run_dir / "narrative_checkpoint.json",
            renderer=renderer,
            run_dir=run_dir,
            story_control_enabled=story_control_enabled,
        )
        _write_json(run_dir / "step_logs.partial.json", step_logs)

    # _real_step_logs temporarily stores CharacterConfig objects while running the simulation.
    # Downstream world-state construction and JSON persistence consume plain dictionaries.
    world_model["characters"] = character_configs

    world_states = []
    world_deltas = []
    event_keyframes = []
    current_state = build_initial_world_state(world_model=world_model, backend_tool=backend_tool)
    world_states.append(current_state)
    write_world_state(run_dir / "world_state_step0.json", current_state)

    judge = WorldDeltaJudge(
        llm_enabled=not dry_run,
        strict_online=strict_online,
        retries=retries,
    )
    for step_log in step_logs:
        step = int(step_log.get("step", 0) or 0)
        delta = judge.judge_step_delta(previous_state=current_state, step_log=step_log, story=story)
        delta = _render_world_refreshes(delta=delta, previous_state=current_state, renderer=renderer, run_dir=run_dir)
        world_deltas.append(delta)
        _write_json(run_dir / "world_deltas" / f"world_delta_step{step}.json", delta)
        current_state = update_world_state_from_delta(previous_state=current_state, delta=delta)
        world_states.append(current_state)
        write_world_state(run_dir / f"world_state_step{step}.json", current_state)

        for role_log in step_log.get("role_logs", []) or []:
            event_context = _event_context_for_role(
                role_log=role_log,
                step_log=step_log,
                current_state=current_state,
            )
            reference_bundle = select_event_reference_images(
                event=event_context,
                current_state=current_state,
                character_portraits=world_model.get("character_portraits", {}),
                max_reference_images=4,
            )
            reference_image_paths = reference_bundle["reference_image_paths"]
            kf = keyframe_renderer.render_event_keyframe(
                step=step,
                role_log=role_log,
                output_dir=run_dir / "narrative_artifacts" / "event_keyframes",
                reference_image_paths=reference_image_paths,
                event_context=event_context,
            )
            kf["reference_selection"] = reference_bundle
            role_log["pipeline_event_keyframe_path"] = kf.get("output_path", "")
            event_keyframes.append(kf)

    result = {
        "backend_mode": backend_mode,
        "backend_tool": backend_tool,
        "dry_run": dry_run,
        "story": story,
        "characters": characters,
        "character_configs": character_configs,
        "character_portraits": world_model.get("character_portraits", {}),
        "world_model": world_model,
        "world_states": world_states,
        "world_deltas": world_deltas,
        "step_logs": step_logs,
        "event_keyframes": event_keyframes,
        "story_control": _read_json_if_exists(run_dir / "story_control.json") if story_control_enabled else {},
    }
    _write_json(run_dir / "story_bundle.json", {"story": story, "characters": characters, "backend_mode": backend_mode})
    _write_json(run_dir / "step_logs.json", step_logs)
    _write_json(run_dir / "pipeline_result.json", result)
    generate_pipeline_html(run_dirs=[run_dir], output_html=run_dir / "visualizations" / "combined_overview.html")
    LOGGER.info("pipeline run status=done backend=%s output=%s", backend_mode, run_dir / "pipeline_result.json")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dual-backend narrative world pipeline.")
    parser.add_argument("--story-file", type=Path, default=None)
    parser.add_argument("--storyline", type=str, default="")
    parser.add_argument("--characters", type=str, default="Journalist,HiddenWitness,Caretaker")
    parser.add_argument(
        "--backend-mode",
        type=str,
        choices=["all_flux", "all_banana", "all_flux2_klein"],
        default="all_flux",
    )
    parser.add_argument("--run-both", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--stop-on-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    story = _read_story(args)
    characters = _parse_characters(args.characters)
    modes = ["all_flux", "all_banana"] if args.run_both else [args.backend_mode]
    run_dirs = []
    for mode in modes:
        run_dirs.append(
            _run_one_backend(
                story=story,
                characters=characters,
                backend_mode=mode,
                max_steps=max(1, int(args.max_steps)),
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                stop_on_complete=args.stop_on_complete,
                strict_online=args.strict_online,
                retries=args.retries,
                story_control_enabled=False,
            )
        )
    if len(run_dirs) > 1:
        generate_pipeline_html(run_dirs=run_dirs, output_html=args.output_dir / "comparison_overview.html")
        print(args.output_dir / "comparison_overview.html")
    else:
        print(run_dirs[0] / "pipeline_result.json")


if __name__ == "__main__":
    main()
