#!/usr/bin/env python3
"""Rerun narrative steps that contained LLM fallbacks.

The original pipeline stores the final simulation checkpoint, not a checkpoint
for every individual step.  This tool reconstructs the simulation state at the
step before the requested node, reruns that step (and optionally all later
steps), then rebuilds the downstream world-state and keyframe artifacts in a
new output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from narrative.asset_reference_selector import select_event_reference_images
from narrative.event_keyframe_renderer import EventKeyframeRenderer
from narrative.pipeline_schemas import normalize_location_view
from narrative.simulation_scaffold import CharacterAgent, StateAnalysisMachine, WorldState
from narrative.world_delta_judge import WorldDeltaJudge
from narrative.world_renderer import WorldRenderer
from narrative.world_state_tracker import update_world_state_from_delta, write_world_state
from scripts.generate_narrative_world_pipeline_html import generate_pipeline_html
from scripts.run_dual_backend_narrative_world_pipeline import (
    _asset_prompt,
    _dry_step_logs,
    _event_context_for_role,
    _render_world_refreshes,
    _write_json,
)

LOGGER = logging.getLogger(__name__)

VALID_SOURCES = {"action", "evaluation", "world_delta", "storyline_completion"}


@dataclass(frozen=True)
class NodeSpec:
    step: int
    actor: str
    source: str


def parse_node_spec(value: str) -> NodeSpec:
    """Parse ``STEP:Actor_Name:SOURCE`` into a normalized node specification."""
    raw = str(value or "").strip()
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid node spec {value!r}; expected STEP:Actor_Name:SOURCE"
        )
    try:
        step = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"Invalid node step in {value!r}") from exc
    actor = parts[1].replace("_", " ").strip()
    source = parts[2].strip().lower()
    if step <= 0 or not actor:
        raise ValueError(f"Invalid node step or actor in {value!r}")
    if source not in VALID_SOURCES:
        choices = ", ".join(sorted(VALID_SOURCES))
        raise ValueError(f"Invalid source {source!r}; expected one of: {choices}")
    return NodeSpec(step=step, actor=actor, source=source)


def select_rerun_range(
    *, target_step: int, max_step: int, rerun_downstream: bool
) -> tuple[int, int]:
    if target_step <= 0:
        raise ValueError("target_step must be positive")
    if max_step < target_step:
        raise ValueError(
            f"target_step={target_step} is after max_step={max_step}"
        )
    return target_step, max_step if rerun_downstream else target_step


def build_default_output_dir(source_pipeline_dir: Path) -> Path:
    """Return a sibling output tree without overwriting the source."""
    source_pipeline_dir = Path(source_pipeline_dir)
    backend_dir = source_pipeline_dir.name
    parent = source_pipeline_dir.parent
    return parent.with_name(f"{parent.name}_fallback_rerun") / backend_dir


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dict(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _read_list(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _load_inputs(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pipeline_result = _read_dict(source_dir / "pipeline_result.json")
    checkpoint = _read_dict(source_dir / "narrative_checkpoint.json")
    step_logs = _read_list(source_dir / "step_logs.json")
    return pipeline_result, checkpoint, step_logs


def _find_step(step_logs: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in step_logs:
        if int(row.get("step", 0) or 0) == step:
            return row
    raise RuntimeError(f"Step {step} not found in step_logs.json")


def _validate_nodes(specs: list[NodeSpec], step_logs: list[dict[str, Any]]) -> None:
    for spec in specs:
        step = _find_step(step_logs, spec.step)
        actors = {
            str(row.get("role_name", "")).strip()
            for row in step.get("role_logs", []) or []
            if isinstance(row, dict)
        }
        if spec.actor not in actors:
            available = ", ".join(sorted(actor for actor in actors if actor))
            raise RuntimeError(
                f"Actor {spec.actor!r} not found at step {spec.step}; "
                f"available actors: {available or 'none'}"
            )


def _public_events_before_step(step_log: dict[str, Any]) -> list[str]:
    role_logs = [row for row in step_log.get("role_logs", []) or [] if isinstance(row, dict)]
    for role in reversed(role_logs):
        environment = role.get("environment_description", {})
        if isinstance(environment, dict):
            events = environment.get("public_events", [])
            if isinstance(events, list):
                return [str(event) for event in events if str(event).strip()]
    return []


def _reconstruct_resume_state(
    *,
    checkpoint: dict[str, Any],
    step_logs: list[dict[str, Any]],
    start_step: int,
) -> dict[str, Any]:
    """Reconstruct the simulation state immediately before ``start_step``."""
    previous_step = start_step - 1
    if previous_step <= 0:
        previous = {
            "step": 0,
            "sam_summary": str(checkpoint.get("prompt", "") or ""),
            "sam_current_environment": str(checkpoint.get("prompt", "") or ""),
            "sam_story_anchor": "",
            "role_logs": [],
        }
    else:
        previous = _find_step(step_logs, previous_step)

    agents = copy.deepcopy(checkpoint.get("agents", []) or [])
    if not isinstance(agents, list) or not agents:
        raise RuntimeError("Checkpoint does not contain agents")

    environment = str(
        previous.get("sam_current_environment")
        or previous.get("sam_summary")
        or checkpoint.get("prompt")
        or ""
    )
    world_state = {
        "step": previous_step,
        "summary": str(previous.get("sam_summary") or checkpoint.get("prompt") or ""),
        "current_environment": environment,
        "story_anchor": str(previous.get("sam_story_anchor") or ""),
        "public_events": _public_events_before_step(previous),
    }

    # Character goals/profiles/relations are stable. The final checkpoint may
    # contain later private state, but action prompting currently depends on
    # the stable identity fields and the reconstructed shared environment.
    initial_by_name = {
        str(item.get("actor") or "").strip(): item
        for item in checkpoint.get("initial_nodes", []) or []
        if isinstance(item, dict)
    }
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        state = agent.get("state")
        if isinstance(state, dict):
            state["current_environment"] = environment
            # The final checkpoint contains later private clues. Reset the
            # fields that influence action prompting to their initial values;
            # the historical step logs do not persist per-step private state.
            initial = initial_by_name.get(str(state.get("name") or "").strip(), {})
            if isinstance(initial, dict):
                state["known_clues"] = list(initial.get("known_clues", []) or [])
                state["state_summary"] = ""

    return {
        "prompt": str(checkpoint.get("prompt") or ""),
        "max_steps": int(checkpoint.get("max_steps", previous_step) or previous_step),
        "completed_steps": previous_step,
        "world_state": world_state,
        "agents": agents,
        "character_portraits": checkpoint.get("character_portraits", {}),
        "keyframe_history_by_actor": {},
        "trace": [],
        "broadcast_log": [],
        "step_logs": [],
        "initial_nodes": checkpoint.get("initial_nodes", []),
        "final_world_state": world_state,
        "narrative_graph": {},
    }


def _prefix_rows(rows: list[dict[str, Any]], before_step: int) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row.get("step", 0) or 0) < before_step
    ]


def _load_world_state(source_dir: Path, pipeline_result: dict[str, Any], step: int) -> dict[str, Any]:
    path = source_dir / f"world_state_step{step}.json"
    if path.exists():
        return _read_dict(path)
    rows = pipeline_result.get("world_states", []) or []
    for row in rows:
        if isinstance(row, dict) and int(row.get("step", 0) or 0) == step:
            return copy.deepcopy(row)
    raise RuntimeError(f"world_state_step{step}.json is missing")


def _copy_prefix_world_artifacts(
    *,
    source_dir: Path,
    output_dir: Path,
    start_step: int,
    world_states: list[dict[str, Any]],
    world_deltas: list[dict[str, Any]],
) -> None:
    for state in world_states:
        step = int(state.get("step", 0) or 0)
        if step < start_step:
            write_world_state(output_dir / f"world_state_step{step}.json", state)
    for delta in world_deltas:
        step = int(delta.get("step", 0) or 0)
        if step < start_step:
            _write_json(output_dir / "world_deltas" / f"world_delta_step{step}.json", delta)

    # Keep old visual assets addressable from the merged JSON while placing
    # newly generated assets in the rerun output tree.
    for relative in [
        Path("narrative_artifacts/world_model"),
        Path("narrative_artifacts/portraits"),
    ]:
        source = source_dir / relative
        target = output_dir / relative
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)


def _rebuild_world_and_keyframes(
    *,
    story: str,
    source_dir: Path,
    output_dir: Path,
    pipeline_result: dict[str, Any],
    step_logs: list[dict[str, Any]],
    start_step: int,
    end_step: int,
    backend_mode: str,
    strict_online: bool,
    retries: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_states = [
        row for row in pipeline_result.get("world_states", []) or []
        if isinstance(row, dict)
    ]
    source_deltas = [
        row for row in pipeline_result.get("world_deltas", []) or []
        if isinstance(row, dict)
    ]
    source_keyframes = [
        row for row in pipeline_result.get("event_keyframes", []) or []
        if isinstance(row, dict)
    ]
    prefix_states = [row for row in source_states if int(row.get("step", 0) or 0) < start_step]
    prefix_deltas = [row for row in source_deltas if int(row.get("step", 0) or 0) < start_step]
    prefix_keyframes = [row for row in source_keyframes if int(row.get("step", 0) or 0) < start_step]
    _copy_prefix_world_artifacts(
        source_dir=source_dir,
        output_dir=output_dir,
        start_step=start_step,
        world_states=prefix_states,
        world_deltas=prefix_deltas,
    )

    current_state = _load_world_state(source_dir, pipeline_result, start_step - 1)
    world_states = [copy.deepcopy(row) for row in prefix_states]
    if not any(int(row.get("step", 0) or 0) == start_step - 1 for row in world_states):
        world_states.append(copy.deepcopy(current_state))
    world_deltas = [copy.deepcopy(row) for row in prefix_deltas]
    event_keyframes = [copy.deepcopy(row) for row in prefix_keyframes]

    renderer = WorldRenderer(backend_mode=backend_mode, dry_run=dry_run)
    keyframe_renderer = EventKeyframeRenderer(backend_mode=backend_mode, dry_run=dry_run)
    backend_tool = str(pipeline_result.get("backend_tool") or "nano_banana")
    world_model = copy.deepcopy(pipeline_result.get("world_model", {}) or {})
    character_portraits = pipeline_result.get("character_portraits", {}) or {}
    judge = WorldDeltaJudge(
        llm_enabled=not dry_run,
        strict_online=strict_online,
        retries=retries,
    )

    logs_by_step = {
        int(row.get("step", 0) or 0): row
        for row in step_logs
        if isinstance(row, dict)
    }
    for step in range(start_step, end_step + 1):
        step_log = logs_by_step.get(step)
        if step_log is None:
            raise RuntimeError(f"Rerun did not produce step {step}")
        delta = judge.judge_step_delta(
            previous_state=current_state,
            step_log=step_log,
            story=story,
        )
        delta = _render_world_refreshes(
            delta=delta,
            previous_state=current_state,
            renderer=renderer,
            run_dir=output_dir,
        )
        world_deltas.append(delta)
        _write_json(output_dir / "world_deltas" / f"world_delta_step{step}.json", delta)
        current_state = update_world_state_from_delta(
            previous_state=current_state,
            delta=delta,
        )
        world_states.append(current_state)
        write_world_state(output_dir / f"world_state_step{step}.json", current_state)

        for role_log in step_log.get("role_logs", []) or []:
            event_context = _event_context_for_role(
                role_log=role_log,
                step_log=step_log,
                current_state=current_state,
            )
            reference_bundle = select_event_reference_images(
                event=event_context,
                current_state=current_state,
                character_portraits=character_portraits,
                max_reference_images=4,
            )
            rendered = keyframe_renderer.render_event_keyframe(
                step=step,
                role_log=role_log,
                output_dir=output_dir / "narrative_artifacts" / "event_keyframes",
                reference_image_paths=reference_bundle["reference_image_paths"],
                event_context=event_context,
            )
            rendered["reference_selection"] = reference_bundle
            role_log["pipeline_event_keyframe_path"] = rendered.get("output_path", "")
            event_keyframes.append(rendered)
    # In target-step-only mode the downstream narrative logs are intentionally
    # reused. Keep their world artifacts in the merged result so the output
    # remains a complete pipeline snapshot rather than a truncated patch.
    for row in source_states:
        step = int(row.get("step", 0) or 0)
        if step > end_step and not any(
            int(existing.get("step", 0) or 0) == step for existing in world_states
        ):
            world_states.append(copy.deepcopy(row))
            write_world_state(output_dir / f"world_state_step{step}.json", row)
    for row in source_deltas:
        step = int(row.get("step", 0) or 0)
        if step > end_step and not any(
            int(existing.get("step", 0) or 0) == step for existing in world_deltas
        ):
            world_deltas.append(copy.deepcopy(row))
            _write_json(output_dir / "world_deltas" / f"world_delta_step{step}.json", row)
    for row in source_keyframes:
        step = int(row.get("step", 0) or 0)
        if step > end_step and not any(
            int(existing.get("step", 0) or 0) == step
            and str(existing.get("role_name", "")) == str(row.get("role_name", ""))
            for existing in event_keyframes
        ):
            event_keyframes.append(copy.deepcopy(row))
    world_states.sort(key=lambda row: int(row.get("step", 0) or 0))
    world_deltas.sort(key=lambda row: int(row.get("step", 0) or 0))
    event_keyframes.sort(
        key=lambda row: (
            int(row.get("step", 0) or 0),
            str(row.get("role_name", "")),
        )
    )
    return world_states, world_deltas, event_keyframes


def rerun_fallback_nodes(
    *,
    pipeline_dir: Path,
    node_specs: list[NodeSpec],
    output_dir: Path | None = None,
    rerun_downstream: bool = False,
    strict_online: bool = True,
    retries: int = 3,
    dry_run: bool = False,
) -> Path:
    source_dir = Path(pipeline_dir).resolve()
    if not source_dir.is_dir():
        raise RuntimeError(f"Pipeline directory does not exist: {source_dir}")
    if output_dir is None:
        output_dir = build_default_output_dir(source_dir)
    output_dir = Path(output_dir).resolve()
    if output_dir == source_dir:
        raise RuntimeError("Refusing to write rerun output into the source pipeline directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory already exists and is non-empty: {output_dir}; "
            "choose another --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result, checkpoint, original_step_logs = _load_inputs(source_dir)
    _validate_nodes(node_specs, original_step_logs)
    max_step = max(
        int(row.get("step", 0) or 0)
        for row in original_step_logs
        if int(row.get("step", 0) or 0) > 0
    )
    start_step, end_step = select_rerun_range(
        target_step=min(spec.step for spec in node_specs),
        max_step=max_step,
        rerun_downstream=rerun_downstream,
    )
    if start_step <= 1:
        raise RuntimeError(
            "Cannot reconstruct a pre-step-1 state from the current checkpoint; "
            "rerun the full pipeline instead"
        )

    story = str(pipeline_result.get("story") or checkpoint.get("prompt") or "")
    characters = [str(x) for x in pipeline_result.get("characters", []) or []]
    resume_state = _reconstruct_resume_state(
        checkpoint=checkpoint,
        step_logs=original_step_logs,
        start_step=start_step,
    )
    rerun_checkpoint = output_dir / "narrative_checkpoint.json"
    if dry_run:
        rerun_step_logs = [
            row
            for row in _dry_step_logs(
                story,
                characters,
                end_step,
            )
            if start_step <= int(row.get("step", 0) or 0) <= end_step
        ]
        _write_json(
            rerun_checkpoint,
            {
                **resume_state,
                "completed_steps": end_step,
                "max_steps": end_step,
                "step_logs": rerun_step_logs,
            },
        )
    else:
        sam = StateAnalysisMachine()
        sam_result = sam.run_narrative_deduction(
            prompt=str(resume_state.get("prompt") or story),
            agents=[
                CharacterAgent.from_dict(item)
                for item in resume_state.get("agents", []) or []
                if isinstance(item, dict)
            ],
            max_steps=end_step,
            output_dir=None,
            checkpoint_path=rerun_checkpoint,
            resume_state=resume_state,
            stop_on_complete=False,
        )
        rerun_step_logs = [
            row for row in sam_result.get("step_logs", []) or []
            if isinstance(row, dict)
        ]
    prefix_logs = _prefix_rows(original_step_logs, start_step)
    suffix_logs = [
        row for row in original_step_logs
        if int(row.get("step", 0) or 0) > end_step
    ]
    merged_step_logs = prefix_logs + rerun_step_logs + ([] if rerun_downstream else suffix_logs)
    _write_json(output_dir / "step_logs.partial.json", merged_step_logs)

    backend_mode = str(pipeline_result.get("backend_mode") or "all_banana")
    world_states, world_deltas, event_keyframes = _rebuild_world_and_keyframes(
        story=story,
        source_dir=source_dir,
        output_dir=output_dir,
        pipeline_result=pipeline_result,
        step_logs=merged_step_logs,
        start_step=start_step,
        end_step=end_step,
        backend_mode=backend_mode,
        strict_online=strict_online,
        retries=retries,
        dry_run=dry_run,
    )

    merged_result = copy.deepcopy(pipeline_result)
    merged_result.update(
        {
            "world_states": world_states,
            "world_deltas": world_deltas,
            "step_logs": merged_step_logs,
            "event_keyframes": event_keyframes,
            "rerun_metadata": {
                "source_pipeline_dir": str(source_dir),
                "requested_nodes": [
                    {
                        "step": spec.step,
                        "actor": spec.actor,
                        "source": spec.source,
                    }
                    for spec in node_specs
                ],
                "rerun_start_step": start_step,
                "rerun_end_step": end_step,
                "rerun_downstream": rerun_downstream,
                "downstream_steps_reused": (
                    [] if rerun_downstream else list(range(start_step + 1, max_step + 1))
                ),
            },
        }
    )
    _write_json(output_dir / "story_bundle.json", {
        "story": story,
        "characters": characters,
        "backend_mode": merged_result.get("backend_mode", ""),
    })
    _write_json(output_dir / "step_logs.json", merged_step_logs)
    _write_json(output_dir / "pipeline_result.json", merged_result)

    # Enrich the checkpoint written by StateAnalysisMachine with the original
    # prefix so it can be inspected or used by a later recovery operation.
    generated_checkpoint = _read_dict(rerun_checkpoint)
    generated_checkpoint["step_logs"] = merged_step_logs
    generated_checkpoint["completed_steps"] = end_step
    generated_checkpoint["max_steps"] = max_step
    generated_checkpoint["rerun_metadata"] = merged_result["rerun_metadata"]
    _write_json(rerun_checkpoint, generated_checkpoint)

    try:
        generate_pipeline_html(
            run_dirs=[output_dir],
            output_html=output_dir / "visualizations" / "combined_overview.html",
        )
    except Exception as exc:  # visualization is auxiliary
        LOGGER.warning("rerun visualization failed: %s", exc)

    manifest = {
        "status": "done",
        "source_pipeline_dir": str(source_dir),
        "output_dir": str(output_dir),
        "requested_nodes": [
            {"step": spec.step, "actor": spec.actor, "source": spec.source}
            for spec in node_specs
        ],
        "rerun_start_step": start_step,
        "rerun_end_step": end_step,
        "rerun_downstream": rerun_downstream,
        "max_source_step": max_step,
        "artifacts": {
            "pipeline_result_json": str(output_dir / "pipeline_result.json"),
            "step_logs_json": str(output_dir / "step_logs.json"),
            "checkpoint_json": str(output_dir / "narrative_checkpoint.json"),
        },
    }
    _write_json(output_dir / "fallback_rerun_manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun narrative steps containing LLM fallback results."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        required=True,
        help="Existing backend directory containing pipeline_result.json and checkpoint.",
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        metavar="STEP:ACTOR:SOURCE",
        help="Repeatable, e.g. 13:Mara_Venn:evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--rerun-downstream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rerun from the earliest requested step through the source max step.",
    )
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    specs = [parse_node_spec(value) for value in args.node]
    planned_output_dir = args.output_dir or build_default_output_dir(args.pipeline_dir)
    try:
        output_dir = rerun_fallback_nodes(
            pipeline_dir=args.pipeline_dir,
            node_specs=specs,
            output_dir=args.output_dir,
            rerun_downstream=args.rerun_downstream,
            strict_online=args.strict_online,
            retries=args.retries,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        source_dir = Path(args.pipeline_dir).resolve()
        failure_dir = Path(planned_output_dir).resolve()
        if failure_dir != source_dir:
            failure_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                failure_dir / "fallback_rerun_manifest.json",
                {
                    "status": "failed",
                    "source_pipeline_dir": str(source_dir),
                    "output_dir": str(failure_dir),
                    "requested_nodes": [
                        {"step": spec.step, "actor": spec.actor, "source": spec.source}
                        for spec in specs
                    ],
                    "rerun_downstream": args.rerun_downstream,
                    "strict_online": args.strict_online,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_artifacts": sorted(
                        str(path.relative_to(failure_dir))
                        for path in failure_dir.rglob("*")
                        if path.is_file() and path.name != "fallback_rerun_manifest.json"
                    ),
                },
            )
        LOGGER.error("fallback rerun failed output=%s error=%s", failure_dir, exc)
        raise
    print(output_dir / "fallback_rerun_manifest.json")


if __name__ == "__main__":
    main()
