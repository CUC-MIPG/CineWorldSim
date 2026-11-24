#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig
from scripts.run_dual_backend_narrative_world_pipeline import _parse_characters, _read_story, _run_one_backend


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _load_pipeline_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid pipeline_result JSON root: {path}")
    return payload


def _world_delta_for_step(payload: dict[str, Any], step: int) -> dict[str, Any]:
    for delta in payload.get("world_deltas", []) or []:
        if isinstance(delta, dict) and int(delta.get("step", 0) or 0) == step:
            return delta
    return {}


def _keyframes_for_step_role(payload: dict[str, Any], step: int, role_name: str) -> list[str]:
    rows: list[str] = []
    for item in payload.get("event_keyframes", []) or []:
        if not isinstance(item, dict):
            continue
        if int(item.get("step", 0) or 0) != step:
            continue
        if str(item.get("role_name", "") or "") != role_name:
            continue
        output_path = str(item.get("output_path", "") or "")
        if output_path:
            rows.append(output_path)
    return rows


def _delta_asset_ids(delta: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ["location_updates", "item_updates"]:
        for row in delta.get(key, []) or []:
            if isinstance(row, dict) and row.get("id"):
                ids.append(str(row["id"]))
    return ids


def build_path_candidates_from_pipeline_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build world-aware candidate nodes for path orchestration."""
    rows: list[dict[str, Any]] = []
    idx = 0
    step_logs = [x for x in payload.get("step_logs", []) or [] if isinstance(x, dict)]
    for step_item in sorted(step_logs, key=lambda x: int(x.get("step", 0) or 0)):
        step = int(step_item.get("step", 0) or 0)
        if step <= 0:
            continue
        scene_summary = str(step_item.get("sam_summary", "") or "")
        delta = _world_delta_for_step(payload, step)
        changed_assets = _delta_asset_ids(delta)
        for role in step_item.get("role_logs", []) or []:
            if not isinstance(role, dict):
                continue
            role_name = str(role.get("role_name", "") or "").strip()
            if not role_name:
                continue
            idx += 1
            node_id = f"step::{step}::{role_name}"
            next_action = role.get("next_action", {})
            action_type = ""
            target = ""
            reason = ""
            location = ""
            if isinstance(next_action, dict):
                action_type = str(next_action.get("action_type") or next_action.get("action") or "").strip()
                target = str(next_action.get("target_character", "") or "").strip()
                reason = str(next_action.get("reason", "") or "").strip()
                location = str(next_action.get("location", "") or "").strip()
            keyframes = _keyframes_for_step_role(payload, step, role_name)
            text = " | ".join(
                item
                for item in [
                    str(role.get("role_output_text", "") or "").strip(),
                    f"action={action_type}" if action_type else "",
                    f"target={target}" if target else "",
                    f"location={location}" if location else "",
                    f"reason={reason}" if reason else "",
                    f"scene_summary={scene_summary}" if scene_summary else "",
                    f"changed_world_assets={', '.join(changed_assets)}" if changed_assets else "",
                ]
                if item
            )
            rows.append(
                {
                    "idx": idx,
                    "node_id": node_id,
                    "step": step,
                    "node_granularity": "single_role_step",
                    "role_name": role_name,
                    "action_type": action_type,
                    "target_character": target,
                    "location": location,
                    "text": text,
                    "scene_summary": scene_summary,
                    "changed_world_assets": changed_assets,
                    "event_keyframe_paths": keyframes,
                }
            )
    return rows


def build_path_candidates_from_narrative_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, node in enumerate(payload.get("selected_nodes", []) or [], start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or node.get("event_id") or f"NN_{idx:03d}")
        characters = [str(x).strip() for x in node.get("characters", []) or [] if str(x).strip()]
        node_granularity = str(node.get("node_granularity") or ("multi_role_interaction" if len(characters) > 1 else "single_role_step"))
        text = " | ".join(
            item
            for item in [
                str(node.get("event_summary", "") or "").strip(),
                f"node_granularity={node_granularity}",
                f"dramatic_function={str(node.get('dramatic_function', '') or '').strip()}"
                if node.get("dramatic_function")
                else "",
                f"event_type={str(node.get('event_type', '') or '').strip()}" if node.get("event_type") else "",
                f"characters={', '.join(node.get('characters', []) or [])}" if node.get("characters") else "",
                f"locations={', '.join(node.get('locations', []) or [])}" if node.get("locations") else "",
                f"items={', '.join(node.get('items', []) or [])}" if node.get("items") else "",
            ]
            if item
        )
        rows.append(
            {
                "idx": idx,
                "node_id": node_id,
                "event_id": str(node.get("event_id", "")),
                "step": int((node.get("source_steps") or [0])[0] or 0),
                "node_granularity": node_granularity,
                "role_name": ",".join(node.get("characters", []) or []),
                "action_type": str(node.get("event_type", "") or ""),
                "text": text,
                "scene_summary": str(node.get("event_summary", "") or ""),
                "selected_node": node,
            }
        )
    return rows


def _run_llm_path(
    *,
    storyline: str,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    strict_online: bool,
    retries: int,
) -> Path:
    try:
        from agents.path_orchestrator_agent import PathOrchestratorAgent

        agent = PathOrchestratorAgent(strict_online=strict_online, retries=retries)
        summary = agent.arrange_candidate_path(
            storyline=storyline,
            candidates=candidates,
            id_field="node_id",
            text_field="text",
            output_list_field="ordered_node_ids",
            output_dir=output_dir,
        )
    except (ModuleNotFoundError, RuntimeError):
        if strict_online:
            raise
        ordered = [str(row.get("node_id", "")) for row in candidates if str(row.get("node_id", ""))]
        summary = {
            "agent": "PathOrchestratorAgent",
            "mode": "fallback_missing_dependency",
            "storyline": storyline,
            "candidate_count": len(ordered),
            "ordered_node_ids": ordered,
            "storyline_fit_summary": "fallback source order because optional online dependency is unavailable.",
            "transition_notes": "Install OpenAI-compatible client dependencies or run in mifage for online ordering.",
            "quality_checks": {
                "coherence": "fallback",
                "coverage": "fallback",
                "pacing": "fallback",
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "path_plan.json", summary)
    ordered = [str(x) for x in summary.get("ordered_node_ids", [])]
    enriched = {
        **summary,
        "node_rank_map": {node_id: i + 1 for i, node_id in enumerate(ordered)},
        "source": "narrative_world_pipeline_result",
    }
    return _write_json(output_dir / "narrative_path_plan.json", enriched)


def _run_mcts_path(
    *,
    storyline: str,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    from agents.path_orchestrator_mcts_agent import PathOrchestratorMCTSAgent

    agent = PathOrchestratorMCTSAgent(
        model=args.mcts_model,
        strict_online=args.strict_online,
        retries=args.retries,
        budget_iters=args.mcts_budget_iters,
        topk_actions=args.mcts_topk_actions,
        cpuct=args.mcts_cpuct,
        rollout_horizon=args.mcts_rollout_horizon,
        rollout_topk=args.mcts_rollout_topk,
        rollout_epsilon=args.mcts_rollout_epsilon,
        progressive_widening_base=args.mcts_pw_base,
        progressive_widening_alpha=args.mcts_pw_alpha,
        pov_switch_penalty=args.pov_switch_penalty,
        over_focus_threshold=args.over_focus_threshold,
        over_focus_penalty=args.over_focus_penalty,
        reward_weight_narrative_causality=args.w_narrative_causality,
        reward_weight_visual_transition=args.w_visual_transition,
        reward_weight_rhythmic_pacing=args.w_rhythmic_pacing,
        random_seed=args.random_seed,
    )
    summary = agent.arrange_candidate_path(
        storyline=storyline,
        candidates=candidates,
        id_field="node_id",
        text_field="text",
        output_list_field="ordered_node_ids",
        output_dir=output_dir,
    )
    ordered = [str(x) for x in summary.get("ordered_node_ids", [])]
    enriched = {
        **summary,
        "node_rank_map": {node_id: i + 1 for i, node_id in enumerate(ordered)},
        "source": "narrative_world_pipeline_result",
    }
    return _write_json(output_dir / "narrative_path_plan_mcts.json", enriched)


def _run_path_for_result(
    *,
    pipeline_result_json: Path,
    storyline: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _load_pipeline_result(pipeline_result_json)
    artifacts: dict[str, str] = {
        "pipeline_result_json": str(pipeline_result_json),
    }
    if args.enable_narrative_node_selector:
        selector = NarrativeNodeSelector(
            NarrativeNodeSelectorConfig(
                selection_mode=args.node_selection_mode,
                score_threshold=args.node_score_threshold,
                min_nodes=args.node_min_nodes,
                max_nodes=args.node_max_nodes,
                llm_enabled=args.node_llm_scoring,
            )
        )
        node_payload = selector.select(payload, source_pipeline_result_json=str(pipeline_result_json))
        narrative_nodes_json = _write_json(output_dir / "narrative_nodes.json", node_payload)
        candidates = build_path_candidates_from_narrative_nodes(node_payload)
        artifacts["narrative_nodes_json"] = str(narrative_nodes_json)
    else:
        candidates = build_path_candidates_from_pipeline_result(payload)
    _write_json(output_dir / "path_candidates.json", candidates)

    runs: list[dict[str, Any]] = []
    artifacts["path_candidates_json"] = str(output_dir / "path_candidates.json")
    if args.path_mode in {"llm", "both"}:
        llm_dir = output_dir / "path_llm_direct"
        llm_path = _run_llm_path(
            storyline=storyline,
            candidates=candidates,
            output_dir=llm_dir,
            strict_online=args.strict_online,
            retries=args.retries,
        )
        artifacts["llm_path_json"] = str(llm_path)
        runs.append({"step": "path_orchestrator_llm_direct", "output_json": str(llm_path)})

    if args.path_mode in {"mcts", "both"}:
        mcts_dir = output_dir / "path_mcts"
        mcts_path = _run_mcts_path(
            storyline=storyline,
            candidates=candidates,
            output_dir=mcts_dir,
            args=args,
        )
        artifacts["mcts_path_json"] = str(mcts_path)
        artifacts["mcts_trace_jsonl"] = str(mcts_dir / "mcts_trace.jsonl")
        runs.append({"step": "path_orchestrator_mcts", "output_json": str(mcts_path)})

    manifest = {
        "status": "done",
        "backend_mode": payload.get("backend_mode", ""),
        "backend_tool": payload.get("backend_tool", ""),
        "storyline": storyline,
        "narrative_node_selector_enabled": bool(args.enable_narrative_node_selector),
        "candidate_count": len(candidates),
        "artifacts": artifacts,
        "runs": runs,
    }
    _write_json(output_dir / "pipeline_manifest.json", manifest)
    return manifest


def _run_world_stage(args: argparse.Namespace, story: str, characters: list[str]) -> list[Path]:
    modes = ["all_flux", "all_banana"] if args.run_both else [args.backend_mode]
    run_dirs: list[Path] = []
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
                strict_online=getattr(args, "strict_online", False),
                retries=getattr(args, "retries", 3),
            )
        )
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated narrative-world pipeline and path orchestration."
    )
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
    parser.add_argument("--existing-pipeline-result-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path-mode", choices=["llm", "mcts", "both"], default="llm")
    parser.add_argument("--enable-narrative-node-selector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--node-selection-mode", choices=["hybrid", "threshold", "top_k"], default="hybrid")
    parser.add_argument("--node-score-threshold", type=float, default=0.55)
    parser.add_argument("--node-min-nodes", type=int, default=3)
    parser.add_argument("--node-max-nodes", type=int, default=8)
    parser.add_argument("--node-llm-scoring", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--mcts-model", type=str, default=None)
    parser.add_argument("--mcts-budget-iters", type=int, default=1)
    parser.add_argument("--mcts-topk-actions", type=int, default=8)
    parser.add_argument("--mcts-cpuct", type=float, default=1.2)
    parser.add_argument("--mcts-rollout-horizon", type=int, default=32)
    parser.add_argument("--mcts-rollout-topk", type=int, default=5)
    parser.add_argument("--mcts-rollout-epsilon", type=float, default=0.2)
    parser.add_argument("--mcts-pw-base", type=float, default=1.5)
    parser.add_argument("--mcts-pw-alpha", type=float, default=0.55)
    parser.add_argument("--pov-switch-penalty", type=float, default=0.05)
    parser.add_argument("--over-focus-threshold", type=int, default=4)
    parser.add_argument("--over-focus-penalty", type=float, default=0.08)
    parser.add_argument("--w-narrative-causality", type=float, default=0.4)
    parser.add_argument("--w-visual-transition", type=float, default=0.3)
    parser.add_argument("--w-rhythmic-pacing", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    story = _read_story(args)
    top_runs: list[dict[str, Any]] = []

    if args.existing_pipeline_result_json is not None:
        manifest = _run_path_for_result(
            pipeline_result_json=args.existing_pipeline_result_json,
            storyline=story,
            output_dir=args.output_dir,
            args=args,
        )
        print(args.output_dir / "pipeline_manifest.json")
        return

    characters = _parse_characters(args.characters)
    run_dirs = _run_world_stage(args, story, characters)
    for run_dir in run_dirs:
        backend_mode = run_dir.name
        path_output_dir = run_dir / "path_orchestration"
        manifest = _run_path_for_result(
            pipeline_result_json=run_dir / "pipeline_result.json",
            storyline=story,
            output_dir=path_output_dir,
            args=args,
        )
        top_runs.append(
            {
                "backend_mode": backend_mode,
                "pipeline_result_json": str(run_dir / "pipeline_result.json"),
                "path_manifest_json": str(path_output_dir / "pipeline_manifest.json"),
                "candidate_count": manifest.get("candidate_count", 0),
            }
        )

    root_manifest = {
        "status": "done",
        "storyline": story,
        "characters": characters,
        "path_mode": args.path_mode,
        "run_both": bool(args.run_both),
        "runs": top_runs,
    }
    _write_json(args.output_dir / "pipeline_manifest.json", root_manifest)
    print(args.output_dir / "pipeline_manifest.json")


if __name__ == "__main__":
    main()
