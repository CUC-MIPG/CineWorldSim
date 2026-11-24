#!/usr/bin/env python3
"""Run a narrative simulation example and dump JSON output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from agents.logging_utils import configure_logging
from narrative.simulation_scaffold import CharacterAgent, OrchestratorAgent, StateAnalysisMachine, WorldModelAgent


def build_example_input() -> tuple[str, list[str]]:
    story = (
        "A quiet museum heist begins at midnight. "
        "The curator notices a missing key artifact, "
        "while a security guard and an unknown visitor each react differently."
    )
    characters = ["Curator", "Guard", "Visitor"]
    return story, characters


def _serialize_character_config(cfg: object) -> dict[str, object]:
    # Support both dataclass CharacterConfig and dict form from checkpoint resume.
    if isinstance(cfg, dict):
        return {
            "name": cfg.get("name", ""),
            "final_goal": cfg.get("final_goal", ""),
            "role_profile": cfg.get("role_profile", ""),
            "known_relations": cfg.get("known_relations", []),
            "current_environment": cfg.get("current_environment", ""),
            "prompt_arm": cfg.get("prompt_arm", ""),
            "model_type": cfg.get("model_type", ""),
            "model_name": cfg.get("model_name", ""),
            "base_url": cfg.get("base_url", ""),
            "llm_enabled": bool(cfg.get("llm_enabled", False)),
            "is_ready": bool(cfg.get("llm_enabled", False) and cfg.get("model_name") and cfg.get("base_url")),
            "system_prompt_path": cfg.get("system_prompt_path", ""),
            "dynamic_system_prompt_preview": str(cfg.get("dynamic_system_prompt", ""))[:500],
        }
    model_cfg = getattr(cfg, "model_config", None)
    return {
        "name": getattr(cfg, "name", ""),
        "final_goal": getattr(cfg, "final_goal", ""),
        "role_profile": getattr(cfg, "role_profile", ""),
        "known_relations": list(getattr(cfg, "known_relations", []) or []),
        "current_environment": getattr(cfg, "current_environment", ""),
        "prompt_arm": getattr(cfg, "prompt_arm", ""),
        "model_type": getattr(model_cfg, "model_type", "") if model_cfg else "",
        "model_name": getattr(model_cfg, "model_name", "") if model_cfg else "",
        "base_url": getattr(model_cfg, "base_url", "") if model_cfg else "",
        "llm_enabled": bool(getattr(model_cfg, "llm_enabled", False)) if model_cfg else False,
        "is_ready": bool(model_cfg.is_ready()) if model_cfg and hasattr(model_cfg, "is_ready") else False,
        "system_prompt_path": getattr(model_cfg, "system_prompt_path", "") if model_cfg else "",
        "dynamic_system_prompt_preview": str(getattr(cfg, "dynamic_system_prompt", "") or "")[:500],
    }


def _recover_base_result_from_checkpoint(checkpoint: dict[str, object]) -> dict[str, object]:
    raw_result = checkpoint.get("result", {})
    if isinstance(raw_result, dict) and (
        raw_result.get("trace")
        or raw_result.get("step_logs")
        or raw_result.get("broadcast_log")
    ):
        return raw_result

    # Interrupted runs may only have top-level trace/step_logs before final merge.
    recovered = {
        "character_portraits": checkpoint.get("character_portraits", {}),
        "keyframe_history_by_actor": checkpoint.get("keyframe_history_by_actor", {}),
        "narrative_graph": checkpoint.get("narrative_graph", {}),
        "trace": checkpoint.get("trace", []),
        "broadcast_log": checkpoint.get("broadcast_log", []),
        "step_logs": checkpoint.get("step_logs", []),
        "initial_nodes": checkpoint.get("initial_nodes", []),
        "final_world_state": checkpoint.get("final_world_state", {}),
        "completed_steps": int(checkpoint.get("completed_steps", 0)),
        "max_steps": int(checkpoint.get("max_steps", 0) or 0),
        "stop_reason": str(checkpoint.get("stop_reason", "")),
    }
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=Path("/data1/AgentG/hhb/CameraMaster/outputs/narrative_simulation_example/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data1/AgentG/hhb/CameraMaster/outputs/narrative_simulation_example"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    configure_logging(args.log_level)

    story, character_names = build_example_input()
    orchestrator = OrchestratorAgent()
    world_model_agent = WorldModelAgent()
    sam = StateAnalysisMachine()

    checkpoint_path = args.output_dir / "narrative_checkpoint.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        plan = checkpoint.get("plan", {})
        world_model = checkpoint.get("world_model", {})
        completed_steps = int(checkpoint.get("completed_steps", 0))
        logger = __import__("logging").getLogger(__name__)
        logger.info("narrative resume status=loaded checkpoint=%s completed_steps=%d", checkpoint_path, completed_steps)
        agents = [CharacterAgent.from_dict(a) for a in checkpoint.get("agents", [])]
        if not plan and "prompt" in checkpoint:
            plan = {
                "prompt": checkpoint["prompt"],
                "character_configs": [],
                "mab_arm_stats": {},
            }
        modeled_prompt = str(checkpoint.get("modeled_prompt", "") or "").strip()
        if not modeled_prompt:
            modeled_prompt = WorldModelAgent.enrich_story_prompt(
                str(plan.get("prompt", story) or story),
                world_model if isinstance(world_model, dict) else {},
            )
        if (not isinstance(world_model, dict) or not world_model) and completed_steps <= 0:
            world_model = world_model_agent.build_world_model(
                story_prompt=str(plan.get("prompt", story) or story),
                output_dir=args.output_dir,
            )
            checkpoint["world_model"] = world_model
            modeled_prompt = world_model_agent.enrich_story_prompt(str(plan.get("prompt", story) or story), world_model)
            checkpoint["modeled_prompt"] = modeled_prompt
            checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if completed_steps >= args.max_steps and checkpoint.get("result"):
            result = checkpoint["result"]
            trace = result.get("trace", [])
            broadcast_log = result.get("broadcast_log", [])
            step_logs = result.get("step_logs", [])
            output = {
                "story_outline": {
                    "story": story,
                    "max_steps": args.max_steps,
                },
                "character_configs": plan.get("character_configs", []),
                "sam_config": {
                    "mab_arm_stats": plan.get("mab_arm_stats", {}),
                },
                "world_model_config": {
                    "model_type": world_model_agent.model_config.model_type,
                    "model_name": world_model_agent.model_config.model_name,
                    "base_url": world_model_agent.model_config.base_url,
                    "llm_enabled": world_model_agent.model_config.llm_enabled,
                    "is_ready": world_model_agent.model_config.is_ready(),
                    "system_prompt_path": world_model_agent.model_config.system_prompt_path,
                },
                "world_model": world_model,
                "character_portraits": result.get("character_portraits", {}),
                "step_logs": step_logs,
                "narrative_graph": result.get("narrative_graph", {}),
                "final_world_state": result.get("final_world_state", {}),
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(args.output_json)
            return
    else:
        plan = orchestrator.understand_input(story, character_names)
        world_model = world_model_agent.build_world_model(
            story_prompt=story,
            output_dir=args.output_dir,
        )
        plan["character_configs"] = world_model_agent.enrich_character_configs(
            plan["character_configs"],
            world_model,
        )
        modeled_prompt = world_model_agent.enrich_story_prompt(story, world_model)
        completed_steps = 0
        checkpoint = {
            "story": story,
            "character_names": character_names,
            "world_model": world_model,
            "plan": {
                "prompt": plan["prompt"],
                "character_configs": [
                    {
                        "name": cfg.name,
                        "final_goal": cfg.final_goal,
                        "role_profile": cfg.role_profile,
                        "known_relations": cfg.known_relations,
                        "current_environment": cfg.current_environment,
                        "prompt_arm": cfg.prompt_arm,
                        "model_type": cfg.model_config.model_type,
                        "model_name": cfg.model_config.model_name,
                        "base_url": cfg.model_config.base_url,
                        "llm_enabled": cfg.model_config.llm_enabled,
                        "system_prompt_path": cfg.model_config.system_prompt_path,
                        "dynamic_system_prompt": cfg.dynamic_system_prompt,
                    }
                    for cfg in plan["character_configs"]
                ],
                "mab_arm_stats": plan.get("mab_arm_stats", {}),
            },
            "modeled_prompt": modeled_prompt,
            "completed_steps": 0,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        agents = orchestrator.build_character_agents(plan["character_configs"])

    base_result = _recover_base_result_from_checkpoint(checkpoint) if isinstance(checkpoint, dict) else {}
    result = sam.run_narrative_deduction(
        prompt=modeled_prompt,
        agents=agents,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
        resume_state=checkpoint if args.resume and checkpoint_path.exists() else None,
    )
    def _world_to_dict(value: object) -> dict[str, object]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "step"):
            return {
                "step": getattr(value, "step", 0),
                "summary": getattr(value, "summary", ""),
                "current_environment": getattr(value, "current_environment", ""),
                "story_anchor": getattr(value, "story_anchor", ""),
                "public_events": list(getattr(value, "public_events", [])),
            }
        return {}

    merged_trace = list(base_result.get("trace", [])) + list(result.get("trace", []))
    merged_broadcast_log = list(base_result.get("broadcast_log", [])) + list(result.get("broadcast_log", []))
    merged_step_logs = list(base_result.get("step_logs", [])) + list(result.get("step_logs", []))
    merged_result = {
        "character_portraits": result.get("character_portraits", base_result.get("character_portraits", {})),
        "keyframe_history_by_actor": result.get("keyframe_history_by_actor", base_result.get("keyframe_history_by_actor", {})),
        "narrative_graph": result.get("narrative_graph", base_result.get("narrative_graph", {})),
        "trace": merged_trace,
        "broadcast_log": merged_broadcast_log,
        "step_logs": merged_step_logs,
        "initial_nodes": result.get("initial_nodes", base_result.get("initial_nodes", [])),
        "final_world_state": _world_to_dict(result.get("final_world_state", base_result.get("final_world_state", {}))),
        "completed_steps": int(result.get("completed_steps", completed_steps)),
        "max_steps": int(result.get("max_steps", args.max_steps)),
        "stop_reason": str(result.get("stop_reason", "")),
    }
    checkpoint["completed_steps"] = int(merged_result.get("completed_steps", completed_steps))
    checkpoint["world_model"] = world_model
    checkpoint["modeled_prompt"] = modeled_prompt
    checkpoint["result"] = merged_result
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = merged_trace
    broadcast_log = merged_broadcast_log
    step_logs = merged_step_logs
    final_world_state = merged_result["final_world_state"]

    steps: list[dict[str, object]] = []
    trace_by_step: dict[int, list[dict[str, object]]] = {}
    broadcast_by_step: dict[int, list[dict[str, object]]] = {}
    for item in trace:
        trace_by_step.setdefault(int(item["step"]), []).append(item)
    for item in broadcast_log:
        broadcast_by_step.setdefault(int(item["step"]), []).append(item)

    completed_total = int(merged_result.get("completed_steps", 0))
    for step_id in range(1, completed_total + 1):
        steps.append(
            {
                "step": step_id,
                "actions": trace_by_step.get(step_id, []),
                "broadcasts": broadcast_by_step.get(step_id, []),
                "story_anchor": final_world_state.get("story_anchor", "") if step_id == completed_total else "",
                "world_summary": final_world_state.get("summary", "") if step_id == completed_total else "",
                "current_environment": final_world_state.get("current_environment", "") if step_id == completed_total else "",
            }
        )

    output = {
        "story_outline": {
            "story": story,
            "modeled_prompt": modeled_prompt,
            "max_steps": args.max_steps,
        },
        "character_configs": [_serialize_character_config(cfg) for cfg in plan.get("character_configs", [])],
        "world_model_config": {
            "model_type": world_model_agent.model_config.model_type,
            "model_name": world_model_agent.model_config.model_name,
            "base_url": world_model_agent.model_config.base_url,
            "llm_enabled": world_model_agent.model_config.llm_enabled,
            "is_ready": world_model_agent.model_config.is_ready(),
            "system_prompt_path": world_model_agent.model_config.system_prompt_path,
        },
        "world_model": world_model,
        "sam_config": {
            "model_type": sam.model_config.model_type,
            "model_name": sam.model_config.model_name,
            "base_url": sam.model_config.base_url,
            "llm_enabled": sam.model_config.llm_enabled,
            "is_ready": sam.model_config.is_ready(),
            "system_prompt_path": sam.model_config.system_prompt_path,
            "mab_arm_stats": plan.get("mab_arm_stats", {}),
        },
        "character_portraits": merged_result.get("character_portraits", {}),
        "step_logs": [
            {
                "step": step_item["step"],
                "role_logs": step_item["role_logs"],
                "sam_story_anchor": step_item["sam_story_anchor"],
                "sam_summary": step_item["sam_summary"],
                "sam_current_environment": step_item["sam_current_environment"],
                "sam_story_phase": step_item.get("sam_story_phase", ""),
                "sam_remaining_gap_text": step_item.get("sam_remaining_gap_text", ""),
                "storyline_completion": step_item.get("storyline_completion", {}),
            }
            for step_item in merged_step_logs
        ],
        "narrative_graph": merged_result.get("narrative_graph", {}),
        "final_world_state": {
            "step": final_world_state.get("step", 0),
            "summary": final_world_state.get("summary", ""),
            "current_environment": final_world_state.get("current_environment", ""),
            "story_anchor": final_world_state.get("story_anchor", ""),
            "public_events": final_world_state.get("public_events", []),
        },
        "run_control": {
            "completed_steps": completed_total,
            "requested_max_steps": args.max_steps,
            "stop_reason": str(merged_result.get("stop_reason", "")),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
