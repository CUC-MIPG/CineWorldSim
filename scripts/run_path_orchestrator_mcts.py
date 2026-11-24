from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.path_orchestrator_mcts_agent import PathOrchestratorMCTSAgent
from agents.schemas import Storyboard


def _load_step_logs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("step_logs"), list):
        return [x for x in data.get("step_logs", []) if isinstance(x, dict)]
    result = data.get("result", {})
    if isinstance(result, dict) and isinstance(result.get("step_logs"), list):
        return [x for x in result.get("step_logs", []) if isinstance(x, dict)]
    return []


def _build_narrative_candidates(step_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for step_item in sorted(step_logs, key=lambda x: int(x.get("step", 0) or 0)):
        step = int(step_item.get("step", 0) or 0)
        if step <= 0:
            continue
        scene_summary = str(step_item.get("sam_summary", "") or "")
        role_logs = step_item.get("role_logs", [])
        if not isinstance(role_logs, list):
            continue
        for role in role_logs:
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
            if isinstance(next_action, dict):
                action_type = str(next_action.get("action_type", "") or "").strip()
                target = str(next_action.get("target_character", "") or "").strip()
                reason = str(next_action.get("reason", "") or "").strip()
            role_text = str(role.get("role_output_text", "") or "").strip()
            text = " | ".join(
                x
                for x in [
                    role_text,
                    f"action={action_type}" if action_type else "",
                    f"target={target}" if target else "",
                    f"reason={reason}" if reason else "",
                    f"env={str(role.get('environment_text', '') or '').strip()}",
                    f"scene_summary={scene_summary}",
                ]
                if x
            )
            rows.append(
                {
                    "idx": idx,
                    "node_id": node_id,
                    "step": step,
                    "role_name": role_name,
                    "action_type": action_type,
                    "text": text,
                }
            )
    return rows


def _load_storyboard(path: Path) -> Storyboard:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid storyboard JSON root: {path}")
    return Storyboard.from_dict(data)


def _build_agent(args: argparse.Namespace) -> PathOrchestratorMCTSAgent:
    return PathOrchestratorMCTSAgent(
        model=args.model,
        strict_online=args.strict_online,
        retries=args.retries,
        budget_iters=args.budget_iters,
        topk_actions=args.topk_actions,
        cpuct=args.cpuct,
        rollout_horizon=args.rollout_horizon,
        rollout_topk=args.rollout_topk,
        rollout_epsilon=args.rollout_epsilon,
        progressive_widening_base=args.pw_base,
        progressive_widening_alpha=args.pw_alpha,
        pov_switch_penalty=args.pov_switch_penalty,
        over_focus_threshold=args.over_focus_threshold,
        over_focus_penalty=args.over_focus_penalty,
        reward_weight_narrative_causality=args.w_narrative_causality,
        reward_weight_visual_transition=args.w_visual_transition,
        reward_weight_rhythmic_pacing=args.w_rhythmic_pacing,
        random_seed=args.random_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCTS Path Orchestrator on storyboard or narrative step logs.")
    parser.add_argument("--storyline", type=str, required=True, help="User storyline objective.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--model", type=str, default=None, help="LLM model override for MCTS scoring.")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--storyboard-json", type=Path, default=None, help="Input storyboard JSON.")
    source_group.add_argument("--narrative-json", type=Path, default=None, help="Narrative result/checkpoint JSON.")

    parser.add_argument(
        "--output-storyboard-json",
        type=Path,
        default=None,
        help="Only for --storyboard-json mode. Default: <output-dir>/storyboard_path_orchestrated_mcts.json",
    )

    parser.add_argument("--budget-iters", type=int, default=300)
    parser.add_argument("--topk-actions", type=int, default=8)
    parser.add_argument("--cpuct", type=float, default=1.2)
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--rollout-topk", type=int, default=5)
    parser.add_argument("--rollout-epsilon", type=float, default=0.2)
    parser.add_argument("--pw-base", type=float, default=1.5)
    parser.add_argument("--pw-alpha", type=float, default=0.55)
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
    agent = _build_agent(args)

    if args.storyboard_json is not None:
        storyboard = _load_storyboard(args.storyboard_json)
        summary = agent.arrange_story_path(
            storyline=args.storyline,
            storyboard=storyboard,
            output_dir=args.output_dir,
        )
        out_storyboard = args.output_storyboard_json or (args.output_dir / "storyboard_path_orchestrated_mcts.json")
        storyboard.save_json(out_storyboard)
        (args.output_dir / "path_orchestrator_mcts_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("=== MCTS Path Orchestration Completed (Storyboard) ===")
        print(f"Storyboard in : {args.storyboard_json}")
        print(f"Storyboard out: {out_storyboard}")
        print(f"Path plan     : {args.output_dir / 'path_plan_mcts.json'}")
        return

    step_logs = _load_step_logs(args.narrative_json)
    candidates = _build_narrative_candidates(step_logs)
    summary = agent.arrange_candidate_path(
        storyline=args.storyline,
        candidates=candidates,
        id_field="node_id",
        text_field="text",
        output_list_field="ordered_node_ids",
        output_dir=args.output_dir,
    )
    ordered = [str(x) for x in summary.get("ordered_node_ids", [])]
    rank_map = {nid: i + 1 for i, nid in enumerate(ordered)}
    enriched = {
        **summary,
        "node_rank_map": rank_map,
        "source_narrative_json": str(args.narrative_json),
    }
    (args.output_dir / "narrative_path_plan_mcts.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== MCTS Path Orchestration Completed (Narrative) ===")
    print(f"Candidates : {len(candidates)}")
    print(f"Path plan  : {args.output_dir / 'narrative_path_plan_mcts.json'}")


if __name__ == "__main__":
    main()
