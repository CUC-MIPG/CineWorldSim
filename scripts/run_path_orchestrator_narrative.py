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
from agents.path_orchestrator_agent import PathOrchestratorAgent


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


def _build_candidates(step_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            role_text = str(role.get("role_output_text", "") or "").strip()
            next_action = role.get("next_action", {})
            if isinstance(next_action, dict):
                act = str(next_action.get("action_type", "") or "").strip()
                target = str(next_action.get("target_character", "") or "").strip()
                if act:
                    role_text = f"{role_text} | action={act} target={target or 'none'}".strip()
            rows.append(
                {
                    "idx": idx,
                    "node_id": node_id,
                    "step": step,
                    "role_name": role_name,
                    "text": role_text,
                    "scene_summary": scene_summary,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PathOrchestrator on narrative step_logs and produce node sequence.")
    parser.add_argument("--storyline", type=str, required=True)
    parser.add_argument("--narrative-json", type=Path, required=True, help="narrative result.json or checkpoint json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_logs = _load_step_logs(args.narrative_json)
    candidates = _build_candidates(step_logs)

    agent = PathOrchestratorAgent(strict_online=args.strict_online, retries=args.retries)
    summary = agent.arrange_candidate_path(
        storyline=args.storyline,
        candidates=candidates,
        id_field="node_id",
        text_field="text",
        output_list_field="ordered_node_ids",
        output_dir=args.output_dir,
    )

    # Add rank map for easier visualization integration.
    ordered = [str(x) for x in summary.get("ordered_node_ids", [])]
    rank_map = {nid: i + 1 for i, nid in enumerate(ordered)}
    enriched = {
        **summary,
        "node_rank_map": rank_map,
        "source_narrative_json": str(args.narrative_json),
    }
    (args.output_dir / "narrative_path_plan.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Narrative Path Orchestration Completed ===")
    print(f"Candidates : {len(candidates)}")
    print(f"Path plan  : {args.output_dir / 'narrative_path_plan.json'}")


if __name__ == "__main__":
    main()

