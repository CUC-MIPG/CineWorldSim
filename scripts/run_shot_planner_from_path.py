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
from agents.shot_planner_agent import ShotPlannerAgent


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


def _load_storyline(path: Path, override: str) -> str:
    if override.strip():
        return override.strip()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        so = data.get("story_outline", {})
        if isinstance(so, dict) and str(so.get("story", "")).strip():
            return str(so.get("story", "")).strip()
        if str(data.get("prompt", "")).strip():
            return str(data.get("prompt", "")).strip()
    return "Generate a coherent cinematic sequence."


def _load_ordered_ids(path_plan_json: Path) -> list[str]:
    data = json.loads(path_plan_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    ordered = data.get("ordered_node_ids", [])
    if not isinstance(ordered, list):
        return []
    return [str(x).strip() for x in ordered if str(x).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile narrative path sequence into video-shot planning JSON.")
    parser.add_argument("--narrative-json", type=Path, required=True, help="Narrative result/checkpoint json.")
    parser.add_argument("--path-plan-json", type=Path, required=True, help="Path plan json with ordered_node_ids.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output shot plan json.")
    parser.add_argument("--storyline", type=str, default="", help="Optional storyline override.")
    parser.add_argument("--model", type=str, default=None, help="SHOT_PLANNER model override.")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    step_logs = _load_step_logs(args.narrative_json)
    ordered_node_ids = _load_ordered_ids(args.path_plan_json)
    storyline = _load_storyline(args.narrative_json, args.storyline)
    if not step_logs:
        raise RuntimeError(f"No step_logs found: {args.narrative_json}")
    if not ordered_node_ids:
        raise RuntimeError(f"No ordered_node_ids found: {args.path_plan_json}")

    agent = ShotPlannerAgent(
        model=args.model,
        strict_online=args.strict_online,
        retries=args.retries,
    )
    payload = agent.compile_narrative_path(
        storyline=storyline,
        step_logs=step_logs,
        ordered_node_ids=ordered_node_ids,
    )
    agent.save_json(payload, args.output_json)
    print(args.output_json)


if __name__ == "__main__":
    main()

