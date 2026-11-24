from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.path_orchestrator_agent import PathOrchestratorAgent
from agents.schemas import Storyboard


def _load_storyboard(path: Path) -> Storyboard:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid storyboard JSON root: {path}")
    return Storyboard.from_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PathOrchestratorAgent on an existing storyboard.")
    parser.add_argument("--storyline", type=str, required=True, help="User storyline objective.")
    parser.add_argument("--storyboard-json", type=Path, required=True, help="Input storyboard JSON path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--output-storyboard-json",
        type=Path,
        default=None,
        help="Optional output storyboard path. Default: <output-dir>/storyboard_path_orchestrated.json",
    )
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    storyboard = _load_storyboard(args.storyboard_json)
    agent = PathOrchestratorAgent(strict_online=args.strict_online, retries=args.retries)
    summary = agent.arrange_story_path(
        storyline=args.storyline,
        storyboard=storyboard,
        output_dir=args.output_dir,
    )

    out_storyboard = args.output_storyboard_json or (args.output_dir / "storyboard_path_orchestrated.json")
    storyboard.save_json(out_storyboard)

    (args.output_dir / "path_orchestrator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Path Orchestration Completed ===")
    print(f"Storyboard in : {args.storyboard_json}")
    print(f"Storyboard out: {out_storyboard}")
    print(f"Path plan     : {args.output_dir / 'path_plan.json'}")


if __name__ == "__main__":
    main()

