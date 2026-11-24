from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Storyboard
from agents.logging_utils import configure_logging
from agents.story_supervisor_agent import StorySupervisorAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StorySupervisorAgent only (generate narration/dialogue/music planning per scene)"
    )
    parser.add_argument("--storyboard-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--shot-min-sec",
        type=float,
        default=None,
        help="Per-shot minimum duration in seconds. Default from SUPERVISOR_SHOT_MIN_SEC or 1.5.",
    )
    parser.add_argument(
        "--shot-max-sec",
        type=float,
        default=None,
        help="Per-shot maximum duration in seconds. Default from SUPERVISOR_SHOT_MAX_SEC or 8.0.",
    )
    parser.add_argument(
        "--scene-sec-per-shot",
        type=float,
        default=None,
        help="Target scene seconds per shot before normalization. Default from SUPERVISOR_SCENE_SEC_PER_SHOT or 3.2.",
    )
    parser.add_argument(
        "--duration-tolerance-sec",
        type=float,
        default=None,
        help="Allowed deviation between shot duration sum and scene duration target. Default from SUPERVISOR_DURATION_TOLERANCE_SEC or 0.5.",
    )
    parser.add_argument(
        "--strict-online",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, fail when LLM API errors occur. If false, use fallback planning.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    if not args.storyboard_json.exists():
        raise SystemExit(f"storyboard-json not found: {args.storyboard_json}")

    storyboard = Storyboard.from_dict(json.loads(args.storyboard_json.read_text(encoding="utf-8")))
    agent = StorySupervisorAgent(
        model=args.model,
        strict_online=args.strict_online,
        retries=args.retries,
        shot_duration_min_sec=args.shot_min_sec,
        shot_duration_max_sec=args.shot_max_sec,
        scene_duration_per_shot_sec=args.scene_sec_per_shot,
        duration_tolerance_sec=args.duration_tolerance_sec,
    )
    plan = agent.generate_story_plan(storyboard=storyboard, output_dir=args.output_dir)

    print("=== Story Supervisor Completed ===")
    print(f"Movie: {plan['movie_title']}")
    print(f"Scenes planned: {plan['scene_count']}")
    print(f"Output: {args.output_dir / 'story_supervisor_plan.json'}")


if __name__ == "__main__":
    main()
