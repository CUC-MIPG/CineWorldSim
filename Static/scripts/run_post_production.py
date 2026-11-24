from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.post_production_agent import PostProductionAgent
from agents.logging_utils import configure_logging
from agents.schemas import Storyboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PostProductionAgent only (pseudocode tools for dubbing/bgm/narration/editing)"
    )
    parser.add_argument("--storyboard-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--story-supervisor-plan",
        type=Path,
        default=None,
        help="Optional path to story_supervisor_plan.json to drive narration/dialogue/music design.",
    )
    parser.add_argument(
        "--final-video-path",
        type=Path,
        default=None,
        help="Optional final video path from video stage. If exists, used as edit input.",
    )
    parser.add_argument(
        "--include-narration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include narration track in final mix.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    if not args.storyboard_json.exists():
        raise SystemExit(f"storyboard-json not found: {args.storyboard_json}")

    storyboard = Storyboard.from_dict(json.loads(args.storyboard_json.read_text(encoding="utf-8")))
    agent = PostProductionAgent()
    supervision_plan = None
    if args.story_supervisor_plan:
        if not args.story_supervisor_plan.exists():
            raise SystemExit(f"story-supervisor-plan not found: {args.story_supervisor_plan}")
        supervision_plan = json.loads(args.story_supervisor_plan.read_text(encoding="utf-8"))

    summary = agent.process(
        storyboard=storyboard,
        output_dir=args.output_dir,
        final_video_path=str(args.final_video_path) if args.final_video_path else "",
        supervision_plan=supervision_plan,
        include_narration=args.include_narration,
    )

    print("=== Post-Production Completed ===")
    print(f"Movie: {summary['movie_title']}")
    print(f"Scenes processed: {summary['scene_count']}")
    print(f"Output: {args.output_dir}")
    print(f"Info JSON: {args.output_dir / 'post_production_info.json'}")


if __name__ == "__main__":
    main()
