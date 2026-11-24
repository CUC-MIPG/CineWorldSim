from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.video_agent import VideoAgent
from agents.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Process offline/queued video tasks from video_info.json")
    parser.add_argument("--task-json", type=Path, required=True, help="Path to video task JSON list")
    parser.add_argument(
        "--strict-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip tasks with no existing images when true",
    )
    parser.add_argument(
        "--strict-online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail on LLM/API errors instead of fallback",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)
    result = agent.process_task_file(args.task_json, strict_images=args.strict_images)

    print("=== Video Task Processing Completed ===")
    print(f"Task JSON: {args.task_json}")
    print(f"Total: {result['total']}")
    print(f"Success: {result['success']}")
    print(f"Failed: {result['failed']}")
    print(f"Skipped: {result['skipped']}")


if __name__ == "__main__":
    main()
