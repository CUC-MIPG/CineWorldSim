from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PROMPT_LIBRARY = ROOT / "prompt_library" / "default"

from agents.orchestrator import run_pipeline
from agents.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume pipeline from checkpoint (Director/Shot/Video + optional PostProduction)"
    )
    parser.add_argument("--storyline", type=str, default="resume")
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_PROMPT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to keep in workflow (sorted by scene_id).",
    )
    parser.add_argument("--dry-run-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--scene-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
    )
    parser.add_argument(
        "--character-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        choices=["reference", "keyframe", "text", "image", "first_last", "last_frame"],
        default="reference",
    )
    parser.add_argument(
        "--reference-assets-json",
        type=Path,
        default=None,
        help="Optional JSON file for character/scene multi-reference images.",
    )
    parser.add_argument(
        "--enable-story-supervisor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-post-production",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    storyboard = run_pipeline(
        storyline=args.storyline,
        prompt_library=args.prompt_library,
        output_root=args.output_dir,
        max_scenes=args.max_scenes,
        dry_run_video=args.dry_run_video,
        strict_online=args.strict_online,
        retries=args.retries,
        resume=True,
        scene_image_policy=args.scene_image_policy,
        character_image_policy=args.character_image_policy,
        video_mode=args.video_mode,
        reference_assets_json=args.reference_assets_json,
        enable_story_supervisor=args.enable_story_supervisor,
        enable_post_production=args.enable_post_production,
    )

    print("=== Resume Completed ===")
    print(f"Title: {storyboard.title}")
    print(f"Scenes: {len(storyboard.scenes)}")
    print(f"Shots: {len(storyboard.shots)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
