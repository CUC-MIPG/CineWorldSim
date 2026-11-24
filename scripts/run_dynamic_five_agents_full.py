from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from Dynamic.agents.orchestrator_dynamic import run_dynamic_pipeline

DEFAULT_PROMPT_LIBRARY = ROOT / "prompt_library" / "default"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dynamic Co-Director-style pre-production + CineWorldSim production/post-production pipeline."
    )
    parser.add_argument("--storyline", type=str, required=True)
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_PROMPT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_five_agents_full_run"))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-shots-per-scene", type=int, default=8)
    parser.add_argument("--scene-image-policy", type=str, choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--character-image-policy", type=str, choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--video-mode", type=str, choices=["reference", "keyframe", "text", "image", "first_last", "last_frame"], default="reference")
    parser.add_argument("--reference-assets-json", type=Path, default=None)
    parser.add_argument("--reuse-video-dir", type=Path, default=None)
    parser.add_argument("--dry-run-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--enable-story-supervisor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-post-production", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-narration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-storyline-refine-attempts", type=int, default=2)
    parser.add_argument("--storyline-score-threshold", type=int, default=85)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    storyboard = run_dynamic_pipeline(
        storyline=args.storyline,
        prompt_library=args.prompt_library,
        output_root=args.output_dir,
        max_scenes=args.max_scenes,
        max_shots_per_scene=args.max_shots_per_scene,
        dry_run_video=args.dry_run_video,
        strict_online=args.strict_online,
        retries=args.retries,
        scene_image_policy=args.scene_image_policy,
        character_image_policy=args.character_image_policy,
        video_mode=args.video_mode,
        reference_assets_json=args.reference_assets_json,
        reuse_video_dir=args.reuse_video_dir,
        enable_story_supervisor=args.enable_story_supervisor,
        enable_post_production=args.enable_post_production,
        include_narration=args.include_narration,
        max_storyline_refine_attempts=args.max_storyline_refine_attempts,
        storyline_score_threshold=args.storyline_score_threshold,
    )

    print("=== Dynamic Pipeline Completed ===")
    print(f"Title: {storyboard.title}")
    print(f"Scenes: {len(storyboard.scenes)}")
    print(f"Shots: {len(storyboard.shots)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
