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
        description="Run pipeline (Director/Shot/Video + optional PostProduction)"
    )
    parser.add_argument("--storyline", type=str, required=True)
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_PROMPT_LIBRARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/three_agents_run"),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to keep in workflow (sorted by scene_id).",
    )
    parser.add_argument("--max-shots-per-scene", type=int, default=8)
    parser.add_argument(
        "--scene-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
        help="auto: generate missing scene images only; always: always generate; never: do not generate scene images.",
    )
    parser.add_argument(
        "--character-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
        help="auto: generate missing character images only; always: always generate; never: do not generate character images.",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        choices=["reference", "keyframe"],
        default="reference",
        help="reference: existing reference-image mode; keyframe: keyframe pipeline (pseudocode placeholder + I2V placeholder).",
    )
    parser.add_argument(
        "--reference-assets-json",
        type=Path,
        default=None,
        help="Optional JSON file for character/scene multi-reference images.",
    )
    parser.add_argument(
        "--dry-run-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only generate video tasks and paths without calling video API.",
    )
    parser.add_argument(
        "--strict-online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail on LLM generation errors instead of local fallback.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--enable-story-supervisor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run StorySupervisorAgent to generate scene-level narration/dialogue/music planning.",
    )
    parser.add_argument(
        "--enable-post-production",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run PostProductionAgent after video stage (pseudocode tools for dubbing/bgm/narration/editing).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from output-dir/pipeline_state.json if present.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    storyboard = run_pipeline(
        storyline=args.storyline,
        prompt_library=args.prompt_library,
        output_root=args.output_dir,
        max_scenes=args.max_scenes,
        max_shots_per_scene=args.max_shots_per_scene,
        dry_run_video=args.dry_run_video,
        strict_online=args.strict_online,
        retries=args.retries,
        resume=args.resume,
        scene_image_policy=args.scene_image_policy,
        character_image_policy=args.character_image_policy,
        video_mode=args.video_mode,
        reference_assets_json=args.reference_assets_json,
        enable_story_supervisor=args.enable_story_supervisor,
        enable_post_production=args.enable_post_production,
    )

    print("=== Pipeline Completed ===")
    print(f"Title: {storyboard.title}")
    print(f"Scenes: {len(storyboard.scenes)}")
    print(f"Shots: {len(storyboard.shots)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
