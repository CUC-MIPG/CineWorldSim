from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PROMPT_LIBRARY = ROOT / "prompt_library" / "default"

from agents.director_agent import DirectorAgent
from agents.logging_utils import configure_logging
from agents.schemas import Storyboard
from agents.shot_agent import ShotAgent


def _apply_max_scenes(storyboard: Storyboard, max_scenes: int | None) -> Storyboard:
    if not max_scenes or max_scenes <= 0:
        return storyboard
    sorted_ids = sorted(storyboard.scenes.keys())
    keep_ids = set(sorted_ids[:max_scenes])
    storyboard.scenes = {sid: storyboard.scenes[sid] for sid in sorted_ids if sid in keep_ids}
    storyboard.shots = {
        shot_id: shot
        for shot_id, shot in storyboard.shots.items()
        if shot.scene_id in keep_ids
    }
    return storyboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run first two agents only (Director -> Shot), without Video generation."
    )
    parser.add_argument("--storyline", type=str, required=True)
    parser.add_argument(
        "--prompt-library",
        type=Path,
        default=DEFAULT_PROMPT_LIBRARY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/first_two_agents_run"),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to keep after Director stage (sorted by scene_id).",
    )
    parser.add_argument("--max-shots-per-scene", type=int, default=8)
    parser.add_argument(
        "--scene-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="never",
        help="Scene image generation policy in Director stage.",
    )
    parser.add_argument(
        "--character-image-policy",
        type=str,
        choices=["auto", "always", "never"],
        default="never",
        help="Character image generation policy in Director stage.",
    )
    parser.add_argument(
        "--reference-assets-json",
        type=Path,
        default=None,
        help="Optional JSON file for character/scene multi-reference images.",
    )
    parser.add_argument(
        "--strict-online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail on LLM/API errors instead of fallback.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    director = DirectorAgent(
        prompt_library=args.prompt_library,
        strict_online=args.strict_online,
        retries=args.retries,
    )

    storyboard = director.build_storyboard(args.storyline)
    storyboard = _apply_max_scenes(storyboard, args.max_scenes)

    if args.reference_assets_json:
        storyboard = director.attach_reference_assets_json(
            storyboard=storyboard,
            reference_assets_json=args.reference_assets_json,
        )

    storyboard = director.ensure_character_images(
        storyboard=storyboard,
        image_dir=output_dir / "characters",
        image_policy=args.character_image_policy,
    )
    storyboard = director.ensure_scene_images(
        storyboard=storyboard,
        image_dir=output_dir / "images",
        image_policy=args.scene_image_policy,
    )

    director_storyboard_json = output_dir / "director_storyboard.json"
    storyboard.save_json(director_storyboard_json)

    shot_agent = ShotAgent(
        prompt_library=args.prompt_library,
        strict_online=args.strict_online,
        retries=args.retries,
    )
    storyboard = shot_agent.loop_storyboard_generation(
        storyboard=storyboard,
        max_shots_per_scene=args.max_shots_per_scene,
    )

    shot_storyboard_json = output_dir / "shot_storyboard.json"
    storyboard.save_json(shot_storyboard_json)

    latest_storyboard_json = output_dir / "storyboard.json"
    storyboard.save_json(latest_storyboard_json)

    print("=== First Two Agents Completed ===")
    print(f"Title: {storyboard.title}")
    print(f"Scenes: {len(storyboard.scenes)}")
    print(f"Shots: {len(storyboard.shots)}")
    print(f"Director output: {director_storyboard_json}")
    print(f"Shot output: {shot_storyboard_json}")
    print(f"Latest storyboard: {latest_storyboard_json}")


if __name__ == "__main__":
    main()
