from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.generation_backends import generate_video_minimax_h3, minimax_h3_settings_from_env


SKILL_ENV_KEYS = [
    "CAMERAMASTER_SKILL_ROOTS",
    "CAMERAMASTER_SKILL_POLICY_PATH",
    "DIRECTOR_SKILL_NAME",
    "SHOT_SKILL_NAME",
    "SUPERVISOR_SKILL_NAME",
    "VIDEO_SKILL_NAME",
]


def main() -> None:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Standalone MiniMax-H3 video provider test.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["text", "image", "reference", "first_last", "last_frame"],
        default="text",
        help="CineWorldSim video mode mapped to MiniMax-H3 task.",
    )
    parser.add_argument("--task", default=None, help="Optional raw MiniMax-H3 task override.")
    parser.add_argument("--image", type=Path, default=None, help="First image/keyframe for image or first_last mode.")
    parser.add_argument("--last-image", type=Path, default=None, help="Last image for first_last or last_frame mode.")
    parser.add_argument("--ref-image", action="append", default=[], help="Reference image for ref2va.")
    parser.add_argument("--ref-video", action="append", default=[], help="Reference video for ref2va.")
    parser.add_argument("--ref-audio", action="append", default=[], help="Reference audio for ref2va.")
    args = parser.parse_args()

    cfg = minimax_h3_settings_from_env()
    image_paths = []
    if args.image:
        image_paths.append(str(args.image))
    if args.last_image and args.mode == "first_last":
        image_paths.append(str(args.last_image))

    ok = generate_video_minimax_h3(
        prompt=args.prompt,
        image_paths=image_paths,
        output_video_path=str(args.output_video),
        video_mode=args.mode,
        task=args.task or cfg["task"],
        first_image_path=str(args.image) if args.image else "",
        last_image_path=str(args.last_image) if args.last_image else "",
        reference_image_paths=[str(x) for x in args.ref_image],
        reference_video_paths=[str(x) for x in args.ref_video],
        reference_audio_paths=[str(x) for x in args.ref_audio],
        model_dir=cfg["model_dir"],
        conda_env=cfg["conda_env"],
        script_path=cfg["script_path"],
        python_bin=cfg["python_bin"],
        cuda_visible_devices=cfg["cuda_visible_devices"],
        width=cfg["width"],
        height=cfg["height"],
        num_frames=cfg["num_frames"],
        steps=cfg["steps"],
        seed=cfg["seed"],
        fps=cfg["fps"],
        parallel_mode=cfg["parallel_mode"],
        device_map=cfg["device_map"],
        max_memory=cfg["max_memory"],
        text_encoder_placement=cfg["text_encoder_placement"],
        text_encoder_use_stream=cfg["text_encoder_use_stream"],
        pytorch_cuda_alloc_conf=cfg["pytorch_cuda_alloc_conf"],
        timeout_sec=cfg["timeout_sec"],
    )
    if not ok:
        raise SystemExit("MiniMax-H3 generation returned False")
    print(args.output_video)
    for key in SKILL_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
