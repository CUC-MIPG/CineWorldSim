from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.generation_backends import (
    generate_image_nano_banana,
    generate_video_wan_r2v,
    nano_banana_settings_from_env,
    wan_r2v_settings_from_env,
)


def test_nano(prompt: str, out_image: Path) -> Path:
    cfg = nano_banana_settings_from_env()
    if not cfg["api_key"]:
        raise RuntimeError("NANO_BANANA_API_KEY is missing in .env")

    print("[nano] model:", cfg["model_name"])
    print("[nano] base_url:", cfg["base_url"])
    res = generate_image_nano_banana(
        prompt=prompt,
        output_path=out_image,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model_name=cfg["model_name"],
        timeout_sec=cfg["timeout_sec"],
    )
    if not res:
        raise RuntimeError("nano_banana returned no image data/url")

    p = Path(res)
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"nano_banana output missing or empty: {p}")
    print(f"[nano] OK -> {p} ({p.stat().st_size} bytes)")
    return p


def test_wan(prompt: str, in_image: Path, out_video: Path) -> Path:
    cfg = wan_r2v_settings_from_env()
    if not cfg["api_key"]:
        raise RuntimeError("WAN_R2V_API_KEY is missing in .env")
    if not in_image.exists():
        raise RuntimeError(f"input image not found: {in_image}")

    print("[wan] model:", cfg["model_name"])
    print("[wan] base_http_api_url:", cfg["base_http_api_url"])
    ok = generate_video_wan_r2v(
        prompt=prompt,
        image_paths=[str(in_image)],
        output_video_path=str(out_video),
        api_key=cfg["api_key"],
        base_http_api_url=cfg["base_http_api_url"],
        model_name=cfg["model_name"],
        resolution=cfg["resolution"],
        duration=cfg["duration"],
        prompt_extend=cfg["prompt_extend"],
        watermark=cfg["watermark"],
        negative_prompt=cfg["negative_prompt"],
        seed=cfg["seed"],
        timeout_sec=cfg["timeout_sec"],
    )
    if not ok:
        raise RuntimeError("wan_r2v returned False (often means input image list invalid)")

    if not out_video.exists() or out_video.stat().st_size == 0:
        raise RuntimeError(f"wan_r2v output missing or empty: {out_video}")
    print(f"[wan] OK -> {out_video} ({out_video.stat().st_size} bytes)")
    return out_video


def main() -> None:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Standalone test for nano_banana image and wan2.6-r2v video tools")
    parser.add_argument(
        "--image-prompt",
        type=str,
        default="A cinematic banana-cat hero portrait, clean background, high detail",
    )
    parser.add_argument(
        "--video-prompt",
        type=str,
        default="The character waves and blinks gently, stable camera push-in, cinematic lighting",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=Path("outputs/tool_unit_tests/nano_image_test.png"),
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=Path("outputs/tool_unit_tests/wan_video_test.mp4"),
    )
    parser.add_argument(
        "--video-input-image",
        type=Path,
        default=None,
        help="Use existing image as wan input. If omitted, uses the generated nano image.",
    )
    parser.add_argument("--skip-nano", action="store_true")
    parser.add_argument("--skip-wan", action="store_true")
    args = parser.parse_args()

    args.image_output.parent.mkdir(parents=True, exist_ok=True)
    args.video_output.parent.mkdir(parents=True, exist_ok=True)

    image_path = args.video_input_image
    if not args.skip_nano:
        image_path = test_nano(args.image_prompt, args.image_output)
    elif image_path is None:
        raise RuntimeError("--skip-nano requires --video-input-image")

    if not args.skip_wan:
        test_wan(args.video_prompt, Path(image_path), args.video_output)


if __name__ == "__main__":
    main()

