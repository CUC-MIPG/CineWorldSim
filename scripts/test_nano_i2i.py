from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.generation_backends import nano_banana_settings_from_env
from agents.http_utils import openai_client_kwargs, requests_get


def _extract_image_data(text: str) -> tuple[str | None, str | None]:
    m = re.search(r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)", text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _extract_image_url(text: str) -> str | None:
    m = re.search(r"https?://[^\s)\]\">]+\.(png|jpe?g|webp)", text, re.IGNORECASE)
    if not m:
        return None
    return m.group(0)


def run_i2i(input_image: Path, output_image: Path, prompt: str) -> Path:
    cfg = nano_banana_settings_from_env()
    if not cfg["api_key"]:
        raise RuntimeError("NANO_BANANA_API_KEY is missing")

    mime, _ = mimetypes.guess_type(str(input_image))
    if not mime:
        mime = "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(input_image.read_bytes()).decode('utf-8')}"

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        **openai_client_kwargs(),
    )
    resp = client.chat.completions.create(
        model=cfg["model_name"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        timeout=480,
    )

    if not resp.choices:
        raise RuntimeError("No choices returned by nano-banana")
    content = resp.choices[0].message.content
    text = content if isinstance(content, str) else str(content)
    print("[nano-i2i] response preview:", text[:300].replace("\n", " "))

    ext, b64 = _extract_image_data(text)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    if b64:
        suffix = f".{ext or 'png'}"
        out = output_image if output_image.suffix else output_image.with_suffix(suffix)
        out.write_bytes(base64.b64decode(b64))
        return out

    url = _extract_image_url(text)
    if url:
        r = requests_get(url, timeout=480)
        r.raise_for_status()
        out = output_image if output_image.suffix else output_image.with_suffix(".png")
        out.write_bytes(r.content)
        return out

    raise RuntimeError("No image data/url found in response")


def main() -> None:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Test nano-banana image-to-image")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=Path("/data1/AgentG/hhb/nano_banana/character_Cassidy.png"),
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("outputs/tool_unit_tests/nano_i2i_test.png"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Keep the same character identity and face. Convert the style to cinematic cyberpunk "
            "concept art, add neon rim light, rainy night background, and higher contrast details."
        ),
    )
    args = parser.parse_args()

    if not args.input_image.exists():
        raise RuntimeError(f"Input image not found: {args.input_image}")

    out = run_i2i(args.input_image, args.output_image, args.prompt)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"I2I output missing or empty: {out}")
    print(f"[nano-i2i] OK -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
