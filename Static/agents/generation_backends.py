from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


def _normalize_base_url(url: str) -> str:
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url.rstrip("/")


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _extract_image_data(content: Any) -> tuple[str | None, str | None]:
    text = _to_text(content)
    match = re.search(
        r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)",
        text,
    )
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _extract_image_url(content: Any) -> str | None:
    text = _to_text(content)
    candidates = re.findall(r"https?://[^\s)\]\">]+", text, re.IGNORECASE)
    for url in candidates:
        lower = url.lower()
        if any(k in lower for k in (".png", ".jpg", ".jpeg", ".webp", "/response_images/")):
            return url
    return None


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_to_data_uri(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported or unknown image format: {path}")
    b64 = _encode_image_base64(path)
    return f"data:{mime_type};base64,{b64}"


def generate_image_flux(
    *,
    prompt: str,
    output_path: Path,
    api_url: str,
    width: int = 480,
    height: int = 854,
    timeout_sec: int = 120,
) -> str | None:
    payload = {
        "output_path": str(output_path),
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
    }
    resp = requests.post(api_url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    img_b64 = data.get("result")
    if not img_b64:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(img_b64))
    return str(output_path)


def generate_image_nano_banana(
    *,
    prompt: str,
    output_path: Path,
    api_key: str,
    base_url: str,
    model_name: str,
    timeout_sec: int = 120,
) -> str | None:
    client = OpenAI(api_key=api_key, base_url=_normalize_base_url(base_url))
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        timeout=timeout_sec,
    )

    if not resp.choices:
        return None
    content = resp.choices[0].message.content

    image_ext, image_b64 = _extract_image_data(content)
    if image_b64:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{image_ext or 'png'}"
        final_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
        final_path.write_bytes(base64.b64decode(image_b64))
        return str(final_path)

    image_url = _extract_image_url(content)
    if image_url:
        out = requests.get(image_url, timeout=timeout_sec)
        out.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
        if not final_path.suffix:
            final_path = final_path.with_suffix(".png")
        final_path.write_bytes(out.content)
        return str(final_path)

    return None


def generate_video_magref(
    *,
    prompt: str,
    image_paths: list[str],
    output_video_path: str,
    api_url: str,
    timeout_sec: int | None = None,
) -> bool:
    images_b64 = [_encode_image_base64(p) for p in image_paths if Path(p).exists()]
    if not images_b64:
        return False

    payload = {"prompt": prompt, "images": images_b64}
    resp = requests.post(api_url, json=payload, timeout=_normalize_timeout(timeout_sec))
    resp.raise_for_status()
    data = resp.json()
    if "video_base64" not in data:
        raise RuntimeError("video_base64 missing")

    out = Path(output_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(data["video_base64"]))
    return True


def _as_bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_timeout(timeout_sec: int | float | None) -> float | None:
    if timeout_sec is None:
        return None
    try:
        t = float(timeout_sec)
    except Exception:
        return None
    if t <= 0:
        return None
    return t


def _optional_timeout_from_env(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"none", "null", "no", "off", "infinite", "inf"}:
        return None
    try:
        val = int(float(text))
    except Exception:
        return default
    if val <= 0:
        return None
    return val


def generate_video_wan_r2v(
    *,
    prompt: str,
    image_paths: list[str],
    output_video_path: str,
    api_key: str,
    base_http_api_url: str,
    model_name: str,
    resolution: str = "720P",
    duration: int = 5,
    prompt_extend: bool = True,
    watermark: bool = False,
    negative_prompt: str = "",
    seed: int = 12345,
    timeout_sec: int | None = None,
) -> bool:
    try:
        import dashscope
        from dashscope import VideoSynthesis
    except ImportError as exc:
        raise RuntimeError("dashscope is required for wan2.6-r2v; please install it.") from exc

    valid_images = [p for p in image_paths if Path(p).exists()]
    if not valid_images:
        return False

    reference_urls = [_image_to_data_uri(p) for p in valid_images]
    dashscope.base_http_api_url = base_http_api_url

    rsp = VideoSynthesis.call(
        api_key=api_key,
        model=model_name,
        prompt=prompt,
        extra_input={"reference_urls": reference_urls},
        resolution=resolution,
        duration=int(duration),
        prompt_extend=bool(prompt_extend),
        watermark=bool(watermark),
        negative_prompt=negative_prompt,
        seed=int(seed),
    )
    if rsp.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"wan2.6-r2v failed: status_code={rsp.status_code}, code={getattr(rsp, 'code', '')}, "
            f"message={getattr(rsp, 'message', '')}"
        )

    video_url = str(getattr(rsp.output, "video_url", "") or "")
    if not video_url:
        raise RuntimeError("wan2.6-r2v returned no video_url")

    download = requests.get(video_url, timeout=_normalize_timeout(timeout_sec))
    download.raise_for_status()
    out = Path(output_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(download.content)
    return True


def image_provider_from_env() -> str:
    return (os.getenv("IMAGE_GEN_PROVIDER") or "flux").strip().lower()


def video_provider_from_env() -> str:
    return (os.getenv("VIDEO_GEN_PROVIDER") or "magref").strip().lower()


def flux_settings_from_env() -> dict[str, Any]:
    return {
        "api_url": os.getenv("FLUX_API_URL", "http://127.0.0.1:5010/flux"),
        "width": int(os.getenv("FLUX_WIDTH", "480")),
        "height": int(os.getenv("FLUX_HEIGHT", "854")),
        "timeout_sec": int(os.getenv("FLUX_TIMEOUT_SEC", "120")),
    }


def nano_banana_settings_from_env() -> dict[str, Any]:
    return {
        "api_key": os.getenv("NANO_BANANA_API_KEY", ""),
        "base_url": os.getenv("NANO_BANANA_BASE_URL", "https://api.xi-ai.cn/v1"),
        "model_name": os.getenv("NANO_BANANA_MODEL_NAME", "gemini-2.5-flash-image-preview"),
        "timeout_sec": int(os.getenv("NANO_BANANA_TIMEOUT_SEC", "120")),
    }


def magref_settings_from_env() -> dict[str, Any]:
    return {
        "api_url": os.getenv("MAGREF_API_URL", "http://localhost:5020/magref_generate"),
        "timeout_sec": _optional_timeout_from_env("MAGREF_TIMEOUT_SEC", None),
    }


def wan_r2v_settings_from_env() -> dict[str, Any]:
    return {
        "api_key": os.getenv("WAN_R2V_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
        "base_http_api_url": os.getenv(
            "WAN_R2V_BASE_HTTP_API_URL",
            os.getenv("DASHSCOPE_BASE_HTTP_API_URL", "https://dashscope.aliyuncs.com/api/v1"),
        ),
        "model_name": os.getenv("WAN_R2V_MODEL_NAME", os.getenv("DASHSCOPE_WAN_MODEL", "wan2.6-r2v-flash")),
        "resolution": os.getenv("WAN_R2V_RESOLUTION", "720P"),
        "duration": int(os.getenv("WAN_R2V_DURATION", "5")),
        "prompt_extend": _as_bool(os.getenv("WAN_R2V_PROMPT_EXTEND"), True),
        "watermark": _as_bool(os.getenv("WAN_R2V_WATERMARK"), False),
        "negative_prompt": os.getenv("WAN_R2V_NEGATIVE_PROMPT", ""),
        "seed": int(os.getenv("WAN_R2V_SEED", "12345")),
        "timeout_sec": _optional_timeout_from_env("WAN_R2V_TIMEOUT_SEC", None),
    }


def generate_keyframe_pseudocode(
    *,
    prompt: str,
    scene_images: list[str],
    character_images: list[str],
    output_keyframe_path: str,
) -> str:
    """
    Pseudocode placeholder for keyframe generation.
    - If source images are available, copy the first one as the pseudo keyframe.
    - Otherwise, write a tiny placeholder PNG and a .todo text note.
    """
    out = Path(output_keyframe_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in (scene_images + character_images) if p and Path(p).exists()]
    if candidates:
        shutil.copyfile(candidates[0], out)
    else:
        # 1x1 transparent PNG
        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9oN8M0sAAAAASUVORK5CYII="
        )
        out.write_bytes(base64.b64decode(tiny_png_b64))

    todo = out.with_suffix(".keyframe.todo.txt")
    todo.write_text(
        json.dumps(
            {
                "tool": "generate_keyframe_pseudocode",
                "prompt": prompt,
                "scene_images": scene_images,
                "character_images": character_images,
                "keyframe_path": str(out),
                "note": "Replace with real keyframe generator backend.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out)


def generate_i2v_from_keyframe_pseudocode(
    *,
    keyframe_path: str,
    prompt: str,
    output_video_path: str,
) -> str:
    """
    Pseudocode placeholder for keyframe-to-video (I2V).
    Writes a TODO artifact and returns its path.
    """
    out = Path(output_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = out.with_suffix(".i2v.todo.txt")
    todo.write_text(
        json.dumps(
            {
                "tool": "generate_i2v_from_keyframe_pseudocode",
                "prompt": prompt,
                "keyframe_path": keyframe_path,
                "output_video_path": output_video_path,
                "note": "Replace with real I2V backend.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(todo)
