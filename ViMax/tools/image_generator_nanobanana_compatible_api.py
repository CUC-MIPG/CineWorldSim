"""Nano Banana image generation through an OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, List

import aiohttp
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from interfaces.image_output import ImageOutput
from tools.image_orientation import ensure_landscape, landscape_guard_requested
from utils.rate_limiter import RateLimiter
from utils.retry import after_func

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


class NanoBananaAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        super().__init__(f"Nano Banana image generation failed with HTTP {status_code}: {payload}")


class NanoBananaResponseError(RuntimeError):
    """A transient or malformed successful response from the image provider."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, NanoBananaAPIError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    if isinstance(exc, NanoBananaResponseError):
        return True
    if isinstance(exc, ValueError) and "landscape" in str(exc).lower():
        return True
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


class ImageGeneratorNanobananaCompatibleAPI:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("NANO_BANANA_API_KEY", "")
        self.base_url = (base_url or os.environ.get("NANO_BANANA_BASE_URL", "https://api.xi-ai.cn/v1")).rstrip("/")
        self.model = model or os.environ.get("NANO_BANANA_MODEL_NAME", "nano-banana-2.1")
        self.timeout_sec = float(timeout_sec or os.environ.get("NANO_BANANA_TIMEOUT_SEC", "480"))
        self.rate_limiter = rate_limiter
        if not self.api_key:
            raise RuntimeError("NANO_BANANA_API_KEY is required for Nano Banana image generation")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_retryable),
        after=after_func,
        reraise=True,
    )
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] | None = None,
        aspect_ratio: str | None = "16:9",
        **kwargs: Any,
    ) -> ImageOutput:
        references = list(reference_image_paths or [])
        if not references and _is_blank_white_frame_prompt(prompt):
            width, height = _frame_dimensions(kwargs.get("size"), aspect_ratio)
            return ImageOutput(
                fmt="pil",
                ext="png",
                data=Image.new("RGB", (width, height), (255, 255, 255)),
            )
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        enforce_landscape = landscape_guard_requested(
            size=kwargs.get("size"), aspect_ratio=aspect_ratio,
            enforce_landscape=kwargs.get("enforce_landscape", True),
            allow_portrait=kwargs.get("allow_portrait", False),
        )
        request_prompt = prompt
        if _originality_guard_enabled():
            request_prompt = _add_originality_guard(request_prompt)
        if enforce_landscape:
            request_prompt += f"\n\nComposition requirement: create a landscape image with an approximate {aspect_ratio or '16:9'} aspect ratio; the width must be greater than the height."

        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        content: list[dict[str, Any]] = [{"type": "text", "text": request_prompt}]
        for path in references:
            if path and Path(path).is_file():
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(path)}})
        status, response = await _post_json(
            f"{self.base_url}/chat/completions", headers=self._headers(),
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "modalities": _response_modalities(),
            },
            timeout=timeout,
        )
        if status >= 400:
            raise NanoBananaAPIError(status, response)
        if not isinstance(response, dict):
            raise NanoBananaResponseError(
                "Nano Banana chat completion returned unexpected payload "
                f"type: {type(response).__name__}"
            )
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise NanoBananaResponseError(_describe_unrecognized_payload(response))
        choice = choices[0]
        message = choice.get("message") or {}
        try:
            image, extension = await _decode_image_response(
                message.get("content"), timeout, alternate_payload=message.get("images")
            )
        except ValueError as exc:
            message_keys = ",".join(map(str, message.keys())) if isinstance(message, dict) else "<not-an-object>"
            finish_reason = choice.get("finish_reason", "<missing>")
            raise NanoBananaResponseError(
                f"{exc}; completion_diagnostic=(finish_reason={finish_reason}, "
                f"message_keys={message_keys})"
            ) from exc
        if enforce_landscape:
            ensure_landscape(image)
        return ImageOutput(fmt="pil", ext=extension, data=image)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


async def _decode_image_response(
    payload: Any,
    timeout: aiohttp.ClientTimeout,
    alternate_payload: Any = None,
) -> tuple[Image.Image, str]:
    text = _content_to_text(payload)
    data_match = re.search(r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)", text)
    if data_match:
        media_type = "image/jpeg" if data_match.group(1) in {"jpeg", "jpg"} else f"image/{data_match.group(1)}"
        raw = base64.b64decode(data_match.group(2), validate=True)
    else:
        url_match = re.search(r"https?://[^\s)\]\\\">]+", text, re.IGNORECASE)
        image_url = None
        if url_match:
            candidate = url_match.group(0).rstrip(".,;:")
            if any(token in candidate.lower() for token in (".png", ".jpg", ".jpeg", ".webp", "/response_images/")):
                image_url = candidate
        if image_url:
            raw, media_type = await _download_image(image_url, timeout)
        elif alternate_payload is not None:
            return await _decode_image_response(alternate_payload, timeout)
        else:
            raise ValueError(_describe_unrecognized_payload(payload))
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        image = opened.copy()
    extension = {"image/jpeg": "jpg", "image/webp": "webp"}.get(media_type, "png")
    return image, extension


def _describe_unrecognized_payload(payload: Any) -> str:
    """Describe a missing image payload without exposing image data or URLs."""
    if not isinstance(payload, (dict, list, str)):
        return f"Nano Banana success response contains no supported image payload (payload_type={type(payload).__name__})"

    if isinstance(payload, str):
        excerpt = _safe_text_excerpt(payload)
        suffix = f"; response_text={excerpt!r}" if excerpt else ""
        return "Nano Banana chat completion contained no image data URI or recognized image URL" + suffix
    if isinstance(payload, list):
        block_types = [str(block.get("type", "<missing>")) for block in payload if isinstance(block, dict)]
        text = _safe_text_excerpt(_content_to_text(payload))
        parts = [f"block_count={len(payload)}", f"block_types={','.join(block_types) or '<none>'}"]
        if text:
            parts.append(f"response_text={text!r}")
        return "Nano Banana chat completion contained unsupported content blocks (" + "; ".join(parts) + ")"

    parts = [f"top_level_keys={','.join(map(str, payload.keys())) or '<none>'}"]
    choices = payload.get("choices")
    if isinstance(choices, list):
        parts.append(f"choices_count={len(choices)}")
        if choices and isinstance(choices[0], dict):
            parts.append(f"choice_item_keys={','.join(map(str, choices[0].keys())) or '<none>'}")
    data = payload.get("data")
    if isinstance(data, list):
        parts.append(f"data_count={len(data)}")
        if data and isinstance(data[0], dict):
            parts.append(f"data_item_keys={','.join(map(str, data[0].keys())) or '<none>'}")

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, (str, int, float)):
            parts.append(f"error_code={str(code)[:100]}")
        if isinstance(message, str):
            # Providers occasionally echo request details; redact credential-like values.
            safe_message = message.replace("\n", " ").replace("\r", " ")[:300]
            safe_message = re.sub(
                r"(?i)(bearer\s+)[^\s,;]+|((?:api[_-]?\s*key|token|secret)\s*[:=]\s*)[^\s,;]+",
                lambda match: (match.group(1) or match.group(2)) + "[REDACTED]",
                safe_message,
            )
            parts.append(f"error_message={safe_message}")
    return "Nano Banana success response contains no supported image payload (" + "; ".join(parts) + ")"


def _response_modalities() -> list[str]:
    raw = os.environ.get("NANO_BANANA_RESPONSE_MODALITIES", "TEXT,IMAGE")
    modalities = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return modalities or ["TEXT", "IMAGE"]


def _is_blank_white_frame_prompt(prompt: str) -> bool:
    text = " ".join(str(prompt).lower().split())
    has_white_frame = any(
        phrase in text
        for phrase in ("white field", "cut to white", "solid white", "blank white", "pure white")
    )
    has_empty_scene = any(
        phrase in text
        for phrase in ("no characters", "no visual elements", "nothing else visible", "entire frame")
    )
    return has_white_frame and has_empty_scene


def _frame_dimensions(size: Any, aspect_ratio: str | None) -> tuple[int, int]:
    if isinstance(size, str):
        match = re.fullmatch(r"\s*(\d+)\s*[x:]\s*(\d+)\s*", size.lower())
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            if width > height:
                return width, height
    if isinstance(aspect_ratio, str):
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", aspect_ratio)
        if match:
            ratio_width, ratio_height = int(match.group(1)), int(match.group(2))
            if ratio_width > ratio_height and ratio_height > 0:
                height = 900
                return round(height * ratio_width / ratio_height), height
    return 1600, 900


def _originality_guard_enabled() -> bool:
    return os.environ.get("NANO_BANANA_ORIGINALITY_GUARD", "true").strip().lower() not in {"0", "false", "no", "off"}


def _add_originality_guard(prompt: str) -> str:
    return (
        "Create a wholly original fictional image based on the scene description below. "
        "Do not reproduce or imitate any existing photograph, movie frame, artwork, "
        "advertisement, or other copyrighted image. Use generic visual details and "
        "invent an original composition.\n\n"
        + prompt
    )


def _safe_text_excerpt(text: str, limit: int = 400) -> str:
    """Keep provider diagnostics useful while excluding common secret-bearing values."""
    safe = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE DATA REDACTED]", text, flags=re.IGNORECASE)
    safe = re.sub(r"https?://[^\s]+", "[URL REDACTED]", safe, flags=re.IGNORECASE)
    safe = re.sub(
        r"(?i)(bearer\s+)[^\s,;]+|((?:api[_-]?\s*key|token|secret)\s*[:=]\s*)[^\s,;]+",
        lambda match: (match.group(1) or match.group(2)) + "[REDACTED]",
        safe,
    )
    safe = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", safe)
    safe = " ".join(safe.split())
    return safe[:limit] + ("..." if len(safe) > limit else "")


async def _post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: aiohttp.ClientTimeout) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers={**headers, "Content-Type": "application/json"}, json=payload) as response:
            return response.status, _parse_body(await response.text())


def _image_to_data_uri(path: str) -> str:
    media_type, _ = mimetypes.guess_type(path)
    if not media_type or not media_type.startswith("image/"):
        media_type = "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _content_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = []
        for block in payload:
            if isinstance(block, dict):
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
                image_url = block.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    parts.append(image_url["url"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(payload)


async def _download_image(url: str, timeout: aiohttp.ClientTimeout) -> tuple[bytes, str]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status >= 400:
                raise NanoBananaAPIError(response.status, await response.text())
            raw = await response.read()
            media_type = response.headers.get("Content-Type", "image/png").split(";", 1)[0]
            return raw, media_type


def _parse_body(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}
