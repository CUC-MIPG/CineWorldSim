from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import aiohttp

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter


_SUPPORTED_TASKS = {"t2va", "fl2va", "fl2va_last_frame", "fl2va_first_last", "ref2va"}
_TASK_ALIASES = {
    "text": "t2va",
    "text_to_video": "t2va",
    "t2v": "t2va",
    "image": "fl2va",
    "image_to_video": "fl2va",
    "i2v": "fl2va",
    "last_frame": "fl2va_last_frame",
    "first_last": "fl2va_first_last",
    "first_last_frame": "fl2va_first_last",
    "reference": "ref2va",
    "reference_to_video": "ref2va",
    "ref": "ref2va",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _resolve_task(task: str | None, reference_count: int) -> str:
    requested = (task or "auto").strip().lower()
    if requested == "auto":
        return "t2va" if reference_count == 0 else "fl2va" if reference_count == 1 else "ref2va"
    resolved = _TASK_ALIASES.get(requested, requested)
    if resolved not in _SUPPORTED_TASKS:
        raise ValueError(f"Unsupported MiniMax H3 task: {task}")
    # The CameraMaster vLLM endpoint implements last-frame variants as first-frame fl2va.
    return "fl2va" if resolved in {"fl2va_last_frame", "fl2va_first_last"} else resolved


def _request_fields(prompt: str, task: str, aspect_ratio: str) -> dict[str, str]:
    width = _env_int("MINIMAX_H3_WIDTH", 1024)
    height = _env_int("MINIMAX_H3_HEIGHT", 576)
    num_frames = _env_int("MINIMAX_H3_NUM_FRAMES", 124)
    fps = _env_int("MINIMAX_H3_FPS", 24)
    steps = _env_int("MINIMAX_H3_STEPS", 30)
    flow_shift = os.environ.get("MINIMAX_H3_VLLM_FLOW_SHIFT", "12")
    seed_value = os.environ.get("MINIMAX_H3_SEED", "42").strip()
    audio_flow_shift = _env_float("MINIMAX_H3_VLLM_AUDIO_FLOW_SHIFT", 3.0)
    duration = _env_float("MINIMAX_H3_VLLM_DURATION", 5.1667)
    extra_params: dict[str, Any] = {"task": task, "audio_flow_shift": audio_flow_shift}
    if duration > 0:
        extra_params["duration"] = duration

    fields = {
        "prompt": prompt,
        "width": str(width),
        "height": str(height),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "flow_shift": flow_shift,
        "extra_params": json.dumps(extra_params, ensure_ascii=False),
    }
    if seed_value.lower() not in {"", "none", "null", "unseeded"}:
        fields["seed"] = seed_value
    if task == "t2va":
        fields["aspect_ratio"] = aspect_ratio
    else:
        fields["num_frames"] = str(num_frames)
    return fields


async def _post_multipart(
    url: str,
    *,
    fields: dict[str, str],
    file_fields: list[tuple[str, str]],
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    form = aiohttp.FormData()
    for name, value in fields.items():
        form.add_field(name, value)

    # Keep file handles alive until aiohttp has finished consuming the multipart body.
    with ExitStack() as stack:
        for name, image_path in file_fields:
            path = Path(image_path)
            handle = stack.enter_context(path.open("rb"))
            form.add_field(
                name,
                handle,
                filename=path.name,
                content_type=mimetypes.guess_type(path.name)[0] or "image/jpeg",
            )
        client_timeout = aiohttp.ClientTimeout(total=timeout if timeout > 0 else None)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, data=form) as response:
                return response.status, dict(response.headers), await response.read()


class VideoGeneratorMinimaxH3VLLM:
    """Generate MiniMax H3 video through CameraMaster's synchronous vLLM API."""

    def __init__(
        self,
        api_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_url = api_url
        self.rate_limiter = rate_limiter
        # The local H3 server can time out when multiple long diffusion jobs
        # overlap. Rate limiting controls request starts, not in-flight work.
        self._generation_lock = asyncio.Lock()

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: list[str] | None = None,
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> VideoOutput:
        started = time.monotonic()
        references = [str(Path(path)) for path in (reference_image_paths or [])]
        task = _resolve_task(os.environ.get("MINIMAX_H3_TASK", "auto"), len(references))
        if task == "fl2va" and not references:
            raise ValueError("MiniMax H3 task fl2va requires a reference image")
        if task == "ref2va" and not references:
            raise ValueError("MiniMax H3 task ref2va requires at least one reference image")
        if any(not Path(path).is_file() for path in references):
            missing = next(path for path in references if not Path(path).is_file())
            raise FileNotFoundError(f"MiniMax H3 reference image does not exist: {missing}")

        if os.environ.get("MINIMAX_H3_VLLM_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
            raise RuntimeError("VideoGeneratorMinimaxH3VLLM requires MINIMAX_H3_VLLM_ENABLED=true")

        api_url = self.api_url or os.environ.get(
            "MINIMAX_H3_VLLM_API_URL",
            "http://127.0.0.1:9001/v1/videos/sync",
        )
        timeout = _env_float("MINIMAX_H3_VLLM_TIMEOUT_SEC", 3600.0)
        aspect_ratio = os.environ.get("MINIMAX_H3_VLLM_ASPECT_RATIO", aspect_ratio)
        fields = _request_fields(prompt, task, aspect_ratio)
        if task == "fl2va":
            file_fields = [("input_reference", references[0])]
        elif task == "ref2va":
            file_fields = [("input_references", path) for path in references]
        else:
            file_fields = []

        progress = kwargs.get("progress")
        if progress is not None:
            progress("video_create", "Submitting MiniMax H3 vLLM generation request", {
                "task": task,
                "reference_image_count": len(file_fields),
            })

        output_path = kwargs.get("output_video_path")
        report_path = Path(output_path).with_suffix(".minimax_h3_vllm.json") if output_path else None
        status = 0
        headers: dict[str, str] = {}
        body = b""
        error = None
        try:
            async with self._generation_lock:
                if self.rate_limiter is not None:
                    await self.rate_limiter.acquire()
                status, headers, body = await _post_multipart(
                    api_url,
                    fields=fields,
                    file_fields=file_fields,
                    timeout=timeout,
                )
                if not 200 <= status < 300:
                    detail = body.decode("utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"MiniMax H3 vLLM request failed with HTTP {status}: {detail}")
                if not body:
                    raise RuntimeError("MiniMax H3 vLLM returned an empty video response")
            return VideoOutput(fmt="bytes", ext="mp4", data=body)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report = {
                    "tool": "minimax_h3_vllm",
                    "ok": error is None and bool(body),
                    "task": task,
                    "api_url": api_url,
                    "method": "POST",
                    "status_code": status,
                    "content_type": headers.get("content-type", ""),
                    "request_fields": list(fields),
                    "reference_image_count": len(file_fields),
                    "bytes": len(body),
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
                if error:
                    report["error"] = error[-4000:]
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if progress is not None and error is None:
                progress("video_completed", "MiniMax H3 vLLM video received", {"task": task, "bytes": len(body)})
