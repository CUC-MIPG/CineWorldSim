from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from .call_logger import new_call_id, text_summary, write_call_event
from .http_utils import openai_client_kwargs, requests_get, requests_post


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
    timeout_sec: int = 480,
) -> str | None:
    call_id = new_call_id("tool-flux")
    start = time.time()
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "flux_api",
            "status": "start",
            "api_url": api_url,
            "output_path": str(output_path),
            "width": int(width),
            "height": int(height),
            "timeout_sec": timeout_sec,
            "prompt_preview": text_summary(prompt),
        }
    )
    payload = {
        "output_path": str(output_path),
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
    }
    try:
        resp = requests_post(api_url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        img_b64 = data.get("result")
        if not img_b64:
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "tool",
                    "tool": "flux_api",
                    "status": "done",
                    "ok": False,
                    "elapsed_sec": round(time.time() - start, 3),
                    "output_path": str(output_path),
                    "reason": "missing result",
                }
            )
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(img_b64))
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux_api",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(output_path),
                "bytes": output_path.stat().st_size if output_path.exists() else 0,
            }
        )
        return str(output_path)
    except Exception as exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux_api",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(output_path),
                "error": text_summary(exc, 1200),
            }
        )
        raise


def generate_image_flux_local(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    conda_env: str,
    script_path: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    device: str = "auto",
    dtype: str = "bfloat16",
    seed: int | None = None,
    timeout_sec: int = 480,
    python_bin: str = "",
) -> str | None:
    call_id = new_call_id("tool-flux-local")
    start = time.time()
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "flux_local",
            "status": "start",
            "model_dir": model_dir,
            "conda_env": conda_env,
            "script_path": script_path,
            "output_path": str(output_path),
            "width": int(width),
            "height": int(height),
            "device": device,
            "timeout_sec": timeout_sec,
            "prompt_preview": text_summary(prompt),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = [python_bin, script_path] if python_bin else ["conda", "run", "-n", conda_env, "python", script_path]
    runtime_device = str(device or "auto")
    env = os.environ.copy()
    cuda_match = re.fullmatch(r"cuda:(\d+)", runtime_device.strip().lower())
    if cuda_match:
        env["CUDA_VISIBLE_DEVICES"] = cuda_match.group(1)
        runtime_device = "cuda:0"
    cmd = [
        *runner,
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--output-path",
        str(output_path),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--num-inference-steps",
        str(int(num_inference_steps)),
        "--guidance-scale",
        str(float(guidance_scale)),
        "--device",
        runtime_device,
        "--dtype",
        str(dtype or "bfloat16"),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_normalize_timeout(timeout_sec),
        check=False,
    )
    if proc.returncode != 0:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux_local",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(output_path),
                "returncode": proc.returncode,
                "stdout_tail": text_summary(proc.stdout[-1200:], 1200),
                "stderr_tail": text_summary(proc.stderr[-1600:], 1600),
            }
        )
        raise RuntimeError(
            "flux_local failed: "
            f"rc={proc.returncode}, stdout={proc.stdout[-800:]}, stderr={proc.stderr[-1200:]}"
        )
    if output_path.exists() and output_path.stat().st_size > 0:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux_local",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(output_path),
                "bytes": output_path.stat().st_size,
            }
        )
        return str(output_path)
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "flux_local",
            "status": "done",
            "ok": False,
            "elapsed_sec": round(time.time() - start, 3),
            "output_path": str(output_path),
            "reason": "missing output",
        }
    )
    return None


def generate_image_flux2_klein(
    *,
    prompt: str,
    output_path: Path,
    api_url: str,
    width: int = 1536,
    height: int = 864,
    num_inference_steps: int = 4,
    seed: int = 42,
    reference_image_paths: list[str] | None = None,
    timeout_sec: int = 1200,
) -> str | None:
    """Call the local FLUX.2 Klein multipart image service.

    The service accepts each reference image as a repeated ``images`` field and
    returns PNG bytes. The response is decoded and re-encoded through Pillow so
    downstream consumers always receive a valid PNG file.
    """
    call_id = new_call_id("tool-flux2-klein")
    start = time.time()
    valid_refs = [
        str(Path(path).resolve())
        for path in reference_image_paths or []
        if path and Path(path).exists()
    ]
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "flux2_klein",
            "status": "start",
            "api_url": api_url,
            "output_path": str(output_path),
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(num_inference_steps),
            "seed": int(seed),
            "reference_image_count": len(valid_refs),
            "timeout_sec": timeout_sec,
            "prompt_preview": text_summary(prompt),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt": prompt,
        "width": str(int(width)),
        "height": str(int(height)),
        "num_inference_steps": str(int(num_inference_steps)),
        "seed": str(int(seed)),
    }
    handles = []
    files = []
    try:
        for path in valid_refs:
            handle = open(path, "rb")
            handles.append(handle)
            mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            files.append(("images", (Path(path).name, handle, mime_type)))

        resp = requests_post(api_url, data=data, files=files, timeout=timeout_sec)
        resp.raise_for_status()
        try:
            from PIL import Image  # type: ignore

            with Image.open(io.BytesIO(resp.content)) as image:
                image.load()
                image.convert("RGB").save(output_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("FLUX.2 Klein returned non-image or invalid image bytes") from exc

        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux2_klein",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(output_path),
                "bytes": output_path.stat().st_size,
                "result_type": "png",
            }
        )
        return str(output_path)
    except Exception as exc:  # noqa: BLE001
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "flux2_klein",
                "status": "failed",
                "ok": False,
                "elapsed_sec": round(time.time() - start, 3),
                "api_url": api_url,
                "output_path": str(output_path),
                "error": text_summary(exc, 2000),
            }
        )
        raise
    finally:
        for handle in handles:
            handle.close()


def generate_image_nano_banana(
    *,
    prompt: str,
    output_path: Path,
    api_key: str,
    base_url: str,
    model_name: str,
    timeout_sec: int = 480,
    reference_image_paths: list[str] | None = None,
) -> str | None:
    call_id = new_call_id("tool-nano-banana")
    start = time.time()
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "nano_banana",
            "status": "start",
            "base_url": _normalize_base_url(base_url),
            "model": model_name,
            "output_path": str(output_path),
            "timeout_sec": timeout_sec,
            "reference_image_count": len(reference_image_paths or []),
            "prompt_preview": text_summary(prompt),
        }
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in reference_image_paths or []:
        if not path or not Path(path).exists():
            continue
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(path)}})

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=api_key,
            base_url=_normalize_base_url(base_url),
            **openai_client_kwargs(),
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            timeout=timeout_sec,
        )
        if not resp.choices:
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "tool",
                    "tool": "nano_banana",
                    "status": "done",
                    "ok": False,
                    "elapsed_sec": round(time.time() - start, 3),
                    "output_path": str(output_path),
                    "reason": "empty openai choices",
                }
            )
            return None
        content_text = resp.choices[0].message.content
    except Exception as first_exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "nano_banana",
                "status": "fallback",
                "base_url": _normalize_base_url(base_url),
                "model": model_name,
                "elapsed_sec": round(time.time() - start, 3),
                "reason": text_summary(first_exc, 1000),
            }
        )
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
        }
        try:
            resp = requests_post(
                f"{_normalize_base_url(base_url)}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "tool",
                    "tool": "nano_banana",
                    "status": "failed",
                    "ok": False,
                    "elapsed_sec": round(time.time() - start, 3),
                    "base_url": _normalize_base_url(base_url),
                    "model": model_name,
                    "output_path": str(output_path),
                    "error": text_summary(exc, 1200),
                }
            )
            raise
        choices = data.get("choices") or []
        if not choices:
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "tool",
                    "tool": "nano_banana",
                    "status": "done",
                    "ok": False,
                    "elapsed_sec": round(time.time() - start, 3),
                    "output_path": str(output_path),
                    "reason": "empty http choices",
                }
            )
            return None
        content_text = choices[0].get("message", {}).get("content", "")

    image_ext, image_b64 = _extract_image_data(content_text)
    if image_b64:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{image_ext or 'png'}"
        final_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
        final_path.write_bytes(base64.b64decode(image_b64))
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "nano_banana",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(final_path),
                "bytes": final_path.stat().st_size if final_path.exists() else 0,
                "result_type": "base64",
            }
        )
        return str(final_path)

    image_url = _extract_image_url(content_text)
    if image_url:
        out = requests_get(image_url, timeout=timeout_sec)
        out.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
        if not final_path.suffix:
            final_path = final_path.with_suffix(".png")
        final_path.write_bytes(out.content)
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "nano_banana",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_path": str(final_path),
                "bytes": final_path.stat().st_size if final_path.exists() else 0,
                "result_type": "url",
            }
        )
        return str(final_path)

    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "nano_banana",
            "status": "done",
            "ok": False,
            "elapsed_sec": round(time.time() - start, 3),
            "output_path": str(output_path),
            "reason": "no image data or image url in response",
        }
    )
    return None


def generate_video_magref(
    *,
    prompt: str,
    image_paths: list[str],
    output_video_path: str,
    api_url: str,
    timeout_sec: int | None = None,
) -> bool:
    call_id = new_call_id("tool-magref")
    start = time.time()
    images_b64 = [_encode_image_base64(p) for p in image_paths if Path(p).exists()]
    if not images_b64:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "magref",
                "status": "done",
                "ok": False,
                "output_video_path": output_video_path,
                "reason": "no valid reference images",
            }
        )
        return False

    payload = {"prompt": prompt, "images": images_b64}
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "magref",
            "status": "start",
            "api_url": api_url,
            "output_video_path": output_video_path,
            "reference_image_count": len(images_b64),
            "timeout_sec": timeout_sec,
            "prompt_preview": text_summary(prompt),
        }
    )
    try:
        resp = requests_post(api_url, json=payload, timeout=_normalize_timeout(timeout_sec))
        resp.raise_for_status()
        data = resp.json()
        if "video_base64" not in data:
            raise RuntimeError("video_base64 missing")

        out = Path(output_video_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data["video_base64"]))
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "magref",
                "status": "done",
                "ok": True,
                "elapsed_sec": round(time.time() - start, 3),
                "output_video_path": str(out),
                "bytes": out.stat().st_size if out.exists() else 0,
            }
        )
        return True
    except Exception as exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "magref",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "output_video_path": output_video_path,
                "error": text_summary(exc, 1200),
            }
        )
        raise


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

    download = requests_get(video_url, timeout=_normalize_timeout(timeout_sec))
    download.raise_for_status()
    out = Path(output_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(download.content)
    return True


def _list_from_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _minimax_task_from_mode(task: str | None, video_mode: str | None, image_paths: list[str]) -> str:
    explicit = (task or "").strip().lower()
    if explicit and explicit != "auto":
        aliases = {
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
        explicit = aliases.get(explicit, explicit)
        if explicit not in {"t2va", "fl2va", "fl2va_last_frame", "fl2va_first_last", "ref2va"}:
            raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
        return explicit

    mode = (video_mode or "").strip().lower()
    if mode in {"text", "t2va", "text_to_video", "t2v"}:
        return "t2va"
    if mode in {"image", "i2v", "fl2va", "image_to_video", "keyframe"}:
        return "fl2va"
    if mode in {"first_last", "first_last_frame", "fl2va_first_last"}:
        return "fl2va_first_last"
    if mode in {"last_frame", "fl2va_last_frame"}:
        return "fl2va_last_frame"
    if mode in {"reference", "ref2va", "reference_to_video"}:
        return "ref2va"
    return "ref2va" if image_paths else "t2va"



def generate_video_minimax_h3_vllm(
    *,
    prompt: str,
    output_video_path: str,
    image_paths: list[str] | None = None,
    video_mode: str = "reference",
    task: str | None = None,
    first_image_path: str = "",
    last_image_path: str = "",
    reference_image_paths: list[str] | None = None,
    reference_audio_paths: list[str] | None = None,
    upload_reference_audio: bool = False,
    api_url: str = "http://127.0.0.1:9001/v1/videos/sync",
    width: int = 1024,
    height: int = 576,
    num_frames: int = 243,
    fps: int = 24,
    steps: int = 30,
    flow_shift: int | float = 12,
    seed: int = 42,
    duration: float | None = None,
    audio_flow_shift: float = 3.0,
    timeout_sec: int = 3600,
    aspect_ratio: str = "16:9",
) -> bool:
    num_frames = 243
    call_id = new_call_id("tool-minimax-vllm")
    start = time.time()
    valid_images = [str(Path(p).resolve()) for p in image_paths or [] if p and Path(p).exists()]
    valid_ref_images = [str(Path(p).resolve()) for p in reference_image_paths or [] if p and Path(p).exists()]
    valid_ref_audios = [str(Path(p).resolve()) for p in reference_audio_paths or [] if p and Path(p).exists()]
    first_image = str(Path(first_image_path).resolve()) if first_image_path and Path(first_image_path).exists() else ""
    if not first_image and valid_images:
        first_image = valid_images[0]

    resolved_task = _minimax_task_from_mode(task, video_mode, valid_images + valid_ref_images)
    if resolved_task == "fl2va" and not first_image:
        raise ValueError("MiniMax-H3 vLLM fl2va requires input_reference/first_image_path.")
    if resolved_task == "ref2va" and not (valid_ref_images or valid_images):
        raise ValueError("MiniMax-H3 vLLM ref2va requires at least one input_references image.")
    if resolved_task in {"fl2va_first_last", "fl2va_last_frame"}:
        # The vLLM sync API examples expose fl2va as first-frame image-to-video.
        resolved_task = "fl2va"

    out = Path(output_video_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    headers_path = out.with_suffix(".minimax_h3_vllm.headers.txt")
    report_path = out.with_suffix(".minimax_h3_vllm.json")

    extra_params = {"task": resolved_task, "audio_flow_shift": float(audio_flow_shift)}
    if duration is not None and float(duration) > 0:
        extra_params["duration"] = float(duration)
    data: dict[str, str] = {
        "prompt": prompt,
        "width": str(int(width)),
        "height": str(int(height)),
        "fps": str(int(fps)),
        "num_inference_steps": str(int(steps)),
        "flow_shift": str(flow_shift),
        "seed": str(int(seed)),
        "extra_params": json.dumps(extra_params, ensure_ascii=False),
    }
    if resolved_task == "t2va" and aspect_ratio:
        data["aspect_ratio"] = aspect_ratio
    else:
        data["num_frames"] = str(int(num_frames))

    open_files = []
    files = []
    try:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "minimax_h3_vllm",
                "status": "start",
                "api_url": api_url,
                "task": resolved_task,
                "video_mode": video_mode,
                "output_video_path": str(out),
                "width": int(width),
                "height": int(height),
                "num_frames": int(num_frames),
                "fps": int(fps),
                "steps": int(steps),
                "timeout_sec": timeout_sec,
                "reference_image_count": len(valid_ref_images or valid_images),
                "reference_audio_count": len(valid_ref_audios),
                "upload_reference_audio": bool(upload_reference_audio),
                "prompt_preview": text_summary(prompt),
            }
        )
        if resolved_task == "fl2va":
            fh = open(first_image, "rb")
            open_files.append(fh)
            files.append(("input_reference", (Path(first_image).name, fh, mimetypes.guess_type(first_image)[0] or "image/jpeg")))
        elif resolved_task == "ref2va":
            for path in valid_ref_images or valid_images:
                fh = open(path, "rb")
                open_files.append(fh)
                files.append(("input_references", (Path(path).name, fh, mimetypes.guess_type(path)[0] or "image/jpeg")))
            if upload_reference_audio:
                for path in valid_ref_audios:
                    fh = open(path, "rb")
                    open_files.append(fh)
                    files.append(("input_references", (Path(path).name, fh, mimetypes.guess_type(path)[0] or "audio/wav")))

        resp = requests_post(api_url, data=data, files=files or None, timeout=timeout_sec)
        headers_path.write_text("".join(f"{k}: {v}\n" for k, v in resp.headers.items()), encoding="utf-8")
        ok = 200 <= resp.status_code < 300
        content_type = resp.headers.get("content-type", "")
        report = {
            "tool": "minimax_h3_vllm",
            "ok": ok,
            "task": resolved_task,
            "video_mode": video_mode,
            "api_url": api_url,
            "output_video_path": str(out),
            "status_code": resp.status_code,
            "content_type": content_type,
            "headers_path": str(headers_path),
            "data": data,
            "reference_images": valid_ref_images or valid_images,
            "reference_audios": valid_ref_audios,
            "upload_reference_audio": bool(upload_reference_audio),
        }
        if not ok:
            report["body_tail"] = resp.text[-4000:]
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            resp.raise_for_status()
        out.write_bytes(resp.content)
        report["bytes"] = out.stat().st_size if out.exists() else 0
        report["ok"] = out.exists() and out.stat().st_size > 0
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "minimax_h3_vllm",
                "status": "done",
                "ok": bool(report["ok"]),
                "elapsed_sec": round(time.time() - start, 3),
                "task": resolved_task,
                "output_video_path": str(out),
                "status_code": resp.status_code,
                "bytes": report.get("bytes", 0),
                "report_path": str(report_path),
            }
        )
        return bool(report["ok"])
    except Exception as exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "minimax_h3_vllm",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "task": resolved_task,
                "output_video_path": str(out),
                "error": text_summary(exc, 1600),
                "report_path": str(report_path),
            }
        )
        raise
    finally:
        for fh in open_files:
            fh.close()

def generate_video_minimax_h3(
    *,
    prompt: str,
    output_video_path: str,
    image_paths: list[str] | None = None,
    video_mode: str = "reference",
    task: str | None = None,
    first_image_path: str = "",
    last_image_path: str = "",
    reference_image_paths: list[str] | None = None,
    reference_video_paths: list[str] | None = None,
    reference_audio_paths: list[str] | None = None,
    model_dir: str,
    conda_env: str,
    script_path: str,
    python_bin: str = "",
    cuda_visible_devices: str = "",
    width: int = 768,
    height: int = 768,
    num_frames: int = 243,
    steps: int = 30,
    seed: int = 42,
    fps: int = 24,
    parallel_mode: str = "single_offload",
    device_map: str = "balanced",
    max_memory: list[str] | None = None,
    text_encoder_placement: str = "offload",
    text_encoder_use_stream: bool = False,
    offload_folder: str = "",
    runtime_device_index: int = 0,
    vae_parallel_mode: str = "none",
    vae_placement: str = "offload",
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    reference_resize_width: int = 0,
    reference_resize_height: int = 0,
    pytorch_cuda_alloc_conf: str = "expandable_segments:True",
    timeout_sec: int | None = None,
) -> bool:
    num_frames = 243
    call_id = new_call_id("tool-minimax-script")
    start = time.time()
    valid_images = [str(Path(p).resolve()) for p in image_paths or [] if p and Path(p).exists()]
    valid_ref_images = [str(Path(p).resolve()) for p in reference_image_paths or [] if p and Path(p).exists()]
    valid_ref_videos = [str(Path(p).resolve()) for p in reference_video_paths or [] if p and Path(p).exists()]
    valid_ref_audios = [str(Path(p).resolve()) for p in reference_audio_paths or [] if p and Path(p).exists()]
    first_image = str(Path(first_image_path).resolve()) if first_image_path and Path(first_image_path).exists() else ""
    last_image = str(Path(last_image_path).resolve()) if last_image_path and Path(last_image_path).exists() else ""
    if not first_image and valid_images:
        first_image = valid_images[0]
    if not last_image and len(valid_images) >= 2:
        last_image = valid_images[-1]

    resolved_task = _minimax_task_from_mode(task, video_mode, valid_images + valid_ref_images)
    if resolved_task == "fl2va" and not first_image:
        raise ValueError("MiniMax-H3 fl2va requires a first image/keyframe.")
    if resolved_task == "fl2va_last_frame" and not last_image:
        raise ValueError("MiniMax-H3 fl2va_last_frame requires a last image.")
    if resolved_task == "fl2va_first_last" and (not first_image or not last_image):
        raise ValueError("MiniMax-H3 fl2va_first_last requires both first and last images.")
    if resolved_task == "ref2va" and not (valid_ref_images or valid_ref_videos or valid_ref_audios or valid_images):
        raise ValueError("MiniMax-H3 ref2va requires at least one image/video/audio reference.")

    out = Path(output_video_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.parent / f"{out.stem}_minimax_h3"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = [python_bin, script_path] if python_bin else ["conda", "run", "-n", conda_env, "python", script_path]
    cmd = [
        *runner,
        "--model-dir",
        str(Path(model_dir).resolve()),
        "--output-dir",
        str(run_dir),
        "--task",
        resolved_task,
        "--prompt",
        prompt,
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--num-frames",
        str(int(num_frames)),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--fps",
        str(int(fps)),
        "--parallel-mode",
        str(parallel_mode),
        "--device-map",
        str(device_map),
        "--text-encoder-placement",
        str(text_encoder_placement),
    ]
    for item in max_memory or []:
        cmd.extend(["--max-memory", str(item)])
    if text_encoder_use_stream:
        cmd.append("--text-encoder-use-stream")
    cmd.extend(["--runtime-device-index", str(int(runtime_device_index))])
    if offload_folder:
        cmd.extend(["--offload-folder", str(Path(offload_folder).resolve())])
    if vae_parallel_mode and vae_parallel_mode != "none":
        cmd.extend(["--vae-parallel-mode", str(vae_parallel_mode)])
    if vae_placement:
        cmd.extend(["--vae-placement", str(vae_placement)])
    if vae_use_tiling:
        cmd.append("--vae-use-tiling")
    if vae_use_slicing:
        cmd.append("--vae-use-slicing")
    if reference_resize_width and reference_resize_height:
        cmd.extend(["--reference-resize-width", str(int(reference_resize_width))])
        cmd.extend(["--reference-resize-height", str(int(reference_resize_height))])

    if resolved_task in {"fl2va", "fl2va_first_last"}:
        cmd.extend(["--image", first_image])
    if resolved_task in {"fl2va_last_frame", "fl2va_first_last"}:
        cmd.extend(["--last-image", last_image])
    if resolved_task == "ref2va":
        for path in valid_ref_images or valid_images:
            cmd.extend(["--ref-image", path])
        for path in valid_ref_videos:
            cmd.extend(["--ref-video", path])
        for path in valid_ref_audios:
            cmd.extend(["--ref-audio", path])

    env = os.environ.copy()
    if cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", pytorch_cuda_alloc_conf or "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", env["PYTORCH_CUDA_ALLOC_CONF"])

    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "minimax_h3",
            "status": "start",
            "task": resolved_task,
            "video_mode": video_mode,
            "output_video_path": str(out),
            "run_dir": str(run_dir),
            "script_path": script_path,
            "model_dir": model_dir,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "width": int(width),
            "height": int(height),
            "num_frames": int(num_frames),
            "fps": int(fps),
            "steps": int(steps),
            "timeout_sec": timeout_sec,
            "reference_image_count": len(valid_ref_images or valid_images),
            "reference_video_count": len(valid_ref_videos),
            "reference_audio_count": len(valid_ref_audios),
            "prompt_preview": text_summary(prompt),
        }
    )
    proc = subprocess.run(
        cmd,
        cwd=str(Path(script_path).resolve().parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_normalize_timeout(timeout_sec),
        check=False,
    )

    source_candidates = [
        run_dir / f"minimax_h3_{resolved_task}_muxed.mp4",
        run_dir / f"minimax_h3_{resolved_task}_video.mp4",
    ]
    source = next((p for p in source_candidates if p.exists() and p.stat().st_size > 0), None)
    report = {
        "tool": "minimax_h3",
        "ok": proc.returncode == 0 and source is not None,
        "task": resolved_task,
        "video_mode": video_mode,
        "output_video_path": str(out),
        "run_dir": str(run_dir),
        "source_video_path": str(source) if source else "",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "cmd": cmd,
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "PYTORCH_ALLOC_CONF": env.get("PYTORCH_ALLOC_CONF", ""),
        },
    }
    out.with_suffix(".minimax_h3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "minimax_h3",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "output_video_path": str(out),
                "returncode": proc.returncode,
                "stdout_tail": text_summary(proc.stdout[-1200:], 1200),
                "stderr_tail": text_summary(proc.stderr[-1800:], 1800),
                "report_path": str(out.with_suffix(".minimax_h3.json")),
            }
        )
        raise RuntimeError(
            "minimax_h3 failed: "
            f"rc={proc.returncode}, stdout={proc.stdout[-1000:]}, stderr={proc.stderr[-1600:]}"
        )
    if source is None:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "tool",
                "tool": "minimax_h3",
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "output_video_path": str(out),
                "returncode": proc.returncode,
                "reason": f"missing mp4 in {run_dir}",
                "report_path": str(out.with_suffix(".minimax_h3.json")),
            }
        )
        raise RuntimeError(f"minimax_h3 did not produce mp4 in {run_dir}")

    shutil.copyfile(source, out)
    write_call_event(
        {
            "call_id": call_id,
            "kind": "tool",
            "tool": "minimax_h3",
            "status": "done",
            "ok": True,
            "elapsed_sec": round(time.time() - start, 3),
            "output_video_path": str(out),
            "source_video_path": str(source),
            "bytes": out.stat().st_size if out.exists() else 0,
            "report_path": str(out.with_suffix(".minimax_h3.json")),
        }
    )
    return out.exists() and out.stat().st_size > 0


def image_provider_from_env() -> str:
    return (os.getenv("IMAGE_GEN_PROVIDER") or "flux").strip().lower()


def video_provider_from_env() -> str:
    return (os.getenv("VIDEO_GEN_PROVIDER") or "magref").strip().lower()


def flux_settings_from_env() -> dict[str, Any]:
    return {
        "api_url": os.getenv("FLUX_API_URL", "http://127.0.0.1:5010/flux"),
        "width": int(os.getenv("FLUX_WIDTH", "480")),
        "height": int(os.getenv("FLUX_HEIGHT", "854")),
        "timeout_sec": int(os.getenv("FLUX_TIMEOUT_SEC", "480")),
    }


def flux_local_settings_from_env() -> dict[str, Any]:
    seed_raw = os.getenv("FLUX_LOCAL_SEED", "").strip()
    seed: int | None = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except Exception:
            seed = None
    return {
        "model_dir": os.getenv("FLUX_LOCAL_MODEL_DIR", "/data1/AgentG/hhb/FLUX/FLUX.2-klein-9B"),
        "conda_env": os.getenv("FLUX_LOCAL_CONDA_ENV", "FLUX"),
        "script_path": os.getenv(
            "FLUX_LOCAL_SCRIPT",
            "/data1/AgentG/hhb/CameraMaster/tool_hub/code/image_generation/providers/flux_local_infer.py",
        ),
        "python_bin": os.getenv("FLUX_LOCAL_PYTHON_BIN", "").strip(),
        "width": int(os.getenv("FLUX_LOCAL_WIDTH", os.getenv("FLUX_WIDTH", "1024"))),
        "height": int(os.getenv("FLUX_LOCAL_HEIGHT", os.getenv("FLUX_HEIGHT", "1024"))),
        "num_inference_steps": int(os.getenv("FLUX_LOCAL_NUM_INFERENCE_STEPS", "4")),
        "guidance_scale": float(os.getenv("FLUX_LOCAL_GUIDANCE_SCALE", "1.0")),
        "device": os.getenv("FLUX_LOCAL_DEVICE", "auto"),
        "dtype": os.getenv("FLUX_LOCAL_DTYPE", "bfloat16"),
        "seed": seed,
        "timeout_sec": int(os.getenv("FLUX_LOCAL_TIMEOUT_SEC", os.getenv("FLUX_TIMEOUT_SEC", "480"))),
    }


def nano_banana_settings_from_env() -> dict[str, Any]:
    return {
        "api_key": os.getenv("NANO_BANANA_API_KEY", ""),
        "base_url": os.getenv("NANO_BANANA_BASE_URL", "https://api.xi-ai.cn/v1"),
        "model_name": os.getenv("NANO_BANANA_MODEL_NAME", "gemini-2.5-flash-image-preview"),
        "timeout_sec": int(os.getenv("NANO_BANANA_TIMEOUT_SEC", "480")),
    }


def flux2_klein_settings_from_env() -> dict[str, Any]:
    seed_raw = os.getenv("FLUX2_KLEIN_SEED", "42").strip()
    try:
        seed = int(seed_raw)
    except Exception:
        seed = 42
    return {
        "api_url": os.getenv("FLUX2_KLEIN_API_URL", "http://127.0.0.1:8000/generate"),
        "health_url": os.getenv("FLUX2_KLEIN_HEALTH_URL", "http://127.0.0.1:8000/health"),
        "width": int(os.getenv("FLUX2_KLEIN_WIDTH", "1536")),
        "height": int(os.getenv("FLUX2_KLEIN_HEIGHT", "864")),
        "num_inference_steps": int(os.getenv("FLUX2_KLEIN_STEPS", "4")),
        "seed": seed,
        "timeout_sec": int(os.getenv("FLUX2_KLEIN_TIMEOUT_SEC", "1200")),
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


def minimax_h3_vllm_settings_from_env() -> dict[str, Any]:
    return {
        "enabled": _as_bool(os.getenv("MINIMAX_H3_VLLM_ENABLED"), True),
        "api_url": os.getenv("MINIMAX_H3_VLLM_API_URL", "http://127.0.0.1:9001/v1/videos/sync"),
        "width": int(os.getenv("MINIMAX_H3_WIDTH", "1024")),
        "height": int(os.getenv("MINIMAX_H3_HEIGHT", "576")),
        "num_frames": 243,
        "fps": int(os.getenv("MINIMAX_H3_FPS", "24")),
        "steps": int(os.getenv("MINIMAX_H3_STEPS", "30")),
        "flow_shift": float(os.getenv("MINIMAX_H3_VLLM_FLOW_SHIFT", "12")),
        "seed": int(os.getenv("MINIMAX_H3_SEED", "42")),
        "duration": 10.0,
        "audio_flow_shift": float(os.getenv("MINIMAX_H3_VLLM_AUDIO_FLOW_SHIFT", "3.0")),
        "upload_reference_audio": _as_bool(os.getenv("MINIMAX_H3_VLLM_UPLOAD_REFERENCE_AUDIO"), False),
        "timeout_sec": int(os.getenv("MINIMAX_H3_VLLM_TIMEOUT_SEC", "3600")),
        "aspect_ratio": os.getenv("MINIMAX_H3_VLLM_ASPECT_RATIO", "16:9"),
    }


def minimax_h3_settings_from_env() -> dict[str, Any]:
    return {
        "model_dir": os.getenv("MINIMAX_H3_MODEL_DIR", "/data1/AgentG/hhb/minimax_h3/MiniMax-H3"),
        "conda_env": os.getenv("MINIMAX_H3_CONDA_ENV", "minimaxh3hf"),
        "script_path": os.getenv("MINIMAX_H3_SCRIPT", "/data1/AgentG/hhb/minimax_h3/generate_minimax_h3_all.py"),
        "python_bin": os.getenv("MINIMAX_H3_PYTHON_BIN", "").strip(),
        "cuda_visible_devices": os.getenv("MINIMAX_H3_CUDA_VISIBLE_DEVICES", "").strip(),
        "task": os.getenv("MINIMAX_H3_TASK", "auto").strip(),
        "width": int(os.getenv("MINIMAX_H3_WIDTH", "768")),
        "height": int(os.getenv("MINIMAX_H3_HEIGHT", "768")),
        "num_frames": 243,
        "steps": int(os.getenv("MINIMAX_H3_STEPS", "30")),
        "seed": int(os.getenv("MINIMAX_H3_SEED", "42")),
        "fps": int(os.getenv("MINIMAX_H3_FPS", "24")),
        "parallel_mode": os.getenv("MINIMAX_H3_PARALLEL_MODE", "single_offload"),
        "device_map": os.getenv("MINIMAX_H3_DEVICE_MAP", "balanced"),
        "max_memory": _list_from_env("MINIMAX_H3_MAX_MEMORY"),
        "text_encoder_placement": os.getenv("MINIMAX_H3_TEXT_ENCODER_PLACEMENT", "offload"),
        "text_encoder_use_stream": _as_bool(os.getenv("MINIMAX_H3_TEXT_ENCODER_USE_STREAM"), False),
        "offload_folder": os.getenv("MINIMAX_H3_OFFLOAD_FOLDER", "").strip(),
        "runtime_device_index": int(os.getenv("MINIMAX_H3_RUNTIME_DEVICE_INDEX", "0")),
        "vae_parallel_mode": os.getenv("MINIMAX_H3_VAE_PARALLEL_MODE", "none").strip(),
        "vae_placement": os.getenv("MINIMAX_H3_VAE_PLACEMENT", "offload").strip(),
        "vae_use_tiling": _as_bool(os.getenv("MINIMAX_H3_VAE_USE_TILING"), False),
        "vae_use_slicing": _as_bool(os.getenv("MINIMAX_H3_VAE_USE_SLICING"), False),
        "reference_resize_width": int(os.getenv("MINIMAX_H3_REFERENCE_RESIZE_WIDTH", "0")),
        "reference_resize_height": int(os.getenv("MINIMAX_H3_REFERENCE_RESIZE_HEIGHT", "0")),
        "pytorch_cuda_alloc_conf": os.getenv("MINIMAX_H3_PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "timeout_sec": _optional_timeout_from_env("MINIMAX_H3_TIMEOUT_SEC", None),
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
