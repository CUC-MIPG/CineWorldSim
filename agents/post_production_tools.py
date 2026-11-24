from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import requests

logger = logging.getLogger(__name__)

_OMNIVOICE_EN_ALLOWED = {
    "american accent",
    "australian accent",
    "british accent",
    "canadian accent",
    "child",
    "chinese accent",
    "elderly",
    "female",
    "high pitch",
    "indian accent",
    "japanese accent",
    "korean accent",
    "low pitch",
    "male",
    "middle-aged",
    "moderate pitch",
    "portuguese accent",
    "russian accent",
    "teenager",
    "very high pitch",
    "very low pitch",
    "whisper",
    "young adult",
}

_OMNIVOICE_ZH_ALLOWED = {
    "东北话",
    "中年",
    "中音调",
    "云南话",
    "低音调",
    "儿童",
    "四川话",
    "女",
    "宁夏话",
    "少年",
    "极低音调",
    "极高音调",
    "桂林话",
    "河南话",
    "济南话",
    "甘肃话",
    "男",
    "石家庄话",
    "老年",
    "耳语",
    "贵州话",
    "陕西话",
    "青岛话",
    "青年",
    "高音调",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _post_provider(env_name: str, default: str = "auto") -> str:
    val = os.getenv(env_name, default).strip().lower()
    if val in {"auto", "omnivoice", "service", "pseudo", "acestep"}:
        return val
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _acestep_project_root() -> Path:
    return Path(os.getenv("ACESTEP_PROJECT_ROOT", "/data1/AgentG/hhb/ACE-Step-1.5")).expanduser()


def _acestep_local_infer_script() -> Path:
    default = _project_root() / "tool_hub" / "code" / "post_production" / "providers" / "acestep_local_infer.py"
    return Path(os.getenv("ACESTEP_LOCAL_INFER_SCRIPT", str(default))).expanduser()


def _acestep_uv_cache_dir() -> Path:
    return Path(os.getenv("ACESTEP_UV_CACHE_DIR", "/tmp/uv-cache")).expanduser()


def _acestep_device() -> str:
    return os.getenv("ACESTEP_DEVICE", "auto").strip() or "auto"


def _acestep_with_lm() -> bool:
    return _env_flag("ACESTEP_WITH_LM", default=False)


def _acestep_thinking() -> bool:
    return _env_flag("ACESTEP_THINKING", default=False)


def _omnivoice_model_dir() -> Path:
    default_dir = _project_root() / "tool_hub" / "models" / "dubbing" / "OmniVoice"
    return Path(os.getenv("OMNIVOICE_MODEL_DIR", str(default_dir))).expanduser()


def _omnivoice_device() -> str:
    raw = os.getenv("OMNIVOICE_DEVICE", "cuda:0").strip() or "cuda:0"
    if raw == "auto":
        return "cuda:0"
    return raw


def _omnivoice_dtype_name() -> str:
    return os.getenv("OMNIVOICE_DTYPE", "float16").strip().lower() or "float16"


def _safe_json_load(path_str: str) -> dict[str, Any]:
    p = Path(path_str).expanduser()
    if not path_str or not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_profile_ref(profile_name: str, text_fallback: str) -> tuple[str | None, str | None]:
    map_path = os.getenv("OMNIVOICE_PROFILE_MAP_JSON", "").strip()
    profile_map = _safe_json_load(map_path) if map_path else {}

    row = profile_map.get(profile_name) if profile_name else None
    if isinstance(row, str):
        audio_path = row.strip()
        if audio_path and Path(audio_path).exists():
            return audio_path, text_fallback
    if isinstance(row, dict):
        audio_path = str(row.get("ref_audio", "")).strip()
        ref_text = str(row.get("ref_text", "")).strip() or text_fallback
        if audio_path and Path(audio_path).exists():
            return audio_path, ref_text

    env_audio = os.getenv("OMNIVOICE_REF_AUDIO", "").strip()
    env_text = os.getenv("OMNIVOICE_REF_TEXT", "").strip() or text_fallback
    if env_audio and Path(env_audio).exists():
        return env_audio, env_text
    return None, None


def _contains_cjk(s: str) -> bool:
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _normalize_omnivoice_instruct(profile_name: str) -> str:
    raw = (profile_name or "").strip()
    if not raw:
        return "male, moderate pitch"

    if _contains_cjk(raw):
        tokens = [x.strip() for x in raw.replace(",", "，").split("，") if x.strip()]
        valid = [x for x in tokens if x in _OMNIVOICE_ZH_ALLOWED]
        if not valid:
            return "男，中音调"
        return "，".join(valid)

    tokens = [x.strip().lower() for x in raw.split(",") if x.strip()]
    valid = [x for x in tokens if x in _OMNIVOICE_EN_ALLOWED]
    if not valid:
        return "male, moderate pitch"
    return ", ".join(valid)


@lru_cache(maxsize=1)
def _load_omnivoice_model() -> Any:
    model_dir = _omnivoice_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"OmniVoice model dir not found: {model_dir}")

    import torch  # lazy import for optional dependency
    from omnivoice import OmniVoice  # type: ignore

    dtype_name = _omnivoice_dtype_name()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name, torch.float16)
    preferred_device = _omnivoice_device()
    device_candidates = [preferred_device]
    if preferred_device.startswith("cuda"):
        if not torch.cuda.is_available():
            device_candidates = ["cpu"]
        else:
            device_candidates.append("cpu")
    elif preferred_device != "cpu":
        device_candidates.append("cpu")

    last_err: Exception | None = None
    for device in device_candidates:
        try:
            run_dtype = dtype
            if device == "cpu" and run_dtype in {torch.float16, torch.bfloat16}:
                run_dtype = torch.float32
            return OmniVoice.from_pretrained(
                str(model_dir),
                device_map=device,
                dtype=run_dtype,
            )
        except Exception as exc:
            last_err = exc
            logger.warning("omnivoice load failed device=%s error=%s", device, exc)
            continue
    if last_err is None:
        raise RuntimeError("OmniVoice failed to load with unknown error")
    raise last_err


def _call_omnivoice_local(
    *,
    text: str,
    profile_name: str,
    output_audio_path: str,
    tool_name: str,
    reference_audio_path: str = "",
    reference_text: str = "",
) -> dict[str, Any]:
    target = Path(output_audio_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s omnivoice status=start model_dir=%s", tool_name, str(_omnivoice_model_dir()))
    model: Any | None = None
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "tool": tool_name,
            "mode": "omnivoice_failed",
            "error": f"import_soundfile_failed:{type(exc).__name__}:{exc}",
            "output_audio_path": str(target),
            "todo_file": "",
        }

    try:
        model = _load_omnivoice_model()
        ref_audio = reference_audio_path.strip()
        ref_text = reference_text.strip()
        if ref_audio and not Path(ref_audio).exists():
            ref_audio = ""
        if ref_audio and not ref_text:
            ref_text = text
        if not ref_audio:
            ref_audio, ref_text = _resolve_profile_ref(profile_name=profile_name, text_fallback=text)
        if ref_audio and ref_text:
            audio = model.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            clone_mode = "with_ref_audio"
        else:
            instruct = _normalize_omnivoice_instruct(profile_name)
            audio = model.generate(
                text=text,
                instruct=instruct,
            )
            clone_mode = "instruction_only"
        if not audio:
            raise RuntimeError("OmniVoice returned empty audio list")
        sf.write(str(target), audio[0], 24000)
        todo_path = target.with_suffix(target.suffix + ".todo.txt")
        if todo_path.exists():
            todo_path.unlink(missing_ok=True)
        logger.info("%s omnivoice status=done output=%s mode=%s", tool_name, str(target), clone_mode)
        return {
            "ok": True,
            "tool": tool_name,
            "mode": "omnivoice_local",
            "local_mode": clone_mode,
            "output_audio_path": str(target),
            "todo_file": "",
            "sample_rate": 24000,
            "model_dir": str(_omnivoice_model_dir()),
            "ref_audio": ref_audio or "",
            "instruct_used": "" if ref_audio else instruct,
            "voice_profile_input": profile_name,
        }
    except Exception as exc:
        if model is not None and not _resolve_profile_ref(profile_name=profile_name, text_fallback=text)[0]:
            try:
                fallback_instruct = "male, moderate pitch"
                logger.warning("%s omnivoice retry with fallback instruct=%s", tool_name, fallback_instruct)
                audio = model.generate(
                    text=text,
                    instruct=fallback_instruct,
                )
                if audio:
                    sf.write(str(target), audio[0], 24000)
                    todo_path = target.with_suffix(target.suffix + ".todo.txt")
                    if todo_path.exists():
                        todo_path.unlink(missing_ok=True)
                    return {
                        "ok": True,
                        "tool": tool_name,
                        "mode": "omnivoice_local",
                        "local_mode": "instruction_retry_fallback",
                        "output_audio_path": str(target),
                        "todo_file": "",
                        "sample_rate": 24000,
                        "model_dir": str(_omnivoice_model_dir()),
                        "ref_audio": "",
                        "instruct_used": fallback_instruct,
                        "voice_profile_input": profile_name,
                        "warning": f"primary_instruct_failed:{type(exc).__name__}",
                    }
            except Exception:
                pass
        logger.warning("%s omnivoice status=failed error=%s", tool_name, exc)
        return {
            "ok": False,
            "tool": tool_name,
            "mode": "omnivoice_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "output_audio_path": str(target),
            "todo_file": "",
        }


def _service_timeout_sec() -> float:
    raw = os.getenv("POST_AUDIO_SERVICE_TIMEOUT_SEC", "480").strip()
    try:
        val = float(raw)
    except Exception:
        return 480.0
    if val <= 0:
        return 480.0
    return val


def _call_audio_service(*, text: str, output_audio_path: str, api_url: str, tool_name: str) -> dict[str, Any]:
    target = Path(output_audio_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}

    logger.info("%s service_call status=start url=%s", tool_name, api_url)
    try:
        resp = requests.post(
            api_url,
            data=json.dumps(payload, ensure_ascii=False),
            headers=headers,
            timeout=_service_timeout_sec(),
        )
        if resp.status_code == 200 and resp.content:
            target.write_bytes(resp.content)
            todo_path = target.with_suffix(target.suffix + ".todo.txt")
            if todo_path.exists():
                todo_path.unlink(missing_ok=True)
            logger.info("%s service_call status=done output=%s bytes=%d", tool_name, str(target), target.stat().st_size)
            return {
                "ok": True,
                "tool": tool_name,
                "output_audio_path": str(target),
                "todo_file": "",
                "mode": "service",
                "api_url": api_url,
                "status_code": resp.status_code,
            }
        logger.warning(
            "%s service_call status=failed status_code=%s body_len=%d",
            tool_name,
            resp.status_code,
            len(resp.content or b""),
        )
        return {
            "ok": False,
            "tool": tool_name,
            "output_audio_path": str(target),
            "todo_file": "",
            "mode": "service_failed",
            "api_url": api_url,
            "status_code": resp.status_code,
            "error": f"bad_status_or_empty_content:{resp.status_code}",
        }
    except Exception as exc:
        logger.warning("%s service_call exception=%s", tool_name, exc)
        return {
            "ok": False,
            "tool": tool_name,
            "output_audio_path": str(target),
            "todo_file": "",
            "mode": "service_failed",
            "api_url": api_url,
            "status_code": -1,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_todo(target_path: Path, tool_name: str, payload: dict[str, Any]) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    todo_path = target_path.with_suffix(target_path.suffix + ".todo.txt")
    content = {
        "tool": tool_name,
        "target_output": str(target_path),
        "status": "pseudo_not_executed",
        "payload": payload,
    }
    todo_path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(todo_path)


def _normalize_duration_list(raw: list[float] | None, expected_len: int) -> list[float | None]:
    if not raw:
        return [None] * expected_len
    out: list[float | None] = []
    for idx in range(expected_len):
        if idx >= len(raw):
            out.append(None)
            continue
        val = raw[idx]
        try:
            d = float(val)
        except Exception:
            out.append(None)
            continue
        out.append(d if d > 0 else None)
    return out


def _build_audio_timeline(
    paths: list[str],
    out_path: Path,
    work_dir: Path,
    tag: str,
    segment_durations: list[float] | None = None,
) -> Path | None:
    if not paths and not segment_durations:
        return None

    expected_len = max(len(paths), len(segment_durations or []))
    if expected_len <= 0:
        return None
    padded_paths = list(paths) + [""] * max(0, expected_len - len(paths))
    durations = _normalize_duration_list(segment_durations, expected_len)

    normalized_dir = work_dir / f"audio_norm_{tag}"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for idx, path_str in enumerate(padded_paths, start=1):
        seg_dur = durations[idx - 1]
        src = Path(path_str) if path_str else None
        dst = normalized_dir / f"{tag}_{idx:04d}.wav"

        if src and src.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
            if seg_dur is not None:
                cmd.extend(
                    [
                        "-af",
                        f"apad=pad_dur={seg_dur:.3f},atrim=0:{seg_dur:.3f}",
                    ]
                )
            cmd.extend(["-c:a", "pcm_s16le", str(dst)])
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                normalized.append(dst)
                continue

        if seg_dur is not None:
            silence_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-t",
                f"{seg_dur:.3f}",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ]
            silence_proc = subprocess.run(silence_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if silence_proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                normalized.append(dst)

    if not normalized:
        return None

    concat_list = work_dir / f"concat_{tag}.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in normalized), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:a",
        "aac",
        str(out_path),
    ]
    proc2 = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc2.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        return None
    return out_path


def _call_acestep_local(
    *,
    style_prompt: str,
    duration_sec: float,
    output_audio_path: str,
) -> dict[str, Any]:
    target = Path(output_audio_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ace_root = _acestep_project_root()
    infer_script = _acestep_local_infer_script()
    if not ace_root.exists():
        return {
            "ok": False,
            "tool": "bgm",
            "mode": "acestep_failed",
            "error": f"acestep_root_not_found:{ace_root}",
            "output_audio_path": str(target),
            "todo_file": "",
        }
    if not infer_script.exists():
        return {
            "ok": False,
            "tool": "bgm",
            "mode": "acestep_failed",
            "error": f"acestep_script_not_found:{infer_script}",
            "output_audio_path": str(target),
            "todo_file": "",
        }

    clamped_duration = max(10.0, min(600.0, float(duration_sec)))
    with tempfile.TemporaryDirectory(prefix="acestep_bgm_") as td:
        tmp_out = Path(td)
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(infer_script),
            "--prompt",
            style_prompt or "Cinematic background score, adaptive dynamics, no vocals.",
            "--duration",
            f"{clamped_duration:.2f}",
            "--audio-format",
            "wav",
            "--device",
            _acestep_device(),
            "--output-dir",
            str(tmp_out),
        ]
        if _acestep_with_lm():
            cmd.append("--with-lm")
            if _acestep_thinking():
                cmd.append("--thinking")

        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(_acestep_uv_cache_dir())
        logger.info("bgm acestep status=start duration=%.2f root=%s", clamped_duration, str(ace_root))
        proc = subprocess.run(
            cmd,
            cwd=str(ace_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning("bgm acestep status=failed rc=%d", proc.returncode)
            return {
                "ok": False,
                "tool": "bgm",
                "mode": "acestep_failed",
                "error": f"acestep_return_code:{proc.returncode}",
                "stderr_tail": proc.stderr[-1200:],
                "stdout_tail": proc.stdout[-1200:],
                "output_audio_path": str(target),
                "todo_file": "",
            }

        wavs = sorted(tmp_out.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wavs:
            return {
                "ok": False,
                "tool": "bgm",
                "mode": "acestep_failed",
                "error": "no_wav_generated",
                "output_audio_path": str(target),
                "todo_file": "",
            }

        shutil.copyfile(wavs[0], target)
        todo_path = target.with_suffix(target.suffix + ".todo.txt")
        if todo_path.exists():
            todo_path.unlink(missing_ok=True)
        logger.info("bgm acestep status=done output=%s", str(target))
        return {
            "ok": True,
            "tool": "bgm",
            "mode": "acestep_local",
            "output_audio_path": str(target),
            "todo_file": "",
            "duration_sec": clamped_duration,
            "source_wav": str(wavs[0]),
        }


def generate_dubbing_pseudocode(
    *,
    script_text: str,
    voice_profile: str,
    output_audio_path: str,
    language: str = "zh-CN",
    reference_audio_path: str = "",
    reference_text: str = "",
) -> dict[str, Any]:
    provider = _post_provider("POST_DUBBING_PROVIDER", default="auto")
    if provider in {"auto", "omnivoice"}:
        local = _call_omnivoice_local(
            text=script_text,
            profile_name=voice_profile,
            output_audio_path=output_audio_path,
            tool_name="dubbing",
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )
        if local.get("ok"):
            return local
        if provider == "omnivoice":
            target = Path(output_audio_path)
            todo = _write_todo(
                target,
                "dubbing",
                {
                    "script_text": script_text,
                    "voice_profile": voice_profile,
                    "language": language,
                    "reference_audio_path": reference_audio_path,
                    "reference_text": reference_text,
                    "provider": provider,
                    "local_error": local.get("error", ""),
                },
            )
            return {
                "ok": True,
                "tool": "dubbing",
                "output_audio_path": str(target),
                "todo_file": todo,
                "mode": "pseudo_fallback",
                "local_error": local.get("error", ""),
            }

    if provider not in {"service", "auto"}:
        target = Path(output_audio_path)
        todo = _write_todo(
            target,
            "dubbing",
            {
                "script_text": script_text,
                "voice_profile": voice_profile,
                "language": language,
                "reference_audio_path": reference_audio_path,
                "reference_text": reference_text,
                "provider": provider,
            },
        )
        return {
            "ok": True,
            "tool": "dubbing",
            "output_audio_path": str(target),
            "todo_file": todo,
            "mode": "pseudo_fallback",
        }

    api_url = os.getenv("DUBBING_GEN_API_URL", "").strip() or os.getenv(
        "NARRATION_GEN_API_URL",
        "http://172.17.34.43:5050/narration_gen",
    )
    online = _call_audio_service(
        text=script_text,
        output_audio_path=output_audio_path,
        api_url=api_url,
        tool_name="dubbing",
    )
    if online.get("ok"):
        return online

    target = Path(output_audio_path)
    todo = _write_todo(
        target,
        "dubbing",
        {
            "script_text": script_text,
            "voice_profile": voice_profile,
            "language": language,
            "reference_audio_path": reference_audio_path,
            "reference_text": reference_text,
        },
    )
    return {
        "ok": True,
        "tool": "dubbing",
        "output_audio_path": str(target),
        "todo_file": todo,
        "mode": "pseudo_fallback",
        "provider": provider,
        "service_error": online.get("error", ""),
    }


def generate_bgm_pseudocode(
    *,
    style_prompt: str,
    duration_sec: float,
    output_audio_path: str,
) -> dict[str, Any]:
    provider = _post_provider("POST_BGM_PROVIDER", default="acestep")
    if provider in {"auto", "acestep"}:
        return _call_acestep_local(
            style_prompt=style_prompt,
            duration_sec=duration_sec,
            output_audio_path=output_audio_path,
        )

    target = Path(output_audio_path)
    return {
        "ok": False,
        "tool": "bgm",
        "output_audio_path": str(target),
        "todo_file": "",
        "mode": "provider_not_supported",
        "error": f"unsupported_bgm_provider:{provider}",
    }


def generate_narration_pseudocode(
    *,
    narration_text: str,
    narrator_profile: str,
    output_audio_path: str,
    language: str = "zh-CN",
) -> dict[str, Any]:
    provider = _post_provider("POST_NARRATION_PROVIDER", default="auto")
    if provider in {"auto", "omnivoice"}:
        local = _call_omnivoice_local(
            text=narration_text,
            profile_name=narrator_profile,
            output_audio_path=output_audio_path,
            tool_name="narration",
        )
        if local.get("ok"):
            return local
        if provider == "omnivoice":
            target = Path(output_audio_path)
            todo = _write_todo(
                target,
                "narration",
                {
                    "narration_text": narration_text,
                    "narrator_profile": narrator_profile,
                    "language": language,
                    "provider": provider,
                    "local_error": local.get("error", ""),
                },
            )
            return {
                "ok": True,
                "tool": "narration",
                "output_audio_path": str(target),
                "todo_file": todo,
                "mode": "pseudo_fallback",
                "local_error": local.get("error", ""),
            }

    if provider not in {"service", "auto"}:
        target = Path(output_audio_path)
        todo = _write_todo(
            target,
            "narration",
            {
                "narration_text": narration_text,
                "narrator_profile": narrator_profile,
                "language": language,
                "provider": provider,
            },
        )
        return {
            "ok": True,
            "tool": "narration",
            "output_audio_path": str(target),
            "todo_file": todo,
            "mode": "pseudo_fallback",
        }

    api_url = os.getenv("NARRATION_GEN_API_URL", "http://172.17.34.43:5050/narration_gen").strip()
    online = _call_audio_service(
        text=narration_text,
        output_audio_path=output_audio_path,
        api_url=api_url,
        tool_name="narration",
    )
    if online.get("ok"):
        return online

    target = Path(output_audio_path)
    todo = _write_todo(
        target,
        "narration",
        {
            "narration_text": narration_text,
            "narrator_profile": narrator_profile,
            "language": language,
        },
    )
    return {
        "ok": True,
        "tool": "narration",
        "output_audio_path": str(target),
        "todo_file": todo,
        "mode": "pseudo_fallback",
        "provider": provider,
        "service_error": online.get("error", ""),
    }


def edit_master_video_pseudocode(
    *,
    input_video_paths: list[str],
    dubbing_paths: list[str],
    narration_paths: list[str],
    bgm_paths: list[str],
    output_video_path: str,
    edit_instructions: str,
    dubbing_segment_durations: list[float] | None = None,
    narration_segment_durations: list[float] | None = None,
    bgm_segment_durations: list[float] | None = None,
    include_narration: bool = True,
) -> dict[str, Any]:
    logger.info("post_tool edit status=start input_videos=%d dubbing=%d narration=%d bgm=%d", len(input_video_paths), len(dubbing_paths), len(narration_paths), len(bgm_paths))
    target = Path(output_video_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_video_paths": input_video_paths,
        "dubbing_paths": dubbing_paths,
        "narration_paths": narration_paths,
        "bgm_paths": bgm_paths,
        "edit_instructions": edit_instructions,
        "dubbing_segment_durations": dubbing_segment_durations or [],
        "narration_segment_durations": narration_segment_durations or [],
        "bgm_segment_durations": bgm_segment_durations or [],
        "include_narration": bool(include_narration),
    }

    valid_inputs = [Path(p) for p in input_video_paths if p and Path(p).exists()]
    if not valid_inputs:
        logger.info("post_tool edit status=skip reason=no_valid_input_video")
        todo = _write_todo(target, "editing", payload)
        return {
            "ok": False,
            "tool": "editing",
            "output_video_path": str(target),
            "todo_file": todo,
            "reason": "no_valid_input_video",
        }

    work_dir = target.parent / "_edit_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = work_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    first_cap = cv2.VideoCapture(str(valid_inputs[0]))
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(first_cap.get(cv2.CAP_PROP_FPS)) or 24.0
    first_cap.release()
    fps = max(1.0, min(60.0, fps))

    normalized_clips: list[Path] = []
    for idx, src in enumerate(valid_inputs, start=1):
        logger.info("post_tool edit normalize clip_index=%d src=%s", idx, str(src))
        dst = normalized_dir / f"clip_{idx:04d}.mp4"
        cap = cv2.VideoCapture(str(src))
        writer = cv2.VideoWriter(
            str(dst),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not cap.isOpened() or not writer.isOpened():
            cap.release()
            writer.release()
            continue

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)

        cap.release()
        writer.release()
        if dst.exists() and dst.stat().st_size > 0:
            normalized_clips.append(dst)
            logger.info("post_tool edit normalize status=done clip=%s", str(dst))

    if not normalized_clips:
        logger.info("post_tool edit status=failed reason=opencv_normalize_failed")
        todo = _write_todo(target, "editing", payload)
        return {
            "ok": False,
            "tool": "editing",
            "output_video_path": str(target),
            "todo_file": todo,
            "reason": "opencv_normalize_failed",
        }

    concat_list = work_dir / "concat_list.txt"
    concat_list.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in normalized_clips),
        encoding="utf-8",
    )
    merged_video = work_dir / "merged_video.mp4"
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(merged_video),
    ]
    concat_proc = subprocess.run(
        concat_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if concat_proc.returncode != 0 or not merged_video.exists():
        logger.info("post_tool edit status=failed reason=ffmpeg_concat_failed")
        todo = _write_todo(target, "editing", payload)
        return {
            "ok": False,
            "tool": "editing",
            "output_video_path": str(target),
            "todo_file": todo,
            "reason": "ffmpeg_concat_failed",
            "stderr_tail": concat_proc.stderr[-800:],
        }

    # Build scene-ordered audio timelines first; this avoids all scene clips being mixed at t=0.
    dubbing_timeline = _build_audio_timeline(
        dubbing_paths,
        out_path=work_dir / "dubbing_timeline.m4a",
        work_dir=work_dir,
        tag="dubbing",
        segment_durations=dubbing_segment_durations,
    )
    narration_timeline = None
    if include_narration:
        narration_timeline = _build_audio_timeline(
            narration_paths,
            out_path=work_dir / "narration_timeline.m4a",
            work_dir=work_dir,
            tag="narration",
            segment_durations=narration_segment_durations,
        )
    bgm_timeline = _build_audio_timeline(
        bgm_paths,
        out_path=work_dir / "bgm_timeline.m4a",
        work_dir=work_dir,
        tag="bgm",
        segment_durations=bgm_segment_durations,
    )

    audio_inputs: list[tuple[str, Path]] = []
    if dubbing_timeline:
        audio_inputs.append(("dubbing", dubbing_timeline))
    if narration_timeline:
        audio_inputs.append(("narration", narration_timeline))
    if bgm_timeline:
        audio_inputs.append(("bgm", bgm_timeline))

    if not audio_inputs:
        shutil.copyfile(merged_video, target)
        logger.info("post_tool edit status=done mode=video_concat_only output=%s", str(target))
        return {
            "ok": True,
            "tool": "editing",
            "output_video_path": str(target),
            "todo_file": "",
            "mode": "video_concat_only",
            "clip_count": len(normalized_clips),
            "audio_track_count": 0,
            "work_dir": str(work_dir),
        }

    mix_output = target
    ffmpeg_cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(merged_video),
    ]
    for _, p in audio_inputs:
        ffmpeg_cmd.extend(["-i", str(p)])

    filter_parts: list[str] = []
    amix_inputs: list[str] = []
    for idx in range(1, len(audio_inputs) + 1):
        label = f"a{idx}"
        role = audio_inputs[idx - 1][0]
        volume = 0.45 if role == "bgm" else 1.0
        filter_parts.append(f"[{idx}:a]volume={volume:.2f}[{label}]")
        amix_inputs.append(f"[{label}]")
    # Keep full video duration by padding mixed audio timeline with silence.
    # Combined with "-shortest", this guarantees output duration follows video stream.
    filter_parts.append(f"{''.join(amix_inputs)}amix=inputs={len(audio_inputs)}:duration=longest,apad[aout]")

    ffmpeg_cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(mix_output),
        ]
    )
    mix_proc = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if mix_proc.returncode != 0 or not mix_output.exists():
        logger.info("post_tool edit audio_mix status=failed, try video-only fallback")
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(merged_video),
            "-c:v",
            "copy",
            "-an",
            str(target),
        ]
        fallback_proc = subprocess.run(
            fallback_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if fallback_proc.returncode != 0 or not target.exists():
            logger.info("post_tool edit status=failed reason=ffmpeg_audio_mix_failed")
            todo = _write_todo(target, "editing", payload)
            return {
                "ok": False,
                "tool": "editing",
                "output_video_path": str(target),
                "todo_file": todo,
                "reason": "ffmpeg_audio_mix_failed",
                "stderr_tail": mix_proc.stderr[-800:],
            }
        return {
            "ok": True,
            "tool": "editing",
            "output_video_path": str(target),
            "todo_file": "",
            "mode": "video_concat_only_fallback",
            "clip_count": len(normalized_clips),
            "audio_track_count": len(audio_inputs),
            "work_dir": str(work_dir),
            "mix_warning": mix_proc.stderr[-400:],
        }

    logger.info("post_tool edit status=done mode=video_plus_audio_mix output=%s", str(target))
    return {
        "ok": True,
        "tool": "editing",
        "output_video_path": str(target),
        "todo_file": "",
        "mode": "video_plus_audio_mix",
        "clip_count": len(normalized_clips),
        "audio_track_count": len(audio_inputs),
        "work_dir": str(work_dir),
    }
