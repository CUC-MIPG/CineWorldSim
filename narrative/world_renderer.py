from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def backend_mode_to_tool(backend_mode: str) -> str:
    if backend_mode == "all_flux":
        return "flux_local"
    if backend_mode == "all_banana":
        return "nano_banana"
    if backend_mode == "all_flux2_klein":
        return "flux2_klein"
    raise ValueError(f"Unsupported backend mode: {backend_mode}")


def _normalize_image_extension(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        return path
    head = path.read_bytes()[:12]
    suffix = path.suffix.lower()
    expected = ""
    if head.startswith(b"\xff\xd8\xff"):
        expected = ".jpg"
    elif head.startswith(b"\x89PNG\r\n\x1a\n"):
        expected = ".png"
    elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        expected = ".webp"
    if not expected or suffix == expected:
        return path
    target = path.with_suffix(expected)
    if target.exists():
        target = path.with_name(f"{path.stem}_{path.stat().st_mtime_ns}{expected}")
    path.rename(target)
    return target


def find_existing_image_path(path: Path) -> Path | None:
    if path.exists() and path.stat().st_size > 0:
        return path
    if path.suffix.lower() in _IMAGE_EXTS:
        for suffix in _IMAGE_EXTS:
            candidate = path.with_suffix(suffix)
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return None


class WorldRenderer:
    def __init__(self, *, backend_mode: str, dry_run: bool = False) -> None:
        self.backend_mode = backend_mode
        self.tool_name = backend_mode_to_tool(backend_mode)
        self.dry_run = dry_run
        self.max_attempts = max(1, int(os.getenv("NARRATIVE_IMAGE_RENDER_MAX_ATTEMPTS", "3")))
        self.retry_sleep_sec = max(0.0, float(os.getenv("NARRATIVE_IMAGE_RENDER_RETRY_SLEEP_SEC", "10")))

    def render_world_asset(
        self,
        *,
        prompt: str,
        output_path: Path,
        reference_image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = find_existing_image_path(output_path)
        if existing and not self.dry_run:
            return {
                "tool": self.tool_name,
                "ok": True,
                "output_path": str(existing),
                "error": "",
                "reused_existing": True,
                "attempts": 0,
            }
        if self.dry_run:
            output_path.write_bytes(_ONE_PIXEL_PNG)
            return {"tool": self.tool_name, "ok": True, "output_path": str(output_path), "error": ""}

        from narrative.simulation_scaffold import WorldModelAgent

        result: dict[str, Any] = {"tool": self.tool_name, "ok": False, "output_path": str(output_path), "error": ""}
        refs = [str(p) for p in reference_image_paths or [] if str(p).strip() and Path(str(p)).exists()]
        if self.tool_name == "nano_banana" and refs:
            from agents.generation_backends import generate_image_nano_banana, nano_banana_settings_from_env

            cfg = nano_banana_settings_from_env()
            if cfg.get("api_key"):
                for attempt in range(1, self.max_attempts + 1):
                    result["attempts"] = attempt
                    try:
                        img = generate_image_nano_banana(
                            prompt=prompt,
                            output_path=output_path,
                            api_key=str(cfg["api_key"]),
                            base_url=str(cfg["base_url"]),
                            model_name=str(cfg["model_name"]),
                            timeout_sec=int(cfg["timeout_sec"]),
                            reference_image_paths=refs,
                        )
                        if img:
                            actual_path = _normalize_image_extension(Path(str(img)))
                            return {
                                "tool": self.tool_name,
                                "ok": True,
                                "output_path": str(actual_path),
                                "error": "",
                                "reference_image_paths": refs,
                                "attempts": attempt,
                            }
                        result["error"] = "nano_banana reference generation failed"
                    except Exception as exc:  # noqa: BLE001
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    if attempt < self.max_attempts:
                        time.sleep(self.retry_sleep_sec)

        if self.tool_name == "flux2_klein":
            from agents.generation_backends import (
                flux2_klein_settings_from_env,
                generate_image_flux2_klein,
            )

            cfg = flux2_klein_settings_from_env()
            for attempt in range(1, self.max_attempts + 1):
                result["attempts"] = attempt
                try:
                    img = generate_image_flux2_klein(
                        prompt=prompt,
                        output_path=output_path,
                        api_url=str(cfg["api_url"]),
                        width=int(cfg["width"]),
                        height=int(cfg["height"]),
                        num_inference_steps=int(cfg["num_inference_steps"]),
                        seed=int(cfg["seed"]),
                        reference_image_paths=refs,
                        timeout_sec=int(cfg["timeout_sec"]),
                    )
                    if img:
                        actual_path = _normalize_image_extension(Path(str(img)))
                        return {
                            "tool": self.tool_name,
                            "ok": True,
                            "output_path": str(actual_path),
                            "error": "",
                            "reference_image_paths": refs,
                            "attempts": attempt,
                        }
                    result["error"] = "flux2_klein generation failed"
                except Exception as exc:  # noqa: BLE001
                    result["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    time.sleep(self.retry_sleep_sec)
            return result

        agent = WorldModelAgent()
        for attempt in range(1, self.max_attempts + 1):
            result = agent._render_with_tool(
                tool_name=self.tool_name,
                prompt=prompt,
                output_path=output_path,
            )
            if result.get("ok"):
                break
            result["attempts"] = attempt
            if attempt < self.max_attempts:
                time.sleep(self.retry_sleep_sec)
        if result.get("ok"):
            actual_path = _normalize_image_extension(Path(str(result.get("output_path") or output_path)))
            result["output_path"] = str(actual_path)
        return result
