from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents.generation_backends import (
    flux2_klein_settings_from_env,
    generate_image_flux2_klein,
    generate_image_nano_banana,
    nano_banana_settings_from_env,
)
from narrative.simulation_scaffold import _safe_name
from narrative.visual_prompt_constraints import build_people_control_clause
from narrative.world_renderer import WorldRenderer, _normalize_image_extension, find_existing_image_path


class EventKeyframeRenderer:
    def __init__(self, *, backend_mode: str, dry_run: bool = False) -> None:
        self.world_renderer = WorldRenderer(backend_mode=backend_mode, dry_run=dry_run)

    @staticmethod
    def build_prompt(*, step: int, role_log: dict[str, Any], event_context: dict[str, Any] | None = None) -> str:
        action = role_log.get("next_action", {}) or {}
        role_name = str(role_log.get("role_name", "") or "Character")
        context = event_context if isinstance(event_context, dict) else {}
        story_description = str(
            context.get("story_description")
            or context.get("video_description")
            or context.get("event_summary")
            or role_log.get("story_description")
            or role_log.get("role_output_text")
            or ""
        ).strip()
        locations = ", ".join(str(x) for x in context.get("locations", []) or [] if str(x).strip())
        items = ", ".join(str(x) for x in context.get("items", []) or [] if str(x).strip())
        return (
            f"Cinematic narrative event keyframe for step {step}, character {role_name}. "
            f"Story event: {story_description}. "
            f"Action: {action.get('action') or action.get('action_type') or ''}. "
            f"Location: {locations or action.get('location') or ''}. "
            f"Relevant items: {items or 'none'}. "
            f"Reason: {action.get('reason') or ''}. "
            "16:9 film still, clear event readability, consistent character identity, stable world-space layout, high detail."
            f"\n{build_people_control_clause([role_name], allow_multi_people=False)}"
        )

    def render_event_keyframe(
        self,
        *,
        step: int,
        role_log: dict[str, Any],
        output_dir: Path,
        reference_image_paths: list[str] | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        role_name = str(role_log.get("role_name", "") or "Character")
        out_path = output_dir / f"step_{step:03d}" / f"{_safe_name(role_name)}_event_keyframe.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = self.build_prompt(step=step, role_log=role_log, event_context=event_context)
        reference_image_paths = [str(x) for x in (reference_image_paths or []) if str(x).strip()]

        existing = find_existing_image_path(out_path)
        if existing and not self.world_renderer.dry_run:
            result = {
                "tool": self.world_renderer.tool_name,
                "ok": True,
                "output_path": str(existing),
                "error": "",
                "reused_existing": True,
                "attempts": 0,
            }
        elif reference_image_paths and not self.world_renderer.dry_run:
            result = self._render_with_reference_images(
                prompt=prompt,
                output_path=out_path,
                reference_image_paths=reference_image_paths,
            )
        else:
            result = self.world_renderer.render_world_asset(prompt=prompt, output_path=out_path)

        result["prompt"] = prompt
        result["step"] = step
        result["role_name"] = role_name
        result["reference_image_paths"] = reference_image_paths
        if isinstance(event_context, dict):
            result["event_context"] = event_context
        return result

    def _render_with_reference_images(
        self,
        *,
        prompt: str,
        output_path: Path,
        reference_image_paths: list[str],
    ) -> dict[str, Any]:
        if self.world_renderer.tool_name == "flux2_klein":
            cfg = flux2_klein_settings_from_env()
        else:
            cfg = nano_banana_settings_from_env()
            if not cfg.get("api_key"):
                return self.world_renderer.render_world_asset(prompt=prompt, output_path=output_path)

        result: dict[str, Any] = {
            "tool": self.world_renderer.tool_name,
            "ok": False,
            "output_path": str(output_path),
            "error": "",
            "attempts": 0,
        }
        for attempt in range(1, self.world_renderer.max_attempts + 1):
            result["attempts"] = attempt
            try:
                if self.world_renderer.tool_name == "flux2_klein":
                    rendered = generate_image_flux2_klein(
                        prompt=prompt,
                        output_path=output_path,
                        api_url=str(cfg["api_url"]),
                        width=int(cfg["width"]),
                        height=int(cfg["height"]),
                        num_inference_steps=int(cfg["num_inference_steps"]),
                        seed=int(cfg["seed"]),
                        timeout_sec=int(cfg["timeout_sec"]),
                        reference_image_paths=reference_image_paths,
                    )
                else:
                    rendered = generate_image_nano_banana(
                        prompt=prompt,
                        output_path=output_path,
                        api_key=str(cfg["api_key"]),
                        base_url=str(cfg["base_url"]),
                        model_name=str(cfg["model_name"]),
                        timeout_sec=int(cfg["timeout_sec"]),
                        reference_image_paths=reference_image_paths,
                    )
                if rendered:
                    actual_path = _normalize_image_extension(Path(str(rendered)))
                    result.update({"ok": True, "output_path": str(actual_path), "error": ""})
                    return result
                result["error"] = "multi_reference generation failed"
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"
            if attempt < self.world_renderer.max_attempts:
                time.sleep(self.world_renderer.retry_sleep_sec)
        return result
