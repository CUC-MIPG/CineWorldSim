#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from narrative.world_renderer import WorldRenderer

LOGGER = logging.getLogger(__name__)


def _needs_repair(record: dict[str, Any]) -> bool:
    output_path = str(record.get("output_path", "") or "").strip()
    if not bool(record.get("ok", False)):
        return True
    if not output_path:
        return True
    path = Path(output_path)
    return not path.exists() or path.stat().st_size <= 0


def _render_record(
    *,
    renderer: WorldRenderer,
    record: dict[str, Any],
    prompt: str,
    output_path: Path,
) -> bool:
    LOGGER.info("asset repair render status=start output=%s", output_path)
    result = renderer.render_world_asset(prompt=prompt, output_path=output_path)
    record.update(result)
    LOGGER.info(
        "asset repair render status=done ok=%s output=%s error=%s",
        bool(result.get("ok")),
        result.get("output_path", str(output_path)),
        str(result.get("error", "") or "")[:500],
    )
    return bool(result.get("ok"))


def repair_pipeline_assets(
    pipeline_result_json: Path,
    *,
    backend_mode: str = "all_banana",
    dry_run: bool = False,
    max_assets: int = 0,
) -> dict[str, Any]:
    payload = json.loads(pipeline_result_json.read_text(encoding="utf-8"))
    renderer = WorldRenderer(backend_mode=backend_mode, dry_run=dry_run)
    repaired = 0
    failed = 0
    skipped = 0
    attempted: list[dict[str, Any]] = []

    def can_continue() -> bool:
        return max_assets <= 0 or repaired + failed < max_assets

    world_model = payload.get("world_model", {}) if isinstance(payload, dict) else {}
    for group_name in ["locations", "items"]:
        for asset in world_model.get(group_name, []) or []:
            if not isinstance(asset, dict):
                continue
            prompt = str(asset.get("image_prompt", "") or asset.get("description", "") or "").strip()
            asset_id = str(asset.get("asset_id") or asset.get("name") or "asset")
            for image in asset.get("images", []) or []:
                if not isinstance(image, dict):
                    continue
                if not _needs_repair(image):
                    skipped += 1
                    continue
                if not can_continue():
                    skipped += 1
                    continue
                output_path = Path(str(image.get("output_path", "") or "")).resolve()
                if not prompt or not str(output_path):
                    failed += 1
                    image["ok"] = False
                    image["error"] = "repair failed: missing prompt or output_path"
                    continue
                ok = _render_record(renderer=renderer, record=image, prompt=prompt, output_path=output_path)
                LOGGER.info("asset repair world kind=%s asset=%s ok=%s", group_name, asset_id, ok)
                repaired += 1 if ok else 0
                failed += 0 if ok else 1
                attempted.append(
                    {
                        "kind": f"world_model.{group_name}",
                        "asset_id": asset_id,
                        "ok": ok,
                        "output_path": str(image.get("output_path", "")),
                    }
                )

    for keyframe in payload.get("event_keyframes", []) or []:
        if not isinstance(keyframe, dict):
            continue
        if not _needs_repair(keyframe):
            skipped += 1
            continue
        if not can_continue():
            skipped += 1
            continue
        prompt = str(keyframe.get("prompt", "") or "").strip()
        output_path = Path(str(keyframe.get("output_path", "") or "")).resolve()
        if not prompt or not str(output_path):
            failed += 1
            keyframe["ok"] = False
            keyframe["error"] = "repair failed: missing prompt or output_path"
            continue
        ok = _render_record(renderer=renderer, record=keyframe, prompt=prompt, output_path=output_path)
        LOGGER.info(
            "asset repair keyframe step=%s role=%s ok=%s",
            keyframe.get("step", ""),
            keyframe.get("role_name", ""),
            ok,
        )
        repaired += 1 if ok else 0
        failed += 0 if ok else 1
        attempted.append(
            {
                "kind": "event_keyframe",
                "step": keyframe.get("step", ""),
                "role_name": keyframe.get("role_name", ""),
                "ok": ok,
                "output_path": str(keyframe.get("output_path", "")),
            }
        )

    pipeline_result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "done" if failed == 0 else "partial",
        "pipeline_result_json": str(pipeline_result_json),
        "backend_mode": backend_mode,
        "dry_run": dry_run,
        "repaired": repaired,
        "failed": failed,
        "skipped": skipped,
        "attempted": attempted,
    }
    summary_path = pipeline_result_json.with_name("asset_repair_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing/failed image assets in an existing pipeline_result.json.")
    parser.add_argument("--pipeline-result-json", type=Path, required=True)
    parser.add_argument("--backend-mode", choices=["all_banana", "all_flux"], default="all_banana")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    summary = repair_pipeline_assets(
        args.pipeline_result_json,
        backend_mode=args.backend_mode,
        dry_run=args.dry_run,
        max_assets=max(0, int(args.max_assets)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
