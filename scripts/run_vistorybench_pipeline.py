#!/usr/bin/env python3
"""Generate one ViStoryBench PNG per fixed benchmark shot.

This runner is intentionally independent from the narrative video pipeline.
It supports Banana and the local FLUX.2 Klein image backend, and never
reorders or merges benchmark shots.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.call_logger import call_context, write_call_event
from agents.generation_backends import (
    flux2_klein_settings_from_env,
    generate_image_flux2_klein,
    generate_image_nano_banana,
    nano_banana_settings_from_env,
)
from agents.vistorybench_anchor_selector import ViStoryBenchAnchorSelector, ViStoryBenchSelectorConfig
from agents.vistorybench_prompt_adapter import build_vistorybench_prompt
from utils.vistorybench_image import normalize_image_to_png

LOGGER = logging.getLogger("vistorybench_pipeline")


def _generate_vistorybench_image(
    *,
    image_backend: str,
    prompt: str,
    output_path: Path,
    reference_image_paths: list[str],
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> str | None:
    if image_backend == "flux2_klein":
        cfg = flux2_klein_settings_from_env()
        return generate_image_flux2_klein(
            prompt=prompt,
            output_path=output_path,
            api_url=str(cfg["api_url"]),
            width=int(width or cfg["width"]),
            height=int(height or cfg["height"]),
            num_inference_steps=int(cfg["num_inference_steps"]),
            seed=int(cfg["seed"] if seed is None else seed),
            timeout_sec=int(cfg["timeout_sec"]),
            reference_image_paths=reference_image_paths,
        )
    if image_backend != "banana":
        raise ValueError(f"Unsupported ViStoryBench image backend: {image_backend}")
    settings = nano_banana_settings_from_env()
    return generate_image_nano_banana(
        prompt=prompt,
        output_path=output_path,
        api_key=str(settings["api_key"]),
        base_url=str(settings["base_url"]),
        model_name=str(settings["model_name"]),
        timeout_sec=int(settings["timeout_sec"]),
        reference_image_paths=reference_image_paths,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _story_ids(dataset_root: Path, requested: list[str] | None, split: str) -> list[str]:
    if requested:
        return [str(item) for item in requested]
    all_ids = sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit())
    if split == "lite":
        lite = {"01", "08", "09", "15", "17", "19", "24", "27", "28", "29", "32", "41", "52", "53", "55", "57", "60", "64", "68", "79"}
        return [item for item in all_ids if item in lite]
    return all_ids


def _load_simulation(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if path.is_dir():
        path = path / "pipeline_result.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_json(path)


def _character_refs(dataset_story_dir: Path, story: dict[str, Any], language: str, names: list[str]) -> list[str]:
    result: list[str] = []
    characters = story.get("Characters", {}) or {}
    for name in names:
        matched_key = None
        for key, info in characters.items():
            if key == name or info.get("name_en") == name or info.get(f"name_{language}") == name:
                matched_key = key
                break
        if matched_key is None:
            continue
        image_dir = dataset_story_dir / "image" / matched_key
        candidates = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}) if image_dir.is_dir() else []
        if candidates:
            result.append(str(candidates[0]))
    return result


def _simulation_assets(node: dict[str, Any], simulation: dict[str, Any], max_refs: int) -> list[str]:
    assets = node.get("visual_assets") or {}
    paths = [str(path) for path in assets.get("event_keyframes", []) or [] if Path(str(path)).exists()]
    return paths[:max_refs]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root
    output_root = args.output_root / args.method / args.mode / args.language / args.timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    call_log_path = args.call_log or output_root / "xi_api_calls.jsonl"
    os.environ["CAMERAMASTER_CALL_LOG_JSONL"] = str(call_log_path)
    settings = nano_banana_settings_from_env()
    if args.image_backend == "banana" and not args.dry_run and not settings.get("api_key"):
        raise RuntimeError("NANO_BANANA_API_KEY is required unless --dry-run is used")
    selector = ViStoryBenchAnchorSelector(
        ViStoryBenchSelectorConfig(
            context_mode=args.context_mode,
            max_prior_events=args.max_prior_events,
            max_future_events=args.max_future_events,
        )
    )
    manifest: dict[str, Any] = {
        "method": args.method,
        "mode": args.mode,
        "image_backend": args.image_backend,
        "language": args.language,
        "timestamp": args.timestamp,
        "context_mode": args.context_mode,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "xi_api_call_log": str(call_log_path),
        "model_tool_call_log": str(call_log_path),
        "stories": [],
    }
    for story_id in _story_ids(dataset_root, args.story_id, args.split):
        story_dir = dataset_root / story_id
        story_path = story_dir / "story.json"
        story = _load_json(story_path)
        simulation = _load_simulation(args.cameramaster_run_dir)
        nodes = selector.build_nodes(story, simulation, story_id=story_id, language=args.language)
        story_output = output_root / story_id
        prompts: list[dict[str, Any]] = []
        generated: list[dict[str, Any]] = []
        for node in nodes:
            index = int(node["shot_index"])
            output_path = story_output / "shots" / f"shot_{index:03d}.png"
            prompt = build_vistorybench_prompt(node, mode=args.mode, language=args.language)
            char_refs = _character_refs(story_dir, story, args.language, node["benchmark_anchor"].get("characters", []))
            sim_refs = _simulation_assets(node, simulation, args.max_reference_images) if args.mode == "vistorybench_anchor_plus_sim" else []
            references = (char_refs + sim_refs)[:args.max_reference_images]
            prompt_record = {"shot_index": index, "prompt": prompt, "references": references, "output_path": str(output_path)}
            prompts.append(prompt_record)
            if output_path.exists() and args.resume:
                generated.append({**prompt_record, "status": "reused"})
                continue
            if args.dry_run:
                generated.append({**prompt_record, "status": "dry_run"})
                continue
            row = {**prompt_record, "status": "failed", "attempts": 0, "error": ""}
            for attempt in range(1, args.retries + 1):
                row["attempts"] = attempt
                try:
                    with call_context(
                        provider="local-flux2-klein" if args.image_backend == "flux2_klein" else "xi-api",
                        tool=args.image_backend,
                        story_id=story_id,
                        shot_index=index,
                        attempt=attempt,
                        model="FLUX.2 Klein 9B" if args.image_backend == "flux2_klein" else str(settings["model_name"]),
                        output_path=str(output_path),
                    ):
                        result = _generate_vistorybench_image(
                            image_backend=args.image_backend,
                            prompt=prompt,
                            output_path=output_path,
                            reference_image_paths=references,
                            width=args.width,
                            height=args.height,
                            seed=args.seed,
                        )
                    if result and Path(result).exists():
                        normalized_path = normalize_image_to_png(result, output_path)
                        row["status"] = "success"
                        row["output_path"] = str(normalized_path)
                        break
                    row["error"] = "image backend returned no output"
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    with call_context(
                        provider="local-flux2-klein" if args.image_backend == "flux2_klein" else "xi-api",
                        tool=args.image_backend,
                        story_id=story_id,
                        shot_index=index,
                        attempt=attempt,
                        model="FLUX.2 Klein 9B" if args.image_backend == "flux2_klein" else str(settings["model_name"]),
                        output_path=str(output_path),
                    ):
                        write_call_event(
                            {
                                "call_id": f"vistorybench-{story_id}-{index:03d}-{attempt}",
                                "kind": "runner",
                                "status": "shot_failed",
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:6000],
                            }
                        )
                    LOGGER.warning("shot failed story=%s shot=%s attempt=%s error=%s", story_id, index, attempt, exc)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep_sec)
            generated.append(row)
        _write_json(story_output / "nodes.json", {"story_id": story_id, "nodes": nodes})
        _write_json(story_output / "prompts.json", prompts)
        story_manifest = {"story_id": story_id, "expected_shots": len(nodes), "generated": generated}
        _write_json(story_output / "generation_manifest.json", story_manifest)
        manifest["stories"].append(story_manifest)
    _write_json(output_root / "generation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--story-id", action="append", default=None)
    parser.add_argument("--split", choices=["full", "lite"], default="full")
    parser.add_argument("--cameramaster-run-dir", type=Path, default=None)
    parser.add_argument("--method", default="cameramaster")
    parser.add_argument("--image-backend", choices=["banana", "flux2_klein"], default="banana")
    parser.add_argument("--mode", choices=["vistorybench_anchor_only", "vistorybench_anchor_plus_sim"], required=True)
    parser.add_argument("--context-mode", choices=["causal", "full_story"], default="causal")
    parser.add_argument("--language", choices=["en", "ch"], default="en")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--call-log", type=Path, default=None, help="JSONL log of model and image-tool calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--max-prior-events", type=int, default=3)
    parser.add_argument("--max-future-events", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    manifest = run(args)
    print(args.output_root / args.method / args.mode / args.language / args.timestamp / "generation_manifest.json")
    print(json.dumps({"stories": len(manifest["stories"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
