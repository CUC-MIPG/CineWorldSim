#!/usr/bin/env python3
"""Run ViMax video generation on the paired MovieBench American Beauty arc_01 plan."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.video_generator_minimax_h3_vllm import VideoGeneratorMinimaxH3VLLM
from utils.video import concatenate_video_files

LOGGER = logging.getLogger("vimax_moviebench_baseline")
CONDITIONS = ("fl2va_event_keyframe_only", "ref2va_semantic_assets")
ALLOWED_TASKS = {"fl2va", "ref2va"}
DEFAULT_PLAN = Path(
    "/data1/AgentG/hhb/CameraMaster/outputs/moviebench_video_eval/"
    "0001_American_Beauty/arc_01/video_comparison_plan.json"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "vimax_moviebench" / "0001_American_Beauty" / "arc_01"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")[:96] or "node"


def build_jobs(plan: Any, condition: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(plan, list) or not plan:
        raise ValueError("Comparison plan must be a non-empty JSON array")
    if condition is not None and condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")

    jobs: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    conditions = [condition] if condition else list(CONDITIONS)
    for row in plan:
        if not isinstance(row, dict):
            raise ValueError("Every comparison-plan row must be an object")
        node_id = str(row.get("node_id") or "").strip()
        prompt = str(row.get("prompt") or "")
        if not node_id or not prompt:
            raise ValueError("Every comparison-plan row requires node_id and prompt")
        if node_id in seen_nodes:
            raise ValueError(f"duplicate node_id: {node_id}")
        seen_nodes.add(node_id)
        if row.get("same_prompt_for_both_groups") is False:
            raise ValueError(f"node {node_id} does not declare the same prompt for both conditions")

        groups = row.get("groups")
        if not isinstance(groups, dict):
            raise ValueError(f"node {node_id} has no groups object")
        for condition_name in conditions:
            group = groups.get(condition_name)
            if not isinstance(group, dict):
                raise ValueError(f"node {node_id} is missing condition {condition_name}")
            task = str(group.get("minimax_task") or "").lower()
            refs = group.get("reference_images")
            if task not in ALLOWED_TASKS or not isinstance(refs, list) or not refs:
                raise ValueError(f"node {node_id} has invalid task or reference_images in {condition_name}")
            jobs.append({
                "rank": int(row.get("rank") or 0),
                "node_id": node_id,
                "condition": condition_name,
                "prompt": prompt,
                "task": task,
                "reference_mode": str(group.get("reference_mode") or condition_name),
                "reference_images": [str(path) for path in refs],
                "duration_sec": float(row.get("duration_sec") or 5),
                "num_frames": int(row.get("num_frames") or 124),
                "mentioned_items": list(group.get("mentioned_items") or []),
                "status": "planned",
            })

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        selected_nodes = list(dict.fromkeys(job["node_id"] for job in jobs))[:limit]
        jobs = [job for job in jobs if job["node_id"] in selected_nodes]
    return jobs


def _job_output_path(output_dir: Path, job: dict[str, Any]) -> Path:
    return output_dir / job["condition"] / f"event_video_{job['rank']:03d}_{_safe_name(job['node_id'])}.mp4"


def _missing_references(jobs: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        f"{job['condition']}:{job['node_id']}": [path for path in job["reference_images"] if not Path(path).is_file()]
        for job in jobs
        if any(not Path(path).is_file() for path in job["reference_images"])
    }


def _write_manifests(output_dir: Path, jobs_by_condition: dict[str, list[dict[str, Any]]]) -> None:
    for condition, jobs in jobs_by_condition.items():
        _write_json(output_dir / condition / "event_video_tasks.json", {"condition": condition, "jobs": jobs})


def _summary(
    *,
    status: str,
    plan_path: Path,
    output_dir: Path,
    jobs: list[dict[str, Any]],
    missing: dict[str, list[str]],
    concat: bool,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        selected = [job for job in jobs if job["condition"] == condition]
        groups[condition] = {
            "total": len(selected),
            "success": sum(job["status"] == "success" for job in selected),
            "failed": sum(job["status"] == "failed" for job in selected),
            "skipped": sum(job["status"] == "skipped" for job in selected),
            "planned": sum(job["status"] == "planned" for job in selected),
            "missing_references": sum(job["status"] == "blocked_missing_references" for job in selected),
            "concat_ok": None,
            "final_video": str(output_dir / condition / "final_comparison.mp4") if concat else None,
        }
    return {
        "status": status,
        "baseline": "ViMax VideoGeneratorMinimaxH3VLLM",
        "movie_id": "0001_American_Beauty",
        "arc_id": "0001_American_Beauty_arc_01",
        "comparison_plan": str(plan_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "planned_jobs": len(jobs),
        "missing_reference_count": sum(len(paths) for paths in missing.values()),
        "missing_references": missing,
        "conditions": groups,
        "parameters": {
            "model": os.environ.get("MINIMAX_H3_VLLM_MODEL", "server-configured; not reported by endpoint"),
            "width": os.environ.get("MINIMAX_H3_WIDTH", "1024"),
            "height": os.environ.get("MINIMAX_H3_HEIGHT", "576"),
            "fps": os.environ.get("MINIMAX_H3_FPS", "24"),
            "num_inference_steps": os.environ.get("MINIMAX_H3_STEPS", "30"),
            "flow_shift": os.environ.get("MINIMAX_H3_VLLM_FLOW_SHIFT", "12"),
            "audio_flow_shift": os.environ.get("MINIMAX_H3_VLLM_AUDIO_FLOW_SHIFT", "3.0"),
            "aspect_ratio": os.environ.get("MINIMAX_H3_VLLM_ASPECT_RATIO", "16:9"),
            "timeout_sec": os.environ.get("MINIMAX_H3_VLLM_TIMEOUT_SEC", "3600"),
            "duration_sec_per_node": sorted({job["duration_sec"] for job in jobs}),
            "num_frames_per_node": sorted({job["num_frames"] for job in jobs}),
            "seed": "uncontrolled (not sent)" if os.environ.get("MINIMAX_H3_SEED", "none").lower() in {"", "none", "null", "unseeded"} else os.environ["MINIMAX_H3_SEED"],
            "api_url": os.environ.get("MINIMAX_H3_VLLM_API_URL", "http://127.0.0.1:9001/v1/videos/sync"),
        },
    }


async def _generate_job(job: dict[str, Any], output_path: Path, retries: int) -> None:
    original = {key: os.environ.get(key) for key in ("MINIMAX_H3_TASK", "MINIMAX_H3_NUM_FRAMES", "MINIMAX_H3_VLLM_DURATION", "MINIMAX_H3_SEED")}
    os.environ["MINIMAX_H3_TASK"] = job["task"]
    os.environ["MINIMAX_H3_NUM_FRAMES"] = str(job["num_frames"])
    os.environ["MINIMAX_H3_VLLM_DURATION"] = str(job["duration_sec"])
    os.environ["MINIMAX_H3_SEED"] = "none"
    generator = VideoGeneratorMinimaxH3VLLM()
    try:
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                output = await generator.generate_single_video(
                    prompt=job["prompt"],
                    reference_image_paths=job["reference_images"],
                    output_video_path=str(output_path),
                )
                output.save(str(output_path))
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    raise RuntimeError("generator produced an empty video file")
                job.update(status="success", output_path=str(output_path), attempts=attempt)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("job failed attempt=%d/%d node=%s condition=%s: %s", attempt, retries, job["node_id"], job["condition"], last_error)
                if attempt < retries:
                    await asyncio.sleep(min(attempt * 2, 10))
        job.update(status="failed", output_path=str(output_path), attempts=retries, error=last_error)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(
    plan_path: Path,
    output_dir: Path,
    *,
    run_generation: bool,
    condition: str | None = None,
    limit: int | None = None,
    retries: int = 3,
    concat: bool = True,
) -> dict[str, Any]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"Comparison plan not found: {plan_path}")
    if retries < 1:
        raise ValueError("--retries must be at least 1")
    jobs = build_jobs(_read_json(plan_path), condition=condition, limit=limit)
    for job in jobs:
        job["output_path"] = str(_job_output_path(output_dir, job))
    missing = _missing_references(jobs)
    if run_generation and missing:
        raise FileNotFoundError(
            "Cannot start baseline generation because planned reference images are missing: "
            + "; ".join(f"{key} -> {', '.join(paths)}" for key, paths in missing.items())
        )
    if run_generation and os.environ.get("MINIMAX_H3_VLLM_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError("Set MINIMAX_H3_VLLM_ENABLED=true before --run")

    jobs_by_condition = {name: [job for job in jobs if job["condition"] == name] for name in CONDITIONS if any(job["condition"] == name for job in jobs)}
    if not run_generation:
        for job in jobs:
            if any(not Path(path).is_file() for path in job["reference_images"]):
                job["status"] = "blocked_missing_references"
        _write_manifests(output_dir, jobs_by_condition)
        result = _summary(status="dry_run", plan_path=plan_path, output_dir=output_dir, jobs=jobs, missing=missing, concat=concat)
        _write_json(output_dir / "baseline_summary.json", result)
        return result

    async def generate_all() -> None:
        for job in jobs:
            output_path = Path(job["output_path"])
            if output_path.is_file() and output_path.stat().st_size:
                job.update(status="skipped", attempts=0)
            else:
                job.update(status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                _write_manifests(output_dir, jobs_by_condition)
                await _generate_job(job, output_path, retries)
            _write_manifests(output_dir, jobs_by_condition)

    asyncio.run(generate_all())
    concat_results: dict[str, bool] = {}
    if concat:
        for condition_name, condition_jobs in jobs_by_condition.items():
            paths = [Path(job["output_path"]) for job in condition_jobs]
            if condition_jobs and all(path.is_file() and path.stat().st_size for path in paths):
                final_path = output_dir / condition_name / "final_comparison.mp4"
                try:
                    concatenate_video_files([str(path) for path in paths], str(final_path))
                    concat_results[condition_name] = final_path.is_file() and final_path.stat().st_size > 0
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("concat failed for %s", condition_name)
                    concat_results[condition_name] = False
                    for job in condition_jobs:
                        job["concat_error"] = f"{type(exc).__name__}: {exc}"
            else:
                concat_results[condition_name] = False

    all_success = all(job["status"] in {"success", "skipped"} for job in jobs)
    concat_success = not concat or all(concat_results.values())
    result = _summary(
        status="done" if all_success and concat_success else "partial",
        plan_path=plan_path,
        output_dir=output_dir,
        jobs=jobs,
        missing=missing,
        concat=concat,
    )
    for name, ok in concat_results.items():
        result["conditions"][name]["concat_ok"] = ok
    _write_manifests(output_dir, jobs_by_condition)
    _write_json(output_dir / "baseline_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="CameraMaster video_comparison_plan.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition", choices=CONDITIONS, help="Run or inspect one condition only")
    parser.add_argument("--limit", type=int, help="Limit to the first N nodes in plan order")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action="store_true", help="Submit video-generation requests; default is no-call dry-run")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.run and os.environ.get("MINIMAX_H3_SEED", "none").lower() not in {"", "none", "null", "unseeded"}:
        LOGGER.warning("Ignoring MINIMAX_H3_SEED for this protocol run; the protocol leaves video seed uncontrolled")
    summary = run(
        args.plan,
        args.output_dir,
        run_generation=args.run,
        condition=args.condition,
        limit=args.limit,
        retries=args.retries,
        concat=args.concat,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.run and summary["status"] != "done":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
