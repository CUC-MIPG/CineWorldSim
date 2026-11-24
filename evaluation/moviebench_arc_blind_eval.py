"""Prepare blinded MovieBench arc reviews and summarize human/LLM score files."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from pathlib import Path
from typing import Any

CONDITIONS = ("fl2va_event_keyframe_only", "ref2va_semantic_assets")
DIMENSIONS = (
    "event_action_fidelity",
    "character_location_consistency",
    "relevant_object_fidelity",
    "temporal_coherence",
    "overall_quality",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    for key in ("tasks", "items", "videos"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("Expected a JSON list or an object containing tasks/items/videos")


def export_blind_tasks(arc_dir: Path, seed: int | None = None) -> dict[str, str]:
    arc_dir = Path(arc_dir).resolve()
    plan_path = arc_dir / "video_comparison_plan.json"
    plan = _rows(_read_json(plan_path))
    plan_by_node = {str(row["node_id"]): row for row in plan}
    if len(plan_by_node) != len(plan):
        raise ValueError("video_comparison_plan.json contains duplicate node_id values")

    found: dict[tuple[str, str], Path] = {}
    missing: list[dict[str, str]] = []
    for condition in CONDITIONS:
        manifest_path = arc_dir / "video_comparison" / condition / "event_video_tasks.json"
        manifest_by_node: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            for row in _rows(_read_json(manifest_path)):
                if row.get("node_id"):
                    manifest_by_node[str(row["node_id"])] = row
        for node_id in plan_by_node:
            row = manifest_by_node.get(node_id)
            video = Path(str((row or {}).get("output_video_path", "")))
            if not video.is_absolute():
                video = (arc_dir / video).resolve()
            if row and row.get("status") in {"done", "success", "succeeded"} and video.is_file():
                found[(condition, node_id)] = video
            else:
                missing.append({"condition": condition, "node_id": node_id, "reason": "missing manifest entry, unsuccessful status, or missing video file"})

    assignments: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    for (condition, node_id), source in found.items():
        code = uuid.uuid4().hex[:10].upper()
        assignments.append({"blind_code": code, "condition": condition, "node_id": node_id})
        alias = arc_dir / "blind_media" / f"{code}{source.suffix or '.mp4'}"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(source)
        plan_row = plan_by_node[node_id]
        items.append({
            "blind_code": code,
            "node_id": node_id,
            "video_path": str(alias.relative_to(arc_dir)),
            "event_summary": plan_row.get("event_summary", ""),
            "story_description": plan_row.get("story_description", ""),
            "prompt": plan_row.get("prompt", ""),
            "duration_sec": plan_row.get("duration_sec"),
            "num_frames": plan_row.get("num_frames"),
        })
    (random.Random(seed) if seed is not None else random.SystemRandom()).shuffle(items)
    tasks_path = arc_dir / "blind_review_tasks.json"
    key_path = arc_dir / "blind_condition_key.json"
    template_path = arc_dir / "blind_scores_template.json"
    missing_nodes = sorted({row["node_id"] for row in missing})
    _write_json(tasks_path, {"dimensions": list(DIMENSIONS), "scale": {"min": 1, "max": 5, "na": "N/A"}, "items": items, "missing": [{"node_id": node_id} for node_id in missing_nodes]})
    _write_json(key_path, {"assignments": assignments, "missing": missing})
    _write_json(template_path, [{"blind_code": item["blind_code"], **{dimension: None for dimension in DIMENSIONS}, "reviewer_id": "", "notes": ""} for item in items])
    return {"tasks_path": str(tasks_path), "key_path": str(key_path), "score_template_path": str(template_path)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_and_summarize(scores_path: Path, key_path: Path) -> dict[str, Any]:
    scores = _rows(_read_json(Path(scores_path)))
    key_payload = _read_json(Path(key_path))
    assignments = key_payload.get("assignments", [])
    by_code = {row["blind_code"]: row for row in assignments}
    if len(by_code) != len(assignments):
        raise ValueError("Condition key contains duplicate blind codes")
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    for row in scores:
        code = str(row.get("blind_code", ""))
        if code not in by_code:
            raise ValueError(f"Unknown blind_code: {code}")
        if code in seen:
            raise ValueError(f"Duplicate score row for blind_code: {code}")
        seen.add(code)
        enriched = {**row, **by_code[code]}
        for dimension in DIMENSIONS:
            value = row.get(dimension)
            if value is None or (isinstance(value, str) and value.strip().lower() in {"n/a", "na", ""}):
                enriched[dimension] = None
            else:
                try:
                    score = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{dimension} must be 1-5 or N/A") from exc
                if score < 1 or score > 5:
                    raise ValueError(f"{dimension} must be 1-5 or N/A")
                enriched[dimension] = score
        valid.append(enriched)

    dimensions: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        groups = {condition: [row[dimension] for row in valid if row["condition"] == condition and row[dimension] is not None] for condition in CONDITIONS}
        dimensions[dimension] = {
            "n_by_condition": {condition: len(groups[condition]) for condition in CONDITIONS},
            "mean_by_condition": {condition: statistics.mean(values) if values else None for condition, values in groups.items()},
        }
        scores_by_node = {(row["condition"], row["node_id"]): row[dimension] for row in valid}
        diffs = [scores_by_node[(CONDITIONS[1], node)] - scores_by_node[(CONDITIONS[0], node)]
                 for node in {row["node_id"] for row in valid}
                 if scores_by_node.get((CONDITIONS[0], node)) is not None and scores_by_node.get((CONDITIONS[1], node)) is not None]
        paired[dimension] = {"n_pairs": len(diffs), "mean_difference_b_minus_a": statistics.mean(diffs) if diffs else None}
    return {"score_rows": len(valid), "unscored_blind_codes": sorted(set(by_code) - seen), "dimensions": dimensions, "paired_differences": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Create blinded review tasks and a separate condition key")
    export.add_argument("arc_dir", type=Path)
    export.add_argument("--seed", type=int)
    summarize = sub.add_parser("summarize", help="Validate scores and report per-dimension summaries")
    summarize.add_argument("scores", type=Path)
    summarize.add_argument("--key", required=True, type=Path)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_blind_tasks(args.arc_dir, args.seed)
        else:
            result = score_and_summarize(args.scores, args.key)
            if args.output:
                _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
