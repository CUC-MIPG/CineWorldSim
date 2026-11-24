"""Aggregate CameraMaster evaluation outputs without collapsing metric families."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in records for name, value in row.items() if isinstance(value, (int, float))})
    return {name: {"mean": statistics.mean([float(row[name]) for row in records if isinstance(row.get(name), (int, float))]), "n": sum(isinstance(row.get(name), (int, float)) for row in records)} for name in names}


def _collect_metric_family(arc_dirs: list[Path], filename: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sources: list[str] = []
    for arc in arc_dirs:
        path = arc / filename
        if not path.is_file():
            continue
        payload = _load(path)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            payload = payload["metrics"]
        if isinstance(payload, dict):
            records.append(payload)
            sources.append(str(path))
    return {"status": "available" if records else "unavailable", "sample_count": len(records), "metrics": _mean_metrics(records) if records else {}, "sources": sources}


def build_report(moviebench_root: Path, external_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    root = Path(moviebench_root).resolve()
    summary_paths = sorted(root.rglob("moviebench_arc_video_eval_summary.json")) if root.exists() else []
    arc_dirs = [path.parent for path in summary_paths]
    summaries = [_load(path) for path in summary_paths]
    total_jobs = success_jobs = failed_jobs = skipped_jobs = 0
    status_counts: dict[str, int] = {}
    for summary in summaries:
        status = str(summary.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        for group in (summary.get("groups") or {}).values():
            total_jobs += int(group.get("total", 0) or 0)
            success_jobs += int(group.get("success", 0) or 0)
            failed_jobs += int(group.get("failed", 0) or 0)
            skipped_jobs += int(group.get("skipped", 0) or 0)
        if not summary.get("groups") and summary.get("planned_video_jobs") is not None:
            total_jobs += int(summary.get("planned_video_jobs", 0) or 0)

    external: dict[str, Any] = {}
    for name, path in (external_paths or {}).items():
        if Path(path).is_file():
            external[name] = _load(Path(path))

    report = {
        "schema_version": "cameramaster-evaluation-report-v1",
        "scope": {"moviebench_root": str(root), "arc_count": len(arc_dirs), "arc_summary_files": [str(path) for path in summary_paths], "included_families": ["track_a", "track_b", "external_benchmarks"]},
        "generation": {"total_jobs": total_jobs, "success_jobs": success_jobs, "failed_jobs": failed_jobs, "skipped_jobs": skipped_jobs, "arc_status_counts": status_counts},
        "track_a": _collect_metric_family(arc_dirs, "track_a_metrics.json"),
        "track_b": _collect_metric_family(arc_dirs, "track_b_metrics.json"),
        "external_benchmarks": external,
        "limitations": [
            "Metric families are reported separately; no uncalibrated overall score is computed.",
            "Missing metric files are reported as unavailable rather than treated as zero.",
            "External benchmark outputs are included verbatim and are not recomputed by this aggregator.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moviebench-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--external", action="append", default=[], metavar="NAME=PATH", help="Attach an existing external benchmark JSON report")
    args = parser.parse_args()
    external: dict[str, Path] = {}
    for item in args.external:
        if "=" not in item:
            parser.error("--external must use NAME=PATH")
        name, path = item.split("=", 1)
        external[name] = Path(path)
    report = build_report(args.moviebench_root, external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
