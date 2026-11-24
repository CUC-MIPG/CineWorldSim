#!/usr/bin/env python3
"""Evaluate existing ViMax benchmark Pipeline outputs with ViCLIP embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vimax_pipeline_eval import (  # noqa: E402
    DeterministicFakeEncoder,
    ViCLIPHTTPEncoder,
    discover_track_a,
    discover_track_b,
    evaluate_sample,
    load_benchmark_cases,
    make_report,
)

LOGGER = logging.getLogger("vimax_pipeline_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "vimax_benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "vimax_benchmark" / "all_cases")
    parser.add_argument("--case", action="append", help="Theme, source filename, or stem; repeatable")
    parser.add_argument("--track", choices=("A", "B", "both"), default="both")
    parser.add_argument("--endpoint", default="http://172.17.34.43:8890/embed")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--expected-dimension", type=int, default=768)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "vimax_benchmark" / "evaluation_report.json")
    parser.add_argument("--track-b-scene-diagnostics", action="store_true",
                        help="Compute separate diagnostic IC/CC using generated Track B scene groups")
    parser.add_argument("--smoke", action="store_true", help="Use a deterministic fake encoder; report is non-reportable")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.num_frames <= 0 or args.resolution <= 0 or args.timeout <= 0:
        raise ValueError("--num-frames, --resolution, and --timeout must be positive")
    cases = load_benchmark_cases(args.benchmark_root)
    selectors = {value.lower().removesuffix(".json") for value in (args.case or [])}
    if selectors:
        cases = [row for row in cases if selectors.intersection(
            {row[0].lower(), row[1].name.lower(), row[1].stem.lower()})]
    if not cases:
        raise ValueError("No benchmark cases matched the requested selector")

    encoder = DeterministicFakeEncoder() if args.smoke else ViCLIPHTTPEncoder(
        args.endpoint, args.num_frames, args.resolution, args.timeout, args.expected_dimension)
    cache_dir = args.cache_dir or (args.output_root / ".viclip_embedding_cache")
    tracks = ("A", "B") if args.track == "both" else (args.track,)
    records = []
    for theme, source, source_case in cases:
        for track in tracks:
            pipeline_root = args.output_root / theme / f"track_{track}" / "vimax_pipeline"
            if track == "A":
                sample = discover_track_a(source_case, pipeline_root)
            else:
                sample = discover_track_b(pipeline_root, args.track_b_scene_diagnostics)
            sample.update({"case": theme, "source_case": str(source.resolve()),
                           "manifest_path": str((args.output_root / theme / "case_manifest.json").resolve()),
                           "manifest_exists": (args.output_root / theme / "case_manifest.json").is_file()})
            sample["track"] = track
            if sample["status"] == "complete":
                sample = evaluate_sample(sample, encoder, cache_dir, f"{theme}/track_{track}",
                                         args.track_b_scene_diagnostics)
            LOGGER.info("%s track %s: %s (%d/%d shots)", theme, track, sample["status"],
                        sample["found_shot_count"], sample["expected_shot_count"])
            records.append(sample)

    report = make_report(records, encoder, {
        "endpoint": None if args.smoke else args.endpoint,
        "num_frames": args.num_frames, "resolution": args.resolution, "timeout_seconds": args.timeout,
        "expected_dimension": args.expected_dimension, "smoke_mode": args.smoke,
        "cache_dir": str(cache_dir.resolve()),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report: {args.report.resolve()}")
    print(f"Complete samples: {report['complete_sample_count']}/{report['sample_count']}")
    for row in report["case_scores"]:
        def display(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.4f}"
        print(f"{row['case']} Track {row['track']} [{row['status']}]: "
              f"IC={display(row['IC'])} CC={display(row['CC'])} GC={display(row['GC'])}")
    return 0 if report["complete_sample_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
