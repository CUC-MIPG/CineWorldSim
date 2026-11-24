from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<start>\d{2}\.\d{2}\.\d{2}\.\d{3})-(?P<end>\d{2}\.\d{2}\.\d{2}\.\d{3})\.json$"
)
CONTINUITY_GAP_THRESHOLD_MS = 1000


@dataclass
class ClipFile:
    path: Path
    movie_id: str
    start_tc: str
    end_tc: str
    start_ms: int
    end_ms: int
    duration_ms: int


def tc_to_ms(tc: str) -> int:
    hh, mm, ss, mmm = [int(x) for x in tc.split(".")]
    return ((hh * 60 + mm) * 60 + ss) * 1000 + mmm


def ms_to_sec(ms: int) -> float:
    return round(ms / 1000.0, 3)


def parse_clip_file(path: Path, movie_id: str) -> ClipFile:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected file name format: {path}")

    start_tc = m.group("start")
    end_tc = m.group("end")
    start_ms = tc_to_ms(start_tc)
    end_ms = tc_to_ms(end_tc)
    duration_ms = end_ms - start_ms
    return ClipFile(
        path=path,
        movie_id=movie_id,
        start_tc=start_tc,
        end_tc=end_tc,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
    )


def collect_groups(root: Path) -> dict[str, list[ClipFile]]:
    groups: dict[str, list[ClipFile]] = {}
    for p in root.rglob("*.json"):
        if not p.is_file():
            continue
        if not FILENAME_RE.match(p.name):
            continue
        rel = p.relative_to(root)
        movie_id = rel.parts[0] if rel.parts else p.parent.name
        clip = parse_clip_file(p, movie_id=movie_id)
        groups.setdefault(movie_id, []).append(clip)

    for movie_id in groups:
        groups[movie_id].sort(key=lambda x: (x.start_ms, x.end_ms, x.path.name))
    return groups


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def enrich_groups(groups: dict[str, list[ClipFile]], write: bool) -> dict[str, int]:
    total = 0
    updated = 0
    skipped = 0

    for movie_id, clips in groups.items():
        prev: ClipFile | None = None
        for idx, clip in enumerate(clips, start=1):
            total += 1
            raw = clip.path.read_text(encoding="utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                prev = clip
                continue

            if not isinstance(payload, dict):
                skipped += 1
                prev = clip
                continue

            gap_ms = None if prev is None else (clip.start_ms - prev.end_ms)
            is_continuous = None if prev is None else (abs(gap_ms) < CONTINUITY_GAP_THRESHOLD_MS)
            prev_file = None if prev is None or not is_continuous else prev.path.name
            prev_end_tc = None if prev is None or not is_continuous else prev.end_tc
            prev_end_ms = None if prev is None or not is_continuous else prev.end_ms
            prev_chronological_file = None if prev is None else prev.path.name
            prev_chronological_end_tc = None if prev is None else prev.end_tc
            prev_chronological_end_ms = None if prev is None else prev.end_ms

            payload["clip_meta"] = {
                "movie_id": movie_id,
                "file_name": clip.path.name,
                "clip_start_tc": clip.start_tc,
                "clip_end_tc": clip.end_tc,
                "clip_start_ms": clip.start_ms,
                "clip_end_ms": clip.end_ms,
                "duration_ms": clip.duration_ms,
                "duration_sec": ms_to_sec(clip.duration_ms),
                "order_index_in_movie": idx,
                "prev_file_name": prev_file,
                "prev_end_tc": prev_end_tc,
                "prev_end_ms": prev_end_ms,
                "prev_chronological_file_name": prev_chronological_file,
                "prev_chronological_end_tc": prev_chronological_end_tc,
                "prev_chronological_end_ms": prev_chronological_end_ms,
                "is_continuous_with_prev": is_continuous,
                "time_diff_from_prev_ms": gap_ms,
                "time_diff_from_prev_sec": None if gap_ms is None else ms_to_sec(gap_ms),
            }

            updated += 1
            if write:
                write_json(clip.path, payload)

            prev = clip

    return {
        "total": total,
        "updated": updated,
        "skipped": skipped,
        "movies": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process MovieBench shot JSON files: parse start/end from filename, "
            "compute duration, assign chronological index in each movie folder, "
            "and write continuity metadata against previous clip."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data1/AgentG/hhb/CameraMaster/MovieBench/Annotation_Shot_Desc_11.15_V2_160movies"
        ),
        help="Root directory containing per-movie subfolders with JSON clips.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to files. Without this flag it runs in dry-run mode.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    groups = collect_groups(root)
    stats = enrich_groups(groups, write=args.write)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] movies={stats['movies']} total={stats['total']} updated={stats['updated']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
