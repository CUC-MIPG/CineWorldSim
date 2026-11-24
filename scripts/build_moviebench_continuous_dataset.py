from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_moviebench_clip_metadata import ClipFile, collect_groups, ms_to_sec


REQUIRED_CLIP_FIELDS = [
    "Characters",
    "Style Elements",
    "Plot",
    "Background Description",
    "Camera Motion",
]


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_continuous(prev: ClipFile, current: ClipFile, threshold_ms: int) -> bool:
    return abs(current.start_ms - prev.end_ms) < threshold_ms


def _split_continuous_sequences(clips: list[ClipFile], threshold_ms: int) -> list[list[ClipFile]]:
    if not clips:
        return []
    sequences: list[list[ClipFile]] = [[clips[0]]]
    for clip in clips[1:]:
        prev = sequences[-1][-1]
        if _is_continuous(prev, clip, threshold_ms):
            sequences[-1].append(clip)
        else:
            sequences.append([clip])
    return sequences


def _clip_payload(clip: ClipFile, source: dict[str, Any]) -> dict[str, Any]:
    row = {field: source.get(field) for field in REQUIRED_CLIP_FIELDS}
    row.update(
        {
            "file_name": clip.path.name,
            "source_json": str(clip.path),
            "clip_start_tc": clip.start_tc,
            "clip_end_tc": clip.end_tc,
            "clip_start_ms": clip.start_ms,
            "clip_end_ms": clip.end_ms,
            "duration_sec": ms_to_sec(clip.duration_ms),
        }
    )
    return row


def _sequence_payload(
    *,
    movie_id: str,
    sequence_index: int,
    clips: list[ClipFile],
    clip_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    first = clips[0]
    last = clips[-1]
    return {
        "movie_id": movie_id,
        "sequence_id": f"{movie_id}_continuous_{sequence_index:04d}",
        "sequence_index": sequence_index,
        "clip_count": len(clips),
        "time_range": {
            "clip_start_tc": first.start_tc,
            "clip_end_tc": last.end_tc,
            "clip_start_ms": first.start_ms,
            "clip_end_ms": last.end_ms,
            "duration_sec": ms_to_sec(last.end_ms - first.start_ms),
        },
        "source_files": [clip.path.name for clip in clips],
        "clips": clip_payloads,
    }


def build_continuous_dataset(
    *,
    root: Path,
    output_root: Path,
    threshold_ms: int = 1000,
    min_clips: int = 1,
) -> dict[str, Any]:
    groups = collect_groups(root)
    output_root.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    total_clips = 0
    total_sequences = 0
    skipped_invalid_json = 0

    for movie_id, clips in sorted(groups.items()):
        movie_output = output_root / movie_id
        movie_output.mkdir(parents=True, exist_ok=True)
        sequences = _split_continuous_sequences(clips, threshold_ms=threshold_ms)
        written_for_movie = 0

        for sequence in sequences:
            loaded: list[tuple[ClipFile, dict[str, Any]]] = []
            for clip in sequence:
                payload = _load_json_object(clip.path)
                if payload is None:
                    skipped_invalid_json += 1
                    continue
                loaded.append((clip, payload))
            if len(loaded) < max(1, int(min_clips)):
                continue

            written_for_movie += 1
            total_sequences += 1
            sequence_clips = [clip for clip, _ in loaded]
            clip_rows = [_clip_payload(clip, payload) for clip, payload in loaded]
            total_clips += len(clip_rows)
            payload = _sequence_payload(
                movie_id=movie_id,
                sequence_index=written_for_movie,
                clips=sequence_clips,
                clip_payloads=clip_rows,
            )
            out_file = movie_output / f"{payload['sequence_id']}.json"
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            index_rows.append(
                {
                    "movie_id": movie_id,
                    "sequence_id": payload["sequence_id"],
                    "clip_count": payload["clip_count"],
                    "clip_start_tc": payload["time_range"]["clip_start_tc"],
                    "clip_end_tc": payload["time_range"]["clip_end_tc"],
                    "duration_sec": payload["time_range"]["duration_sec"],
                    "file": str(out_file),
                }
            )

    index = {
        "source_root": str(root),
        "output_root": str(output_root),
        "continuity_rule": {
            "threshold_ms": int(threshold_ms),
            "description": "Two neighboring clips are continuous when abs(current.start_ms - previous.end_ms) < threshold_ms.",
        },
        "min_clips": max(1, int(min_clips)),
        "movies": len(groups),
        "total_sequences": total_sequences,
        "total_clips": total_clips,
        "skipped_invalid_json": skipped_invalid_json,
        "sequences": index_rows,
    }
    (output_root / "continuous_dataset_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a MovieBench continuous-clip dataset copy. Each continuous clip run is saved as one JSON."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("MovieBench/Annotation_Shot_Desc_11.15_V2_160movies"),
        help="Source MovieBench annotation root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("MovieBench/Continuous_Clip_Groups_11.15_V2_160movies"),
        help="Output root for grouped continuous-sequence JSON files.",
    )
    parser.add_argument("--threshold-ms", type=int, default=1000)
    parser.add_argument("--min-clips", type=int, default=1)
    args = parser.parse_args()

    index = build_continuous_dataset(
        root=args.root,
        output_root=args.output_root,
        threshold_ms=max(1, int(args.threshold_ms)),
        min_clips=max(1, int(args.min_clips)),
    )
    print("=== MovieBench Continuous Dataset Built ===")
    print(f"source_root: {index['source_root']}")
    print(f"output_root: {index['output_root']}")
    print(f"movies: {index['movies']}")
    print(f"total_sequences: {index['total_sequences']}")
    print(f"total_clips: {index['total_clips']}")
    print(f"skipped_invalid_json: {index['skipped_invalid_json']}")
    print(f"index: {Path(index['output_root']) / 'continuous_dataset_index.json'}")


if __name__ == "__main__":
    main()
