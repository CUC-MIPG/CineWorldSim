from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_moviebench_event_descriptions_from_continuous_shots import build_event_descriptions


def _write_sequence(path: Path, sequence_id: str, clip_count: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    clips = [
        {
            "Characters": {"Hero": "focused"},
            "Style Elements": "tense cinematic lighting",
            "Plot": f"shot {idx} plot",
            "Background Description": "industrial room",
            "Camera Motion": "slow push in",
            "clip_start_tc": f"00.00.{idx:02d}.000",
            "clip_end_tc": f"00.00.{idx + 1:02d}.000",
            "duration_sec": 1.0,
        }
        for idx in range(clip_count)
    ]
    payload = {
        "movie_id": "MOVIE_A",
        "sequence_id": sequence_id,
        "clip_count": clip_count,
        "time_range": {
            "clip_start_tc": "00.00.00.000",
            "clip_end_tc": f"00.00.{clip_count:02d}.000",
            "duration_sec": float(clip_count),
        },
        "source_files": [f"clip_{idx}.json" for idx in range(clip_count)],
        "clips": clips,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def test_build_event_descriptions_filters_sequences_over_three_shots(tmp_path):
    input_root = tmp_path / "continuous"
    output_root = tmp_path / "events"
    seq4 = input_root / "MOVIE_A" / "MOVIE_A_continuous_0001.json"
    seq3 = input_root / "MOVIE_A" / "MOVIE_A_continuous_0002.json"
    _write_sequence(seq4, "MOVIE_A_continuous_0001", 4)
    _write_sequence(seq3, "MOVIE_A_continuous_0002", 3)
    index = {
        "sequences": [
            {"movie_id": "MOVIE_A", "sequence_id": "MOVIE_A_continuous_0001", "clip_count": 4, "file": str(seq4)},
            {"movie_id": "MOVIE_A", "sequence_id": "MOVIE_A_continuous_0002", "clip_count": 3, "file": str(seq3)},
        ]
    }
    (input_root / "continuous_dataset_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = build_event_descriptions(input_root=input_root, output_root=output_root, min_shots=4, dry_run=True)

    assert result["total_candidates"] == 1
    assert result["written"] == 1
    event_files = sorted((output_root / "MOVIE_A").glob("*.json"))
    assert len(event_files) == 1
    event = json.loads(event_files[0].read_text(encoding="utf-8"))
    assert event["source_sequence_id"] == "MOVIE_A_continuous_0001"
    assert event["shot_count"] == 4
    assert event["event_description"]
    assert event["llm"]["status"] == "dry_run"
    assert len(event["shots"]) == 4
    for key in ["Characters", "Style Elements", "Plot", "Background Description", "Camera Motion"]:
        assert key in event["shots"][0]


def test_dry_run_event_description_includes_every_shot_plot(tmp_path):
    input_root = tmp_path / "continuous"
    output_root = tmp_path / "events"
    seq10 = input_root / "MOVIE_A" / "MOVIE_A_continuous_0001.json"
    _write_sequence(seq10, "MOVIE_A_continuous_0001", 10)
    (input_root / "continuous_dataset_index.json").write_text(
        json.dumps(
            {
                "sequences": [
                    {"movie_id": "MOVIE_A", "sequence_id": "MOVIE_A_continuous_0001", "clip_count": 10, "file": str(seq10)}
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    build_event_descriptions(input_root=input_root, output_root=output_root, min_shots=4, dry_run=True)

    event_file = next((output_root / "MOVIE_A").glob("*.json"))
    event = json.loads(event_file.read_text(encoding="utf-8"))
    assert "shot 0 plot" in event["event_description"]
    assert "shot 9 plot" in event["event_description"]
