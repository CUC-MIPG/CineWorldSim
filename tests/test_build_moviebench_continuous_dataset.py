from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_moviebench_continuous_dataset import build_continuous_dataset


def _write_clip(path: Path, plot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "Characters": {"Hero": "calm"},
                "Style Elements": "warm cinematic lighting",
                "Plot": plot,
                "Background Description": "a room",
                "Camera Motion": "static",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_continuous_dataset_groups_clips_and_keeps_required_fields(tmp_path):
    source = tmp_path / "source"
    movie = source / "MOVIE_A"
    _write_clip(movie / "MOVIE_A_00.00.00.000-00.00.10.000.json", "first")
    _write_clip(movie / "MOVIE_A_00.00.10.500-00.00.20.000.json", "second")
    _write_clip(movie / "MOVIE_A_00.00.21.000-00.00.30.000.json", "third")
    _write_clip(movie / "MOVIE_A_00.00.30.500-00.00.40.000.json", "fourth")
    output = tmp_path / "continuous"

    index = build_continuous_dataset(root=source, output_root=output, threshold_ms=1000, min_clips=1)

    assert index["movies"] == 1
    assert index["total_clips"] == 4
    assert index["total_sequences"] == 2
    files = sorted((output / "MOVIE_A").glob("*.json"))
    assert len(files) == 2
    first_group = json.loads(files[0].read_text(encoding="utf-8"))
    second_group = json.loads(files[1].read_text(encoding="utf-8"))
    assert first_group["clip_count"] == 2
    assert second_group["clip_count"] == 2
    clip = first_group["clips"][0]
    for key in [
        "Characters",
        "Style Elements",
        "Plot",
        "Background Description",
        "Camera Motion",
        "clip_start_tc",
        "clip_end_tc",
        "duration_sec",
    ]:
        assert key in clip
    assert clip["Plot"] == "first"
    assert clip["duration_sec"] == 10.0
