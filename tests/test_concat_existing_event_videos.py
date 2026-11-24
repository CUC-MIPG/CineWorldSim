from __future__ import annotations

import json
from pathlib import Path

from scripts.concat_existing_event_videos import collect_existing_video_paths


def test_collect_existing_video_paths_uses_task_order_and_skips_missing(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    summary = tmp_path / "event_video_summary.json"
    summary.write_text(
        json.dumps(
            {
                "generated": [
                    {"status": "done", "output_video_path": str(first)},
                    {"status": "failed", "output_video_path": str(tmp_path / "failed.mp4")},
                ]
            }
        ),
        encoding="utf-8",
    )
    tasks = tmp_path / "event_video_tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {"output_video_path": str(second)},
                {"output_video_path": str(first)},
            ]
        ),
        encoding="utf-8",
    )

    assert collect_existing_video_paths(summary, tasks) == [str(first.resolve())]
