import json
from pathlib import Path

import pytest

from prepare_vimax_sss_case import build_case_inputs


def test_build_case_inputs_extracts_story_and_preserves_track_a_clip_order(tmp_path):
    output_root = tmp_path / "outputs"
    track_a = output_root / "track_A"
    track_b = output_root / "track_B"
    (track_a / "video").mkdir(parents=True)
    (track_b / "video").mkdir(parents=True)
    ids_a = ["scene_001_shot_001", "scene_001_shot_002"]
    ids_b = ["NN_001", "NN_002"]
    for index, node_id in enumerate(ids_a, 1):
        (track_a / "video" / f"event_video_{index:03d}_{node_id}.mp4").touch()
    for index, node_id in enumerate(ids_b, 1):
        (track_b / "video" / f"event_video_{index:03d}_{node_id}.mp4").touch()
    (track_a / "video" / "final_orchestrated_events.mp4").touch()
    (track_b / "video" / "final_orchestrated_events.mp4").touch()
    (track_a / "path_plan.json").write_text(json.dumps({"ordered_node_ids": ids_a}), encoding="utf-8")
    (track_b / "path_plan.json").write_text(json.dumps({"ordered_node_ids": ids_b}), encoding="utf-8")
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"story_overview": "Reference story."}), encoding="utf-8")

    result = build_case_inputs(case, output_root)

    assert result["summary"] == "Reference story."
    sample_a = next(row for row in result["adapter"]["samples"] if row["track"] == "A")
    assert sample_a["ordered_node_ids"] == ids_a
    assert [Path(path).name for path in sample_a["source_clip_paths"]] == [
        "event_video_001_scene_001_shot_001.mp4",
        "event_video_002_scene_001_shot_002.mp4",
    ]
    assert result["summary"] not in json.dumps(result["adapter"])


def test_build_case_inputs_fails_on_missing_clip(tmp_path):
    output_root = tmp_path / "outputs"
    for track in ("A", "B"):
        (output_root / f"track_{track}" / "video").mkdir(parents=True)
        (output_root / f"track_{track}" / "path_plan.json").write_text(
            json.dumps({"ordered_node_ids": ["node_001"]}), encoding="utf-8"
        )
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"story_overview": "Reference story."}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected exactly one clip"):
        build_case_inputs(case, output_root)
