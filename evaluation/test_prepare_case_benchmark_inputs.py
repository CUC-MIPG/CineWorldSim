import json
from pathlib import Path

import pytest

from prepare_case_benchmark_inputs import build_track_record, prepare_case


def test_build_track_record_follows_plan_order_and_maps_video_metadata(tmp_path):
    videos = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    nodes = [
        {"node_id": "n1", "event_id": "e1", "characters": ["A"], "locations": ["room"]},
        {"node_id": "n2", "event_id": "e2", "characters": ["B"], "locations": ["street"]},
    ]
    generated = [
        {"node_id": "n2", "event_id": "e2", "output_video_path": str(videos[1]), "story_description": "Second event.", "video_description": "B leaves.", "dialogue": "none", "duration_sec": 5},
        {"node_id": "n1", "event_id": "e1", "output_video_path": str(videos[0]), "story_description": "First event.", "video_description": "A enters.", "dialogue": "A: Hello.", "duration_sec": 5},
    ]
    path_plan = {"ordered_node_ids": ["n1", "n2"]}
    narrative_nodes = {"storyline": "Sequence synopsis.", "selected_nodes": nodes}

    result = build_track_record("A", "movie_001", generated, path_plan, narrative_nodes)

    assert result["ordered_node_ids"] == ["n1", "n2"]
    assert [clip["output_video_path"] for clip in result["clips"]] == [str(videos[0]), str(videos[1])]
    assert result["generated_story"] == "First event.\nDialogue: A: Hello.\n\nSecond event."
    assert [shot["node_id"] for shot in result["storyboard"]] == ["n1", "n2"]
    assert result["prompt"] == "Sequence synopsis."
    assert result["prompt_source"] == "narrative_nodes.storyline"


def test_build_track_record_uses_track_b_context_storyline():
    generated = [{"node_id": "NN_1", "output_video_path": "/clips/one.mp4", "story_description": "One.", "dialogue": "none"}]
    path_plan = {"ordered_node_ids": ["NN_1"]}
    narrative_nodes = {
        "storyline": "Global storyline.",
        "track_b_context": {"storyline": "Track B context."},
        "selected_nodes": [{"node_id": "NN_1", "event_id": "EVT_1"}],
    }

    result = build_track_record("B", "movie_001", generated, path_plan, narrative_nodes)

    assert result["prompt"] == "Track B context."
    assert result["prompt_source"] == "track_b_context.storyline"


def test_prepare_case_exports_benchmark_inputs_and_uses_movie_directory_name(tmp_path, monkeypatch):
    case_root = tmp_path / "outputs" / "moviebench_video_pipeline" / "0001_American_Beauty" / "arc_001_real"
    case_root.mkdir(parents=True)
    clip_paths = [case_root / f"track_{track}.mp4" for track in ("A", "B")]
    for path in clip_paths:
        path.touch()

    track_runs = []
    for track, clip_path in zip(("A", "B"), clip_paths):
        artifact_dir = case_root / f"track_{track}"
        artifact_dir.mkdir()
        (artifact_dir / "nodes.json").write_text(json.dumps({
            "storyline": f"Track {track} storyline.",
            "selected_nodes": [{"node_id": f"n{track}", "event_id": f"e{track}"}],
        }), encoding="utf-8")
        (artifact_dir / "plan.json").write_text(json.dumps({"ordered_node_ids": [f"n{track}"]}), encoding="utf-8")
        track_runs.append({"track": track, "samples": [{"video_summary": {
            "failed": 0,
            "skipped": 0,
            "generated": [{
                "node_id": f"n{track}",
                "output_video_path": str(clip_path),
                "story_description": f"Track {track} event.",
            }],
            "narrative_nodes_json": str(artifact_dir / "nodes.json"),
            "path_plan_json": str(artifact_dir / "plan.json"),
        }}]})
    (case_root / "moviebench_pipeline_manifest.json").write_text(
        json.dumps({"track_runs": track_runs}), encoding="utf-8"
    )

    def fake_concat(paths, list_path, output_path):
        output_path.touch()
        return {"duration_sec": 60.0, "has_audio": True}

    monkeypatch.setattr("prepare_case_benchmark_inputs._concat_videos", fake_concat)
    output_dir = tmp_path / "prepared"
    report = prepare_case(case_root, output_dir)

    assert report["tracks"]["A"]["sample_id"] == "0001_American_Beauty__TrackA"
    assert report["tracks"]["B"]["sample_id"] == "0001_American_Beauty__TrackB"
    assert report["tracks"]["B"]["duration_status"] == "minute_scale_candidate"
    import pandas as pd
    rows = pd.read_parquet(output_dir / "constory" / "cameramaster_stories.parquet")
    assert list(rows["generated_story"]) == ["Track A event.", "Track B event."]
    assert len(json.loads((output_dir / "directorbench" / "inputs.json").read_text())["samples"]) == 2

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_case(case_root, output_dir)


@pytest.mark.parametrize(
    ("ordered_ids", "generated_ids", "node_ids", "message"),
    [
        (["n1", "n1"], ["n1"], ["n1"], "duplicate.*ordered"),
        (["n1"], ["n1", "n1"], ["n1"], "duplicate.*generated"),
        (["n2"], ["n1"], ["n1"], "no generated record"),
        (["n1"], ["n1"], [], "no narrative node"),
    ],
)
def test_build_track_record_rejects_unreconciled_node_ids(ordered_ids, generated_ids, node_ids, message):
    generated = [
        {"node_id": node_id, "output_video_path": f"/{node_id}.mp4", "story_description": "Text."}
        for node_id in generated_ids
    ]
    narrative_nodes = {"storyline": "Story.", "selected_nodes": [{"node_id": node_id} for node_id in node_ids]}

    with pytest.raises(ValueError, match=message):
        build_track_record("A", "movie_001", generated, {"ordered_node_ids": ordered_ids}, narrative_nodes)
