import json
from pathlib import Path

import numpy as np
import pytest

from scripts.vimax_pipeline_eval import (
    DeterministicFakeEncoder,
    ViCLIPHTTPEncoder,
    aggregate_dataset_metrics,
    aggregate_viclip_metrics,
    discover_track_a,
    discover_track_b,
    evaluate_sample,
    make_report,
)


def _write_video(path: Path, content: bytes = b"video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _source_case():
    return {
        "scenes": [
            {"scene_num": 2, "shots": [{"shot_id": 1}, {"shot_id": 0}]},
            {"scene_num": 1, "shots": [{"shot_id": 1}, {"shot_id": 0}]},
        ]
    }


def test_track_a_maps_contiguous_generated_indices_to_sorted_source_shots(tmp_path):
    pipeline = tmp_path / "track_A" / "vimax_pipeline"
    for idx in range(4):
        _write_video(pipeline / "shots" / str(idx) / "video.mp4")

    sample = discover_track_a(_source_case(), pipeline)

    assert sample["status"] == "complete"
    assert [shot["shot_id"] for shot in sample["shots"]] == ["0", "1", "2", "3"]
    assert [shot["scene_id"] for shot in sample["shots"]] == ["1", "1", "2", "2"]
    assert sample["scene_label_source"] == "source_positional"


def test_track_a_count_mismatch_is_incomplete_and_has_no_mapped_scene_labels(tmp_path):
    pipeline = tmp_path / "track_A" / "vimax_pipeline"
    _write_video(pipeline / "shots" / "0" / "video.mp4")

    sample = discover_track_a(_source_case(), pipeline)

    assert sample["status"] == "incomplete"
    assert sample["metrics"] is None
    assert sample["scene_label_source"] is None
    assert sample["expected_shot_count"] == 4
    assert sample["found_shot_count"] == 1


def test_track_b_uses_scene_and_shot_tuple_ids_without_source_labels(tmp_path):
    pipeline = tmp_path / "track_B" / "vimax_pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "script.json").write_text(json.dumps(["scene 0", "scene 1"]))
    for scene_idx in range(2):
        scene_dir = pipeline / f"scene_{scene_idx}"
        scene_dir.mkdir()
        storyboard = [{"idx": 0}, {"idx": 1}]
        (scene_dir / "storyboard.json").write_text(json.dumps(storyboard))
        for shot_idx in range(2):
            _write_video(scene_dir / "shots" / str(shot_idx) / "video.mp4")

    sample = discover_track_b(pipeline)

    assert sample["status"] == "complete"
    assert [shot["shot_id"] for shot in sample["shots"]] == ["0:0", "0:1", "1:0", "1:1"]
    assert all(shot["scene_id"] is None for shot in sample["shots"])
    assert sample["metrics"]["IC"]["score"] is None
    assert sample["metrics"]["CC"]["score"] is None
    assert sample["metrics"]["GC"]["pair_count"] == 6


def test_track_b_inventory_must_match_script_and_storyboards(tmp_path):
    pipeline = tmp_path / "track_B"
    pipeline.mkdir()
    (pipeline / "script.json").write_text(json.dumps(["scene 0"]))
    scene = pipeline / "scene_0"
    scene.mkdir()
    (scene / "storyboard.json").write_text(json.dumps([{"idx": 0}, {"idx": 1}]))
    _write_video(scene / "shots" / "0" / "video.mp4")

    sample = discover_track_b(pipeline)

    assert sample["status"] == "incomplete"
    assert sample["metrics"] is None
    assert any("1" in item for item in sample["missing_shots"])


def test_metrics_and_dataset_aggregation_use_pair_counts():
    metrics = aggregate_viclip_metrics(
        ["a", "b", "c"], ["x", "x", "y"],
        np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
    )
    assert metrics["IC"]["score"] == pytest.approx(1.0)
    assert metrics["CC"]["score"] == pytest.approx(0.0)
    assert metrics["GC"]["pair_count"] == 3

    rows = [
        {"track": "A", "status": "complete", "metrics": {"GC": {"score": 0.2, "pair_count": 1}}},
        {"track": "A", "status": "complete", "metrics": {"GC": {"score": 0.8, "pair_count": 3}}},
    ]
    aggregate = aggregate_dataset_metrics(rows)
    assert aggregate["A"]["GC"]["story_macro_mean"] == pytest.approx(0.5)
    assert aggregate["A"]["GC"]["pair_weighted_mean"] == pytest.approx(0.65)


def test_zero_byte_video_makes_track_a_incomplete(tmp_path):
    pipeline = tmp_path / "track_A" / "vimax_pipeline"
    _write_video(pipeline / "shots" / "0" / "video.mp4", b"")

    sample = discover_track_a({"scenes": [{"scene_num": 0, "shots": [{"shot_id": 0}]}]}, pipeline)

    assert sample["status"] == "incomplete"
    assert sample["metrics"] is None
    assert any("zero-byte" in diagnostic for diagnostic in sample["diagnostics"])


def test_track_b_generated_scene_diagnostics_are_separate(tmp_path, monkeypatch):
    from scripts import vimax_pipeline_eval as evaluator

    pipeline = tmp_path / "track_B"
    pipeline.mkdir()
    (pipeline / "script.json").write_text(json.dumps(["scene"]))
    scene = pipeline / "scene_0"
    scene.mkdir()
    (scene / "storyboard.json").write_text(json.dumps([{"idx": 0}, {"idx": 1}]))
    for i in range(2):
        _write_video(scene / "shots" / str(i) / "video.mp4")
    monkeypatch.setattr(evaluator, "_validate_video", lambda *args: None)

    sample = discover_track_b(pipeline, track_b_scene_diagnostics=True)
    evaluated = evaluate_sample(sample, DeterministicFakeEncoder(), tmp_path / "cache", "case/B", True)

    assert evaluated["metrics"]["IC"]["score"] is None
    assert evaluated["metrics"]["GC"]["score"] is not None
    assert evaluated["scene_diagnostics"]["scene_label_source"] == "pipeline_generated"
    assert evaluated["scene_diagnostics"]["metrics"]["IC"]["score"] is not None


def test_embedding_cache_avoids_reencoding_and_report_is_nonofficial(tmp_path, monkeypatch):
    from scripts import vimax_pipeline_eval as evaluator

    pipeline = tmp_path / "track_A" / "vimax_pipeline"
    for idx in range(2):
        _write_video(pipeline / "shots" / str(idx) / "video.mp4")
    sample = discover_track_a({"scenes": [{"scene_num": 0, "shots": [{"shot_id": 0}, {"shot_id": 1}]}]}, pipeline)
    monkeypatch.setattr(evaluator, "_validate_video", lambda *args: None)

    encoder = DeterministicFakeEncoder()
    cache = tmp_path / "cache"
    first = evaluate_sample(dict(sample), encoder, cache, "case/A")
    second = evaluate_sample(dict(sample), encoder, cache, "case/A")
    report = make_report([second], encoder, {"smoke_mode": True})

    assert first["cache"] == {"hits": 0, "misses": 2}
    assert second["cache"] == {"hits": 2, "misses": 0}
    assert report["results_are_official_vimax_bench"] is False
    assert report["encoder"]["reportable"] is False


def test_http_encoder_validates_embedding_and_metadata(monkeypatch, tmp_path):
    import requests

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [1.0, 0.0], "dimension": 2, "sampled_frames": 8,
                    "resolution": 224, "normalized": True}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    path = _write_video(tmp_path / "sample.mp4")
    encoder = ViCLIPHTTPEncoder(expected_dimension=2)

    assert np.linalg.norm(encoder.encode_video(path)) == pytest.approx(1.0)

    class BadResponse(Response):
        def json(self):
            return {"embedding": [1.0, 0.0], "dimension": 3, "sampled_frames": 8,
                    "resolution": 224, "normalized": True}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: BadResponse())
    with pytest.raises(ValueError, match="metadata"):
        encoder.encode_video(path)


def test_cli_writes_report_and_returns_nonzero_when_all_selected_samples_incomplete(tmp_path):
    from scripts.run_vimax_pipeline_eval import main

    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    source = benchmark / "case.json"
    source.write_text(json.dumps({"scenes": [{"scene_num": 0, "shots": [{"shot_id": 0}]}]}))
    (benchmark / "benchmark_index.json").write_text(json.dumps({
        "stories": [{"theme": "case", "file": "case.json"}],
    }))
    report = tmp_path / "report.json"

    result = main([
        "--benchmark-root", str(benchmark), "--output-root", str(tmp_path / "outputs"),
        "--case", "case", "--track", "A", "--smoke", "--report", str(report),
    ])

    assert result == 2
    data = json.loads(report.read_text())
    assert data["complete_sample_count"] == 0
    assert data["case_scores"][0]["status"] == "incomplete"
    assert data["results_are_official_vimax_bench"] is False
    assert set(data["case_scores"][0]) == {"case", "track", "status", "IC", "CC", "GC", "diagnostic"}
