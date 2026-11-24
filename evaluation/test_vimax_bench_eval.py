import json
from pathlib import Path

import numpy as np
import pytest

from vimax_bench_adapter import build_vimax_samples, load_scene_map
from run_vimax_bench_eval import _feature_cache_path, aggregate_dataset_metrics, evaluate
from vimax_metrics import (
    ViCLIPHTTPEncoder,
    aggregate_viclip_metrics,
    cosine_similarity,
    extract_uniform_indices,
    normalize_features,
)
from vimax_llm_judge import aggregate_judge_batches, validate_judge_payload


CASE_INPUTS = Path(__file__).parent / "adapted_cases/0001_American_Beauty_arc_001_real/directorbench/inputs.json"


def test_extract_uniform_indices_always_returns_requested_count():
    assert extract_uniform_indices(1, 8).tolist() == [0] * 8
    assert extract_uniform_indices(10, 4).tolist() == [0, 3, 6, 9]
    with pytest.raises(ValueError):
        extract_uniform_indices(0, 8)


def test_vimax_adapter_splits_source_clips_and_preserves_storyboard_metadata(tmp_path):
    samples = build_vimax_samples(CASE_INPUTS, track="A", shots_per_sample=8)
    assert len(samples) == 2
    assert [len(s["shots"]) for s in samples] == [8, 5]
    assert samples[0]["shots"][0]["video_path"].endswith(".mp4")
    assert samples[0]["shots"][0]["shot_id"]
    assert samples[0]["shots"][0]["storyboard_description"]
    assert samples[0]["protocol"]["shot_count_status"] == "official_range"
    assert samples[0]["protocol"]["scene_label_status"] == "missing"
    assert samples[0]["protocol"]["official_vimax_result_eligible"] is False


def test_scene_map_is_applied_and_scene_metrics_become_eligible(tmp_path):
    samples = build_vimax_samples(CASE_INPUTS, track="A", shots_per_sample=8,
                                  scene_map={f"{i:02d}": i // 4 for i in range(8)})
    # The adapter accepts node IDs as keys; use the generated IDs to make a complete map.
    shot_ids = [shot["shot_id"] for shot in samples[0]["shots"]]
    scene_map = {shot_id: (idx // 4) for idx, shot_id in enumerate(shot_ids)}
    samples = build_vimax_samples(CASE_INPUTS, track="A", shots_per_sample=8,
                                  scene_map=scene_map)
    assert samples[0]["protocol"]["scene_label_status"] == "complete"
    assert samples[0]["protocol"]["official_vimax_result_eligible"] is True


def test_load_scene_map_reads_json_object(tmp_path):
    path = tmp_path / "scene_map.json"
    path.write_text(json.dumps({"shot-1": "scene-1"}), encoding="utf-8")
    assert load_scene_map(path) == {"shot-1": "scene-1"}


def test_adapter_accepts_already_normalized_shot_scene_json(tmp_path):
    path = tmp_path / "normalized.json"
    path.write_text(json.dumps({"samples": [{
        "sample_id": "official_like", "shots": [
            {"shot_id": "a", "scene_id": "s1", "video_path": "/tmp/a.mp4", "storyboard_description": "one"},
            {"shot_id": "b", "scene_id": "s2", "video_path": "/tmp/b.mp4", "storyboard_description": "two"},
        ]
    }]}), encoding="utf-8")
    samples = build_vimax_samples(path)
    assert len(samples) == 1
    assert samples[0]["shots"][1]["scene_id"] == "s2"


def test_aggregate_metrics_matches_vimax_pair_definitions():
    features = normalize_features(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    result = aggregate_viclip_metrics(
        ["s1", "s2", "s3"], ["scene-a", "scene-a", "scene-b"], features
    )
    assert result["IC"]["pair_count"] == 1
    assert result["CC"]["pair_count"] == 2
    assert result["GC"]["pair_count"] == 3
    assert result["IC"]["score"] == pytest.approx(1.0)
    assert result["CC"]["score"] == pytest.approx(0.0)
    assert result["GC"]["score"] == pytest.approx(1.0 / 3.0)


def test_cosine_similarity_clips_float32_rounding_overshoot():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0000001, 0.0])) == 1.0


def test_http_encoder_posts_video_and_validates_verified_embed_contract(tmp_path, monkeypatch):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"video")
    captured = {}
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            vector = np.zeros(768, dtype=np.float32)
            vector[0] = 1
            return {
                "embedding": vector.tolist(), "dimension": 768, "sampled_frames": 8,
                "resolution": 224, "model": "ViCLIP", "normalized": True,
                "embedding_l2_norm": 1.00000000058904,
            }

    def fake_post(url, *, data, files, timeout):
        captured.update({"url": url, "data": data, "files": files, "timeout": timeout})
        return Response()

    monkeypatch.setattr("vimax_metrics.requests.post", fake_post)
    encoder = ViCLIPHTTPEncoder("http://viclip/embed", num_frames=8, resolution=224, timeout=25)
    embedding = encoder.encode_video(video)
    assert embedding.shape == (768,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert captured["url"] == "http://viclip/embed"
    assert captured["data"] == {"num_frames": "8", "resolution": "224"}
    assert set(captured["files"]) == {"file"}
    assert captured["timeout"] == 25


@pytest.mark.parametrize("payload, message", [
    ({"embedding": [1.0], "dimension": 1, "sampled_frames": 8, "resolution": 224, "model": "ViCLIP", "normalized": True}, "dimension"),
    ({"embedding": [0.0] * 768, "dimension": 768, "sampled_frames": 8, "resolution": 224, "model": "ViCLIP", "normalized": True}, "norm"),
])
def test_http_encoder_rejects_invalid_embed_payloads(tmp_path, monkeypatch, payload, message):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"video")
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("vimax_metrics.requests.post", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match=message):
        ViCLIPHTTPEncoder("http://viclip/embed").encode_video(video)


def test_embedding_cache_key_includes_video_version_and_backend_config(tmp_path):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"first")
    cache = tmp_path / "cache"
    config_a = {"backend": "http", "endpoint": "http://one/embed", "num_frames": 8, "resolution": 224}
    config_b = {**config_a, "endpoint": "http://two/embed"}
    key_a = _feature_cache_path(cache, str(video), config_a)
    key_b = _feature_cache_path(cache, str(video), config_b)
    video.write_bytes(b"changed-size")
    key_changed_video = _feature_cache_path(cache, str(video), config_a)
    assert key_a != key_b
    assert key_a != key_changed_video


def test_evaluator_uses_http_encoder_and_records_endpoint(tmp_path, monkeypatch):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"samples": [{"sample_id": "story", "shots": [
        {"shot_id": "a", "scene_id": "scene1", "video_path": str(videos[0])},
        {"shot_id": "b", "scene_id": "scene2", "video_path": str(videos[1])},
    ]}]}), encoding="utf-8")
    class FakeHTTPEncoder:
        def __init__(self, endpoint, num_frames, resolution, timeout):
            assert endpoint == "http://viclip/embed"
            assert (num_frames, resolution, timeout) == (8, 224, 30)

        def encode_video(self, path):
            return np.array([1.0, 0.0], dtype=np.float32) if str(path).endswith("a.mp4") else np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr("run_vimax_bench_eval.ViCLIPHTTPEncoder", FakeHTTPEncoder)
    report = evaluate(inputs, tmp_path / "report.json", cache_dir=tmp_path / "cache",
                      backend="http", embed_url="http://viclip/embed", timeout=30)
    assert report["backend"] == "http"
    assert report["samples"][0]["feature_extraction"]["endpoint"] == "http://viclip/embed"
    assert report["samples"][0]["metrics"]["GC"]["score"] == pytest.approx(0.0)


def test_dataset_aggregation_reports_story_macro_and_pair_weighted_means():
    records = [
        {"metrics": {name: {"score": 0.2, "pair_count": 1} for name in ("IC", "CC", "GC")}},
        {"metrics": {name: {"score": 0.8, "pair_count": 3} for name in ("IC", "CC", "GC")}},
        {"metrics": {name: {"score": None, "pair_count": 0} for name in ("IC", "CC", "GC")}},
    ]
    aggregate = aggregate_dataset_metrics(records)
    assert aggregate["GC"]["story_macro_mean"] == pytest.approx(0.5)
    assert aggregate["GC"]["pair_weighted_mean"] == pytest.approx(0.65)
    assert aggregate["GC"]["story_count"] == 2
    assert aggregate["GC"]["pair_count"] == 4


def test_missing_scene_labels_do_not_turn_ic_cc_into_zero():
    features = normalize_features(np.eye(3))
    result = aggregate_viclip_metrics(["a", "b", "c"], [None, None, None], features)
    assert result["IC"]["score"] is None
    assert result["CC"]["score"] is None
    assert result["GC"]["pair_count"] == 3
    assert "scene labels" in result["warnings"][0].lower()


def test_llm_judge_payload_requires_integer_five_point_scores():
    payload = {
        "scores": {key: 4 for key in ("CBC", "NC", "PPR", "VSU", "STQ")},
        "rationale": {}, "key_strengths": [], "key_weaknesses": [], "improvement_suggestions": [],
    }
    normalized = validate_judge_payload(payload, metric_names=("CBC", "NC", "PPR", "VSU", "STQ"))
    assert normalized["scores"]["CBC"] == 4
    assert aggregate_judge_batches([normalized, normalized])["average"] == pytest.approx(4.0)
