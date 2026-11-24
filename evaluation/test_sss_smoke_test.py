import json
import sys
from types import SimpleNamespace

import pytest

import sss_smoke_test
from sss_smoke_test import calculate_sss, extract_json_object


def test_calculate_sss_uses_category_balanced_claim_coverage():
    claims = [
        {"claim_id": "e1", "category": "core_event", "text": "A leaves home."},
        {"claim_id": "e2", "category": "core_event", "text": "B confronts A."},
        {"claim_id": "r1", "category": "character_relation", "text": "A is B's sibling."},
        {"claim_id": "c1", "category": "conflict", "text": "They dispute the inheritance."},
        {"claim_id": "g1", "category": "plot_direction", "text": "A returns home."},
    ]
    matches = [
        {"claim_id": "e1", "status": "supported"},
        {"claim_id": "e2", "status": "absent"},
        {"claim_id": "r1", "status": "supported"},
        {"claim_id": "c1", "status": "partial"},
        {"claim_id": "g1", "status": "not_applicable"},
    ]

    result = calculate_sss(claims, matches)

    assert result["category_scores"]["core_event"]["score"] == pytest.approx(0.5)
    assert result["category_scores"]["character_relation"]["score"] == 1.0
    assert result["category_scores"]["conflict"]["score"] == 0.5
    assert result["category_scores"]["plot_direction"]["score"] is None
    assert result["sss"] == pytest.approx(2 / 3)
    assert result["sss_100"] == pytest.approx(66.6667, abs=0.001)


def test_calculate_sss_rejects_duplicate_or_missing_matches():
    claims = [{"claim_id": "e1", "category": "core_event", "text": "A leaves."}]
    with pytest.raises(ValueError, match="exactly one match"):
        calculate_sss(claims, [])
    with pytest.raises(ValueError, match="duplicate"):
        calculate_sss(claims, [
            {"claim_id": "e1", "status": "supported"},
            {"claim_id": "e1", "status": "absent"},
        ])


def test_alignment_requires_evidence_for_positive_or_contradictory_matches():
    claim = [{"claim_id": "e1", "category": "core_event", "text": "A leaves."}]
    with pytest.raises(ValueError, match="Evidence is required"):
        sss_smoke_test._validate_matches(
            {"matches": [{"claim_id": "e1", "status": "supported", "evidence": ""}]}, claim
        )


def test_extract_json_object_accepts_fenced_json():
    payload = extract_json_object('Result:\n```json\n{"story": "A walks."}\n```')
    assert payload == {"story": "A walks."}


def test_extract_json_object_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON object"):
        extract_json_object("not json")


def test_chat_completion_logs_request_context_and_http_error(tmp_path, monkeypatch):
    class FakeResponse:
        status_code = 404
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            error = sss_smoke_test.requests.HTTPError("404 Client Error")
            error.response = self
            raise error

        def json(self):
            return {"error": {"type": "model_not_found", "code": "model_not_found", "message": "unknown model"}}

    monkeypatch.setattr(sss_smoke_test.requests, "post", lambda *args, **kwargs: FakeResponse())
    log_path = tmp_path / "api_requests.jsonl"

    with pytest.raises(RuntimeError, match="grounding.shot_0003"):
        sss_smoke_test._chat_completion(
            "https://api.example/v1", "secret", "model-x", [], 10,
            request_context="grounding.shot_0003", request_log_path=log_path,
        )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request_started", "request_failed"]
    assert records[-1]["context"] == "grounding.shot_0003"
    assert records[-1]["http_status"] == 404
    assert records[-1]["error_code"] == "model_not_found"
    assert "secret" not in log_path.read_text(encoding="utf-8")


def test_adapter_input_is_labeled_proxy(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"samples": [
        {"track": "A", "prompt": "Track A."},
        {"track": "B", "prompt": "Track B fallback."},
    ]}), encoding="utf-8")
    summary, source, provenance = sss_smoke_test.load_summary(None, inputs, None, "B")
    assert summary == "Track B fallback."
    assert source == str(inputs.resolve())
    assert provenance == "proxy"


def test_l4_global_defaults_to_short_synopsis(tmp_path):
    l4_dir = tmp_path / "l4_global"
    l4_dir.mkdir()
    (l4_dir / "movie_bible.json").write_text(json.dumps({"result": {
        "movie_synopsis": {"short_synopsis": "Short L4 summary."},
        "global_storyline": {"story": "Long L4 story."},
    }}), encoding="utf-8")
    summary, source, provenance = sss_smoke_test.load_summary(None, None, l4_dir, "B")
    assert summary == "Short L4 summary."
    assert source.endswith("l4_global/movie_bible.json")
    assert provenance == "l4_global"

    full, _, _ = sss_smoke_test.load_summary(None, None, l4_dir, "B", "global_storyline")
    assert full == "Long L4 story."


def test_grounding_reconstruction_uses_video_evidence_without_reference_text(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    sent = []

    def chat_call(messages, context):
        sent.extend(messages)
        return '{"claims":[{"claim_id":"c1","shot_id":"s1","text":"A person enters.","grounded":true,"evidence":"door opens"}]}'

    result = sss_smoke_test._ground_reconstruction(
        [{"shot_id": "s1", "caption": "A person enters."}],
        {"s1": [{"path": str(image), "timestamp_sec": 1.0}]},
        chat_call,
        tmp_path / "out",
    )
    assert result["hallucination_rate"] == 0.0
    prompt_text = " ".join(part.get("text", "") for part in sent[0]["content"])
    assert "REFERENCE SECRET" not in prompt_text
    assert "A person enters." in prompt_text


def test_score_story_pair_keeps_embedding_primary_and_diagnostics_separate(tmp_path, monkeypatch):
    embedding_calls = iter([[1.0, 0.0], [0.8, 0.6]])
    monkeypatch.setattr(sss_smoke_test, "get_embedding", lambda *args, **kwargs: next(embedding_calls))
    chat_responses = iter([
        '{"scores":{"characters":4,"conflict":4,"key_events":5,"relationships_goals":3,"ending":4},"rationale":{}}',
        '{"events":[{"event_id":"e1","text":"A person enters."}]}',
        '{"matches":[{"event_id":"e1","status":"recovered","matched_reconstructed_event_ids":["event_001"],"evidence":"entry is captioned"}]}',
    ])
    result = sss_smoke_test._score_story_pair(
        "A person enters.", "A person enters.", [{"event_id": "event_001", "description": "A person enters."}],
        lambda messages, context: next(chat_responses), "http://api/v1", "key", "text-embedding-3-small", 10, tmp_path,
    )
    assert result["primary"]["metric"] == "SSS_emb"
    assert result["primary"]["score"] == pytest.approx(0.8)
    assert result["diagnostics"]["event_reconstruction_coverage"]["coverage"] == 1.0
    assert result["diagnostics"]["structured_alignment"]["score"] == pytest.approx(0.8)


def test_hierarchical_evaluation_reconstructs_and_grounds_before_loading_reference(tmp_path, monkeypatch):
    events = []
    frame = {"index": 1, "timestamp_sec": 0.5, "path": str(tmp_path / "frame.jpg")}
    (tmp_path / "frame.jpg").write_bytes(b"jpeg")
    reconstruction = {
        "L1": [{"shot_id": "s1", "caption": "A person enters."}],
        "L2": [{"event_id": "e1", "description": "A person enters."}],
        "L3": [{"scene_id": "sc1", "description": "A person enters."}],
        "L4": {"story": "A person enters."},
    }
    def fake_reconstruct(*args, **kwargs):
        events.append("reconstruct")
        args[3].mkdir(parents=True, exist_ok=True)
        (args[3] / "frame_manifest.json").write_text(json.dumps({"shots": [
            {"shot_id": "s1", "frames": [frame]}
        ]}), encoding="utf-8")
        return reconstruction

    def fake_ground(*args, **kwargs):
        events.append("ground")
        return {"hallucination_rate": 0.0}

    def fake_score(reference, story, l2, *args, **kwargs):
        events.append("score")
        assert reference == "REFERENCE STORY"
        assert story == "A person enters."
        assert l2 == reconstruction["L2"]
        return {"primary": {"metric": "SSS_emb", "score": 1.0}, "diagnostics": {}}

    monkeypatch.setattr(sss_smoke_test, "run_hierarchical_reconstruction", fake_reconstruct)
    monkeypatch.setattr(sss_smoke_test, "_ground_reconstruction", fake_ground)
    monkeypatch.setattr(sss_smoke_test, "_score_story_pair", fake_score)
    video = tmp_path / "video.mp4"
    video.touch()
    result = sss_smoke_test.run_evaluation(
        video, None, [], "http://api/v1", "key", "vision-model", 10, tmp_path / "out",
        mode="v2", segments=[{"shot_id": "s1", "start_sec": 0, "end_sec": 1}],
        frames_per_shot=1, summary_loader=lambda: (events.append("reference") or ("REFERENCE STORY", "ref", "file")),
    )
    assert events == ["reconstruct", "ground", "reference", "score"]
    assert result["primary"]["metric"] == "SSS_emb"
    assert result["diagnostics"]["caption_grounding"]["hallucination_rate"] == 0.0


def test_cli_hierarchical_dry_run_records_segments_without_api(tmp_path, monkeypatch, capsys):
    video = tmp_path / "sample.mp4"
    video.touch()
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text("Input story.", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sss_smoke_test, "_probe_duration", lambda path: 10.0)
    monkeypatch.setattr(sss_smoke_test, "resolve_segments", lambda *args, **kwargs: (
        [{"shot_id": "shot_0001", "shot_index": 0, "start_sec": 0.0, "end_sec": 10.0}],
        {"segmentation": "automatic", "detector": "mock"},
    ))
    monkeypatch.setattr(sss_smoke_test, "extract_segment_frames", lambda *args, **kwargs: [
        {"index": 1, "timestamp_sec": 5.0, "path": str(output_dir / "frame.jpg")}
    ])
    monkeypatch.setattr(sss_smoke_test, "_chat_completion", lambda *args, **kwargs: pytest.fail("unexpected API call"))
    monkeypatch.setattr(sss_smoke_test, "sample_video_frames", lambda path, frame_dir, count: [
        {"index": 1, "timestamp_sec": 5.0, "path": str(frame_dir / "frame_001.jpg")}
    ])
    monkeypatch.setattr(sys, "argv", [
        "sss_smoke_test.py", "--video", str(video), "--summary-file", str(summary_file),
        "--summary-provenance", "proxy", "--mode", "v2", "--output-dir", str(output_dir),
    ])
    assert sss_smoke_test.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "v2"
    assert manifest["segmentation"]["detector"] == "mock"
    assert manifest["frame_count"] == 1
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_cli_prepares_frames_without_calling_api(tmp_path, monkeypatch, capsys):
    video = tmp_path / "sample.mp4"
    video.touch()
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text("An input summary.", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sss_smoke_test, "_probe_duration", lambda path: 12.0)
    monkeypatch.setattr(sss_smoke_test, "sample_video_frames", lambda path, frame_dir, count: [
        {"index": 1, "timestamp_sec": 6.0, "path": str(frame_dir / "frame_001.jpg")}
    ])
    monkeypatch.setattr(sss_smoke_test, "_chat_completion", lambda *args, **kwargs: pytest.fail("unexpected API call"))
    monkeypatch.setattr(sys, "argv", [
        "sss_smoke_test.py", "--video", str(video), "--summary-file", str(summary_file),
        "--summary-provenance", "proxy", "--output-dir", str(output_dir),
    ])

    assert sss_smoke_test.main() == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_calls_enabled"] is False
    assert manifest["summary_exposed_to_reconstruction"] is False
    assert manifest["summary_char_count"] == len("An input summary.")
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
