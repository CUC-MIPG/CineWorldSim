import json
from pathlib import Path
import json

import pytest

import subprocess

from sss_hierarchical import (
    build_l1_prompt,
    build_l2_prompt,
    build_l3_prompt,
    build_l4_prompt,
    build_l4_normalization_prompt,
    detect_shots,
    extract_segment_frames,
    sample_timestamps,
    run_hierarchical_reconstruction,
    resolve_segments,
    track_a_segments,
    validate_l1_payload,
    validate_l2_payload,
    validate_l3_payload,
    validate_l4_payload,
)


def test_sample_timestamps_place_frames_at_segment_bin_midpoints():
    assert sample_timestamps(10, 20, 3) == pytest.approx([11.6666667, 15.0, 18.3333333])


@pytest.mark.parametrize("start,end,count", [(0, 0, 1), (2, 1, 1), (0, 1, 0)])
def test_sample_timestamps_reject_invalid_segment_or_count(start, end, count):
    with pytest.raises(ValueError):
        sample_timestamps(start, end, count)


def test_track_a_segments_follow_adapter_clip_order_and_only_return_visual_metadata():
    payload = {"samples": [{
        "track": "A",
        "video_path": "/videos/track_A.mp4",
        "source_clip_paths": ["/clips/z.mp4", "/clips/a.mp4"],
        "ordered_node_ids": ["node-z", "node-a"],
        "storyboard_path": "/references/storyboard.json",
        "prompt": "private reference story",
    }]}
    durations = {"/clips/z.mp4": 2.0, "/clips/a.mp4": 3.0}

    segments = track_a_segments(payload, "A", "/videos/track_A.mp4", durations.__getitem__)

    assert [(row["shot_id"], row["start_sec"], row["end_sec"]) for row in segments] == [
        ("node-z", 0.0, 2.0), ("node-a", 2.0, 5.0)
    ]
    assert all("prompt" not in row and "storyboard" not in row for row in segments)


def test_track_a_segments_fail_closed_on_missing_or_mismatched_mapping():
    with pytest.raises(ValueError, match="exactly one"):
        track_a_segments({"samples": []}, "A", "/videos/track_A.mp4", lambda _: 1.0)
    payload = {"samples": [{"track": "A", "video_path": "/videos/other.mp4", "source_clip_paths": ["a"], "ordered_node_ids": ["n"]}]}
    with pytest.raises(ValueError, match="does not match"):
        track_a_segments(payload, "A", "/videos/track_A.mp4", lambda _: 1.0)


def test_ffmpeg_scene_detection_returns_ordered_ranges_and_records_detector(monkeypatch):
    def fake_run(command, **kwargs):
        assert "select='gt(scene,0.3000)'" in " ".join(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="pts_time:3.25\npts_time:7.5\n")

    monkeypatch.setattr("sss_hierarchical.subprocess.run", fake_run)
    shots, metadata = detect_shots(Path("video.mp4"), 10.0, threshold=0.3)
    assert shots == [
        {"shot_index": 0, "start_sec": 0.0, "end_sec": 3.25},
        {"shot_index": 1, "start_sec": 3.25, "end_sec": 7.5},
        {"shot_index": 2, "start_sec": 7.5, "end_sec": 10.0},
    ]
    assert metadata["detector"] == "ffmpeg_scene_filter"


def test_missing_scene_detector_requires_explicit_uniform_fallback(monkeypatch):
    def fail_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="ffmpeg failed")

    monkeypatch.setattr("sss_hierarchical.subprocess.run", fail_run)
    with pytest.raises(RuntimeError, match="scene detection failed"):
        detect_shots(Path("video.mp4"), 9.0)
    shots, metadata = detect_shots(Path("video.mp4"), 9.0, uniform_fallback_sec=4.0)
    assert [row["end_sec"] for row in shots] == [4.0, 8.0, 9.0]
    assert metadata["fallback"] is True


def test_extract_segment_frames_uses_midpoints_and_creates_bounded_images(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"frame")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("sss_hierarchical.subprocess.run", fake_run)
    rows = extract_segment_frames(Path("video.mp4"), {"shot_id": "s1", "start_sec": 10, "end_sec": 20}, tmp_path, 3)
    assert [row["timestamp_sec"] for row in rows] == pytest.approx([11.6666667, 15, 18.3333333], abs=0.0005)
    assert len(commands) == 3
    assert all("-frames:v" in command and "1" in command for command in commands)
    assert all("scale='min(768,iw)':-2" in " ".join(command) for command in commands)


def test_hierarchy_prompts_only_use_video_evidence_or_prior_reconstructions():
    segment = {
        "shot_id": "shot-1", "start_sec": 2.0, "end_sec": 5.0,
        "storyboard_description": "MUST NOT LEAK original storyboard",
        "prompt": "MUST NOT LEAK input story",
    }
    l1_prompt = build_l1_prompt(segment)
    assert "shot-1" in l1_prompt and "2.000" in l1_prompt and "5.000" in l1_prompt
    assert "MUST NOT LEAK" not in l1_prompt
    assert "original storyboard" not in l1_prompt

    l1 = [{"shot_id": "shot-1", "caption": "A woman enters a room."}]
    l2_prompt = build_l2_prompt(l1)
    assert "A woman enters a room." in l2_prompt
    assert "original storyboard" not in l2_prompt
    l2 = [{"event_id": "event-1", "description": "A woman enters and sits."}]
    assert "A woman enters and sits." in build_l3_prompt(l2)
    l3 = [{"scene_id": "scene-1", "description": "A woman enters and sits in a quiet room."}]
    l4_prompt = build_l4_prompt(l3, max_words=180)
    assert "180 words" in l4_prompt
    assert "A woman enters and sits" in l4_prompt
    assert "not expose the reference" in build_l4_normalization_prompt("raw story", 120).lower()


def test_hierarchy_validators_reject_empty_or_overlong_outputs():
    l1 = validate_l1_payload({"caption": "A woman enters.", "evidence": ["door opens"]}, "shot-1")
    assert l1["shot_id"] == "shot-1"
    with pytest.raises(ValueError, match="caption"):
        validate_l1_payload({"caption": "", "evidence": []}, "shot-1")
    l2 = validate_l2_payload({"events": [{"description": "She enters."}]})
    assert l2[0]["event_id"] == "event_001"
    l3 = validate_l3_payload({"scenes": [{"description": "Interior scene."}]})
    assert l3[0]["scene_id"] == "scene_001"
    with pytest.raises(ValueError, match="word limit"):
        validate_l4_payload({"story": "one two three four five"}, max_words=4)


def test_hierarchical_runner_persists_all_levels_prompts_and_raw_responses(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not-a-real-jpeg")
    monkeypatch.setattr("sss_hierarchical.extract_segment_frames", lambda video, segment, frame_dir, count: [
        {"index": 1, "timestamp_sec": (segment["start_sec"] + segment["end_sec"]) / 2, "path": str(frame)}
    ])
    responses = iter([
        '{"caption":"A person enters.","evidence":["door opens"]}',
        '{"caption":"They sit down.","evidence":["person sits"]}',
        '{"events":[{"description":"A person enters and sits.","shot_ids":["s1","s2"]}]}',
        '{"scenes":[{"description":"A person enters a room and sits.","event_ids":["event_001"]}]}',
        '{"story":"A person enters a room and sits.","characters":["person"],"conflict":"","event_progression":["enters","sits"],"ending":"sits"}',
    ])
    sent = []

    def chat_call(messages, context):
        sent.append(messages)
        return next(responses)

    result = run_hierarchical_reconstruction(
        [{"shot_id": "s1", "start_sec": 0, "end_sec": 2}, {"shot_id": "s2", "start_sec": 2, "end_sec": 4}],
        "video.mp4", tmp_path / "frames", tmp_path / "out", chat_call,
    )
    assert len(sent) == 5
    assert result["L4"]["word_count"] == 7
    assert json.loads((tmp_path / "out/hierarchical_reconstruction.json").read_text())["L1"][0]["shot_id"] == "s1"
    assert (tmp_path / "out/raw_responses/l2.txt").read_text().startswith('{"events"')
    assert (tmp_path / "out/prompts/l4.txt").is_file()


def test_resolve_segments_routes_v2_and_track_b_to_auto_detection(monkeypatch):
    calls = []

    def fake_detect(video_path, duration, threshold, uniform_fallback_sec=None):
        calls.append((str(video_path), duration, threshold))
        return ([{"shot_index": 0, "start_sec": 0.0, "end_sec": duration}], {"detector": "test"})

    monkeypatch.setattr("sss_hierarchical.detect_shots", fake_detect)
    shots, metadata = resolve_segments("v2", "A", "video.mp4", 10, None, lambda _: 1)
    assert shots[0]["end_sec"] == 10
    assert metadata["segmentation"] == "automatic"
    shots_b, metadata_b = resolve_segments("v3", "B", "video.mp4", 8, None, lambda _: 1)
    assert shots_b[0]["end_sec"] == 8
    assert metadata_b["track"] == "B"
    assert len(calls) == 2


def test_resolve_segments_requires_track_a_mapping_for_v3():
    with pytest.raises(ValueError, match="requires --track-adapter-inputs"):
        resolve_segments("v3", "A", "track_A.mp4", 5, None, lambda _: 1)
