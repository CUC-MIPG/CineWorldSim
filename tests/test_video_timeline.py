from __future__ import annotations

import json

from agents.video_timeline import build_video_timeline, write_video_timeline


def test_build_video_timeline_uses_shot_intervals_for_dialogue_and_narration():
    task = {
        "rank": 5,
        "node_id": "NN_006",
        "event_id": "EVT_006",
        "output_video_path": "videos/event_video_005_NN_006.mp4",
        "duration_sec": 10,
        "num_frames": 243,
        "characters": ["Ilya Trent", "Nora Hale"],
        "event_shot_plan": [
            {
                "shot_index": 1,
                "start_timecode": "00:00.000",
                "start_sec": 0.0,
                "duration_sec": 3.0,
                "dialogue": "Ilya Trent: Hold the pulse steady.",
            },
            {
                "shot_index": 2,
                "start_timecode": "00:03.000",
                "start_sec": 3.0,
                "duration_sec": 7.0,
                "dialogue": "none",
                "narration": "The signal disappears beneath the storm static.",
            },
        ],
    }

    timeline = build_video_timeline(task)

    assert timeline["timing_basis"] == "shot_interval"
    assert timeline["video_filename"] == "event_video_005_NN_006.mp4"
    assert timeline["segments"] == [
        {
            "index": 1,
            "type": "dialogue",
            "speaker": "Ilya Trent",
            "character_slot": "",
            "text": "Hold the pulse steady.",
            "shot_index": 1,
            "start_sec": 0.0,
            "end_sec": 3.0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:03.000",
            "source": "shot_plan",
        },
        {
            "index": 2,
            "type": "narration",
            "speaker": "",
            "character_slot": "",
            "text": "The signal disappears beneath the storm static.",
            "shot_index": 2,
            "start_sec": 3.0,
            "end_sec": 10.0,
            "start_timecode": "00:00:03.000",
            "end_timecode": "00:00:10.000",
            "source": "shot_plan",
        },
    ]


def test_build_video_timeline_does_not_turn_story_description_into_narration():
    task = {
        "rank": 1,
        "node_id": "NN_001",
        "output_video_path": "videos/event_video_001_NN_001.mp4",
        "duration_sec": 5,
        "story_description": "This is visual action and must not become a subtitle.",
        "event_shot_plan": [
            {
                "shot_index": 1,
                "start_timecode": "00:00.000",
                "start_sec": 0.0,
                "duration_sec": 5.0,
                "dialogue": "none",
            }
        ],
    }

    assert build_video_timeline(task)["segments"] == []


def test_write_video_timeline_uses_video_stem_and_is_json_serializable(tmp_path):
    task = {
        "rank": 2,
        "node_id": "NN_002",
        "output_video_path": str(tmp_path / "event_video_002_NN_002.mp4"),
        "duration_sec": 5,
        "event_shot_plan": [
            {
                "shot_index": 1,
                "start_sec": 0.0,
                "duration_sec": 5.0,
                "dialogue": "none",
            }
        ],
    }

    path = write_video_timeline(task)

    assert path.name == "event_video_002_NN_002.timeline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["video_path"] == str(tmp_path / "event_video_002_NN_002.mp4")
    assert payload["segments"] == []


def test_build_video_timeline_uses_explicit_task_narration_without_story_description():
    task = {
        "rank": 3,
        "node_id": "NN_003",
        "output_video_path": "videos/event_video_003_NN_003.mp4",
        "duration_sec": 5,
        "story_description": "Visual action only.",
        "narration": {"text": "The hidden route opens beneath the chamber."},
        "event_shot_plan": [
            {
                "shot_index": 1,
                "start_sec": 0.0,
                "duration_sec": 5.0,
                "dialogue": "none",
            }
        ],
    }

    segments = build_video_timeline(task)["segments"]

    assert len(segments) == 1
    assert segments[0]["type"] == "narration"
    assert segments[0]["text"] == "The hidden route opens beneath the chamber."
