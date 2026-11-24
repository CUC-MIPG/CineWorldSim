from __future__ import annotations

import json
from pathlib import Path
import sys
import types

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

from agents.video_agent import VideoAgent


def _img(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (20, 40, 60)).save(path)
    return str(path)


def _write_nodes_and_plan(tmp_path: Path, nodes: list[dict]) -> tuple[Path, Path, Path]:
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"
    nodes_json.write_text(json.dumps({"storyline": "A test storyline.", "selected_nodes": nodes}), encoding="utf-8")
    plan_json.write_text(json.dumps({"storyline": "A test storyline.", "ordered_node_ids": [n["node_id"] for n in nodes]}), encoding="utf-8")
    return nodes_json, plan_json, out_dir


def test_concat_videos_accepts_string_output_path(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")
    output = tmp_path / "nested" / "combined.mp4"

    agent = VideoAgent.__new__(VideoAgent)
    assert agent.concat_videos([str(source)], str(output)) is True
    assert output.read_bytes() == b"video-bytes"


def test_video_reference_bundle_supports_explicit_modes(tmp_path: Path) -> None:
    event_kf = _img(tmp_path / "Hero_event_keyframe.png")
    portrait = _img(tmp_path / "Hero_portrait.png")
    room = _img(tmp_path / "Room_medium.png")
    key = _img(tmp_path / "Key_initial.png")
    node = {
        "event_summary": "Hero lifts the brass key in the room.",
        "characters": ["Hero"],
        "locations": ["Room"],
        "items": ["Brass Key"],
        "visual_assets": {
            "event_keyframes": [event_kf],
            "character_reference_images": [portrait],
            "location_images": [room],
            "item_images": [key],
        },
    }

    event_only = VideoAgent._node_reference_bundle({**node, "video_reference_mode": "event_keyframe_only"}, max_reference_images=4)
    semantic = VideoAgent._node_reference_bundle({**node, "video_reference_mode": "semantic_assets"}, max_reference_images=4)
    all_allowed = VideoAgent._node_reference_bundle({**node, "video_reference_mode": "all_allowed"}, max_reference_images=4)

    assert event_only["reference_mode"] == "event_keyframe_only"
    assert event_only["reference_image_paths"] == [event_kf]
    assert semantic["reference_image_paths"] == [portrait, room, key]
    assert all_allowed["reference_image_paths"] == [event_kf, portrait, room, key]


def test_multishot_planner_uses_llm_duration_and_frame_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    event_kf = _img(tmp_path / "Hero_event_keyframe.png")
    room = _img(tmp_path / "Room_medium.png")
    nodes = [
        {
            "node_id": "NN_001",
            "event_id": "EVT_001",
            "story_description": "Hero opens the vault door and sees blue light spill across the floor.",
            "event_summary": "Hero opens the vault door.",
            "event_type": "world_state_change",
            "story_phase": "development",
            "characters": ["Hero"],
            "locations": ["Vault"],
            "items": [],
            "visual_assets": {"event_keyframes": [event_kf], "location_images": [room]},
        }
    ]
    nodes_json, plan_json, out_dir = _write_nodes_and_plan(tmp_path, nodes)

    class FakeShotPlanner:
        def generate(self, **kwargs):
            return {
                "status": "done",
                "duration_sec": 10,
                "num_frames": 243,
                "shots": [
                    {"shot_index": 1, "start_timecode": "00:00.000", "duration_sec": 4, "video_description": "Hero reaches the vault.", "cinematography": "wide push", "dialogue": "none", "bgm": "low pulse"},
                    {"shot_index": 2, "start_timecode": "00:04.000", "duration_sec": 6, "video_description": "Blue light spills out.", "cinematography": "close tilt", "dialogue": "none", "bgm": "rising pulse"},
                ],
                "artifact_path": "",
                "meta_path": "",
                "error": "",
            }

    agent = VideoAgent.__new__(VideoAgent)
    agent.shot_planning_tool = FakeShotPlanner()
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        multi_shot_events=True,
        event_duration_sec=None,
    )

    assert summary["success"] == 1
    task = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))[0]
    assert task["duration_sec"] == 10
    assert task["num_frames"] == 243
    assert task["event_shot_plan"][1]["start_timecode"] == "00:04.000"
    assert "[Shot 2] At 00:04.000" in task["prompt"]


def test_event_requests_default_to_ten_seconds_and_243_frames(tmp_path: Path):
    event_kf = _img(tmp_path / "Hero_event_keyframe.png")
    nodes = [{
        "node_id": "NN_001",
        "event_summary": "Hero opens a door.",
        "visual_assets": {"event_keyframes": [event_kf]},
    }]
    nodes_json, plan_json, out_dir = _write_nodes_and_plan(tmp_path, nodes)

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        insert_auxiliary_videos=False,
    )

    assert summary["success"] == 1
    task = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))[0]
    assert task["duration_sec"] == 10
    assert task["num_frames"] == 243


def test_multishot_plan_caps_at_four_and_normalizes_total_duration():
    node = {"event_summary": "A five-beat action sequence."}
    shots = [
        {"shot_index": i, "start_timecode": "", "start_sec": 0, "duration_sec": 2,
         "video_description": f"Beat {i}"}
        for i in range(1, 6)
    ]

    normalized = VideoAgent._normalize_shot_plan_rows(node, shots, duration_sec=10)

    assert len(normalized) == 4
    assert sum(row["duration_sec"] for row in normalized) == 10
    assert [row["start_sec"] for row in normalized] == [0, 2.5, 5, 7.5]


def test_auxiliary_establishing_and_transition_tasks_are_inserted_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    hero = _img(tmp_path / "Hero_event_keyframe.png")
    room = _img(tmp_path / "Room_medium.png")
    changed_room = _img(tmp_path / "world_updates" / "locations" / "Room" / "step_002_close.png")
    nodes = [
        {
            "node_id": "NN_001",
            "event_id": "EVT_001",
            "event_summary": "Hero enters the room.",
            "characters": ["Hero"],
            "locations": ["Room"],
            "items": [],
            "visual_assets": {"event_keyframes": [hero], "location_images": [room]},
        },
        {
            "node_id": "NN_002",
            "event_id": "EVT_002",
            "event_summary": "The room wall cracks open behind Hero.",
            "characters": ["Hero"],
            "locations": ["Room"],
            "items": [],
            "world_update_images": [changed_room],
            "visual_assets": {"event_keyframes": [hero], "location_images": [room], "world_update_images": [changed_room]},
        },
    ]
    nodes_json, plan_json, out_dir = _write_nodes_and_plan(tmp_path, nodes)

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        insert_auxiliary_videos=True,
    )

    assert summary["success"] == 4
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert [task["task_type"] for task in tasks] == ["establishing", "event", "world_transition", "event"]
    assert tasks[0]["insert_before_node_id"] == "NN_001"
    assert tasks[2]["insert_before_node_id"] == "NN_002"
    assert tasks[0]["reference_images"] == [room]
    assert tasks[2]["reference_images"] == [changed_room]
