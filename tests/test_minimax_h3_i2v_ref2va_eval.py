from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.run_minimax_h3_i2v_ref2va_eval import build_comparison_tasks, run_comparison


def _img(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (20, 40, 60)).save(path)
    return str(path)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    event_one = _img(tmp_path / "event" / "NN_001_event_keyframe.png")
    event_two = _img(tmp_path / "event" / "NN_002_event_keyframe_a.png")
    event_three = _img(tmp_path / "event" / "NN_002_event_keyframe_b.png")
    portrait = _img(tmp_path / "assets" / "Hero_portrait.png")
    location = _img(tmp_path / "assets" / "Room_medium.png")
    item = _img(tmp_path / "assets" / "Brass_Key_initial.png")
    nodes = {
        "storyline": "A test story.",
        "selected_nodes": [
            {
                "node_id": "NN_001",
                "event_summary": "Hero enters the Room.",
                "characters": ["Hero"],
                "locations": ["Room"],
                "items": ["Brass Key"],
                "visual_assets": {
                    "event_keyframes": [event_one],
                    "character_reference_images": [portrait],
                    "location_images": [location],
                    "item_images": [item],
                },
            },
            {
                "node_id": "NN_002",
                "event_summary": "Hero lifts the Brass Key in the Room.",
                "characters": ["Hero"],
                "locations": ["Room"],
                "items": ["Brass Key"],
                "visual_assets": {
                    "event_keyframes": [event_two, event_three],
                    "character_reference_images": [portrait],
                    "location_images": [location],
                    "item_images": [item],
                },
            },
        ],
    }
    plan = {"storyline": "A test story.", "ordered_node_ids": ["NN_001", "NN_002"]}
    nodes_path = tmp_path / "narrative_nodes.json"
    plan_path = tmp_path / "narrative_path_plan.json"
    nodes_path.write_text(json.dumps(nodes), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return nodes_path, plan_path


def test_build_comparison_tasks_uses_ref2va_for_all_event_keyframes(tmp_path: Path) -> None:
    nodes_path, plan_path = _inputs(tmp_path)

    tasks = build_comparison_tasks(nodes_path, plan_path, tmp_path / "out", max_reference_images=4)

    first = tasks["fl2va_event_keyframe_only"][0]
    second = tasks["fl2va_event_keyframe_only"][1]
    semantic = tasks["ref2va_semantic_assets"]

    assert first["minimax_task"] == "ref2va"
    assert first["reference_images"] and len(first["reference_images"]) == 1
    assert second["minimax_task"] == "ref2va"
    assert len(second["reference_images"]) == 2
    assert all(task["minimax_task"] == "ref2va" for task in semantic)
    assert all("event_keyframe" not in " ".join(task["reference_images"]) for task in semantic)


def test_run_comparison_uses_same_prompt_and_records_two_modes(tmp_path: Path, monkeypatch) -> None:
    nodes_path, plan_path = _inputs(tmp_path)
    calls: list[dict] = []

    def fake_send(self, **kwargs):
        calls.append(kwargs)
        output = Path(kwargs["video_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return True

    monkeypatch.setattr("scripts.run_minimax_h3_i2v_ref2va_eval.VideoAgent.send_generation_request", fake_send)

    summary = run_comparison(
        nodes_path=nodes_path,
        plan_path=plan_path,
        output_root=tmp_path / "comparison",
        dry_run=False,
        max_reference_images=4,
    )

    assert summary["groups"]["fl2va_event_keyframe_only"]["success"] == 2
    assert summary["groups"]["ref2va_semantic_assets"]["success"] == 2
    assert len(calls) == 4
    assert calls[0]["prompt"] == calls[2]["prompt"]
    assert calls[0]["minimax_task"] == "ref2va"
    assert calls[0]["video_mode"] == "reference"
    assert calls[0]["first_image_path"] == ""
    assert calls[0]["reference_image_paths"] == calls[0]["image_paths"]
    assert calls[1]["minimax_task"] == "ref2va"
    assert calls[2]["minimax_task"] == "ref2va"
