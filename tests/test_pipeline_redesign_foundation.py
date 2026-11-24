from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakeRenderer:
    backend_mode = "all_banana"
    tool_name = "nano_banana"
    dry_run = False

    def __init__(self) -> None:
        self.calls = []

    def render_world_asset(self, *, prompt: str, output_path: Path, reference_image_paths=None):
        self.calls.append(
            {
                "prompt": prompt,
                "output_path": str(output_path),
                "reference_image_paths": list(reference_image_paths or []),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"img")
        return {
            "tool": self.tool_name,
            "ok": True,
            "output_path": str(output_path),
            "error": "",
            "reference_image_paths": list(reference_image_paths or []),
        }


def test_pipeline_schema_duration_frame_mapping() -> None:
    from narrative.pipeline_schemas import LOCATION_VIEW_NAMES, frames_for_duration_sec, normalize_duration_frame_plan

    assert LOCATION_VIEW_NAMES == ("extreme_close", "close", "medium", "wide")
    assert frames_for_duration_sec(5) == 124
    assert frames_for_duration_sec(10) == 243
    with pytest.raises(ValueError):
        frames_for_duration_sec(8)

    assert normalize_duration_frame_plan({"duration_sec": 10, "shots": []})["num_frames"] == 243
    assert normalize_duration_frame_plan({"duration_sec": 5, "num_frames": 243, "shots": []})["num_frames"] == 124


def test_initial_asset_prompts_encode_subject_and_location_view_requirements() -> None:
    from narrative.initial_asset_renderer import (
        build_character_initial_prompt,
        build_item_initial_prompt,
        build_location_view_prompts,
    )

    character_prompt = build_character_initial_prompt(
        {"name": "Mara", "role_profile": "storm archive journalist", "final_goal": "find the route"},
        storyline="A lighthouse adventure.",
    )
    item_prompt = build_item_initial_prompt(
        {"name": "Sealed Letter", "description": "wax-sealed rescue note"},
        storyline="A lighthouse adventure.",
    )
    location_prompts = build_location_view_prompts(
        {"name": "Flooded Archive", "description": "a half-submerged lighthouse archive"},
        storyline="A lighthouse adventure.",
    )

    assert "maximum subject occupancy" in character_prompt.lower()
    assert "minimal background" in character_prompt.lower()
    assert "pedestrians" in character_prompt.lower()
    assert "maximum subject occupancy" in item_prompt.lower()
    assert "minimal background" in item_prompt.lower()
    assert set(location_prompts) == {"wide", "medium", "close", "extreme_close"}
    assert "wide establishing" in location_prompts["wide"].lower()
    assert "medium view" in location_prompts["medium"].lower()
    assert "close view" in location_prompts["close"].lower()
    assert "extreme close" in location_prompts["extreme_close"].lower()


def test_render_initial_assets_creates_character_views_items_and_chained_location_views(tmp_path: Path) -> None:
    from narrative.initial_asset_renderer import render_initial_assets

    renderer = FakeRenderer()
    world_model = {
        "summary": "storm archive",
        "locations": [
            {"name": "Flooded Archive", "description": "a half-submerged lighthouse archive", "asset_id": "Flooded_Archive"}
        ],
        "items": [
            {"name": "Sealed Letter", "description": "wax-sealed rescue note", "asset_id": "Sealed_Letter"}
        ],
    }
    character_configs = [
        {"name": "Mara", "role_profile": "journalist", "final_goal": "find the route"},
        {"name": "Ivo", "role_profile": "caretaker", "final_goal": "hide the route"},
    ]

    result = render_initial_assets(
        story="A lighthouse adventure.",
        character_configs=character_configs,
        world_model=world_model,
        renderer=renderer,
        run_dir=tmp_path,
    )

    assert set(result["character_portraits"]) == {"Mara", "Ivo"}
    for path in result["character_portraits"].values():
        assert Path(path).exists()
    assert result["world_model"]["items"][0]["initial_image_path"]
    assert Path(result["world_model"]["items"][0]["initial_image_path"]).exists()
    location = result["world_model"]["locations"][0]
    assert set(location["views"]) == {"wide", "medium", "close", "extreme_close"}
    assert location["last_image_path"] == location["views"]["wide"]["output_path"]
    assert [call["reference_image_paths"] for call in renderer.calls if "locations" in call["output_path"]] == [
        [],
        [location["views"]["extreme_close"]["output_path"]],
        [location["views"]["close"]["output_path"]],
        [location["views"]["medium"]["output_path"]],
    ]
    assert set(result["world_model"]["characters"][0]["views"]) == {"front_full", "side", "portrait"}
    assert result["world_model"]["characters"][0]["portrait_path"] == result["character_portraits"]["Mara"]
    assert (tmp_path / "narrative_artifacts" / "initial_assets" / "initial_assets.json").exists()


def test_world_delta_judge_uses_llm_physical_change_decision() -> None:
    from narrative.world_delta_judge import WorldDeltaJudge

    captured = {}

    class FakeLLM:
        def run_json(self, *, system_prompt, user_prompt, temperature=None, timeout=None):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return {
                "summary": "The archive wall becomes visibly cracked.",
                "location_updates": [
                    {
                        "id": "Flooded_Archive",
                        "changed": True,
                        "physical_change": "a new crack crosses the blue wall tiles",
                        "changed_properties": ["shape", "texture"],
                        "view_to_update": "close",
                        "new_state_text": "The blue wall tiles now have a visible crack.",
                        "image_refresh_required": True,
                        "source_event_roles": ["Mara"],
                    }
                ],
                "character_updates": [
                    {
                        "id": "Mara",
                        "changed": True,
                        "physical_change": "her coat changes from blue to red",
                        "changed_properties": ["clothing", "color"],
                        "new_state_text": "Mara wears a red coat.",
                        "image_refresh_required": True,
                        "source_event_roles": ["Mara"],
                    }
                ],
                "item_updates": [
                    {"id": "Sealed_Letter", "changed": False, "image_refresh_required": False}
                ],
            }

    previous_state = {
        "step": 0,
        "locations": [{"id": "Flooded_Archive", "name": "Flooded Archive", "current_state_text": "intact blue wall"}],
        "characters": [{"id": "Mara", "name": "Mara", "current_state_text": "wearing a blue coat"}],
        "items": [{"id": "Sealed_Letter", "name": "Sealed Letter", "current_state_text": "sealed"}],
    }
    step_log = {
        "step": 1,
        "sam_summary": "Mara strikes the wall and a crack appears across the blue tiles.",
        "role_logs": [{"role_name": "Mara", "next_action": {"action": "strike wall", "location": "Flooded Archive"}}],
    }

    delta = WorldDeltaJudge(llm_tool=FakeLLM(), llm_enabled=True).judge_step_delta(
        previous_state=previous_state,
        step_log=step_log,
        story="A lighthouse adventure.",
    )

    assert "physical appearance" in captured["system"].lower()
    assert "no people" in captured["system"].lower()
    assert delta["location_updates"] == [
        {
            "step": 1,
            "id": "Flooded_Archive",
            "changed": True,
            "physical_change": "a new crack crosses the blue wall tiles",
            "changed_properties": ["shape", "texture"],
            "view_to_update": "close",
            "new_state_text": "The blue wall tiles now have a visible crack.",
            "image_refresh_required": True,
            "reason": "llm_physical_world_state_change",
            "source_event_roles": ["Mara"],
        }
    ]
    assert delta["item_updates"] == []
    assert delta["character_updates"][0]["changed_properties"] == ["clothing", "color"]
    assert delta["character_updates"][0]["source_event_roles"] == ["Mara"]


def test_world_state_tracker_updates_only_selected_location_view() -> None:
    from narrative.world_state_tracker import build_initial_world_state, update_world_state_from_delta

    world_model = {
        "summary": "archive",
        "locations": [
            {
                "asset_id": "Flooded_Archive",
                "name": "Flooded Archive",
                "description": "archive",
                "views": {
                    "wide": {"output_path": "/tmp/wide.png", "prompt": "wide"},
                    "medium": {"output_path": "/tmp/medium.png", "prompt": "medium"},
                    "close": {"output_path": "/tmp/close.png", "prompt": "close"},
                    "extreme_close": {"output_path": "/tmp/extreme.png", "prompt": "extreme"},
                },
            }
        ],
        "items": [
            {
                "asset_id": "Sealed_Letter",
                "name": "Sealed Letter",
                "description": "sealed note",
                "initial_image_path": "/tmp/letter.png",
            }
        ],
    }
    state = build_initial_world_state(world_model=world_model, backend_tool="nano_banana")
    delta = {
        "step": 1,
        "summary": "The wall cracks.",
        "location_updates": [
            {
                "step": 1,
                "id": "Flooded_Archive",
                "view_to_update": "close",
                "new_state_text": "The close wall tiles are cracked.",
                "new_image_path": "/tmp/close_step1.png",
                "image_refresh_required": True,
                "reason": "llm_physical_world_state_change",
            }
        ],
        "item_updates": [],
    }

    updated = update_world_state_from_delta(previous_state=state, delta=delta)
    loc = updated["locations"][0]
    assert loc["views"]["wide"]["last_image_path"] == "/tmp/wide.png"
    assert loc["views"]["close"]["last_image_path"] == "/tmp/close_step1.png"
    assert loc["changed_views_this_step"] == ["close"]
    assert loc["last_image_path"] == "/tmp/close_step1.png"
    assert updated["items"][0]["views"]["isolated"]["last_image_path"] == "/tmp/letter.png"
