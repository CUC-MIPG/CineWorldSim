from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from agents.vistorybench_anchor_selector import (
    ViStoryBenchAnchorSelector,
    ViStoryBenchSelectorConfig,
)
from agents.vistorybench_prompt_adapter import build_vistorybench_prompt
from utils.export_vistorybench_outputs import validate_story_output
from utils.vistorybench_image import normalize_image_to_png


def _story() -> dict:
    return {
        "Shots": [
            {
                "index": 1,
                "Setting Description": {"en": "A dark station"},
                "Plot Correspondence": {"en": "The detective enters"},
                "Shot Perspective Design": {"en": "Wide shot"},
                "Static Shot Description": {"en": "The detective enters the station"},
                "Characters Appearing": {"en": ["Detective"]},
            },
            {
                "index": 2,
                "Setting Description": {"en": "A dark station"},
                "Plot Correspondence": {"en": "The detective finds a key"},
                "Shot Perspective Design": {"en": "Close shot"},
                "Static Shot Description": {"en": "The detective holds a key"},
                "Characters Appearing": {"en": ["Detective"]},
            },
        ]
    }


def _simulation() -> dict:
    return {
        "step_logs": [
            {
                "step": 1,
                "sam_summary": "The detective enters the station.",
                "role_logs": [{
                    "role_name": "Detective",
                    "role_output_text": "The detective enters.",
                    "next_action": {"action": "enter station", "location": "dark station"},
                }],
            },
            {
                "step": 3,
                "sam_summary": "Future information that must not leak.",
                "role_logs": [{
                    "role_name": "Detective",
                    "role_output_text": "Future action.",
                    "next_action": {"action": "open vault", "location": "dark station"},
                }],
            },
        ],
        "world_states": [{"step": 1, "summary": "dark station"}],
    }


def test_selector_preserves_fixed_shots_and_excludes_future_context() -> None:
    nodes = ViStoryBenchAnchorSelector().build_nodes(_story(), _simulation(), story_id="01")
    assert [node["shot_index"] for node in nodes] == [1, 2]
    assert len(nodes) == 2
    assert all(event["step"] <= node["shot_index"] for node in nodes for event in node["simulation_context"]["related_events"])


def test_prompt_modes_keep_simulation_context_isolated() -> None:
    node = ViStoryBenchAnchorSelector().build_nodes(_story(), _simulation(), story_id="01")[0]
    anchor_prompt = build_vistorybench_prompt(node, mode="vistorybench_anchor_only")
    sim_prompt = build_vistorybench_prompt(node, mode="vistorybench_anchor_plus_sim")
    assert "[Continuity Context]" not in anchor_prompt
    assert "[Continuity Context]" in sim_prompt
    assert "benchmark shot requirement has priority" in sim_prompt.lower()


def test_validator_requires_exact_shot_set(tmp_path: Path) -> None:
    story_dir = tmp_path / "01"
    shot_dir = story_dir / "shots"
    shot_dir.mkdir(parents=True)
    (shot_dir / "shot_001.png").write_bytes(b"png")
    report = validate_story_output(_story(), story_dir)
    assert report["ok"] is False
    assert report["missing"] == [2]


def test_normalize_image_to_png_converts_jpeg_content_with_png_extension(tmp_path: Path) -> None:
    image_path = tmp_path / "shot_001.png"
    image = Image.new("RGB", (12, 8), color=(200, 30, 10))
    image.save(image_path, format="JPEG")

    normalized = normalize_image_to_png(image_path)

    with Image.open(normalized) as result:
        assert result.format == "PNG"
        assert result.size == (12, 8)


def test_validator_rejects_jpeg_bytes_named_as_png(tmp_path: Path) -> None:
    story_dir = tmp_path / "01"
    shot_dir = story_dir / "shots"
    shot_dir.mkdir(parents=True)
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(shot_dir / "shot_001.png", format="JPEG")
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(shot_dir / "shot_002.png", format="PNG")

    report = validate_story_output(_story(), story_dir)

    assert any("not PNG encoded" in error for error in report["errors"])
