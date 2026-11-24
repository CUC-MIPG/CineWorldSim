from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.simulation_scaffold import StateAnalysisMachine, WorldModelAgent


def test_narrative_portrait_and_keyframe_prompts_block_extra_people(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def fake_generate_image_nano_banana(**kwargs):
        captured.append(kwargs["prompt"])
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"image")
        return str(kwargs["output_path"])

    monkeypatch.setattr("agents.generation_backends.generate_image_nano_banana", fake_generate_image_nano_banana)
    monkeypatch.setattr(
        "agents.generation_backends.nano_banana_settings_from_env",
        lambda: {"api_key": "x", "base_url": "http://example.com", "model_name": "model", "timeout_sec": 1},
    )

    machine = StateAnalysisMachine()
    character = SimpleNamespace(
        state=SimpleNamespace(
            name="Hero",
            role_profile="A lone investigator.",
            final_goal="Find the truth.",
            known_relations=["Guard"],
        ),
        portrait_path="",
    )
    action = SimpleNamespace(
        action_type="open door",
        target_character="none",
        reason="to enter the room",
        monologue="I need to see inside.",
        location="Hallway",
    )

    portrait = machine._generate_character_portrait(character=character, output_dir=tmp_path / "portraits", storyline="A test story.")
    keyframe = machine._generate_action_keyframe(
        character=character,
        action=action,
        prev_keyframe_path=None,
        output_dir=tmp_path / "keyframes",
        step=1,
        storyline="A test story.",
    )

    assert Path(portrait).exists()
    assert Path(keyframe).exists()
    assert len(captured) == 2
    assert "people_policy:" in captured[0].lower()
    assert "pedestrians" in captured[0].lower()
    assert "unrelated people" in captured[0].lower()
    assert "people_policy:" in captured[1].lower()
    assert "pedestrians" in captured[1].lower()
    assert "unrelated people" in captured[1].lower()


def test_world_image_prompt_blocks_extra_people() -> None:
    prompt = WorldModelAgent._format_world_image_prompt(
        "Cinematic establishing shot of a station, rich detail, no character close-up.",
        subject_type="location",
    )
    assert "people_policy:" in prompt.lower()
    assert "pedestrians" in prompt.lower()
    assert "unrelated people" in prompt.lower()
