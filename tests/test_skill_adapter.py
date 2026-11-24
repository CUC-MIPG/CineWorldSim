from __future__ import annotations

from agents.skills.adapter import SkillAdapter
from agents.skills.types import SkillManifest


def test_skill_adapter_builds_h3_prompt_with_metadata() -> None:
    manifest = SkillManifest(
        name="h3-prompt-writing",
        description="H3 prompt writing",
        kind="prompt",
        source_root="/tmp/skills",
        skill_path="/tmp/skills/h3-prompt-writing",
    )

    result = SkillAdapter().build(
        "VideoAgent",
        manifest,
        {
            "node": {
                "event_summary": "Hero raises a lantern in the chamber.",
                "characters": ["Hero"],
                "locations": ["Chamber"],
                "items": ["Lantern"],
            },
            "storyline": "A test story.",
            "prompt_style": "official_ref",
        },
    )

    assert result["prompt"]
    assert "subject_definitions:" in result["prompt"]
    assert result["skill_name"] == "h3-prompt-writing"
    assert result["skill_kind"] == "prompt"
    assert result["skill_source"] == "/tmp/skills"
    assert result["prompt_style"] == "official_ref"


def test_skill_adapter_prefers_story_description_for_compact_prompt() -> None:
    manifest = SkillManifest(
        name="cameraatist-event-video-prompt",
        description="Compact event prompt",
        kind="prompt",
        source_root="/tmp/skills",
        skill_path="/tmp/skills/cameraatist-event-video-prompt",
    )

    result = SkillAdapter().build(
        "VideoAgent",
        manifest,
        {
            "node": {
                "story_description": "Hero confronts the guard after the reveal.",
                "video_description": "Reasoning-style event summary that should not drive the prompt.",
                "event_summary": "Hero confronts the guard.",
                "characters": ["Hero", "Guard"],
                "locations": ["Corridor"],
                "items": ["Lantern"],
            },
            "storyline": "A test story.",
        },
    )

    assert "Hero confronts the guard after the reveal." in result["prompt"]
    assert "Reasoning-style event summary" not in result["prompt"]
