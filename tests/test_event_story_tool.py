from __future__ import annotations

import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

from agents.event_story_tool import EventStoryTool


class _FakeLLMTool:
    def __init__(self, payload: dict[str, str], *, should_fail: bool = False) -> None:
        self.payload = payload
        self.should_fail = should_fail

    def run_json(self, **kwargs):
        del kwargs
        if self.should_fail:
            raise RuntimeError("temporary LLM failure")
        return self.payload


def test_event_story_tool_generates_story_description_and_artifacts(tmp_path):
    tool = EventStoryTool(
        llm_tool=_FakeLLMTool(
            {"story_description": "Character1 confronts Character2 in the corridor."}
        )
    )

    result = tool.generate(
        node={
            "event_summary": "A character reaches a corridor and speaks to another character.",
            "video_description": "An event-level summary of the scene logic.",
            "characters": ["Hero", "Guide"],
            "locations": ["Corridor"],
        },
        storyline="A test story.",
        output_dir=tmp_path / "story_artifacts",
        rank=1,
        node_id="NN_001",
    )

    assert result["status"] == "done"
    assert result["story_description"] == "Character1 confronts Character2 in the corridor."
    assert Path(result["artifact_path"]).exists()
    assert Path(result["meta_path"]).exists()
    assert json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8")) == {
        "story_description": "Character1 confronts Character2 in the corridor."
    }


def test_event_story_tool_falls_back_to_video_description_when_llm_fails(tmp_path):
    tool = EventStoryTool(llm_tool=_FakeLLMTool({}, should_fail=True))

    result = tool.generate(
        node={
            "event_summary": "Fallback event.",
            "video_description": "A calm establishing beat.",
            "characters": ["Hero"],
            "locations": ["Room"],
        },
        storyline="Fallback story.",
        output_dir=tmp_path / "story_artifacts",
        rank=1,
        node_id="NN_001",
    )

    assert result["status"] == "fallback"
    assert result["story_description"] == "A calm establishing beat."
