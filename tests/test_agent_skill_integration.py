from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.director_agent import DirectorAgent
from agents.shot_agent import ShotAgent
from agents.story_supervisor_agent import StorySupervisorAgent
from agents.skills.policy import AgentSkillPolicy
from agents.skills.registry import SkillRegistry
from agents.skills.resolver import SkillResolver


def _write_skill(root: Path, name: str, template_name: str, template_text: str) -> None:
    skill_dir = root / name
    template_dir = skill_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {name}
kind: prompt
prompt_template_paths:
  - templates/{template_name}
---
""",
        encoding="utf-8",
    )
    (template_dir / template_name).write_text(template_text, encoding="utf-8")


def _write_prompt_library(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompt_query_template.py").write_text(
        """QUERY_DICT = {
    "Scene_Creation_Prompt": "PROMPT_LIBRARY_SCENE",
    "Story_design_Prompt": "PROMPT_LIBRARY_STORY",
    "Shot_Creation_Prompt": "PROMPT_LIBRARY_SHOT",
}
""",
        encoding="utf-8",
    )


def _fake_client(content: str):
    class _Response:
        def __init__(self, value: str):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=value))]

    class _Completions:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(content)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    return _Client()


def test_agents_use_shared_skill_resolver_and_registry_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_library = tmp_path / "prompt_library"
    _write_prompt_library(prompt_library)

    skill_root = tmp_path / "skills"
    _write_skill(skill_root, "cameraatist-scene-creation", "scene.txt", "SCENE_SKILL_TEMPLATE")
    _write_skill(skill_root, "cameraatist-story-design", "story.txt", "STORY_SKILL_TEMPLATE")
    _write_skill(skill_root, "cameraatist-shot-creation", "shot.txt", "SHOT_SKILL_TEMPLATE")
    _write_skill(skill_root, "cameraatist-shot-planning", "planning.txt", "PLANNING_SKILL_TEMPLATE")

    policy = AgentSkillPolicy(
        {
            "DirectorAgent": {"cameraatist-scene-creation", "cameraatist-story-design"},
            "ShotAgent": {"cameraatist-shot-creation"},
            "StorySupervisorAgent": {"cameraatist-shot-planning"},
        },
        {
            "DirectorAgent": "cameraatist-story-design",
            "ShotAgent": "cameraatist-shot-creation",
            "StorySupervisorAgent": "cameraatist-shot-planning",
        },
    )
    registry = SkillRegistry([skill_root])
    registry.load()
    resolver = SkillResolver(registry, policy)

    director_client = _fake_client(json.dumps({"Story": "story", "Scenes": {"scene1": "scene"}, "Characters": {"Hero": "hero"}}))
    shot_client = _fake_client(json.dumps({"Shots": {"content": "shot content", "scene": "SC001"}}))
    supervisor_client = _fake_client(
        json.dumps(
            {
                "scene_id": "SC001",
                "scene_storyline_summary": "summary",
                "shot_plans": [{"shot_id": "SC001_SH001", "sequence": 1, "duration_sec": 2.0, "purpose": "p", "transition_to_next": "t"}],
                "narration": {"text": "n", "duration_sec": 1.0, "tone": "tone"},
                "dialogue": [{"speaker": "A", "text": "hi", "duration_sec": 1.0}],
                "music": {"description": "m", "mood": "mo", "duration_sec": 1.0},
                "editing_notes": "notes",
            }
        )
    )

    monkeypatch.setattr("agents.director_agent.make_client", lambda scope="DIRECTOR": director_client)
    monkeypatch.setattr("agents.shot_agent.make_client", lambda scope="SHOT": shot_client)
    monkeypatch.setattr("agents.story_supervisor_agent.make_client", lambda scope="SUPERVISOR": supervisor_client)

    director = DirectorAgent(prompt_library=prompt_library, strict_online=False, skill_resolver=resolver)
    shot = ShotAgent(prompt_library=prompt_library, strict_online=False, skill_resolver=resolver)
    supervisor = StorySupervisorAgent(strict_online=False, skill_resolver=resolver)

    fast = director.fast_story_cot("A story about a lantern.")
    structured = director.story_design_structured(json.dumps(fast, ensure_ascii=False))
    shot_payload = shot.slow_shot_cot("A story about a lantern.", "A room with a lantern.", "None", "SC001")
    plan = supervisor.generate_scene_plan(
        scene=SimpleNamespace(scene_id="SC001", location="Room", time_of_day="night", plot_description="A lantern glows."),
        shots=[SimpleNamespace(shot_id="SC001_SH001", sequence=1, auto_prompt=lambda: "A shot prompt.")],
    )

    assert set(fast) >= {"Story", "Scenes", "Characters"}
    assert isinstance(structured, dict)
    assert set(shot_payload) == {"Shots"}
    assert set(plan) >= {"scene_id", "shot_plans"}

    assert director_client.chat.completions.calls[0]["messages"][0]["content"] == "SCENE_SKILL_TEMPLATE"
    assert director_client.chat.completions.calls[1]["messages"][0]["content"] == "STORY_SKILL_TEMPLATE"
    assert shot_client.chat.completions.calls[0]["messages"][0]["content"] == "SHOT_SKILL_TEMPLATE"
    assert supervisor_client.chat.completions.calls[0]["messages"][0]["content"] == "PLANNING_SKILL_TEMPLATE"


def test_agents_fall_back_to_compatibility_prompts_when_exact_skill_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_library = tmp_path / "prompt_library"
    _write_prompt_library(prompt_library)

    skill_root = tmp_path / "skills"
    _write_skill(skill_root, "cameraatist-story-design", "story.txt", "STORY_SKILL_TEMPLATE")

    policy = AgentSkillPolicy(
        {
            "DirectorAgent": {"cameraatist-scene-creation", "cameraatist-story-design"},
            "ShotAgent": {"cameraatist-shot-creation"},
            "StorySupervisorAgent": {"cameraatist-shot-planning"},
        },
        {
            "DirectorAgent": "cameraatist-story-design",
            "ShotAgent": "cameraatist-shot-creation",
            "StorySupervisorAgent": "cameraatist-shot-planning",
        },
    )
    registry = SkillRegistry([skill_root])
    registry.load()
    resolver = SkillResolver(registry, policy)

    director_client = _fake_client(json.dumps({"Story": "story", "Scenes": {"scene1": "scene"}, "Characters": {"Hero": "hero"}}))
    shot_client = _fake_client(json.dumps({"Shots": {"content": "shot content", "scene": "SC001"}}))
    supervisor_client = _fake_client(
        json.dumps(
            {
                "scene_id": "SC001",
                "scene_storyline_summary": "summary",
                "shot_plans": [{"shot_id": "SC001_SH001", "sequence": 1, "duration_sec": 2.0, "purpose": "p", "transition_to_next": "t"}],
                "narration": {"text": "n", "duration_sec": 1.0, "tone": "tone"},
                "dialogue": [{"speaker": "A", "text": "hi", "duration_sec": 1.0}],
                "music": {"description": "m", "mood": "mo", "duration_sec": 1.0},
                "editing_notes": "notes",
            }
        )
    )

    monkeypatch.setattr("agents.director_agent.make_client", lambda scope="DIRECTOR": director_client)
    monkeypatch.setattr("agents.shot_agent.make_client", lambda scope="SHOT": shot_client)
    monkeypatch.setattr("agents.story_supervisor_agent.make_client", lambda scope="SUPERVISOR": supervisor_client)

    director = DirectorAgent(prompt_library=prompt_library, strict_online=False, skill_resolver=resolver)
    shot = ShotAgent(prompt_library=prompt_library, strict_online=False, skill_resolver=resolver)
    supervisor = StorySupervisorAgent(strict_online=False, skill_resolver=resolver)

    director.fast_story_cot("A story about a lantern.")
    director.story_design_structured(json.dumps({"Story": "story"}, ensure_ascii=False))
    shot.slow_shot_cot("A story about a lantern.", "A room with a lantern.", "None", "SC001")
    plan = supervisor.generate_scene_plan(
        scene=SimpleNamespace(scene_id="SC001", location="Room", time_of_day="night", plot_description="A lantern glows."),
        shots=[SimpleNamespace(shot_id="SC001_SH001", sequence=1, auto_prompt=lambda: "A shot prompt.")],
    )

    assert director_client.chat.completions.calls[0]["messages"][0]["content"] == "PROMPT_LIBRARY_SCENE"
    assert director_client.chat.completions.calls[1]["messages"][0]["content"] == "STORY_SKILL_TEMPLATE"
    assert shot_client.chat.completions.calls[0]["messages"][0]["content"] == "PROMPT_LIBRARY_SHOT"
    assert "电影剧情监制" in supervisor_client.chat.completions.calls[0]["messages"][0]["content"]
    assert set(plan) >= {"scene_id", "shot_plans"}
