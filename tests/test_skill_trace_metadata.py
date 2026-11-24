from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
from scripts.run_four_genre_demo_pipeline import prepare_child_env


def _img(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (30, 50, 70)).save(path)
    return str(path)


def _write_skill(root: Path, name: str, template_name: str, template_text: str) -> Path:
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
    template_path = template_dir / template_name
    template_path.write_text(template_text, encoding="utf-8")
    return template_path


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


def test_video_agent_uses_env_skill_manifest_and_records_trace_metadata(tmp_path, monkeypatch):
    prompt_library = tmp_path / "prompt_library"
    _write_prompt_library(prompt_library)

    skill_root = tmp_path / "skills"
    template_text = "VIDEO_TRACE_SKILL_TEMPLATE"
    _write_skill(skill_root, "video-trace-skill", "video.txt", template_text)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """VideoAgent:
  default: video-trace-skill
  allowed:
    - video-trace-skill
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CAMERAMASTER_SKILL_ROOTS", str(skill_root))
    monkeypatch.setenv("CAMERAMASTER_SKILL_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("VIDEO_SKILL_NAME", "video-trace-skill")
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")

    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A trace metadata test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero enters a glowing room and lifts a lantern.",
                        "event_type": "information_reveal",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Glowing Room"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {"storyline": "A trace metadata test.", "ordered_node_ids": ["NN_001"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    agent = VideoAgent(strict_online=False)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
    )

    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[0]["skill_name"] == "video-trace-skill"
    assert tasks[0]["skill_source"] == str(skill_root)
    assert tasks[0]["skill_kind"] == "prompt"
    assert tasks[0]["prompt"] == template_text


def test_prepare_child_env_preserves_skill_override_variables(tmp_path):
    call_log = tmp_path / "logs" / "model_tool_calls.jsonl"
    base_env = {
        "CAMERAMASTER_SKILL_ROOTS": "/tmp/skills-a:/tmp/skills-b",
        "CAMERAMASTER_SKILL_POLICY_PATH": "/tmp/policy.yaml",
        "DIRECTOR_SKILL_NAME": "director-skill",
        "SHOT_SKILL_NAME": "shot-skill",
        "SUPERVISOR_SKILL_NAME": "supervisor-skill",
        "VIDEO_SKILL_NAME": "video-skill",
    }

    env = prepare_child_env(base_env=base_env, call_log_jsonl=call_log)

    for key, value in base_env.items():
        assert env[key] == value
