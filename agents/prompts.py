from __future__ import annotations

import importlib.util
from pathlib import Path

from agents.skills.types import SkillManifest


def load_prompt_library_query_dict(prompt_library: Path) -> dict:
    prompt_file = prompt_library / "prompt_query_template.py"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Missing prompt_query_template.py: {prompt_file}")

    spec = importlib.util.spec_from_file_location("cameraatist_prompt_query_template", str(prompt_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load prompt_query_template module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    query_dict = getattr(module, "QUERY_DICT", None)
    if not isinstance(query_dict, dict):
        raise RuntimeError("QUERY_DICT missing or invalid in prompt_query_template.py")
    return query_dict


def load_cameraatist_query_dict(cameraatist_dir: Path) -> dict:
    # Backward-compatible alias.
    return load_prompt_library_query_dict(cameraatist_dir)


def build_scene_creation_system_prompt(query_dict: dict) -> str:
    return query_dict["Scene_Creation_Prompt"]


def build_shot_creation_system_prompt(query_dict: dict) -> str:
    return query_dict["Shot_Creation_Prompt"]


def build_story_design_system_prompt(query_dict: dict) -> str:
    return query_dict["Story_design_Prompt"]


def load_skill_prompt_template_text(skill: SkillManifest) -> str | None:
    skill_dir = Path(skill.skill_path)
    for relative_path in skill.prompt_template_paths:
        template_path = Path(relative_path)
        if not template_path.is_absolute():
            template_path = skill_dir / template_path
        if not template_path.exists():
            continue
        text = template_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None
