from pathlib import Path
import os
import pytest

from agents.skills.registry import SkillRegistry


def _minimax_skill_root() -> Path:
    configured = os.environ.get("MINIMAX_H3_SKILLS_ROOT", "").strip()
    if configured:
        root = Path(configured)
    else:
        root = (
            Path(__file__).resolve().parents[2]
            / "minimax_h3"
            / "minimaxh3"
            / "MiniMax-H3"
            / "skills"
        )
    if not root.exists():
        pytest.skip(f"MiniMax-H3 skill root not available: {root}")
    return root


def test_registry_normalizes_front_matter_and_meta_yaml(tmp_path):
    root = tmp_path / "skills"
    skill_dir = root / "example-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: example-skill
description: Example skill
kind: prompt
compatible_agents:
  - VideoAgent
prompt_template_paths:
  - templates/prompt.txt
fallback_policy: compact
---

# Example
""",
        encoding="utf-8",
    )
    (skill_dir / "meta.yaml").write_text(
        """kind: workflow
compatible_agents:
  - DirectorAgent
prompt_template_paths:
  - templates/override.txt
""",
        encoding="utf-8",
    )

    registry = SkillRegistry([root])
    registry.load()

    manifest = registry.get("example-skill")
    assert manifest is not None
    assert manifest.name == "example-skill"
    assert manifest.description == "Example skill"
    assert manifest.kind == "prompt"
    assert manifest.source_root == str(root)
    assert manifest.skill_path == str(skill_dir)
    assert manifest.compatible_agents == ("VideoAgent",)
    assert manifest.prompt_template_paths == ("templates/prompt.txt",)
    assert manifest.fallback_policy == "compact"


def test_registry_discovers_local_and_minimax_h3_skill_roots():
    local_root = Path("docs/skills")
    minimax_root = _minimax_skill_root()

    registry = SkillRegistry([local_root, minimax_root])
    registry.load()

    names = {manifest.name for manifest in registry.list()}
    assert "h3-prompt-writing" in names
    assert "brand-promo-video-generator" in names
    assert registry.get("h3-prompt-writing").source_root == str(local_root)


def test_registry_can_find_h3_prompt_writing_in_minimax_tree():
    root = _minimax_skill_root()
    registry = SkillRegistry([root])
    registry.load()

    manifest = registry.get("h3-prompt-writing")
    assert manifest is not None
    assert manifest.skill_path == str(root / "h3-prompt-writing")
