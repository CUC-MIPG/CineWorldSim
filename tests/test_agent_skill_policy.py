from __future__ import annotations

import pytest
from pathlib import Path

from agents.skills.policy import AgentSkillPolicy
from agents.skills.registry import SkillRegistry
from agents.skills.resolver import SkillResolver


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "agent_skill_policies.yaml"


def _write_skill(root: Path, name: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name}\n---\n",
        encoding="utf-8",
    )


def _registry_with(root: Path, *skill_names: str) -> SkillRegistry:
    root.mkdir(parents=True, exist_ok=True)
    for skill_name in skill_names:
        _write_skill(root, skill_name)
    registry = SkillRegistry([root])
    registry.load()
    return registry


def test_policy_loads_allowlists_and_defaults() -> None:
    policy = AgentSkillPolicy.from_yaml(POLICY_PATH)

    assert policy.allows("VideoAgent", "h3-prompt-writing")
    assert not policy.allows("ShotAgent", "h3-prompt-writing")
    assert policy.default_skill("DirectorAgent") == "cameraatist-story-design"


def test_resolver_falls_back_to_allowed_default_when_requested_skill_missing_from_registry(
    tmp_path: Path,
) -> None:
    policy = AgentSkillPolicy(
        {
            "VideoAgent": {
                "cameraatist-event-video-prompt",
                "h3-prompt-writing",
            }
        },
        {"VideoAgent": "h3-prompt-writing"},
    )
    registry = _registry_with(tmp_path / "skills", "h3-prompt-writing")
    resolver = SkillResolver(registry, policy)

    manifest = resolver.resolve("VideoAgent", "cameraatist-event-video-prompt", {})

    assert manifest.name == "h3-prompt-writing"


def test_resolver_rejects_disallowed_requested_skill_when_default_is_not_allowed(
    tmp_path: Path,
) -> None:
    policy = AgentSkillPolicy(
        {"ShotAgent": {"cameraatist-shot-creation"}},
        {"ShotAgent": "cameraatist-shot-planning"},
    )
    registry = _registry_with(tmp_path / "skills", "cameraatist-shot-planning")
    resolver = SkillResolver(registry, policy)

    with pytest.raises(ValueError, match="not allowed or unavailable"):
        resolver.resolve("ShotAgent", "h3-prompt-writing", {})
