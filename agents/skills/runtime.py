from __future__ import annotations

import os
from pathlib import Path

from agents.skills.policy import AgentSkillPolicy
from agents.skills.registry import SkillRegistry
from agents.skills.resolver import SkillResolver


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILL_POLICY_PATH = ROOT / "configs" / "agent_skill_policies.yaml"

_AGENT_SKILL_ENV = {
    "DirectorAgent": "DIRECTOR_SKILL_NAME",
    "ShotAgent": "SHOT_SKILL_NAME",
    "StorySupervisorAgent": "SUPERVISOR_SKILL_NAME",
    "VideoAgent": "VIDEO_SKILL_NAME",
}


def parse_skill_roots(raw: str | None) -> list[Path]:
    text = str(raw or "").strip()
    if not text:
        return []
    normalized = text.replace(os.pathsep, ",")
    return [Path(piece.strip()) for piece in normalized.split(",") if piece.strip()]


def skill_name_override(agent_name: str) -> str:
    env_name = _AGENT_SKILL_ENV.get(agent_name, "")
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def build_skill_resolver_from_env() -> SkillResolver | None:
    roots = parse_skill_roots(os.getenv("CAMERAMASTER_SKILL_ROOTS"))
    if not roots:
        return None

    registry = SkillRegistry(roots)
    registry.load()
    raw_policy_path = os.getenv("CAMERAMASTER_SKILL_POLICY_PATH", "").strip()
    policy_path = Path(raw_policy_path).expanduser() if raw_policy_path else DEFAULT_SKILL_POLICY_PATH
    if not policy_path.exists():
        return None
    policy = AgentSkillPolicy.from_yaml(policy_path)
    return SkillResolver(registry, policy)
