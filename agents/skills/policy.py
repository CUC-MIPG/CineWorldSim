from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class AgentSkillPolicy:
    def __init__(
        self,
        allowlists: Mapping[str, set[str] | frozenset[str]],
        defaults: Mapping[str, str | None] | None = None,
    ):
        self._allowlists = {
            agent_name: frozenset(skill for skill in skills if skill)
            for agent_name, skills in allowlists.items()
        }
        self._defaults = {
            agent_name: default
            for agent_name, default in (defaults or {}).items()
            if default
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentSkillPolicy":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Agent skill policy must be a mapping: {path}")

        allowlists: dict[str, set[str]] = {}
        defaults: dict[str, str | None] = {}
        for agent_name, raw_policy in data.items():
            if not isinstance(agent_name, str):
                raise ValueError(f"Agent skill policy key must be a string: {path}")
            if not isinstance(raw_policy, dict):
                raise ValueError(f"Agent skill policy for {agent_name} must be a mapping")

            raw_allowed = raw_policy.get("allowed", [])
            if not isinstance(raw_allowed, list):
                raise ValueError(f"allowed for {agent_name} must be a list")
            allowlists[agent_name] = {
                skill_name.strip()
                for skill_name in raw_allowed
                if isinstance(skill_name, str) and skill_name.strip()
            }

            default = raw_policy.get("default")
            defaults[agent_name] = default.strip() if isinstance(default, str) and default.strip() else None

        return cls(allowlists, defaults)

    def allows(self, agent_name: str, skill_name: str) -> bool:
        return skill_name in self._allowlists.get(agent_name, frozenset())

    def default_skill(self, agent_name: str) -> str | None:
        return self._defaults.get(agent_name)
