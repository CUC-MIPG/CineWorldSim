from __future__ import annotations

from typing import Any

from agents.skills.policy import AgentSkillPolicy
from agents.skills.registry import SkillRegistry
from agents.skills.types import SkillManifest


class SkillResolver:
    def __init__(self, registry: SkillRegistry, policy: AgentSkillPolicy):
        self.registry = registry
        self.policy = policy

    def resolve_exact(
        self,
        agent_name: str,
        requested_skill: str | None,
        context: dict[str, Any],
    ) -> SkillManifest | None:
        del context  # Reserved for future context-sensitive resolution.

        requested = requested_skill.strip() if isinstance(requested_skill, str) else ""
        if not requested or not self.policy.allows(agent_name, requested):
            return None
        return self.registry.get(requested)

    def resolve(
        self,
        agent_name: str,
        requested_skill: str | None,
        context: dict[str, Any],
    ) -> SkillManifest:
        del context  # Reserved for future context-sensitive resolution.

        requested = requested_skill.strip() if isinstance(requested_skill, str) else ""
        if requested and self.policy.allows(agent_name, requested):
            manifest = self.registry.get(requested)
            if manifest is not None:
                return manifest

        default = self.policy.default_skill(agent_name)
        if default and self.policy.allows(agent_name, default):
            manifest = self.registry.get(default)
            if manifest is not None:
                return manifest

        raise ValueError(
            f"Requested skill {requested_skill!r} is not allowed or unavailable, "
            f"and no allowed default could be resolved for agent={agent_name!r}"
        )
