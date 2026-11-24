from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    kind: str
    source_root: str
    skill_path: str
    manifest_path: str = ""
    compatible_agents: tuple[str, ...] = ()
    prompt_template_paths: tuple[str, ...] = ()
    fallback_policy: str = "fallback"
