from __future__ import annotations

import logging
from pathlib import Path

from agents.skills.loader import discover_skill_manifests
from agents.skills.types import SkillManifest

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, roots: list[Path]):
        self.roots = [Path(root) for root in roots]
        self._skills: dict[str, SkillManifest] = {}

    def load(self) -> None:
        self._skills.clear()
        for root in self.roots:
            for manifest in discover_skill_manifests(root):
                existing = self._skills.get(manifest.name)
                if existing is not None:
                    logger.warning(
                        "Duplicate skill name %s ignored from %s; keeping %s",
                        manifest.name,
                        manifest.skill_path,
                        existing.skill_path,
                    )
                    continue
                self._skills[manifest.name] = manifest

    def get(self, name: str) -> SkillManifest | None:
        return self._skills.get(name)

    def list(self) -> list[SkillManifest]:
        return [self._skills[name] for name in sorted(self._skills)]
