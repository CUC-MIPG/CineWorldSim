from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agents.skills.types import SkillManifest

logger = logging.getLogger(__name__)


def load_skill_manifest(source_root: Path, skill_path: Path) -> SkillManifest | None:
    skill_file = skill_path / "SKILL.md"
    meta_file = skill_path / "meta.yaml"
    front_matter = _load_skill_front_matter(skill_file)
    meta = _load_yaml_file(meta_file)
    data = {**meta, **front_matter}

    name = _normalize_text(data.get("name")) or skill_path.name
    description = (
        _normalize_text(data.get("description"))
        or _normalize_text(data.get("summary-en"))
        or _normalize_text(data.get("desc-en"))
        or ""
    )
    kind = _normalize_text(data.get("kind")) or "skill"
    compatible_agents = _normalize_sequence(data.get("compatible_agents"))
    prompt_template_paths = _normalize_sequence(data.get("prompt_template_paths"))
    fallback_policy = _normalize_text(data.get("fallback_policy")) or "fallback"

    return SkillManifest(
        name=name,
        description=description,
        kind=kind,
        source_root=str(source_root),
        skill_path=str(skill_path),
        manifest_path=str(skill_file if front_matter else meta_file if meta else skill_file),
        compatible_agents=compatible_agents,
        prompt_template_paths=prompt_template_paths,
        fallback_policy=fallback_policy,
    )


def discover_skill_manifests(source_root: Path) -> list[SkillManifest]:
    manifests: list[SkillManifest] = []
    if not source_root.exists():
        return manifests

    seen_paths: set[Path] = set()
    for pattern in ("**/SKILL.md", "**/meta.yaml"):
        for manifest_file in sorted(source_root.glob(pattern)):
            skill_path = manifest_file.parent
            if skill_path in seen_paths:
                continue
            seen_paths.add(skill_path)
            manifest = load_skill_manifest(source_root, skill_path)
            if manifest is not None:
                manifests.append(manifest)
    return manifests


def _load_skill_front_matter(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid YAML front matter fence in {path}")
    try:
        return _coerce_mapping(yaml.safe_load(parts[1]))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML front matter in {path}: {exc}") from exc


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _coerce_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML file in {path}: {exc}") from exc


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return ()
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())
