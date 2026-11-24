"""Narrative simulation scaffold.

Responsibility split:
1) OrchestratorAgent: understand input + create character agents/configuration only
2) StateAnalysisMachine: create initial nodes + run deduction loop + action evaluation +
   story anchoring + environment sync
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.call_logger import new_call_id, text_summary, write_call_event
from agents.http_utils import openai_client_kwargs
from narrative.story_control import StoryController
from narrative.visual_prompt_constraints import build_people_control_clause

LOGGER = logging.getLogger(__name__)


def _read_text_file(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def _load_local_env_file() -> None:
    """Best-effort .env loader so scripts work without manual `source .env`."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Tolerate fenced blocks / extra wrappers.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())[:80] or "item"


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}_v{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _split_csv_values(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _parse_confidence(value: Any, default: float = 0.6) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return default
    text = str(value).strip().lower()
    if not text:
        return default
    mapping = {
        "high": 0.85,
        "medium": 0.6,
        "mid": 0.6,
        "low": 0.35,
        "very_high": 0.95,
        "very low": 0.2,
        "very_high_confidence": 0.95,
        "very_low": 0.2,
    }
    if text in mapping:
        return mapping[text]
    if text.endswith("%"):
        try:
            return max(0.0, min(1.0, float(text[:-1].strip()) / 100.0))
        except Exception:
            return default
    try:
        val = float(text)
        return max(0.0, min(1.0, val if val <= 1.0 else val / 100.0))
    except Exception:
        return default


def _is_ssl_retryable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    tokens = [
        "sslerror",
        "httpsconnectionpool",
        "max retries exceeded",
        "_ssl.c",
        "certificate",
        "tls",
        "wrong version number",
        "eof occurred in violation of protocol",
    ]
    return any(tok in text for tok in tokens)


@dataclass
class ModelConfig:
    model_type: str
    model_name: str
    base_url: str
    api_key: str
    system_prompt_path: str
    system_prompt: str = ""
    timeout_sec: int = 480
    llm_enabled: bool = False
    call_history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @classmethod
    def from_env(cls, prefix: str, default_model_type: str) -> "ModelConfig":
        system_prompt_path = os.getenv(f"{prefix}_SYSTEM_PROMPT", "")
        model_name = os.getenv(f"{prefix}_MODEL_NAME", "") or os.getenv("OPENAI_MODEL", "")
        base_url = (
            os.getenv(f"{prefix}_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or os.getenv("OPENAI_API_BASE", "")
        )
        api_key = os.getenv(f"{prefix}_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        return cls(
            model_type=os.getenv(f"{prefix}_MODEL_TYPE", default_model_type),
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            system_prompt_path=system_prompt_path,
            system_prompt=_read_text_file(system_prompt_path),
            timeout_sec=int(os.getenv(f"{prefix}_TIMEOUT_SEC", "480")),
            llm_enabled=str(os.getenv(f"{prefix}_ENABLE_LLM", "true")).lower() in ("1", "true", "yes", "on"),
        )

    def is_ready(self) -> bool:
        return self.llm_enabled and bool(self.model_name and self.base_url and self.api_key)

    def _record_call(
        self,
        *,
        call_id: str,
        status: str,
        call_label: str,
        error: str = "",
        fallback_used: bool = False,
    ) -> None:
        self.call_history.append(
            {
                "call_id": call_id,
                "status": status,
                "call_label": call_label,
                "model": self.model_name,
                "fallback_used": bool(fallback_used),
                "error": error,
            }
        )

    def chat_json(
        self,
        user_prompt: str,
        fallback: Dict[str, Any],
        temperature: float = 0.3,
        call_label: str = "",
    ) -> Dict[str, Any]:
        if not self.is_ready():
            call_id = new_call_id("llm-skip")
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "llm",
                    "status": "fallback",
                    "scope": "narrative_simulation",
                    "model": self.model_name,
                    "reason": "model config not ready",
                    "prompt_preview": text_summary(user_prompt),
                }
            )
            self._record_call(
                call_id=call_id,
                status="fallback",
                call_label=call_label,
                error="model config not ready",
                fallback_used=True,
            )
            return fallback
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("openai package unavailable: %s", exc)
            call_id = new_call_id("llm-import-fallback")
            self._record_call(
                call_id=call_id,
                status="fallback",
                call_label=call_label,
                error=text_summary(exc, 1200),
                fallback_used=True,
            )
            return fallback

        ssl_retry_sleep_sec = int(os.getenv("NARRATIVE_SSL_RETRY_SLEEP_SEC", "60"))
        max_attempts = 2  # first attempt + one retry for transient SSL/network handshake failure
        for attempt in range(1, max_attempts + 1):
            call_id = new_call_id("llm-narrative")
            start = time.time()
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **openai_client_kwargs(),
                )
                write_call_event(
                    {
                        "call_id": call_id,
                        "kind": "llm",
                        "status": "start",
                        "scope": "narrative_simulation",
                        "model": self.model_name,
                        "base_url": self.base_url,
                        "timeout": self.timeout_sec,
                        "temperature": temperature,
                        "attempt": attempt,
                        "prompt_preview": text_summary(user_prompt),
                    }
                )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt or "Return JSON only."},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    timeout=self.timeout_sec,
                )
                content = response.choices[0].message.content if response.choices else ""
                usage = getattr(response, "usage", None)
                write_call_event(
                    {
                        "call_id": call_id,
                        "kind": "llm",
                        "status": "done",
                        "scope": "narrative_simulation",
                        "model": self.model_name,
                        "elapsed_sec": round(time.time() - start, 3),
                        "attempt": attempt,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                )
                parsed = _extract_json_object(content or "")
                if parsed is None:
                    reason = "model returned non-JSON content"
                    LOGGER.warning("%s, using fallback", reason)
                    write_call_event(
                        {
                            "call_id": call_id,
                            "kind": "llm",
                            "status": "fallback",
                            "scope": "narrative_simulation",
                            "model": self.model_name,
                            "reason": reason,
                        }
                    )
                    self._record_call(
                        call_id=call_id,
                        status="fallback",
                        call_label=call_label,
                        error=reason,
                        fallback_used=True,
                    )
                    return fallback
                self._record_call(
                    call_id=call_id,
                    status="success",
                    call_label=call_label,
                )
                return parsed
            except Exception as exc:  # pragma: no cover
                write_call_event(
                    {
                        "call_id": call_id,
                        "kind": "llm",
                        "status": "failed",
                        "scope": "narrative_simulation",
                        "model": self.model_name,
                        "elapsed_sec": round(time.time() - start, 3),
                        "attempt": attempt,
                        "error": text_summary(exc, 1200),
                    }
                )
                retryable = _is_ssl_retryable_error(exc)
                if attempt < max_attempts and retryable:
                    LOGGER.warning(
                        "model call SSL/network error, sleeping %ss then retrying once: %s",
                        ssl_retry_sleep_sec,
                        exc,
                    )
                    time.sleep(float(ssl_retry_sleep_sec))
                    continue
                LOGGER.warning("model call failed, using fallback: %s", exc)
                self._record_call(
                    call_id=call_id,
                    status="fallback",
                    call_label=call_label,
                    error=text_summary(exc, 1200),
                    fallback_used=True,
                )
                return fallback
        return fallback


@dataclass
class CharacterState:
    name: str
    final_goal: str = ""
    role_profile: str = ""
    known_relations: List[str] = field(default_factory=list)
    current_environment: str = ""
    state_summary: str = ""
    known_clues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "final_goal": self.final_goal,
            "role_profile": self.role_profile,
            "known_relations": list(self.known_relations),
            "current_environment": self.current_environment,
            "state_summary": self.state_summary,
            "known_clues": list(self.known_clues),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterState":
        return cls(
            name=str(data.get("name", "")),
            final_goal=str(data.get("final_goal", "")),
            role_profile=str(data.get("role_profile", "")),
            known_relations=[str(x) for x in data.get("known_relations", []) if str(x).strip()],
            current_environment=str(data.get("current_environment", "")),
            state_summary=str(data.get("state_summary", "")),
            known_clues=[str(x) for x in data.get("known_clues", []) if str(x).strip()],
        )


@dataclass
class CharacterConfig:
    name: str
    final_goal: str
    role_profile: str
    known_relations: List[str]
    current_environment: str
    model_config: ModelConfig
    prompt_arm: str = ""
    dynamic_system_prompt: str = ""
    portrait_path: str = ""


class WorldModelAgent:
    """Build a story world model focused on locations and important props/clues."""

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.model_config = model_config or ModelConfig.from_env("NARRATIVE_WORLD_MODEL", "LLM")

    @staticmethod
    def _max_locations() -> int:
        try:
            return max(1, int(os.getenv("NARRATIVE_WORLD_MODEL_MAX_LOCATIONS", "8")))
        except Exception:
            return 8

    @staticmethod
    def _max_items() -> int:
        try:
            return max(1, int(os.getenv("NARRATIVE_WORLD_MODEL_MAX_ITEMS", "12")))
        except Exception:
            return 12

    @staticmethod
    def _tool_sequence() -> List[str]:
        raw = os.getenv("NARRATIVE_WORLD_MODEL_IMAGE_TOOLS", "nano_banana,flux_local")
        allowed = {"nano_banana", "flux_local", "flux"}
        tools = [tool for tool in _split_csv_values(raw) if tool in allowed]
        return tools or ["nano_banana"]

    @staticmethod
    def _format_world_image_prompt(base_prompt: str, *, subject_type: str) -> str:
        prompt = str(base_prompt or "").strip()
        suffix = (
            " Use a cinematic 16:9 frame composition, wide horizontal layout, "
            "clear spatial readability, high detail, coherent film lighting, "
            f"and keep the focus on the {subject_type} itself."
        )
        if "16:9" in prompt or "wide horizontal" in prompt.lower():
            return f"{prompt}\n{build_people_control_clause([], allow_multi_people=False)}".strip()
        return f"{prompt}{suffix}\n{build_people_control_clause([], allow_multi_people=False)}".strip()

    @staticmethod
    def _count_mentions(story_prompt: str, name: str) -> int:
        needle = str(name or "").strip().lower()
        if not needle:
            return 0
        return max(1, story_prompt.lower().count(needle))

    def _fallback_extract(self, story_prompt: str) -> Dict[str, Any]:
        story_l = story_prompt.lower()
        location_keywords = [
            ("museum", "Museum Interior"),
            ("gallery", "Gallery Hall"),
            ("city", "City Street"),
            ("apartment", "Apartment Interior"),
            ("house", "House Interior"),
            ("office", "Office Space"),
            ("station", "Transit Station"),
            ("warehouse", "Warehouse"),
            ("forest", "Forest Path"),
            ("school", "School Building"),
            ("hospital", "Hospital Corridor"),
        ]
        item_keywords = [
            ("artifact", "Key Artifact"),
            ("key", "Key"),
            ("phone", "Phone"),
            ("letter", "Letter"),
            ("ring", "Ring"),
            ("book", "Book"),
            ("gun", "Weapon"),
            ("knife", "Knife"),
            ("bag", "Bag"),
            ("document", "Document"),
            ("camera", "Camera"),
            ("mask", "Mask"),
        ]

        locations: List[Dict[str, Any]] = []
        for token, label in location_keywords:
            if token in story_l:
                locations.append(
                    {
                        "name": label,
                        "description": f"A cinematic story space centered on {label.lower()} within the given plot.",
                        "evidence": token,
                        "mentions_estimate": self._count_mentions(story_prompt, token),
                        "image_prompt": self._format_world_image_prompt(
                            f"Cinematic establishing shot of {label.lower()}, rich environmental detail, "
                            "story-driven atmosphere, no character close-up.",
                            subject_type="location",
                        ),
                    }
                )
        if not locations:
            locations.append(
                {
                    "name": "Primary Story Space",
                    "description": "The main dramatic environment implied by the story.",
                    "evidence": "global story context",
                    "mentions_estimate": 1,
                    "image_prompt": self._format_world_image_prompt(
                        "Cinematic establishing shot of the primary story space, rich spatial depth, "
                        "clear environmental storytelling, no character close-up.",
                        subject_type="location",
                    ),
                }
            )

        items: List[Dict[str, Any]] = []
        for token, label in item_keywords:
            if token in story_l:
                items.append(
                    {
                        "name": label,
                        "description": f"An important story object or clue related to {token}.",
                        "evidence": token,
                        "mentions_estimate": self._count_mentions(story_prompt, token),
                        "image_prompt": self._format_world_image_prompt(
                            f"Cinematic clue object study of a {label.lower()}, isolated but story-relevant, "
                            "high texture detail, no human hands.",
                            subject_type="object",
                        ),
                    }
                )

        summary = (
            "World model extracted from fallback heuristics. "
            f"Locations={len(locations)}, items={len(items)}."
        )
        return {
            "summary": summary,
            "locations": locations[: self._max_locations()],
            "items": items[: self._max_items()],
        }

    @classmethod
    def _normalize_records(
        cls,
        records: Any,
        *,
        story_prompt: str,
        fallback_prefix: str,
        fallback_prompt_prefix: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(records, list):
            return []
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for idx, item in enumerate(records, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip() or f"{fallback_prefix} {idx}"
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            description = str(item.get("description", "") or "").strip() or name
            evidence = str(item.get("evidence", "") or "").strip()
            prompt = str(item.get("image_prompt", "") or "").strip() or (
                f"{fallback_prompt_prefix}: {description}. Story context: {story_prompt}. Cinematic, highly detailed."
            )
            prompt = cls._format_world_image_prompt(
                prompt,
                subject_type="location" if fallback_prefix.lower().startswith("location") else "object",
            )
            try:
                mentions_estimate = max(1, int(item.get("mentions_estimate", 1)))
            except Exception:
                mentions_estimate = 1
            rows.append(
                {
                    "name": name,
                    "description": description,
                    "evidence": evidence,
                    "mentions_estimate": mentions_estimate,
                    "image_prompt": prompt,
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def extract_world_blueprint(self, story_prompt: str) -> Dict[str, Any]:
        fallback = self._fallback_extract(story_prompt)
        user_prompt = (
            "Read the full story and extract the world spaces and important repeat-mentioned props/clues.\n"
            "Return JSON only with fields: summary, locations, items.\n"
            "locations: list of {name, description, evidence, mentions_estimate, image_prompt}\n"
            "items: list of {name, description, evidence, mentions_estimate, image_prompt}\n"
            "Only include story-relevant spaces and objects that are important, repeated, or likely required for spatial modeling. "
            "For image_prompt values, do not introduce pedestrians, crowds, bystanders, or unrelated people unless the story explicitly requires them."
            f"\nStory:\n{story_prompt}"
        )
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.2,
            call_label="world_blueprint",
        )
        summary = str(payload.get("summary", fallback["summary"])).strip() or fallback["summary"]
        locations = self._normalize_records(
            payload.get("locations", fallback["locations"]),
            story_prompt=story_prompt,
            fallback_prefix="Location",
            fallback_prompt_prefix="Cinematic establishing shot",
            limit=self._max_locations(),
        )
        items = self._normalize_records(
            payload.get("items", fallback["items"]),
            story_prompt=story_prompt,
            fallback_prefix="Item",
            fallback_prompt_prefix="Cinematic object study",
            limit=self._max_items(),
        )
        return {
            "summary": summary,
            "locations": locations or fallback["locations"],
            "items": items or fallback["items"],
        }

    def _render_with_tool(
        self,
        *,
        tool_name: str,
        prompt: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        from agents.generation_backends import (
            flux_local_settings_from_env,
            flux_settings_from_env,
            generate_image_flux,
            generate_image_flux_local,
            generate_image_flux2_klein,
            flux2_klein_settings_from_env,
            generate_image_nano_banana,
            nano_banana_settings_from_env,
        )

        try:
            if tool_name == "nano_banana":
                cfg = nano_banana_settings_from_env()
                if not cfg["api_key"]:
                    raise RuntimeError("NANO_BANANA_API_KEY is missing")
                img = generate_image_nano_banana(
                    prompt=prompt,
                    output_path=output_path,
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    model_name=cfg["model_name"],
                    timeout_sec=cfg["timeout_sec"],
                )
            elif tool_name == "flux_local":
                cfg = flux_local_settings_from_env()
                img = generate_image_flux_local(
                    prompt=prompt,
                    output_path=output_path,
                    model_dir=cfg["model_dir"],
                    conda_env=cfg["conda_env"],
                    script_path=cfg["script_path"],
                    width=cfg["width"],
                    height=cfg["height"],
                    num_inference_steps=cfg["num_inference_steps"],
                    guidance_scale=cfg["guidance_scale"],
                    device=cfg["device"],
                    dtype=cfg["dtype"],
                    seed=cfg["seed"],
                    timeout_sec=cfg["timeout_sec"],
                    python_bin=cfg["python_bin"],
                )
            elif tool_name == "flux2_klein":
                cfg = flux2_klein_settings_from_env()
                img = generate_image_flux2_klein(
                    prompt=prompt,
                    output_path=output_path,
                    api_url=cfg["api_url"],
                    width=cfg["width"],
                    height=cfg["height"],
                    num_inference_steps=cfg["num_inference_steps"],
                    seed=cfg["seed"],
                    timeout_sec=cfg["timeout_sec"],
                )
            else:
                cfg = flux_settings_from_env()
                img = generate_image_flux(
                    prompt=prompt,
                    output_path=output_path,
                    api_url=cfg["api_url"],
                    width=cfg["width"],
                    height=cfg["height"],
                    timeout_sec=cfg["timeout_sec"],
                )
            return {
                "tool": tool_name,
                "ok": bool(img),
                "output_path": str(img or output_path),
                "error": "",
            }
        except Exception as exc:
            LOGGER.warning("world_model render failed tool=%s output=%s error=%s", tool_name, output_path, exc)
            return {
                "tool": tool_name,
                "ok": False,
                "output_path": str(output_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def build_world_model(self, *, story_prompt: str, output_dir: str | Path) -> Dict[str, Any]:
        blueprint = self.extract_world_blueprint(story_prompt)
        tools = self._tool_sequence()
        world_root = Path(output_dir) / "narrative_artifacts" / "world_model"
        world_root.mkdir(parents=True, exist_ok=True)

        def _render_records(records: List[Dict[str, Any]], subdir: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for rec in records:
                slug = _safe_name(rec.get("name", subdir))
                record_dir = world_root / subdir / slug
                record_dir.mkdir(parents=True, exist_ok=True)
                images: List[Dict[str, Any]] = []
                for tool_name in tools:
                    image_path = record_dir / f"{tool_name}.png"
                    images.append(
                        self._render_with_tool(
                            tool_name=tool_name,
                            prompt=str(rec.get("image_prompt", "")),
                            output_path=image_path,
                        )
                    )
                rows.append({**rec, "asset_id": slug, "images": images})
            return rows

        locations = _render_records(list(blueprint.get("locations", [])), "locations")
        items = _render_records(list(blueprint.get("items", [])), "items")
        bundle = {
            "summary": str(blueprint.get("summary", "")),
            "story_prompt": story_prompt,
            "image_tools": tools,
            "locations": locations,
            "items": items,
            "output_dir": str(world_root),
            "llm_model_name": self.model_config.model_name,
            "llm_enabled": self.model_config.llm_enabled,
        }
        bundle["context_text"] = self.build_context_text(bundle)
        return bundle

    @staticmethod
    def build_context_text(bundle: Dict[str, Any]) -> str:
        if not isinstance(bundle, dict):
            return ""
        sections: List[str] = []
        summary = str(bundle.get("summary", "") or "").strip()
        if summary:
            sections.append(f"World summary: {summary}")

        loc_rows = bundle.get("locations", [])
        if isinstance(loc_rows, list) and loc_rows:
            loc_text = "; ".join(
                f"{str(row.get('name', '')).strip()}: {str(row.get('description', '')).strip()}"
                for row in loc_rows
                if isinstance(row, dict)
            )
            if loc_text:
                sections.append(f"Locations: {loc_text}")

        item_rows = bundle.get("items", [])
        if isinstance(item_rows, list) and item_rows:
            item_text = "; ".join(
                f"{str(row.get('name', '')).strip()}: {str(row.get('description', '')).strip()}"
                for row in item_rows
                if isinstance(row, dict)
            )
            if item_text:
                sections.append(f"Important props/clues: {item_text}")

        return "\n".join(sections).strip()

    @classmethod
    def enrich_character_configs(
        cls,
        character_configs: List[CharacterConfig],
        world_model: Dict[str, Any],
    ) -> List[CharacterConfig]:
        context = cls.build_context_text(world_model)
        if not context:
            return character_configs
        enriched: List[CharacterConfig] = []
        for cfg in character_configs:
            env_text = str(cfg.current_environment or "").strip()
            new_env = f"{env_text}\n\nWorld spatial model:\n{context}".strip() if env_text else context
            dyn_prompt = str(cfg.dynamic_system_prompt or "").strip()
            if dyn_prompt:
                dyn_prompt = f"{dyn_prompt}\n\nKnown world model:\n{context}".strip()
            enriched.append(replace(cfg, current_environment=new_env, dynamic_system_prompt=dyn_prompt))
        return enriched

    @classmethod
    def enrich_story_prompt(cls, story_prompt: str, world_model: Dict[str, Any]) -> str:
        context = cls.build_context_text(world_model)
        if not context:
            return story_prompt
        return f"{story_prompt}\n\nWorld spatial model:\n{context}".strip()


@dataclass
class WorldState:
    step: int = 0
    summary: str = ""
    current_environment: str = ""
    story_anchor: str = ""
    public_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "summary": self.summary,
            "current_environment": self.current_environment,
            "story_anchor": self.story_anchor,
            "public_events": list(self.public_events),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        return cls(
            step=int(data.get("step", 0)),
            summary=str(data.get("summary", "")),
            current_environment=str(data.get("current_environment", "")),
            story_anchor=str(data.get("story_anchor", "")),
            public_events=[str(x) for x in data.get("public_events", []) if str(x).strip()],
        )


@dataclass
class ProposedAction:
    actor: str
    action_type: str
    target_character: Optional[str] = None
    reason: str = ""
    monologue: str = ""
    location: str = ""
    content: str = ""
    confidence: float = 0.5
    obstacles: List[str] = field(default_factory=list)
    strategy: str = ""


@dataclass
class ActionResult:
    accepted: bool
    reason: str
    world_delta: Dict[str, Any] = field(default_factory=dict)
    actor_state_delta: Dict[str, Any] = field(default_factory=dict)
    gap_analysis: str = ""
    person_progress_analysis: str = ""
    event_summary_analysis: str = ""
    story_phase: str = ""
    remaining_gap_text: str = ""


@dataclass
class BroadcastEvent:
    step: int
    actor: str
    content: str
    visibility: str = "public"


class CharacterAgent:
    """Standard template for all character agents."""

    def __init__(self, state: CharacterState, model_config: ModelConfig) -> None:
        self.state = state
        self.model_config = model_config
        self.portrait_path: str = ""

    def perceive(self, world: WorldState, visible_observation: Dict[str, Any] | None = None) -> Dict[str, Any]:
        observation = {
            "step": world.step,
            "world_summary": world.summary,
            "current_environment": world.current_environment,
            "story_anchor": world.story_anchor,
            "public_events": list(world.public_events),
            "self_state": self.state,
            "model_config": self.model_config,
        }
        if isinstance(visible_observation, dict):
            observation["visible_observation"] = dict(visible_observation)
        return observation

    def propose_action(self, observation: Dict[str, Any]) -> ProposedAction:
        relation_text = ", ".join(self.state.known_relations) if self.state.known_relations else "none"
        visible = observation.get("visible_observation")
        visible = visible if isinstance(visible, dict) else {}
        visible_environment = str(visible.get("visible_environment") or self.state.current_environment)
        changes = "; ".join(str(x) for x in visible.get("salient_changes", []) or []) or "none"
        risks = "; ".join(str(x) for x in visible.get("immediate_risks", []) or []) or "none"
        conditions = "; ".join(str(x) for x in visible.get("condition_details", []) or []) or "none"
        perceived_events = "; ".join(str(x) for x in visible.get("perceived_events", []) or []) or "none"
        private_memory = str(visible.get("private_memory_summary") or self.state.state_summary or "none")
        user_prompt = (
            f"Identity: {self.state.name}\n"
            f"Character profile: {self.state.role_profile}\n"
            f"Final goal: {self.state.final_goal}\n"
            f"Known relations: {relation_text}\n"
            f"Current visible environment: {visible_environment}\n"
            f"Salient visible changes: {changes}\n"
            f"Observable condition severity and effects: {conditions}\n"
            f"Immediate risks: {risks}\n"
            f"Perceived past events: {perceived_events}\n"
            f"Private memory: {private_memory}\n\n"
            "Using only what you perceive or remember, decide for yourself which present conditions obstruct your personal goal, how strongly, and why. "
            "Choose your own strategy and ONE concrete next action. Explain why it helps now and why any other obstacle can wait. "
            "Consider what earlier attempts actually achieved. Avoid repeating a failed preparation without new evidence. "
            "A residual risk need not be eliminated if you can reasonably perform a short, bounded action toward your goal; judge that yourself. "
            "You may inspect, prepare, mitigate, or advance the goal; do not assume future events or an external story plan.\n"
            "Return JSON only. Required fields: obstacles (list of concrete observations and their effects on your goal), "
            "strategy, action, target_character, reason, monologue, location, confidence."
        )
        fallback = {
            "action": "investigate nearby clue",
            "target_character": "",
            "reason": f"Move one step closer to the goal: {self.state.final_goal}.",
            "monologue": "I should verify the scene before making a risky move.",
            "location": visible_environment or "unknown",
            "confidence": 0.6,
            "obstacles": [],
            "strategy": "Inspect current conditions before committing to an irreversible step.",
        }
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.5,
            call_label="character_action",
        )
        action_text = str(payload.get("action", fallback["action"]))
        target_character = str(payload.get("target_character", fallback["target_character"])).strip() or None
        reason = str(payload.get("reason", fallback["reason"]))
        monologue = str(payload.get("monologue", fallback["monologue"]))
        location = str(payload.get("location", fallback["location"]))
        obstacles = [str(x).strip() for x in payload.get("obstacles", []) if str(x).strip()] if isinstance(payload.get("obstacles"), list) else []
        strategy = str(payload.get("strategy", fallback["strategy"]))
        content = (
            f"{self.state.name} chooses to '{action_text}' at {location}, "
            f"targeting {target_character or 'no specific person'}. "
            f"Obstacles: {'; '.join(obstacles)}. Strategy: {strategy}. "
            f"Reason: {reason}. Inner thought: {monologue}."
        )
        return ProposedAction(
            actor=self.state.name,
            action_type=action_text,
            target_character=target_character,
            reason=reason,
            monologue=monologue,
            location=location,
            content=content,
            confidence=_parse_confidence(payload.get("confidence", fallback["confidence"]), default=fallback["confidence"]),
            obstacles=obstacles,
            strategy=strategy,
        )

    def update_private_state(self, result: ActionResult) -> None:
        if not result.accepted:
            return
        new_env = result.actor_state_delta.get("current_environment")
        if isinstance(new_env, str) and new_env:
            self.state.current_environment = new_env
        new_summary = result.actor_state_delta.get("state_summary")
        if isinstance(new_summary, str) and new_summary:
            self.state.state_summary = new_summary
        new_clues = result.actor_state_delta.get("known_clues")
        if isinstance(new_clues, list):
            for clue in new_clues:
                clue_text = str(clue)
                if clue_text and clue_text not in self.state.known_clues:
                    self.state.known_clues.append(clue_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "model_config": {
                "model_type": self.model_config.model_type,
                "model_name": self.model_config.model_name,
                "base_url": self.model_config.base_url,
                "api_key": self.model_config.api_key,
                "system_prompt_path": self.model_config.system_prompt_path,
                "system_prompt": self.model_config.system_prompt,
                "timeout_sec": self.model_config.timeout_sec,
                "llm_enabled": self.model_config.llm_enabled,
            },
            "portrait_path": self.portrait_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterAgent":
        state = CharacterState.from_dict(data.get("state", {}))
        mc = data.get("model_config", {})
        model_config = ModelConfig(
            model_type=str(mc.get("model_type", "VLM")),
            model_name=str(mc.get("model_name", "")),
            base_url=str(mc.get("base_url", "")),
            api_key=str(mc.get("api_key", "")),
            system_prompt_path=str(mc.get("system_prompt_path", "")),
            system_prompt=str(mc.get("system_prompt", "")),
            timeout_sec=int(mc.get("timeout_sec", 480)),
            llm_enabled=bool(mc.get("llm_enabled", False)),
        )
        agent = cls(state=state, model_config=model_config)
        agent.portrait_path = str(data.get("portrait_path", ""))
        return agent


class StateAnalysisMachine:
    """Run narrative deduction, validate actions, and broadcast events."""

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.model_config = model_config or ModelConfig.from_env("NARRATIVE_SAM", "LLM")

    def _generate_character_portrait(
        self,
        *,
        character: CharacterAgent,
        output_dir: Path,
        storyline: str,
    ) -> str:
        from agents.generation_backends import (
            generate_image_nano_banana,
            nano_banana_settings_from_env,
        )

        out_path = output_dir / f"{_safe_name(character.state.name)}_portrait.png"
        out_path = _ensure_unique_path(out_path)
        LOGGER.info("narrative portrait status=start character=%s output=%s", character.state.name, out_path)
        prompt = (
            f"Character portrait of {character.state.name}. "
            f"Identity: {character.state.role_profile}. "
            f"Goal: {character.state.final_goal}. "
            f"Known relations: {', '.join(character.state.known_relations) or 'none'}. "
            f"Story context: {storyline}. "
            "Generate a clean cinematic portrait with strong character readability."
            f"\n{build_people_control_clause([character.state.name], allow_multi_people=False)}"
        )
        cfg = nano_banana_settings_from_env()
        if not cfg["api_key"]:
            raise RuntimeError("NANO_BANANA_API_KEY is required for narrative portrait generation")
        img = generate_image_nano_banana(
            prompt=prompt,
            output_path=out_path,
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model_name=cfg["model_name"],
            timeout_sec=cfg["timeout_sec"],
        )
        if not img:
            raise RuntimeError(f"Failed to generate portrait for {character.state.name}")
        LOGGER.info("narrative portrait status=done character=%s output=%s", character.state.name, img)
        return img

    def _generate_action_keyframe(
        self,
        *,
        character: CharacterAgent,
        action: ProposedAction,
        prev_keyframe_path: str | None,
        output_dir: Path,
        step: int,
        storyline: str,
    ) -> str:
        from agents.generation_backends import (
            generate_image_nano_banana,
            nano_banana_settings_from_env,
        )

        step_dir = output_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        out_path = _ensure_unique_path(step_dir / f"{_safe_name(character.state.name)}_keyframe.png")
        LOGGER.info(
            "narrative keyframe status=start step=%d character=%s output=%s prev=%s",
            step,
            character.state.name,
            out_path,
            prev_keyframe_path or "",
        )
        prompt = (
            f"Cinematic keyframe for character {character.state.name}. "
            f"Action: {action.action_type}. "
            f"Target: {action.target_character or 'none'}. "
            f"Reason: {action.reason}. "
            f"Monologue: {action.monologue}. "
            f"Location: {action.location}. "
            f"Story context: {storyline}. "
            "Render a cinematic action keyframe with clear pose, scene continuity, and story relevance. "
            "This is a new narrative moment. Keep identity continuity, but do not duplicate the previous keyframe's exact composition, pose, or framing."
            f"\n{build_people_control_clause([character.state.name], allow_multi_people=False)}"
        )
        refs: list[str] = [p for p in [prev_keyframe_path, character.portrait_path] if p]
        cfg = nano_banana_settings_from_env()
        if not cfg["api_key"]:
            raise RuntimeError("NANO_BANANA_API_KEY is required for narrative keyframe generation")
        img = generate_image_nano_banana(
            prompt=prompt,
            output_path=out_path,
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model_name=cfg["model_name"],
            timeout_sec=cfg["timeout_sec"],
            reference_image_paths=refs,
        )
        if not img:
            raise RuntimeError(f"Failed to generate keyframe for {character.state.name} at step {step}")
        LOGGER.info("narrative keyframe status=done step=%d character=%s output=%s", step, character.state.name, img)
        return img

    @staticmethod
    def _build_visual_graph(
        *,
        storyline: str,
        character_configs: list[CharacterConfig],
        step_logs: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            import networkx as nx  # type: ignore
        except Exception:
            nx = None  # type: ignore
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.switch_backend("Agg")
        except Exception:
            plt = None  # type: ignore

        if nx is None:
            nodes: list[dict[str, Any]] = [{"id": "story_start", "kind": "story", "label": "Story Start", "text": storyline}]
            edges: list[dict[str, Any]] = []
            for cfg in character_configs:
                role_node = {
                    "id": f"char::{cfg.name}",
                    "kind": "character",
                    "label": cfg.name,
                    "goal": cfg.final_goal,
                    "profile": cfg.role_profile,
                    "portrait_path": cfg.portrait_path,
                }
                nodes.append(role_node)
                edges.append({"source": "story_start", "target": role_node["id"], "relation": "introduce"})
            for step_item in step_logs:
                step_id = int(step_item["step"])
                step_node_id = f"step::{step_id}"
                nodes.append({"id": step_node_id, "kind": "step", "label": f"Step {step_id}"})
                edges.append({"source": "story_start", "target": step_node_id, "relation": "progress"})
                for role_log in step_item.get("role_logs", []):
                    role_name = str(role_log.get("role_name", ""))
                    role_node_id = f"char::{role_name}"
                    edges.append({"source": role_node_id, "target": step_node_id, "relation": "acts_in"})
                    kf = str(role_log.get("action_keyframe_path", "") or "")
                    if kf:
                        kf_node_id = f"kf::{step_id}::{role_name}"
                        nodes.append({"id": kf_node_id, "kind": "keyframe", "label": f"{role_name} keyframe", "image_path": kf})
                        edges.append({"source": step_node_id, "target": kf_node_id, "relation": "visualize"})
            return {"graph_png": "", "nodes": nodes, "edges": edges, "render_status": "networkx_unavailable"}

        graph = nx.DiGraph()
        start_id = "story_start"
        graph.add_node(start_id, kind="story", label="Story Start", text=storyline)

        for cfg in character_configs:
            node_id = f"char::{cfg.name}"
            graph.add_node(
                node_id,
                kind="character",
                label=cfg.name,
                goal=cfg.final_goal,
                profile=cfg.role_profile,
                portrait_path=cfg.portrait_path,
            )
            graph.add_edge(start_id, node_id, relation="introduce")

        for step_item in step_logs:
            step_id = int(step_item["step"])
            step_node = f"step::{step_id}"
            graph.add_node(step_node, kind="step", label=f"Step {step_id}")
            graph.add_edge(start_id, step_node, relation="progress")
            for role_log in step_item.get("role_logs", []):
                role_name = str(role_log.get("role_name", ""))
                role_node = f"char::{role_name}"
                if graph.has_node(role_node):
                    graph.add_edge(role_node, step_node, relation="acts_in")
                    kf = role_log.get("action_keyframe_path", "")
                    if kf:
                        kf_node = f"kf::{step_id}::{role_name}"
                        graph.add_node(
                            kf_node,
                            kind="keyframe",
                            label=f"{role_name} keyframe",
                            image_path=kf,
                        )
                        graph.add_edge(step_node, kf_node, relation="visualize")

        png_path = ""
        render_status = "ok"
        if plt is not None:
            graph_dir = output_dir / "narrative_graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            png_target = graph_dir / "narrative_network.png"
            plt.figure(figsize=(16, 10))
            pos = nx.spring_layout(graph, seed=42, k=0.9)
            node_colors = []
            for _, data in graph.nodes(data=True):
                kind = data.get("kind", "other")
                node_colors.append({
                    "story": "#ffcc66",
                    "character": "#7fc8f8",
                    "step": "#a8e6cf",
                    "keyframe": "#ff8b94",
                }.get(kind, "#dddddd"))
            nx.draw_networkx_nodes(graph, pos, node_size=900, node_color=node_colors)
            nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="->", arrowsize=14, width=1.2)
            labels = {n: data.get("label", n) for n, data in graph.nodes(data=True)}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(png_target, dpi=200, bbox_inches="tight")
            plt.close()
            png_path = str(png_target)
        else:
            render_status = "matplotlib_unavailable"

        return {
            "graph_png": png_path,
            "nodes": [
                {"id": str(n), **{k: _to_jsonable(v) for k, v in data.items()}}
                for n, data in graph.nodes(data=True)
            ],
            "edges": [
                {"source": str(u), "target": str(v), **{k: _to_jsonable(val) for k, val in data.items()}}
                for u, v, data in graph.edges(data=True)
            ],
            "render_status": render_status,
        }

    def create_initial_nodes(self, agents: List[CharacterAgent], world: WorldState) -> List[Dict[str, Any]]:
        return [
            {
                "actor": agent.state.name,
                "final_goal": agent.state.final_goal,
                "role_profile": agent.state.role_profile,
                "current_environment": agent.state.current_environment,
                "known_clues": list(agent.state.known_clues),
                "initial_step": world.step,
                "model_type": agent.model_config.model_type,
                "model_name": agent.model_config.model_name,
            }
            for agent in agents
        ]

    def evaluate(self, action: ProposedAction, world: WorldState, alignment: Dict[str, Any] | None = None) -> ActionResult:
        if not action.action_type:
            return ActionResult(accepted=False, reason="missing_action_type")
        if alignment is not None and not alignment.get("feasible", False):
            return ActionResult(accepted=False, reason=f"World feasibility check: {alignment.get('reason', 'unsupported action')}")
        user_prompt = self.build_event_understanding_prompt(world, action) + (
            (f"\nPrivate world action alignment: {alignment}. Only report this step's actual effects. "
             "If the action is accepted and physically performed, report its immediate observable outcome. "
             "Do not say a directly performed wipe, repositioning, or inspection has no confirmed effect unless a specific obstacle caused failure. "
             "A wipe can temporarily clear sweat without resolving heat; an inspection can confirm absence of nearby shade. "
             "Distinguish immediate effect, persistent hazard, and future uncertainty. "
             "Do not invent successful actions, mark a goal complete prematurely, or turn a historical event into a current action.\n") if alignment else "\n"
        ) + (
            "\nReturn JSON fields: accepted(boolean), reason, world_delta{public_events[]}, "
            "actor_state_delta{state_summary,current_environment,known_clues[]}, "
            "gap_analysis, person_progress_analysis, event_summary_analysis, story_phase, remaining_gap_text."
        )
        fallback = {
            "accepted": True,
            "reason": "accepted",
            "world_delta": {"public_events": [f"{action.actor}: {action.content}"]},
            "actor_state_delta": {
                "state_summary": f"{action.actor} advances the story at step {world.step}.",
                "current_environment": world.current_environment,
                "known_clues": [action.content],
            },
            "gap_analysis": "The story is still in progress and needs more events to fully satisfy the outline.",
            "person_progress_analysis": f"{action.actor} has advanced one step toward the current goal.",
            "event_summary_analysis": "This step introduces or advances the current event thread.",
            "story_phase": "development",
            "remaining_gap_text": "More conflict escalation and resolution are still needed.",
        }
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.2,
            call_label="action_evaluation",
        )
        world_delta = payload.get("world_delta", fallback["world_delta"])
        actor_state_delta = payload.get("actor_state_delta", fallback["actor_state_delta"])
        result = ActionResult(
            accepted=bool(payload.get("accepted", True)),
            reason=str(payload.get("reason", "accepted")),
            world_delta=world_delta if isinstance(world_delta, dict) else fallback["world_delta"],
            actor_state_delta=actor_state_delta if isinstance(actor_state_delta, dict) else fallback["actor_state_delta"],
            gap_analysis=str(payload.get("gap_analysis", fallback["gap_analysis"])),
            person_progress_analysis=str(payload.get("person_progress_analysis", fallback["person_progress_analysis"])),
            event_summary_analysis=str(payload.get("event_summary_analysis", fallback["event_summary_analysis"])),
            story_phase=str(payload.get("story_phase", fallback["story_phase"])),
            remaining_gap_text=str(payload.get("remaining_gap_text", fallback["remaining_gap_text"])),
        )
        if result.accepted and alignment is not None:
            chosen = action.action_type.lower()
            if re.search(r"\bwip(?:e|ing)\b", chosen) and any(word in chosen for word in ("sweat", "brow", "eyes")):
                # A feasible, accepted wipe is a physical act. Its local
                # effect is temporary; desert heat can cause new sweat later.
                events = result.world_delta.get("public_events", [])
                if not isinstance(events, list):
                    events = []
                events = [str(event) for event in events if not re.search(
                    r"no successful (?:wip|sweat removal)|sweat removal (?:is )?unconfirmed",
                    str(event), re.IGNORECASE,
                )]
                direct_effect = "The wipe temporarily clears sweat from the eye area; heat exposure continues and new sweat may form."
                events.append(direct_effect)
                result.world_delta["public_events"] = events
                result.world_delta["confirmed_direct_effect"] = direct_effect
                result.actor_state_delta["state_summary"] = (
                    f"{result.actor_state_delta.get('state_summary', '')} {direct_effect}"
                ).strip()
                result.reason = re.sub(
                    r"No successful sweat removal or improvement in vision is established\.?",
                    "The immediate wipe cleared sweat temporarily; longer-term vision and heat relief remain uncertain.",
                    result.reason,
                    flags=re.IGNORECASE,
                )
        return result

    def apply(self, result: ActionResult, world: WorldState) -> None:
        if not result.accepted:
            return
        events = result.world_delta.get("public_events", [])
        if isinstance(events, list):
            world.public_events.extend(str(x) for x in events)

    def broadcast(self, *, step: int, action: ProposedAction, result: ActionResult) -> Optional[BroadcastEvent]:
        if not result.accepted:
            return None
        return BroadcastEvent(
            step=step,
            actor=action.actor,
            content=action.content or action.action_type,
            visibility="public",
        )

    def anchor_story_progress(self, world: WorldState, action: ProposedAction, result: ActionResult, alignment: Dict[str, Any] | None = None) -> None:
        if not result.accepted:
            return
        user_prompt = (
            f"{self.model_config.system_prompt}\n\n"
            f"At step {world.step}, action accepted.\n"
            f"Actor: {action.actor}\n"
            f"Action: {action.action_type}\n"
            f"Target character: {action.target_character or 'none'}\n"
            f"Reason: {action.reason}\n"
            f"Monologue: {action.monologue}\n"
            f"Location: {action.location}\n"
            f"Content: {action.content}\n"
            f"Private action alignment and permitted effects: {alignment or {}}\n"
            f"Public events: {world.public_events}\n"
            "Only update current conditions and this accepted action's real effects. A past event may be remembered but may not happen again. "
            "Preserve confirmed immediate physical effects from the accepted event even when the broader hazard persists; do not change a completed wipe back to unconfirmed. "
            "Do not insert future story stages or retroactive achievements.\n"
            "Return JSON fields: story_anchor, summary, current_environment, person_progress_analysis, "
            "event_summary_analysis, story_phase, remaining_gap_text."
        )
        fallback = {
            "story_anchor": f"step {world.step}: {action.actor} -> {action.content}",
            "summary": f"step {world.step}: {action.actor} -> {action.content}",
            "current_environment": (
                f"environment updated after {action.actor} action; "
                f"events={len(world.public_events)}; anchor=step {world.step}"
            ),
            "person_progress_analysis": f"{action.actor} progressed via {action.action_type}.",
            "event_summary_analysis": "Event thread updated for the current step.",
            "story_phase": "development",
            "remaining_gap_text": "The story outline still needs more events to fully resolve.",
        }
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.2,
            call_label="story_anchor",
        )
        world.story_anchor = str(payload.get("story_anchor", fallback["story_anchor"]))
        world.summary = str(payload.get("summary", fallback["summary"]))
        world.current_environment = str(payload.get("current_environment", fallback["current_environment"]))
        result.person_progress_analysis = str(payload.get("person_progress_analysis", fallback["person_progress_analysis"]))
        result.event_summary_analysis = str(payload.get("event_summary_analysis", fallback["event_summary_analysis"]))
        result.story_phase = str(payload.get("story_phase", fallback["story_phase"]))
        result.remaining_gap_text = str(payload.get("remaining_gap_text", fallback["remaining_gap_text"]))

    def sync_environment(self, agents: List[CharacterAgent], world: WorldState) -> None:
        for agent in agents:
            agent.state.current_environment = world.current_environment

    def build_event_understanding_prompt(self, world: WorldState, action: ProposedAction) -> str:
        return (
            f"{self.model_config.system_prompt}\n\n"
            f"World summary: {world.summary}\n"
            f"Current environment: {world.current_environment}\n"
            f"Story anchor: {world.story_anchor}\n"
            f"Actor: {action.actor}\n"
            f"Action type: {action.action_type}\n"
            f"Action content: {action.content}\n"
        ).strip()

    @staticmethod
    def _heuristic_story_completion(step: int, summary: str, remaining_gap_text: str, story_phase: str) -> Dict[str, Any]:
        summary_l = (summary or "").lower()
        gap_l = (remaining_gap_text or "").lower()
        phase_l = (story_phase or "").lower()
        done_tokens = [
            "resolved",
            "resolution",
            "conclusion",
            "ending",
            "closed",
            "complete",
            "completed",
            "finale",
        ]
        unresolved_tokens = [
            "need more",
            "still need",
            "in progress",
            "unfinished",
            "not yet",
            "missing",
            "more conflict",
        ]
        done_hit = any(tok in summary_l for tok in done_tokens) or any(tok in phase_l for tok in done_tokens)
        unresolved_hit = any(tok in gap_l for tok in unresolved_tokens)
        is_complete = bool(done_hit and not unresolved_hit and step >= 2)
        score = 0.9 if is_complete else 0.45
        reason = "heuristic completion signal from story summary/phase." if is_complete else "story likely still in progress."
        return {
            "is_story_complete": is_complete,
            "completion_score": score,
            "completion_reason": reason,
            "remaining_gap_text": remaining_gap_text or "More narrative events are needed.",
            "next_focus": "Drive toward resolution with concrete causal events." if not is_complete else "",
        }

    def evaluate_storyline_completion(
        self,
        *,
        story_outline: str,
        world: WorldState,
        step_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        role_lines: List[str] = []
        role_logs = step_record.get("role_logs", []) or []
        for item in role_logs:
            role_name = str(item.get("role_name", "") or "").strip()
            action_type = str((item.get("next_action", {}) or {}).get("action_type", "") or "").strip()
            target = str((item.get("next_action", {}) or {}).get("target_character", "") or "").strip()
            role_output = str(item.get("role_output_text", "") or "").strip()
            role_lines.append(
                f"- role={role_name}; action={action_type}; target={target or 'none'}; output={role_output}"
            )
        step_text = "\n".join(role_lines) if role_lines else "none"
        fallback = self._heuristic_story_completion(
            step=world.step,
            summary=world.summary,
            remaining_gap_text=str(step_record.get("sam_remaining_gap_text", "") or ""),
            story_phase=str(step_record.get("sam_story_phase", "") or ""),
        )
        user_prompt = (
            f"{self.model_config.system_prompt}\n\n"
            "You are the state analysis machine that decides whether the narrative has already satisfied the full story outline.\n"
            f"Story outline: {story_outline}\n"
            f"Current step: {world.step}\n"
            f"Current world summary: {world.summary}\n"
            f"Current story anchor: {world.story_anchor}\n"
            f"Current environment: {world.current_environment}\n"
            f"Per-role events in this step:\n{step_text}\n\n"
            "Return JSON only with fields: "
            "is_story_complete(boolean), completion_score(float 0-1), completion_reason, remaining_gap_text, next_focus."
        )
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.1,
            call_label="storyline_completion",
        )
        score = _parse_confidence(payload.get("completion_score", fallback["completion_score"]), default=fallback["completion_score"])
        return {
            "is_story_complete": bool(payload.get("is_story_complete", fallback["is_story_complete"])),
            "completion_score": score,
            "completion_reason": str(payload.get("completion_reason", fallback["completion_reason"])),
            "remaining_gap_text": str(payload.get("remaining_gap_text", fallback["remaining_gap_text"])),
            "next_focus": str(payload.get("next_focus", fallback["next_focus"])),
        }

    def run_narrative_deduction(
        self,
        *,
        prompt: str,
        agents: List[CharacterAgent],
        max_steps: int = 20,
        output_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        resume_state: Dict[str, Any] | None = None,
        stop_on_complete: bool = True,
        story_control_enabled: bool = False,
    ) -> Dict[str, Any]:
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        if resume_state:
            world = WorldState.from_dict(resume_state.get("world_state", {}))
            saved_agents = resume_state.get("agents", [])
            if isinstance(saved_agents, list) and len(saved_agents) == len(agents):
                agents = [CharacterAgent.from_dict(a) for a in saved_agents]
            completed_steps = int(resume_state.get("completed_steps", world.step))
            if not world.summary.strip():
                world.summary = "The ongoing scene retains only its established, observable conditions."
            if not world.current_environment.strip():
                world.current_environment = "The immediate setting remains available for cautious observation."
        else:
            opening_environment = "The opening setting is present but only immediately observable conditions are established."
            # The full premise is private to StoryController.  Placing it in
            # the initial public world state lets every future event leak into
            # the first action and defeats causal story progression.
            world = WorldState(
                step=0,
                summary="The story is beginning; only the current observable situation is established.",
                current_environment=opening_environment,
                story_anchor="",
            )
            completed_steps = 0
        story_controller = StoryController(
            story_outline=prompt,
            model_config=self.model_config,
            saved_state=(resume_state or {}).get("story_control") if story_control_enabled else None,
        ) if story_control_enabled else None
        if story_controller and not resume_state:
            opening = story_controller.opening_public_state()
            world.current_environment = opening["environment"]
            for agent in agents:
                agent.state.current_environment = opening["environment"]
                agent.state.final_goal = opening["goal"]
                agent.state.role_profile = opening["role"]
                # Orchestrator used the full outline to build its system
                # prompt. Replace it before the actor's first model call.
                agent.model_config.system_prompt = (
                    f"You are {agent.state.name}. Your current personal goal is: {opening['goal']}. "
                    "Reason from your present observations and memory. Decide for yourself how to handle obstacles. "
                    "Do not assume knowledge of later story events or settings."
                )
        LOGGER.info(
            "narrative run status=start max_steps=%d characters=%d completed_steps=%d stop_on_complete=%s story_control=%s",
            max_steps,
            len(agents),
            completed_steps,
            stop_on_complete,
            bool(story_controller),
        )
        initial_nodes = self.create_initial_nodes(agents, world)
        output_root = Path(output_dir) if output_dir else None
        artifact_dir = output_root / "narrative_artifacts" if output_root else None
        portrait_dir = artifact_dir / "portraits" if artifact_dir else None
        keyframe_dir = artifact_dir / "keyframes" if artifact_dir else None
        if portrait_dir:
            portrait_dir.mkdir(parents=True, exist_ok=True)
        if keyframe_dir:
            keyframe_dir.mkdir(parents=True, exist_ok=True)

        for agent in agents:
            if portrait_dir:
                agent.portrait_path = self._generate_character_portrait(
                    character=agent,
                    output_dir=portrait_dir,
                    storyline=prompt,
                )
            else:
                agent.portrait_path = agent.portrait_path or ""
            LOGGER.info(
                "narrative portrait linked character=%s portrait=%s",
                agent.state.name,
                agent.portrait_path or "",
            )

        trace: List[Dict[str, Any]] = []
        broadcast_log: List[BroadcastEvent] = []
        step_logs: List[Dict[str, Any]] = []
        last_keyframe_by_actor: Dict[str, str] = {}
        keyframe_history_by_actor: Dict[str, List[str]] = {}

        def _save_checkpoint(step_value: int) -> None:
            if not checkpoint_file:
                return
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "prompt": prompt,
                "max_steps": max_steps,
                "completed_steps": step_value,
                "world_state": world.to_dict(),
                "agents": [a.to_dict() for a in agents],
                "character_portraits": {a.state.name: a.portrait_path for a in agents},
                "keyframe_history_by_actor": keyframe_history_by_actor,
                "trace": trace,
                "broadcast_log": [e.__dict__ for e in broadcast_log],
                "step_logs": step_logs,
                "initial_nodes": initial_nodes,
                "final_world_state": world.to_dict(),
                "narrative_graph": {},
                "story_control": story_controller.snapshot() if story_controller else {},
            }
            checkpoint_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("narrative checkpoint saved path=%s step=%d", checkpoint_file, step_value)

        start_step = max(completed_steps + 1, 1)
        stop_reason = "max_steps_reached"
        completed_step = completed_steps
        for step in range(start_step, max_steps + 1):
            world.step = step
            world_tick: Dict[str, Any] = {}
            if story_controller and step > 1:
                world_tick = story_controller.advance_world(world=world)
                if world_tick.get("changed"):
                    world.current_environment = str(world_tick["current_environment"])
                    world.public_events.append(str(world_tick["public_event"]))
                    world.summary = f"{world.summary}\nCurrent external change: {world_tick['public_event']}"
                    self.sync_environment(agents, world)
                    handoff = story_controller.apply_scene_transition(
                        world=world, agents=agents, world_tick=world_tick,
                    )
                    if handoff:
                        world_tick["scene_handoff"] = handoff
            LOGGER.info("narrative step status=start step=%d world_summary=%s", step, world.summary[:160])
            step_record: Dict[str, Any] = {
                "step": step,
                "role_logs": [],
                "world_tick": world_tick,
            }
            for agent in agents:
                LOGGER.info("narrative role status=start step=%d actor=%s", step, agent.state.name)
                agent_call_start = len(agent.model_config.call_history)
                sam_call_start = len(self.model_config.call_history)
                world_observation = story_controller.build_agent_observation(
                    agent_state=agent.state,
                    world=world,
                ) if story_controller else None
                public_observation = (
                    {key: value for key, value in world_observation.items() if key != "available_affordances"}
                    if world_observation else None
                )
                obs = agent.perceive(world, public_observation)
                action = agent.propose_action(obs)
                action_alignment = story_controller.match_action(
                    action=action, observation=world_observation or {}, world=world,
                ) if story_controller else None
                LOGGER.info(
                    "narrative role action step=%d actor=%s action=%s target=%s location=%s",
                    step,
                    agent.state.name,
                    action.action_type,
                    action.target_character or "",
                    action.location,
                )
                result = self.evaluate(action, world, action_alignment)
                LOGGER.info(
                    "narrative role eval step=%d actor=%s accepted=%s reason=%s",
                    step,
                    agent.state.name,
                    result.accepted,
                    result.reason,
                )
                self.apply(result, world)
                self.anchor_story_progress(world, action, result, action_alignment)
                direct_effect = result.world_delta.get("confirmed_direct_effect") if result.accepted else None
                if isinstance(direct_effect, str) and direct_effect:
                    world.summary = f"{world.summary}\nCurrent confirmed effect: {direct_effect}"
                    world.current_environment = f"{world.current_environment}\nCurrent confirmed effect: {direct_effect}"
                agent.update_private_state(result)
                event = self.broadcast(step=step, action=action, result=result)
                if event is not None:
                    broadcast_log.append(event)
                self.sync_environment(agents, world)

                model_calls = [
                    *agent.model_config.call_history[agent_call_start:],
                    *self.model_config.call_history[sam_call_start:],
                ]
                fallback_calls = [
                    item
                    for item in model_calls
                    if isinstance(item, dict) and bool(item.get("fallback_used"))
                ]

                action_keyframe_path = ""
                if keyframe_dir:
                    action_keyframe_path = self._generate_action_keyframe(
                        character=agent,
                        action=action,
                        prev_keyframe_path=last_keyframe_by_actor.get(agent.state.name),
                        output_dir=keyframe_dir,
                        step=step,
                        storyline=prompt,
                    )
                    last_keyframe_by_actor[agent.state.name] = action_keyframe_path
                    keyframe_history_by_actor.setdefault(agent.state.name, []).append(action_keyframe_path)
                    LOGGER.info(
                        "narrative keyframe history actor=%s count=%d latest=%s",
                        agent.state.name,
                        len(keyframe_history_by_actor.get(agent.state.name, [])),
                        action_keyframe_path,
                    )
                LOGGER.info(
                    "narrative role status=done step=%d actor=%s keyframe=%s",
                    step,
                    agent.state.name,
                    action_keyframe_path or "",
                )

                step_record["role_logs"].append(
                    {
                        "role_name": agent.state.name,
                        "environment_description": {
                            "world_summary": obs.get("world_summary", ""),
                            "current_environment": obs.get("current_environment", ""),
                            "story_anchor": obs.get("story_anchor", ""),
                            "public_events": obs.get("public_events", []),
                            "visible_observation": obs.get("visible_observation", {}),
                        },
                        "role_output_text": action.content,
                        "action_alignment": action_alignment or {},
                        "action_accepted": result.accepted,
                        "actual_step_events": result.world_delta.get("public_events", []) if result.accepted else [],
                        "action_keyframe_path": action_keyframe_path,
                        "portrait_path": agent.portrait_path,
                        "next_action": {
                            "action_type": action.action_type,
                            "target_character": action.target_character,
                            "reason": action.reason,
                            "monologue": action.monologue,
                            "location": action.location,
                            "confidence": action.confidence,
                            "obstacles": action.obstacles,
                            "strategy": action.strategy,
                        },
                        "sam_environment_update_text": world.current_environment,
                        "gap_analysis_text": result.gap_analysis,
                        "person_progress_analysis_text": result.person_progress_analysis,
                        "event_summary_analysis_text": result.event_summary_analysis,
                        "story_phase_text": result.story_phase,
                        "remaining_gap_text": result.remaining_gap_text,
                        "model_calls": model_calls,
                        "fallback_used": bool(fallback_calls),
                        "fallback_reasons": [
                            str(item.get("error", ""))
                            for item in fallback_calls
                            if str(item.get("error", "")).strip()
                        ],
                    }
                )

                trace.append(
                    {
                        "step": step,
                        "actor": action.actor,
                        "action_type": action.action_type,
                        "content": action.content,
                        "accepted": result.accepted,
                        "reason": result.reason,
                    }
                )
            step_record["sam_story_anchor"] = world.story_anchor
            step_record["sam_summary"] = world.summary
            step_record["sam_current_environment"] = world.current_environment
            if step_record.get("role_logs"):
                last_role_log = step_record["role_logs"][-1]
                step_record["sam_remaining_gap_text"] = str(last_role_log.get("remaining_gap_text", "") or "")
                step_record["sam_story_phase"] = str(last_role_log.get("story_phase_text", "") or "")
            else:
                step_record["sam_remaining_gap_text"] = ""
                step_record["sam_story_phase"] = ""
            completion_call_start = len(self.model_config.call_history)
            story_control_assessment: Dict[str, Any] = {}
            if story_controller:
                story_control_assessment = story_controller.assess(
                    step=step,
                    world_summary=world.summary,
                    current_environment=world.current_environment,
                    public_events=world.public_events,
                    step_actions=step_record.get("role_logs", []),
                )
                step_record["story_control"] = story_control_assessment
                # A continuous stage shares this scene's physical state. Once
                # its predecessor is evidenced, update only the local goal and
                # carried memory; an independent stage waits for a world tick
                # that establishes an actual transition.
                next_index = story_controller.active_scene_index + 1
                if next_index < len(story_controller.scene_plan):
                    next_scene = story_controller.scene_plan[next_index]
                    if (next_scene["relationship_to_previous"] == "continuous"
                            and story_controller.scene_plan[next_index - 1]["beat_id"]
                            in story_controller.covered_beat_ids):
                        handoff = story_controller.apply_scene_transition(
                            world=world, agents=agents,
                            world_tick={"scene_transition": True, "target_beat_id": next_scene["beat_id"]},
                        )
                        if handoff:
                            step_record["scene_handoff"] = handoff
            completion = self.evaluate_storyline_completion(
                story_outline=prompt,
                world=world,
                step_record=step_record,
            )
            if story_controller:
                completion = story_controller.gate_completion(completion)
                step_record["story_control"]["completion_gate_open"] = bool(completion.get("is_story_complete", False))
            step_record["storyline_completion"] = completion
            step_model_calls = self.model_config.call_history[completion_call_start:]
            step_record["model_calls"] = step_model_calls
            step_record["fallback_used"] = bool(
                any(
                    isinstance(item, dict) and bool(item.get("fallback_used"))
                    for item in step_model_calls
                )
                or any(
                    isinstance(role, dict) and bool(role.get("fallback_used"))
                    for role in step_record.get("role_logs", [])
                )
            )
            step_logs.append(step_record)
            completed_step = step
            LOGGER.info(
                "narrative step status=done step=%d story_anchor=%s complete=%s score=%.2f",
                step,
                world.story_anchor[:160],
                completion.get("is_story_complete", False),
                float(completion.get("completion_score", 0.0)),
            )
            _save_checkpoint(step)
            if story_controller and story_controller.terminal_ready:
                # The final required beat is the hard stop. No subsequent
                # character action can open a new painting or story thread.
                stop_reason = "story_control_terminal_beat"
                LOGGER.info("narrative run status=early_stop step=%d reason=%s", step, stop_reason)
                break
            if completion.get("is_story_complete", False) and stop_on_complete:
                stop_reason = "storyline_completed"
                LOGGER.info("narrative run status=early_stop step=%d reason=%s", step, stop_reason)
                break

        graph = self._build_visual_graph(
            storyline=prompt,
            character_configs=[
                CharacterConfig(
                    name=a.state.name,
                    final_goal=a.state.final_goal,
                    role_profile=a.state.role_profile,
                    known_relations=a.state.known_relations,
                    current_environment=a.state.current_environment,
                    model_config=a.model_config,
                    portrait_path=a.portrait_path,
                )
                for a in agents
            ],
            step_logs=step_logs,
            output_dir=artifact_dir if artifact_dir else Path("."),
        )
        LOGGER.info(
            "narrative graph status=done render_status=%s png=%s nodes=%d edges=%d",
            graph.get("render_status", ""),
            graph.get("graph_png", ""),
            len(graph.get("nodes", [])),
            len(graph.get("edges", [])),
        )
        LOGGER.info("narrative run status=done steps=%d stop_reason=%s", len(step_logs), stop_reason)
        _save_checkpoint(completed_step)

        return {
            "prompt": prompt,
            "characters": [a.state.name for a in agents],
            "character_portraits": {a.state.name: a.portrait_path for a in agents},
            "keyframe_history_by_actor": keyframe_history_by_actor,
            "initial_nodes": initial_nodes,
            "final_world_state": world,
            "completed_steps": completed_step,
            "max_steps": max_steps,
            "stop_reason": stop_reason,
            "trace": trace,
            "broadcast_log": [e.__dict__ for e in broadcast_log],
            "step_logs": step_logs,
            "narrative_graph": graph if artifact_dir else {},
            "story_control": story_controller.snapshot() if story_controller else {},
        }


class OrchestratorAgent:
    """Only understand user input and create character agents."""

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.model_config = model_config or ModelConfig.from_env("NARRATIVE_ORCHESTRATOR", "LLM")
        self._arm_stats: Dict[str, Dict[str, float]] = {}
        self._prompt_arms = self._build_prompt_arms()
        for arm_name in self._prompt_arms:
            self._arm_stats[arm_name] = {"count": 0.0, "total_reward": 0.0}

    @staticmethod
    def _build_prompt_arms() -> Dict[str, str]:
        return {
            "goal_first": (
                "Prioritize goal completion. Propose actions with clear objective progress and measurable outcomes."
            ),
            "conflict_driver": (
                "Prioritize dramatic tension. Propose actions that escalate conflict while keeping causal coherence."
            ),
            "clue_hunter": (
                "Prioritize evidence and information asymmetry. Propose actions to discover, verify, or hide clues."
            ),
            "alliance_builder": (
                "Prioritize social coordination. Propose actions that negotiate, persuade, or align with other roles."
            ),
            "stealth_deception": (
                "Prioritize concealment and misdirection. Propose subtle actions that protect intent and identity."
            ),
        }

    @staticmethod
    def _text_score(text: str, keywords: List[str]) -> float:
        lower = (text or "").lower()
        return float(sum(1 for k in keywords if k in lower))

    def _estimate_arm_reward(self, arm_name: str, *, story: str, role_profile: str, final_goal: str, name: str) -> float:
        source = f"{story} {role_profile} {final_goal} {name}"
        if arm_name == "goal_first":
            return 1.0 + self._text_score(source, ["goal", "finish", "complete", "resolve", "target"])
        if arm_name == "conflict_driver":
            return 1.0 + self._text_score(source, ["conflict", "threat", "fight", "chase", "risk", "suspect"])
        if arm_name == "clue_hunter":
            return 1.0 + self._text_score(source, ["clue", "evidence", "mystery", "trace", "investigate", "witness"])
        if arm_name == "alliance_builder":
            return 1.0 + self._text_score(source, ["team", "ally", "help", "coordinate", "support", "guard"])
        if arm_name == "stealth_deception":
            return 1.0 + self._text_score(source, ["hide", "stealth", "deceive", "secret", "unknown", "visitor"])
        return 1.0

    def _select_prompt_arm(self, *, story: str, role_profile: str, final_goal: str, name: str) -> str:
        # UCB-like arm selection with heuristic bootstrap reward.
        import math

        total_count = sum(v["count"] for v in self._arm_stats.values()) + 1.0
        best_arm = "goal_first"
        best_score = float("-inf")

        for arm_name, stats in self._arm_stats.items():
            count = stats["count"]
            avg_reward = stats["total_reward"] / count if count > 0 else 0.0
            explore_bonus = math.sqrt((2.0 * math.log(total_count + 1.0)) / (count + 1.0))
            prior_reward = self._estimate_arm_reward(
                arm_name,
                story=story,
                role_profile=role_profile,
                final_goal=final_goal,
                name=name,
            )
            score = avg_reward + explore_bonus + 0.25 * prior_reward
            if score > best_score:
                best_score = score
                best_arm = arm_name

        # Bootstrap update using heuristic reward as pseudo-feedback.
        pseudo_reward = self._estimate_arm_reward(
            best_arm,
            story=story,
            role_profile=role_profile,
            final_goal=final_goal,
            name=name,
        )
        self._arm_stats[best_arm]["count"] += 1.0
        self._arm_stats[best_arm]["total_reward"] += pseudo_reward
        return best_arm

    def _build_dynamic_character_system_prompt(
        self,
        *,
        story: str,
        name: str,
        role_profile: str,
        final_goal: str,
        known_relations: List[str],
        arm_name: str,
    ) -> str:
        arm_instruction = self._prompt_arms.get(arm_name, self._prompt_arms["goal_first"])
        relation_text = ", ".join(known_relations) if known_relations else "no known relations"
        base = (
            "You are a character agent.\n"
            f"Your identity is {name}.\n"
            f"You are described as: {role_profile}.\n"
            f"Your final goal is: {final_goal}.\n"
            f"Your known relations are: {relation_text}.\n"
            "You must reason only from current environment, known relations, and known events.\n"
            "You must output JSON with fields: action, target_character (optional), reason, monologue, location, confidence.\n"
        )
        dynamic = (
            f"Story context: {story}\n"
            f"Strategy arm: {arm_name}\n"
            f"Strategy instruction: {arm_instruction}\n"
        )
        return f"{base}\n{dynamic}".strip()

    def build_character_agents(self, character_configs: List[CharacterConfig]) -> List[CharacterAgent]:
        agents: List[CharacterAgent] = []
        for cfg in character_configs:
            agent = CharacterAgent(
                CharacterState(
                    name=cfg.name,
                    final_goal=cfg.final_goal,
                    role_profile=cfg.role_profile,
                    current_environment=cfg.current_environment,
                ),
                model_config=cfg.model_config,
            )
            agent.portrait_path = cfg.portrait_path
            agents.append(agent)
        return agents

    def understand_input(self, prompt: str, character_names: Optional[List[str]] = None) -> Dict[str, Any]:
        names = [name.strip() for name in (character_names or []) if name and name.strip()]
        if not names:
            names = ["Protagonist", "Support", "Opponent"]
        configs = [
            self._build_character_config(name=name, prompt=prompt)
            for name in names
        ]
        return {
            "prompt": prompt,
            "character_configs": configs,
            "orchestrator_model": self.model_config,
            "max_steps": 20,
            "mab_arm_stats": self._arm_stats,
        }

    def _build_character_config(self, *, name: str, prompt: str) -> CharacterConfig:
        role_profile, final_goal, known_relations = self._derive_character_identity_and_goal(name=name, prompt=prompt)
        arm_name = self._select_prompt_arm(
            story=prompt,
            role_profile=role_profile,
            final_goal=final_goal,
            name=name,
        )
        base_model_cfg = ModelConfig.from_env("NARRATIVE_CHARACTER_AGENT", "VLM")
        dynamic_prompt = self._build_dynamic_character_system_prompt(
            story=prompt,
            name=name,
            role_profile=role_profile,
            final_goal=final_goal,
            known_relations=known_relations,
            arm_name=arm_name,
        )
        cfg = replace(
            base_model_cfg,
            system_prompt=dynamic_prompt,
        )
        return CharacterConfig(
            name=name,
            final_goal=final_goal,
            role_profile=role_profile,
            known_relations=known_relations,
            current_environment=prompt,
            model_config=cfg,
            prompt_arm=arm_name,
            dynamic_system_prompt=dynamic_prompt,
        )

    def _derive_character_identity_and_goal(self, *, name: str, prompt: str) -> tuple[str, str, List[str]]:
        fallback_role = f"{name} is a narrative participant in the story."
        fallback_goal = f"Advance the story toward the user-defined ending for {name}."
        fallback_relations: List[str] = []
        user_prompt = (
            f"Story: {prompt}\n"
            f"Character name: {name}\n"
            "Infer a concise role profile, a concrete final goal, and known relations for this character.\n"
            "Return JSON fields: role_profile, final_goal, known_relations."
        )
        payload = self.model_config.chat_json(
            user_prompt=user_prompt,
            fallback={"role_profile": fallback_role, "final_goal": fallback_goal, "known_relations": fallback_relations},
            temperature=0.2,
            call_label="character_identity",
        )
        role_profile = str(payload.get("role_profile", fallback_role)).strip() or fallback_role
        final_goal = str(payload.get("final_goal", fallback_goal)).strip() or fallback_goal
        known_relations = payload.get("known_relations", fallback_relations)
        if not isinstance(known_relations, list):
            known_relations = fallback_relations
        known_relations = [str(item).strip() for item in known_relations if str(item).strip()]
        return role_profile, final_goal, known_relations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    orchestrator = OrchestratorAgent()
    sam = StateAnalysisMachine()
    plan = orchestrator.understand_input(
        prompt="A mystery starts in a rainy city.",
        character_names=["Detective", "Witness", "Suspect"],
    )
    agents = orchestrator.build_character_agents(plan["character_configs"])
    output = sam.run_narrative_deduction(
        prompt=plan["prompt"],
        agents=agents,
        max_steps=2,
    )
    print(
        f"Simulation done. steps={len(output['trace'])}, "
        f"broadcasts={len(output['broadcast_log'])}"
    )
