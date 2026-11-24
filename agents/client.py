from __future__ import annotations

import os
import time
from typing import Any
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised in minimal tool-only envs
    OpenAI = None  # type: ignore[assignment]

from .http_utils import openai_client_kwargs
from .call_logger import new_call_id, text_summary, write_call_event


def _first_non_empty(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def resolve_llm_env(scope: str) -> tuple[str | None, str | None, str | None]:
    load_dotenv(override=False)
    p = scope.upper()
    api_key = _first_non_empty(
        os.getenv(f"{p}_API_KEY"),
        os.getenv(f"{p}_OPENAI_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    )
    base_url = _first_non_empty(
        os.getenv(f"{p}_BASE_URL"),
        os.getenv(f"{p}_OPENAI_BASE_URL"),
        os.getenv(f"{p}_OPENAI_API_BASE"),
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("OPENAI_API_BASE"),
    )
    model_name = _first_non_empty(
        os.getenv(f"{p}_MODEL_NAME"),
        os.getenv(f"{p}_MODEL"),
        os.getenv(f"{p}_OPENAI_MODEL"),
        os.getenv("OPENAI_MODEL"),
    )
    return api_key, base_url, model_name


def make_client(scope: str = "OPENAI") -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is required for LLM-backed agents. Install it or run inside the configured agent env.")
    load_dotenv(override=False)
    api_key, base_url, _ = resolve_llm_env(scope)
    if not api_key:
        raise RuntimeError(
            f"{scope.upper()}_API_KEY (or OPENAI_API_KEY fallback) is required"
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs.update(openai_client_kwargs())
    return _LoggedOpenAIClient(OpenAI(**kwargs), scope=scope, base_url=base_url or "")


def pick_model(default: str = "gpt-5.5", scope: str = "OPENAI") -> str:
    load_dotenv(override=False)
    _, _, model_name = resolve_llm_env(scope)
    return model_name or default


class _LoggedChatCompletions:
    def __init__(self, inner: Any, *, scope: str, base_url: str) -> None:
        self._inner = inner
        self._scope = scope
        self._base_url = base_url

    def create(self, *args: Any, **kwargs: Any) -> Any:
        call_id = new_call_id("llm")
        start = time.time()
        messages = kwargs.get("messages") or []
        write_call_event(
            {
                "call_id": call_id,
                "kind": "llm",
                "status": "start",
                "scope": self._scope,
                "base_url": self._base_url,
                "model": kwargs.get("model", ""),
                "timeout": kwargs.get("timeout"),
                "temperature": kwargs.get("temperature"),
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "response_format": kwargs.get("response_format"),
                "prompt_preview": text_summary(messages[-1].get("content", "") if messages and isinstance(messages[-1], dict) else ""),
            }
        )
        try:
            resp = self._inner.create(*args, **kwargs)
            usage = getattr(resp, "usage", None)
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "llm",
                    "status": "done",
                    "scope": self._scope,
                    "base_url": self._base_url,
                    "model": kwargs.get("model", ""),
                    "elapsed_sec": round(time.time() - start, 3),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            )
            return resp
        except Exception as exc:
            write_call_event(
                {
                    "call_id": call_id,
                    "kind": "llm",
                    "status": "failed",
                    "scope": self._scope,
                    "base_url": self._base_url,
                    "model": kwargs.get("model", ""),
                    "elapsed_sec": round(time.time() - start, 3),
                    "error": text_summary(exc, 1200),
                }
            )
            raise


class _LoggedChat:
    def __init__(self, inner: Any, *, scope: str, base_url: str) -> None:
        self.completions = _LoggedChatCompletions(inner.completions, scope=scope, base_url=base_url)


class _LoggedOpenAIClient:
    def __init__(self, inner: Any, *, scope: str, base_url: str) -> None:
        self._inner = inner
        self.chat = _LoggedChat(inner.chat, scope=scope, base_url=base_url)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
