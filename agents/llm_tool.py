from __future__ import annotations

import json
import re
from typing import Any, Callable

from .client import make_client, pick_model
from .retry import call_with_retries


def extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON root must be an object")
    return parsed


class LLMJsonTool:
    """Small reusable wrapper for LLM calls that must return a JSON object."""

    def __init__(
        self,
        *,
        scope: str = "OPENAI",
        model: str | None = None,
        retries: int = 3,
        client: Any | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.scope = scope
        self.model = model or pick_model("gpt-5.5", scope=scope)
        self.retries = max(1, int(retries or 1))
        self.client = client
        self.client_factory = client_factory or make_client

    def _client(self) -> Any:
        if self.client is None:
            self.client = self.client_factory(self.scope)
        return self.client

    def run_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if timeout is not None:
                kwargs["timeout"] = timeout
            response = self._client().chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("LLM response content must be a string")
            return extract_json_object(content)

        return call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
