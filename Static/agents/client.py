from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI


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
    load_dotenv(override=False)
    api_key, base_url, _ = resolve_llm_env(scope)
    if not api_key:
        raise RuntimeError(
            f"{scope.upper()}_API_KEY (or OPENAI_API_KEY fallback) is required"
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def pick_model(default: str = "gpt-5.5", scope: str = "OPENAI") -> str:
    load_dotenv(override=False)
    _, _, model_name = resolve_llm_env(scope)
    return model_name or default
