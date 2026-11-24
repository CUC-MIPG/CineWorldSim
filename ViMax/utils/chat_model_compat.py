from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI


class UsageCompatibleChatOpenAI(ChatOpenAI):
    """Normalize omitted token-detail counts before LangChain usage accounting."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ):
        response_data = response if isinstance(response, dict) else response.model_dump()
        service_tier = response_data.get("service_tier")
        usage = response_data.get("usage")

        if service_tier in {"priority", "flex"} and isinstance(usage, dict):
            usage_data = dict(usage)
            for details_key, count_key in (
                ("prompt_tokens_details", "cached_tokens"),
                ("completion_tokens_details", "reasoning_tokens"),
            ):
                details = usage_data.get(details_key)
                details_data = dict(details) if isinstance(details, dict) else {}
                if details_data.get(count_key) is None:
                    details_data[count_key] = 0
                usage_data[details_key] = details_data

            if isinstance(response, dict):
                response = {**response_data, "usage": usage_data}
            else:
                usage_model = type(response.usage).model_validate(usage_data)
                response = response.model_copy(update={"usage": usage_model})

        return super()._create_chat_result(response, generation_info)


def init_compatible_chat_model(**kwargs: Any):
    """Use the null-safe OpenAI client for OpenAI-compatible chat providers."""
    if kwargs.get("model_provider") != "openai":
        return init_chat_model(**kwargs)

    openai_kwargs = dict(kwargs)
    openai_kwargs.pop("model_provider", None)
    return UsageCompatibleChatOpenAI(**openai_kwargs)
