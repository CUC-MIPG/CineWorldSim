from unittest.mock import patch

from openai.types.chat import ChatCompletion

from utils.chat_model_compat import (
    UsageCompatibleChatOpenAI,
    init_compatible_chat_model,
)


def test_factory_uses_null_safe_client_for_openai_provider():
    model = init_compatible_chat_model(
        model="gpt-5.6-sol",
        model_provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
    )

    assert isinstance(model, UsageCompatibleChatOpenAI)


def test_factory_delegates_other_providers_to_langchain():
    sentinel = object()
    with patch("utils.chat_model_compat.init_chat_model", return_value=sentinel) as factory:
        result = init_compatible_chat_model(
            model="MiniMax-M3",
            model_provider="minimax",
        )

    assert result is sentinel
    factory.assert_called_once_with(model="MiniMax-M3", model_provider="minimax")


def test_null_tier_token_details_do_not_break_successful_chat_response():
    model = UsageCompatibleChatOpenAI(
        model="gpt-5.6-sol",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-sol",
        "service_tier": "priority",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Story."},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": None},
        },
    }

    result = model._create_chat_result(response)

    message = result.generations[0].message
    assert message.content == "Story."
    assert message.usage_metadata["input_tokens"] == 12
    assert message.usage_metadata["output_tokens"] == 3
    assert message.usage_metadata["input_token_details"]["priority_cache_read"] == 0
    assert message.usage_metadata["input_token_details"]["priority"] == 12
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] is None


def test_null_details_object_is_treated_as_zero_for_service_tier_accounting():
    model = UsageCompatibleChatOpenAI(
        model="gpt-5.6-sol",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-sol",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Story."},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
    }

    result = model._create_chat_result(response)

    message = result.generations[0].message
    assert message.usage_metadata["input_token_details"]["flex_cache_read"] == 0
    assert message.usage_metadata["output_token_details"]["flex_reasoning"] == 0


def test_sdk_response_keeps_refusal_metadata_when_usage_is_normalized():
    model = UsageCompatibleChatOpenAI(
        model="gpt-5.6-sol",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    response = ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-sol",
        "service_tier": "priority",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I cannot help with that request.",
                "refusal": "I cannot help with that request.",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": None},
        },
    })

    result = model._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs["refusal"] == "I cannot help with that request."
