from __future__ import annotations

from types import SimpleNamespace

from agents.llm_tool import LLMJsonTool


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class _FakeClient:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_llm_json_tool_returns_json_object_and_requests_json_format():
    client = _FakeClient('{"Character1": "We should talk.", "Character2": "none"}')
    tool = LLMJsonTool(client=client, model="test-model", retries=1)

    payload = tool.run_json(
        system_prompt="Return dialogue JSON.",
        user_prompt="Generate dialogue.",
    )

    assert payload == {"Character1": "We should talk.", "Character2": "none"}
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"] == {"type": "json_object"}


def test_llm_json_tool_strips_markdown_json_fences():
    client = _FakeClient('```json\n{"ok": true}\n```')
    tool = LLMJsonTool(client=client, model="test-model", retries=1)

    assert tool.run_json(system_prompt="x", user_prompt="y") == {"ok": True}
