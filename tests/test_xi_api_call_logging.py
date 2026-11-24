from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from agents.call_logger import call_context
from agents.http_utils import openai_client_kwargs, requests_post


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_openai_httpx_error_hook_records_xi_api_error_and_context(tmp_path, monkeypatch):
    log_path = tmp_path / "xi_calls.jsonl"
    monkeypatch.setenv("CAMERAMASTER_CALL_LOG_JSONL", str(log_path))
    client = openai_client_kwargs()["http_client"]
    request = httpx.Request(
        "POST",
        "https://api.xi-ai.cn/v1/chat/completions",
        headers={"Authorization": "Bearer must-not-be-logged", "x-request-id": "req-123"},
    )
    response = httpx.Response(
        500,
        request=request,
        headers={"content-type": "application/json", "x-request-id": "req-123"},
        json={"error": {"message": "upstream generation failed", "code": "upstream_error"}},
    )

    with call_context(story_id="01", shot_index=7, model="nano-banana-2"):
        for hook in client.event_hooks["response"]:
            hook(response)

    events = _read_events(log_path)
    assert len(events) == 1
    event = events[0]
    assert event["status"] == "http_error"
    assert event["status_code"] == 500
    assert event["url"] == "https://api.xi-ai.cn/v1/chat/completions"
    assert event["story_id"] == "01"
    assert event["shot_index"] == 7
    assert "upstream generation failed" in event["response_body"]
    assert "must-not-be-logged" not in json.dumps(event)


def test_requests_post_records_http_error_body_as_failed(tmp_path, monkeypatch):
    log_path = tmp_path / "xi_calls.jsonl"
    monkeypatch.setenv("CAMERAMASTER_CALL_LOG_JSONL", str(log_path))
    monkeypatch.setattr(
        "agents.http_utils.requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=503,
            headers={"content-type": "application/json"},
            content=b'{"error":"channel unavailable"}',
            text='{"error":"channel unavailable"}',
        ),
    )

    response = requests_post("https://api.xi-ai.cn/v1/chat/completions", json={"model": "nano-banana-2"})

    assert response.status_code == 503
    events = _read_events(log_path)
    error_event = events[-1]
    assert error_event["status"] == "failed"
    assert error_event["status_code"] == 503
    assert "channel unavailable" in error_event["body_tail"]
