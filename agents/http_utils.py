from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import requests

from .call_logger import new_call_id, text_summary, write_call_event


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def verify_ssl() -> bool | str:
    raw = os.getenv("CAMERAMASTER_SSL_VERIFY", "").strip()
    if not raw:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    return raw


def requests_session() -> requests.Session:
    session = requests.Session()
    session.verify = verify_ssl()
    return session


def requests_get(url: str, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("verify", verify_ssl())
    call_id = new_call_id("http-get")
    start = time.time()
    write_call_event(
        {
            "call_id": call_id,
            "kind": "http",
            "method": "GET",
            "status": "start",
            "url": url,
            "timeout": kwargs.get("timeout"),
        }
    )
    try:
        resp = requests.get(url, **kwargs)
        is_error = resp.status_code >= 400
        write_call_event(
            {
                "call_id": call_id,
                "kind": "http",
                "method": "GET",
                "status": "failed" if is_error else "done",
                "url": url,
                "status_code": resp.status_code,
                "elapsed_sec": round(time.time() - start, 3),
                "content_type": resp.headers.get("content-type", ""),
                "bytes": len(resp.content or b""),
                "body_tail": text_summary(resp.text[-4000:], 4000) if is_error else "",
            }
        )
        return resp
    except Exception as exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "http",
                "method": "GET",
                "status": "failed",
                "url": url,
                "elapsed_sec": round(time.time() - start, 3),
                "error": text_summary(exc, 1000),
            }
        )
        raise


def requests_post(url: str, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("verify", verify_ssl())
    call_id = new_call_id("http-post")
    start = time.time()
    data = kwargs.get("data")
    json_payload = kwargs.get("json")
    files = kwargs.get("files")
    write_call_event(
        {
            "call_id": call_id,
            "kind": "http",
            "method": "POST",
            "status": "start",
            "url": url,
            "timeout": kwargs.get("timeout"),
            "has_json": json_payload is not None,
            "has_data": data is not None,
            "file_count": len(files or []) if isinstance(files, list) else (1 if files else 0),
            "payload_keys": sorted(list(json_payload.keys())) if isinstance(json_payload, dict) else [],
            "data_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
        }
    )
    try:
        resp = requests.post(url, **kwargs)
        is_error = resp.status_code >= 400
        write_call_event(
            {
                "call_id": call_id,
                "kind": "http",
                "method": "POST",
                "status": "failed" if is_error else "done",
                "url": url,
                "status_code": resp.status_code,
                "elapsed_sec": round(time.time() - start, 3),
                "content_type": resp.headers.get("content-type", ""),
                "bytes": len(resp.content or b""),
                "body_tail": text_summary(resp.text[-4000:], 4000) if is_error else "",
            }
        )
        return resp
    except Exception as exc:
        write_call_event(
            {
                "call_id": call_id,
                "kind": "http",
                "method": "POST",
                "status": "failed",
                "url": url,
                "elapsed_sec": round(time.time() - start, 3),
                "error": text_summary(exc, 1000),
            }
        )
        raise


def openai_client_kwargs() -> dict[str, Any]:
    """
    Build shared kwargs for OpenAI client with SSL verification policy.
    """
    cfg: dict[str, Any] = {}
    try:
        import httpx  # type: ignore

        def log_api_error(response: Any) -> None:
            try:
                request = response.request
                url = str(request.url)
                hostname = (urlparse(url).hostname or "").lower()
                if response.status_code < 400 or not (hostname == "xi-ai.cn" or hostname.endswith(".xi-ai.cn")):
                    return
                response.read()
                request_id = ""
                for header in ("x-request-id", "request-id", "x-transaction-id"):
                    request_id = response.headers.get(header) or request.headers.get(header) or ""
                    if request_id:
                        break
                write_call_event(
                    {
                        "call_id": new_call_id("xi-api-httpx"),
                        "kind": "httpx",
                        "provider": "xi-api",
                        "method": request.method,
                        "status": "http_error",
                        "status_code": response.status_code,
                        "url": url,
                        "request_id": request_id,
                        "content_type": response.headers.get("content-type", ""),
                        "response_body": text_summary(response.text, 6000),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                write_call_event(
                    {
                        "call_id": new_call_id("xi-api-httpx-log"),
                        "kind": "httpx",
                        "provider": "xi-api",
                        "status": "logging_failed",
                        "error": text_summary(exc, 1000),
                    }
                )

        cfg["http_client"] = httpx.Client(
            verify=verify_ssl(),
            event_hooks={"response": [log_api_error]},
        )
    except Exception:
        pass
    return cfg
