from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_LOCK = threading.Lock()
_CALL_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("cameramaster_call_context", default={})


def _log_path() -> Path | None:
    raw = os.getenv("CAMERAMASTER_CALL_LOG_JSONL", "").strip()
    if not raw:
        return None
    return Path(raw)


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def write_call_event(event: dict[str, Any]) -> None:
    path = _log_path()
    if path is None:
        return
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "time": time.time(),
        **_CALL_CONTEXT.get(),
        **{k: _safe_json(v) for k, v in event.items()},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # Logging must never break generation.
        return


@contextmanager
def call_context(**fields: Any):
    """Attach non-secret task identifiers to all calls made in this scope."""
    current = dict(_CALL_CONTEXT.get())
    current.update(fields)
    token = _CALL_CONTEXT.set(current)
    try:
        yield
    finally:
        _CALL_CONTEXT.reset(token)


def new_call_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def text_summary(text: Any, limit: int = 600) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"
