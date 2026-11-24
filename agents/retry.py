from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _is_ssl_retryable_error(err: Exception) -> bool:
    text = str(err or "").lower()
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


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    delay_sec: float = 1.0,
) -> T:
    last_err: Exception | None = None
    ssl_sleep_sec = float(os.getenv("SSL_RETRY_SLEEP_SEC", "60"))
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i >= attempts:
                break
            sleep_for = ssl_sleep_sec if _is_ssl_retryable_error(e) else delay_sec
            time.sleep(sleep_for)
    assert last_err is not None
    raise last_err
