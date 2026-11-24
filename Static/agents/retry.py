from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    delay_sec: float = 1.0,
) -> T:
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i >= attempts:
                break
            time.sleep(delay_sec)
    assert last_err is not None
    raise last_err
