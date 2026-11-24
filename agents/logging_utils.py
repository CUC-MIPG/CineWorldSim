from __future__ import annotations

import logging
import os


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    raw = (level or os.getenv("CAMERAMASTER_LOG_LEVEL") or "INFO").strip().upper()
    numeric = getattr(logging, raw, logging.INFO)
    logging.basicConfig(level=numeric, format=_FORMAT, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

