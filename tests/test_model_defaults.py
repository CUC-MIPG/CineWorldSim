from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.client import pick_model


def test_default_model_is_gpt_55_when_no_scoped_model_is_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL_DEFAULT_SCOPE_MODEL", raising=False)

    assert pick_model(scope="MODEL_DEFAULT_SCOPE") == "gpt-5.5"


def test_explicit_gpt_4o_default_argument_remains_supported(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.delenv("LEGACY_SCOPE_MODEL", raising=False)

    assert pick_model("gpt-4o", scope="LEGACY_SCOPE") == "gpt-4o"
