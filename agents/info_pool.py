from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


class InfoPool:
    """
    Workflow-wide information pool.

    Records every agent output as append-only events and keeps a latest snapshot.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.root_dir / "events.jsonl"
        self.snapshot_file = self.root_dir / "latest_snapshot.json"

    def _append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(_to_jsonable(event), ensure_ascii=False)
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_snapshot(self) -> dict[str, Any]:
        if not self.snapshot_file.exists():
            return {
                "updated_at": None,
                "run_id": None,
                "latest_by_agent": {},
                "latest_by_stage": {},
            }
        return json.loads(self.snapshot_file.read_text(encoding="utf-8"))

    def _save_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_file.write_text(
            json.dumps(_to_jsonable(snapshot), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def record_pipeline_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        now = _utc_now_iso()
        event = {
            "timestamp": now,
            "run_id": run_id,
            "kind": "pipeline_event",
            "event_type": event_type,
            "payload": payload,
        }
        self._append_event(event)

        snapshot = self._load_snapshot()
        snapshot["updated_at"] = now
        snapshot["run_id"] = run_id
        snapshot["last_pipeline_event"] = {
            "event_type": event_type,
            "timestamp": now,
            "payload": _to_jsonable(payload),
        }
        self._save_snapshot(snapshot)

    def record_agent_output(
        self,
        *,
        run_id: str,
        stage_name: str,
        agent_name: str,
        output: dict[str, Any],
    ) -> None:
        now = _utc_now_iso()
        event = {
            "timestamp": now,
            "run_id": run_id,
            "kind": "agent_output",
            "stage_name": stage_name,
            "agent_name": agent_name,
            "output": output,
        }
        self._append_event(event)

        snapshot = self._load_snapshot()
        snapshot["updated_at"] = now
        snapshot["run_id"] = run_id
        snapshot.setdefault("latest_by_agent", {})
        snapshot.setdefault("latest_by_stage", {})
        snapshot["latest_by_agent"][agent_name] = {
            "timestamp": now,
            "stage_name": stage_name,
            "output": _to_jsonable(output),
        }
        snapshot["latest_by_stage"][stage_name] = {
            "timestamp": now,
            "agent_name": agent_name,
            "output": _to_jsonable(output),
        }
        self._save_snapshot(snapshot)
