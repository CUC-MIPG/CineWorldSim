from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repair_banana_pipeline_assets import repair_pipeline_assets


def test_repair_pipeline_assets_updates_failed_world_and_keyframe_records(tmp_path):
    result_json = tmp_path / "pipeline_result.json"
    world_out = tmp_path / "world.png"
    keyframe_out = tmp_path / "keyframe.png"
    result_json.write_text(
        json.dumps(
            {
                "world_model": {
                    "locations": [
                        {
                            "asset_id": "Station",
                            "image_prompt": "cinematic station",
                            "images": [{"tool": "nano_banana", "ok": False, "output_path": str(world_out)}],
                        }
                    ],
                    "items": [],
                },
                "event_keyframes": [
                    {
                        "tool": "nano_banana",
                        "ok": True,
                        "output_path": str(keyframe_out),
                        "prompt": "cinematic keyframe",
                        "step": 1,
                        "role_name": "Actor",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = repair_pipeline_assets(result_json, backend_mode="all_banana", dry_run=True)

    updated = json.loads(result_json.read_text(encoding="utf-8"))
    assert summary["repaired"] == 2
    assert world_out.exists()
    assert keyframe_out.exists()
    assert updated["world_model"]["locations"][0]["images"][0]["ok"] is True
    assert updated["event_keyframes"][0]["ok"] is True
