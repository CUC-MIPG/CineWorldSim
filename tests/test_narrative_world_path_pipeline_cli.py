from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_narrative_world_path_pipeline_uses_existing_pipeline_result(tmp_path: Path) -> None:
    pipeline_result = tmp_path / "pipeline_result.json"
    pipeline_result.write_text(
        json.dumps(
            {
                "backend_mode": "all_banana",
                "backend_tool": "nano_banana",
                "story": "A journalist discovers a cracked compass in a lighthouse.",
                "characters": ["Journalist", "Caretaker"],
                "world_states": [{"step": 0, "summary": "A lighthouse world."}],
                "world_deltas": [],
                "event_keyframes": [],
                "step_logs": [
                    {
                        "step": 1,
                        "sam_summary": "The journalist finds the compass.",
                        "role_logs": [
                            {
                                "role_name": "Journalist",
                                "role_output_text": "The journalist examines the cracked compass.",
                                "next_action": {
                                    "action_type": "examine compass",
                                    "target_character": "",
                                    "reason": "to uncover the truth",
                                },
                            },
                            {
                                "role_name": "Caretaker",
                                "role_output_text": "The caretaker blocks the doorway.",
                                "next_action": {
                                    "action_type": "block doorway",
                                    "target_character": "Journalist",
                                    "reason": "to hide the archive",
                                },
                            },
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_narrative_world_path_pipeline.py",
            "--existing-pipeline-result-json",
            str(pipeline_result),
            "--storyline",
            "A journalist discovers a cracked compass in a lighthouse.",
            "--output-dir",
            str(output_dir),
            "--path-mode",
            "llm",
            "--no-strict-online",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = output_dir / "pipeline_manifest.json"
    path_plan = output_dir / "path_llm_direct" / "narrative_path_plan.json"
    assert manifest.exists()
    assert path_plan.exists()

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    plan_payload = json.loads(path_plan.read_text(encoding="utf-8"))
    assert manifest_payload["status"] == "done"
    assert manifest_payload["runs"][0]["step"] == "path_orchestrator_llm_direct"
    assert manifest_payload["artifacts"]["pipeline_result_json"] == str(pipeline_result)
    assert "narrative_nodes_json" in manifest_payload["artifacts"]
    assert plan_payload["ordered_node_ids"] == ["NN_001"]
