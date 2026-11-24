from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_full_pipeline_stage_visualization import normalize_path


def test_normalize_path_resolves_repaired_image_extension(tmp_path):
    cwd = tmp_path
    repaired = tmp_path / "narrative_artifacts" / "step_001" / "Hero_event_keyframe.jpg"
    repaired.parent.mkdir(parents=True, exist_ok=True)
    repaired.write_bytes(b"jpg")

    stale = Path("narrative_artifacts") / "step_001" / "Hero_event_keyframe.png"

    assert normalize_path(stale, cwd) == repaired.resolve()

from scripts.generate_full_pipeline_stage_visualization import generate_html


def test_full_visualization_exposes_redesign_metadata(tmp_path):
    run_dir = tmp_path / "run" / "all_banana"
    video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
    run_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    for rel in [
        "portraits/Mara_initial_nano_banana.png",
        "world_model/locations/Vault/wide_nano_banana.png",
        "world_model/locations/Vault/close_nano_banana.png",
        "world_model/items/Compass/initial_nano_banana.png",
        "world_updates/locations/Vault/step_002_close.png",
        "event_keyframes/step_002/Mara_event_keyframe.png",
    ]:
        path = run_dir / "narrative_artifacts" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"img")

    pipeline = {
        "backend_mode": "all_banana",
        "backend_tool": "nano_banana",
        "story": "A vault adventure.",
        "characters": ["Mara"],
        "character_configs": [{"name": "Mara", "final_goal": "Recover the compass."}],
        "character_portraits": {
            "Mara": str(run_dir / "narrative_artifacts/portraits/Mara_initial_nano_banana.png")
        },
        "world_model": {
            "summary": "Vault world.",
            "locations": [
                {
                    "asset_id": "Vault",
                    "name": "Vault",
                    "description": "Ancient vault.",
                    "views": {
                        "wide": str(run_dir / "narrative_artifacts/world_model/locations/Vault/wide_nano_banana.png"),
                        "close": str(run_dir / "narrative_artifacts/world_model/locations/Vault/close_nano_banana.png"),
                    },
                    "view_prompts": {"wide": "wide view", "close": "close view"},
                    "images": [],
                }
            ],
            "items": [
                {
                    "asset_id": "Compass",
                    "name": "Compass",
                    "description": "A bronze compass.",
                    "initial_image_path": str(run_dir / "narrative_artifacts/world_model/items/Compass/initial_nano_banana.png"),
                    "prompt": "maximize item occupancy",
                    "images": [],
                }
            ],
        },
        "step_logs": [{"step": 2, "role_logs": [], "sam_summary": "Mara opens the vault."}],
        "world_states": [
            {
                "step": 2,
                "summary": "Vault wall changed.",
                "locations": [
                    {
                        "id": "Vault",
                        "name": "Vault",
                        "changed_this_step": True,
                        "changed_view": "close",
                        "views": {
                            "wide": str(run_dir / "narrative_artifacts/world_model/locations/Vault/wide_nano_banana.png"),
                            "close": str(run_dir / "narrative_artifacts/world_updates/locations/Vault/step_002_close.png"),
                        },
                        "last_image_path": str(run_dir / "narrative_artifacts/world_updates/locations/Vault/step_002_close.png"),
                        "change_reason": "physical cracks appeared",
                    }
                ],
                "items": [],
            }
        ],
        "world_deltas": [
            {
                "step": 2,
                "decision_source": "llm",
                "decision_rationale": "Only the close wall texture changed.",
                "location_updates": [
                    {
                        "id": "Vault",
                        "view_to_update": "close",
                        "physical_change": "wall texture cracked",
                    }
                ],
                "item_updates": [],
            }
        ],
        "event_keyframes": [
            {
                "step": 2,
                "role_name": "Mara",
                "output_path": str(run_dir / "narrative_artifacts/event_keyframes/step_002/Mara_event_keyframe.png"),
                "reference_selection": {
                    "selected": ["character:Mara", "location:Vault", "item:Compass"],
                    "excluded_assets": ["unmentioned item"],
                    "reason": "content-driven references",
                },
            }
        ],
    }
    (run_dir / "pipeline_result.json").write_text(json.dumps(pipeline), encoding="utf-8")
    (run_dir / "step_logs.json").write_text(json.dumps(pipeline["step_logs"]), encoding="utf-8")
    (run_dir / "path_orchestration").mkdir()
    (run_dir / "path_orchestration" / "narrative_nodes.json").write_text(json.dumps({"selected_nodes": []}), encoding="utf-8")
    (run_dir / "path_orchestration" / "path_llm_direct").mkdir()
    (run_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").write_text(
        json.dumps({"ordered_node_ids": []}), encoding="utf-8"
    )
    (video_dir / "event_video_summary.json").write_text(json.dumps({"status": "dry_run", "success": 2, "total": 2}), encoding="utf-8")
    (video_dir / "event_video_tasks.json").write_text(
        json.dumps(
            [
                {"task_type": "establishing", "node_id": "establishing::Vault", "duration_sec": 5, "num_frames": 124},
                {
                    "task_type": "event",
                    "node_id": "NN_001",
                    "reference_mode": "semantic_assets",
                    "reference_selection": {"mode": "semantic_assets", "reason": "mentioned assets only"},
                    "duration_sec": 10,
                    "num_frames": 243,
                    "shot_planning": {"shot_count": 3, "duration_sec": 10},
                    "reference_images": [
                        str(run_dir / "narrative_artifacts/event_keyframes/step_002/Mara_event_keyframe.png")
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    output_html = tmp_path / "viz.html"
    generate_html(run_dir, video_dir, output_html, tmp_path)

    text = output_html.read_text(encoding="utf-8")
    assert "Initial Character Portraits" in text
    assert "Location Views" in text
    assert "wide" in text and "close" in text
    assert "World Delta Decisions" in text
    assert "decision_source" in text
    assert "view_to_update" in text
    assert "Reference Selection" in text
    assert "content-driven references" in text
    assert "establishing" in text
    assert "semantic_assets" in text
    assert "duration 10s" in text
    assert "frames 243" in text
