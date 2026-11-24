from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_narrative_world_pipeline_html import generate_pipeline_html


def test_generate_pipeline_html_contains_time_and_space_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "all_flux"
    run_dir.mkdir()
    payload = {
        "backend_mode": "all_flux",
        "story": "A storm mystery.",
        "characters": ["Journalist"],
        "world_states": [{"step": 0, "locations": [], "items": []}],
        "step_logs": [{"step": 1, "role_logs": [], "sam_summary": "A door opens."}],
        "event_keyframes": [],
        "world_deltas": [],
    }
    (run_dir / "pipeline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    output_html = tmp_path / "combined_overview.html"

    generate_pipeline_html(run_dirs=[run_dir], output_html=output_html)

    text = output_html.read_text(encoding="utf-8")
    assert "Narrative Time" in text
    assert "World Space" in text
    assert "Backend Comparison" in text
    assert "all_flux" in text
    assert "A storm mystery." in text


def test_generate_pipeline_html_exposes_redesign_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "all_banana"
    run_dir.mkdir()
    for rel in [
        "portrait.png",
        "wide.png",
        "close.png",
        "item.png",
        "kf.png",
    ]:
        (tmp_path / rel).write_bytes(b"img")
    payload = {
        "backend_mode": "all_banana",
        "story": "A vault adventure.",
        "characters": ["Mara"],
        "character_portraits": {"Mara": str(tmp_path / "portrait.png")},
        "world_model": {
            "locations": [
                {
                    "name": "Vault",
                    "views": {"wide": str(tmp_path / "wide.png"), "close": str(tmp_path / "close.png")},
                }
            ],
            "items": [{"name": "Compass", "initial_image_path": str(tmp_path / "item.png")}],
        },
        "world_states": [
            {
                "step": 1,
                "locations": [
                    {
                        "name": "Vault",
                        "changed_this_step": True,
                        "changed_views_this_step": ["close"],
                        "views": {"wide": str(tmp_path / "wide.png"), "close": str(tmp_path / "close.png")},
                    }
                ],
                "items": [],
            }
        ],
        "step_logs": [{"step": 1, "role_logs": [], "sam_summary": "Mara opens the vault."}],
        "event_keyframes": [
            {
                "step": 1,
                "role_name": "Mara",
                "output_path": str(tmp_path / "kf.png"),
                "reference_selection": {"reason": "content-driven references"},
            }
        ],
        "world_deltas": [{"step": 1, "decision_source": "llm", "location_updates": [{"view_to_update": "close"}]}],
    }
    (run_dir / "pipeline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    output_html = tmp_path / "combined_overview.html"

    generate_pipeline_html(run_dirs=[run_dir], output_html=output_html)

    text = output_html.read_text(encoding="utf-8")
    assert "Initial Character Portraits" in text
    assert "Location Views" in text
    assert "World Delta Decisions" in text
    assert "Reference Selection" in text
    assert "content-driven references" in text
