from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_state_tracker import build_initial_world_state, update_world_state_from_delta


def test_build_initial_world_state_from_world_model() -> None:
    world_model = {
        "summary": "A storm island world.",
        "locations": [
            {
                "asset_id": "Lighthouse",
                "name": "Lighthouse",
                "description": "A broken coastal tower.",
                "images": [{"tool": "flux_local", "ok": True, "output_path": "/tmp/lighthouse.png"}],
            }
        ],
        "items": [
            {
                "asset_id": "Compass",
                "name": "Compass",
                "description": "A cracked compass.",
                "images": [{"tool": "flux_local", "ok": True, "output_path": "/tmp/compass.png"}],
            }
        ],
    }

    state = build_initial_world_state(world_model=world_model, backend_tool="flux_local")

    assert state["step"] == 0
    assert state["summary"] == "A storm island world."
    assert state["locations"][0]["id"] == "Lighthouse"
    assert state["locations"][0]["current_state_text"] == "A broken coastal tower."
    assert state["locations"][0]["last_image_path"] == "/tmp/lighthouse.png"
    assert state["items"][0]["id"] == "Compass"
    assert state["items"][0]["last_image_path"] == "/tmp/compass.png"


def test_update_world_state_from_delta_marks_changed_elements() -> None:
    previous = {
        "step": 0,
        "summary": "A storm island world.",
        "locations": [
            {
                "id": "Lighthouse",
                "name": "Lighthouse",
                "description": "A broken coastal tower.",
                "current_state_text": "Dark and locked.",
                "last_image_path": "/tmp/lighthouse.png",
                "changed_this_step": False,
            }
        ],
        "items": [],
    }
    delta = {
        "step": 1,
        "summary": "The lighthouse has been opened.",
        "location_updates": [
            {
                "id": "Lighthouse",
                "new_state_text": "The door is open and rain enters the stairwell.",
                "image_refresh_required": True,
                "reason": "door opened",
            }
        ],
        "item_updates": [],
    }

    state = update_world_state_from_delta(previous_state=previous, delta=delta)

    assert state["step"] == 1
    assert state["summary"] == "The lighthouse has been opened."
    assert state["locations"][0]["current_state_text"] == "The door is open and rain enters the stairwell."
    assert state["locations"][0]["changed_this_step"] is True
    assert state["locations"][0]["change_reason"] == "door opened"
