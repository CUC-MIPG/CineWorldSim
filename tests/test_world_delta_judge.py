from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_delta_judge import WorldDeltaJudge


def test_world_delta_judge_fails_closed_when_llm_disabled() -> None:
    judge = WorldDeltaJudge(model_config=None, llm_enabled=False)
    previous_state = {
        "step": 0,
        "locations": [{"id": "Lighthouse", "name": "Lighthouse", "current_state_text": "Locked door."}],
        "items": [{"id": "Compass", "name": "Compass", "current_state_text": "Hidden in a drawer."}],
    }
    step_log = {
        "step": 1,
        "role_logs": [
            {
                "role_name": "Journalist",
                "next_action": {
                    "action": "opens the lighthouse door and takes the compass",
                    "location": "Lighthouse",
                    "reason": "The compass is needed as evidence.",
                },
            }
        ],
        "sam_summary": "The lighthouse door is opened and the compass is taken.",
    }

    delta = judge.judge_step_delta(previous_state=previous_state, step_log=step_log)

    assert delta["step"] == 1
    assert delta["location_updates"] == []
    assert delta["item_updates"] == []
    assert delta["decision_source"] == "disabled_or_fallback"


def test_world_delta_judge_filters_non_changed_llm_rows() -> None:
    class FakeLLM:
        def run_json(self, **kwargs):
            return {
                "summary": "No visible world change.",
                "location_updates": [
                    {"id": "Lighthouse", "changed": False, "image_refresh_required": False}
                ],
                "item_updates": [
                    {"id": "Compass", "changed": False, "image_refresh_required": False}
                ],
            }

    previous_state = {
        "step": 0,
        "locations": [{"id": "Lighthouse", "name": "Lighthouse", "current_state_text": "Locked door."}],
        "items": [{"id": "Compass", "name": "Compass", "current_state_text": "Hidden in a drawer."}],
    }
    step_log = {"step": 1, "sam_summary": "The journalist learns about the compass."}

    delta = WorldDeltaJudge(llm_tool=FakeLLM(), llm_enabled=True).judge_step_delta(
        previous_state=previous_state,
        step_log=step_log,
        story="A lighthouse story.",
    )

    assert delta["location_updates"] == []
    assert delta["item_updates"] == []
    assert delta["decision_source"] == "llm"
