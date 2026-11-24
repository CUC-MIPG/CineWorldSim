from narrative.story_control import StoryController


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def chat_json(self, **kwargs):
        self.prompt = kwargs["user_prompt"]
        return self.response


def controller(response):
    contract = {
        "immutable_entities": ["The Painter", "one landscape painting"],
        "hard_beats": [
            {"beat_id": "B3", "title": "Whiteout", "requirement": "Keep painting in a blizzard", "preconditions": [], "terminal": False},
            {"beat_id": "B4", "title": "Finish", "requirement": "Apply the final stroke that completes the same painting", "preconditions": ["B3"], "terminal": True},
        ],
        "terminal_constraints": [],
    }
    model = FakeModel(response)
    control = StoryController(
        story_outline="The painter finishes the same landscape after three weather trials.",
        model_config=model,
        saved_state={
            "version": StoryController.VERSION,
            "contract": contract,
            "scene_plan": [],
            "covered_beat_ids": ["B3"],
            "beat_evidence": {"B3": {"evidence_step": 7}},
        },
    )
    return control, model


def action(action_type, events):
    return [{
        "role_name": "The Painter",
        "action_accepted": True,
        "next_action": {"action_type": action_type},
        "actual_step_events": events,
    }]


def assess(control, actions):
    return control.assess(
        step=19,
        world_summary="The story may be over.",
        current_environment="A snowy coastal overlook.",
        public_events=[],
        step_actions=actions,
    )


def test_final_make_stroke_is_accepted_from_reasoned_action_and_event():
    control, model = controller({
        "beat_evidence": [],
        "terminal_decision": {
            "beat_id": "B4", "matched": True,
            "reason": "The painter makes the stroke and the fresh pigment closes the last gap, completing the same painting.",
            "evidence_text": "Fresh pigment closes the last gap in the painting.",
        },
        "contradictions": [],
    })
    result = assess(control, action(
        "Make one precise brush stroke that closes the remaining hairline gap.",
        ["Fresh pigment closes the last gap in the painting."],
    ))

    assert result["newly_covered_beat_ids"] == ["B4"]
    assert result["terminal_decision"]["accepted"] is True
    assert "fresh pigment" in control.beat_evidence["B4"]["reason"].lower()
    assert control.terminal_ready
    assert "do not require a fixed verb" in model.prompt


def test_withdrawing_brush_does_not_cover_terminal_beat():
    control, _ = controller({
        "beat_evidence": [{"beat_id": "B4", "evidence_text": "The painting was already complete."}],
        "terminal_decision": {
            "beat_id": "B4", "matched": False,
            "reason": "The painter only withdraws the brush; no finishing mark occurs in this step.",
            "evidence_text": "",
        },
        "contradictions": [],
    })
    result = assess(control, action(
        "Withdraw the brush from the finished painting.",
        ["The brush leaves the canvas without adding pigment."],
    ))

    assert result["newly_covered_beat_ids"] == []
    assert "only withdraws" in result["terminal_decision"]["reason"]
    assert not control.terminal_ready


def test_missing_reason_cannot_cover_terminal_beat():
    control, _ = controller({
        "beat_evidence": [{"beat_id": "B4", "evidence_text": "A brush stroke closes the gap."}],
        "terminal_decision": {
            "beat_id": "B4", "matched": True,
            "reason": "", "evidence_text": "A brush stroke closes the gap.",
        },
        "contradictions": [],
    })
    result = assess(control, action(
        "Make one final brush stroke.",
        ["A brush stroke closes the gap."],
    ))

    assert result["newly_covered_beat_ids"] == []
    assert result["terminal_decision"]["accepted"] is False
    assert "nonempty reason" in result["terminal_decision"]["validation_reason"]


def test_model_match_without_recorded_result_does_not_cover_terminal_beat():
    control, _ = controller({
        "beat_evidence": [],
        "terminal_decision": {
            "beat_id": "B4", "matched": True,
            "reason": "The final mark seems complete.",
            "evidence_text": "The final mark completes the painting.",
        },
        "contradictions": [],
    })
    result = assess(control, action("Make a final brush stroke.", []))

    assert result["newly_covered_beat_ids"] == []
    assert result["terminal_decision"]["accepted"] is False
    assert "no accepted resulting event" in result["terminal_decision"]["validation_reason"].lower()
