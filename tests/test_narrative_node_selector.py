from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sample_pipeline_result(tmp_path: Path) -> dict:
    location_img = tmp_path / "lighthouse.jpg"
    item_img = tmp_path / "compass.jpg"
    keyframe_img = tmp_path / "journalist_keyframe.jpg"
    for path in [location_img, item_img, keyframe_img]:
        path.write_bytes(b"fake-image")
    return {
        "story": "A journalist discovers a cracked compass in a lighthouse.",
        "backend_mode": "all_banana",
        "backend_tool": "nano_banana",
        "world_states": [
            {
                "step": 1,
                "locations": [
                    {
                        "id": "Abandoned_Lighthouse",
                        "name": "Abandoned Lighthouse",
                        "last_image_path": str(location_img),
                    }
                ],
                "items": [
                    {
                        "id": "Cracked_Compass",
                        "name": "Cracked Compass",
                        "last_image_path": str(item_img),
                    }
                ],
            }
        ],
        "world_deltas": [
            {
                "step": 1,
                "location_updates": [{"id": "Abandoned_Lighthouse", "new_image_path": str(location_img)}],
                "item_updates": [{"id": "Cracked_Compass", "new_image_path": str(item_img)}],
            }
        ],
        "event_keyframes": [
            {
                "step": 1,
                "role_name": "Journalist",
                "output_path": str(keyframe_img),
            }
        ],
        "step_logs": [
            {
                "step": 1,
                "sam_summary": "The journalist discovers the cracked compass while the caretaker blocks the door.",
                "sam_story_phase": "development",
                "storyline_completion": {"completion_score": 0.45, "is_story_complete": False},
                "role_logs": [
                    {
                        "role_name": "Journalist",
                        "role_output_text": "The journalist examines the cracked compass.",
                        "next_action": {
                            "action_type": "examine compass",
                            "location": "Abandoned Lighthouse",
                            "reason": "to uncover the truth",
                        },
                    },
                    {
                        "role_name": "Caretaker",
                        "role_output_text": "The caretaker blocks the doorway.",
                        "next_action": {
                            "action_type": "block doorway",
                            "target_character": "Journalist",
                            "location": "Abandoned Lighthouse",
                            "reason": "to hide the flooded archive",
                        },
                    },
                ],
            }
        ],
    }


def test_narrative_node_selector_groups_scores_and_binds_assets(tmp_path: Path) -> None:
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.5, min_nodes=1, max_nodes=5, llm_enabled=False)
    )

    result = selector.select(_sample_pipeline_result(tmp_path), source_pipeline_result_json="pipeline_result.json")

    assert result["candidate_count"] == 1
    assert result["selected_count"] == 1
    node = result["selected_nodes"][0]
    assert node["node_id"] == "NN_001"
    assert node["event_id"] == "EVT_001"
    assert node["source_node_ids"] == ["step::1::Journalist", "step::1::Caretaker"]
    assert node["characters"] == ["Journalist", "Caretaker"]
    assert node["locations"] == ["Abandoned Lighthouse"]
    assert node["items"] == ["Cracked Compass"]
    assert node["score"] >= 0.5
    assert node["score_breakdown"]["dramatic_conflict"] > 0
    assert node["visual_assets"]["location_images"]
    assert node["visual_assets"]["item_images"]
    assert node["visual_assets"]["event_keyframes"]
    assert node["visual_assets"]["multi_person_reference_allowed"] is True


def test_narrative_node_selector_binds_character_portraits_from_payload(tmp_path: Path) -> None:
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    portrait = tmp_path / "journalist_portrait.png"
    portrait.write_bytes(b"fake-portrait")
    payload = _sample_pipeline_result(tmp_path)
    payload["character_portraits"] = {"Journalist": str(portrait)}

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.5, min_nodes=1, max_nodes=5, llm_enabled=False)
    )
    result = selector.select(payload, source_pipeline_result_json="pipeline_result.json")

    node = result["selected_nodes"][0]
    assets = node["visual_assets"]
    assert str(portrait) in assets["character_reference_images"]


def test_narrative_node_selector_keeps_same_step_roles_separate_without_interaction() -> None:
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.5, min_nodes=1, max_nodes=5, llm_enabled=False)
    )
    payload = _sample_pipeline_result(Path("/tmp"))
    role_candidates = [
        {
            "candidate_id": "CAND_001",
            "source_node_id": "step::1::Hero",
            "step": 1,
            "role_name": "Hero",
            "role_output_text": "Hero walks to the doorway.",
            "action_type": "walk",
            "target_character": "",
            "location": "Hallway",
            "reason": "",
            "monologue": "",
            "sam_summary": "Hero walks to the doorway while the guard stays nearby.",
            "story_phase": "development",
            "completion_score": 0.4,
        },
        {
            "candidate_id": "CAND_002",
            "source_node_id": "step::1::Guard",
            "step": 1,
            "role_name": "Guard",
            "role_output_text": "Guard watches from the corridor.",
            "action_type": "watch",
            "target_character": "",
            "location": "Hallway",
            "reason": "",
            "monologue": "",
            "sam_summary": "Hero walks to the doorway while the guard stays nearby.",
            "story_phase": "development",
            "completion_score": 0.4,
        },
    ]

    events = selector.group_candidates_into_event_candidates(payload, role_candidates)

    assert len(events) == 2
    assert events[0]["characters"] == ["Hero"]
    assert events[1]["characters"] == ["Guard"]


def test_narrative_node_selector_merges_same_step_interacting_roles() -> None:
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.5, min_nodes=1, max_nodes=5, llm_enabled=False)
    )
    payload = _sample_pipeline_result(Path("/tmp"))
    role_candidates = [
        {
            "candidate_id": "CAND_001",
            "source_node_id": "step::1::Hero",
            "step": 1,
            "role_name": "Hero",
            "role_output_text": "Hero asks the Guard to open the door.",
            "action_type": "ask",
            "target_character": "Guard",
            "location": "Hallway",
            "reason": "",
            "monologue": "",
            "sam_summary": "Hero asks the Guard to open the door.",
            "story_phase": "development",
            "completion_score": 0.7,
        },
        {
            "candidate_id": "CAND_002",
            "source_node_id": "step::1::Guard",
            "step": 1,
            "role_name": "Guard",
            "role_output_text": "Guard blocks the door and answers back.",
            "action_type": "block",
            "target_character": "Hero",
            "location": "Hallway",
            "reason": "",
            "monologue": "",
            "sam_summary": "Hero asks the Guard to open the door.",
            "story_phase": "development",
            "completion_score": 0.7,
        },
    ]

    events = selector.group_candidates_into_event_candidates(payload, role_candidates)

    assert len(events) == 1
    assert events[0]["characters"] == ["Hero", "Guard"]
    assert events[0]["node_granularity"] == "multi_role_interaction"


def test_narrative_node_selector_cli_writes_json(tmp_path: Path) -> None:
    pipeline_result = tmp_path / "pipeline_result.json"
    pipeline_result.write_text(json.dumps(_sample_pipeline_result(tmp_path), ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "nodes"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_narrative_node_selector.py",
            "--pipeline-result-json",
            str(pipeline_result),
            "--output-dir",
            str(output_dir),
            "--score-threshold",
            "0.5",
            "--min-nodes",
            "1",
            "--max-nodes",
            "5",
            "--no-llm-scoring",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    out = output_dir / "narrative_nodes.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selected_nodes"][0]["event_id"] == "EVT_001"


def test_narrative_node_selector_uses_openai_scope_for_llm_scoring(monkeypatch, tmp_path: Path) -> None:
    import agents.client as client_mod
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    called = {"make_client": [], "pick_model": []}

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse(
                json.dumps(
                    {
                        "dramatic_conflict": 9,
                        "goal_progress": 8,
                        "emotional_turning_point": 7,
                        "information_reveal": 10,
                        "visual_potential": 8,
                        "causal_importance": 9,
                        "redundancy_penalty": 1,
                        "selection_reason": "High-value conflict and clue reveal.",
                    }
                )
            )

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    def fake_make_client(*, scope: str = "OPENAI"):
        called["make_client"].append(scope)
        return _FakeClient()

    def fake_pick_model(default: str = "gpt-4o", scope: str = "OPENAI"):
        called["pick_model"].append(scope)
        return "gpt-4o"

    monkeypatch.setattr(client_mod, "make_client", fake_make_client)
    monkeypatch.setattr(client_mod, "pick_model", fake_pick_model)

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.1, min_nodes=1, max_nodes=5, llm_enabled=True)
    )
    result = selector.select(_sample_pipeline_result(tmp_path), source_pipeline_result_json="pipeline_result.json")

    assert called["make_client"] == ["OPENAI"]
    assert called["pick_model"] == ["OPENAI"]
    assert result["selected_nodes"][0]["score_source"] == "llm"
    assert result["selected_nodes"][0]["score"] > 0.7


def test_narrative_node_selector_llm_prompt_acts_as_story_editor(monkeypatch, tmp_path: Path) -> None:
    import agents.client as client_mod
    from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

    captured = {"system": "", "user": ""}

    class _FakeResponse:
        choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Msg",
                        (),
                        {
                            "content": json.dumps(
                                {
                                    "dramatic_conflict": 8,
                                    "goal_progress": 8,
                                    "emotional_turning_point": 7,
                                    "information_reveal": 9,
                                    "visual_potential": 8,
                                    "causal_importance": 8,
                                    "redundancy_penalty": 2,
                                    "selection_reason": "It should be kept for event-to-multishot planning.",
                                }
                            )
                        },
                    )()
                },
            )()
        ]

    class _FakeCompletions:
        def create(self, **kwargs):
            messages = kwargs["messages"]
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(client_mod, "make_client", lambda scope="OPENAI": _FakeClient())
    monkeypatch.setattr(client_mod, "pick_model", lambda default="gpt-4o", scope="OPENAI": "gpt-4o")

    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(score_threshold=0.1, min_nodes=1, max_nodes=5, llm_enabled=True)
    )
    selector.select(_sample_pipeline_result(tmp_path), source_pipeline_result_json="pipeline_result.json")

    combined = captured["system"] + "\n" + captured["user"]
    assert "story editor" in combined.lower()
    assert "director" in combined.lower()
    assert "mundane transitional actions" in combined.lower()
    assert "event-to-multishot" in combined.lower()
    assert "not a shot" in combined.lower()
