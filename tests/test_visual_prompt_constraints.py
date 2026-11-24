from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.visual_prompt_constraints import build_people_control_clause, event_needs_multi_person_references
from narrative.visual_prompt_constraints import build_node_selection_prompt_policy, build_path_orchestration_prompt_policy


def test_single_subject_clause_blocks_unrelated_people() -> None:
    clause = build_people_control_clause(["Hero"], allow_multi_people=False)
    assert "people_policy:" in clause.lower()
    assert "pedestrians" in clause.lower()
    assert "unrelated people" in clause.lower()
    assert "hero" in clause.lower()


def test_multi_person_detection_requires_interaction_cues() -> None:
    node = {
        "characters": ["Hero", "Guard"],
        "event_summary": "Hero confronts the guard at the door.",
    }
    assert event_needs_multi_person_references(node) is True


def test_single_person_event_does_not_require_multi_person_references() -> None:
    node = {
        "characters": ["Hero"],
        "event_summary": "Hero studies the door alone.",
    }
    assert event_needs_multi_person_references(node) is False


def test_prompt_policies_explain_atomic_nodes_and_merge_rules() -> None:
    node_policy = build_node_selection_prompt_policy().lower()
    path_policy = build_path_orchestration_prompt_policy().lower()
    assert "one role action per step" in node_policy
    assert "same step" in node_policy
    assert "same location" in node_policy
    assert "atomic narrative node" in path_policy
    assert "do not split or merge" in path_policy
