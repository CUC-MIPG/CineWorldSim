from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from narrative.moviebench_llm_builder import (
    atomic_write_json,
    build_l1_prompt,
    build_l2_prompt,
    compact_arc_fragment_for_consolidation,
    compact_l1_shot_for_l2,
    compact_l2_event_for_l3,
    compact_l3_scene_for_l4,
    compact_movie_bible_for_arc,
    canonicalize_l3_scenes,
    normalize_l2_event_ids,
    validate_l1_collection,
    validate_l2_collection,
    validate_l3_collection,
    validate_track_b_no_leakage,
    package_track_a_and_b,
    run_unit,
)


def test_atomic_write_json_replaces_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "unit.json"
    atomic_write_json(target, {"status": "done"})

    assert json.loads(target.read_text()) == {"status": "done"}
    assert not list(target.parent.glob("*.tmp"))


def test_l1_collection_requires_one_result_per_source_shot() -> None:
    source_ids = ["shot_001", "shot_002"]
    valid = [
        {"l1_shot_id": "shot_001", "source_evidence": {}},
        {"l1_shot_id": "shot_002", "source_evidence": {}},
    ]
    assert validate_l1_collection(source_ids, valid)["complete"] is True

    invalid = validate_l1_collection(source_ids, valid[:1])
    assert invalid["complete"] is False
    assert invalid["missing_ids"] == ["shot_002"]


def test_l2_collection_requires_exactly_once_shot_assignment() -> None:
    result = validate_l2_collection(
        ["shot_001", "shot_002"],
        [{"source_shot_ids": ["shot_001"]}, {"source_shot_ids": ["shot_001", "shot_002"]}],
    )
    assert result["complete"] is False
    assert result["duplicated_ids"] == ["shot_001"]


def test_normalize_l2_event_ids_uses_sequence_and_position() -> None:
    events = normalize_l2_event_ids(
        "0001_American_Beauty_continuous_0006",
        [{"l2_event_id": "hallucinated"}, {"l2_event_id": "hallucinated"}],
    )
    assert [event["l2_event_id"] for event in events] == [
        "0001_American_Beauty_continuous_0006_l2_01",
        "0001_American_Beauty_continuous_0006_l2_02",
    ]


def test_normalized_l2_ids_are_globally_unique_across_sequences() -> None:
    first = normalize_l2_event_ids("seq_001", [{"l2_event_id": "L2E01"}])
    second = normalize_l2_event_ids("seq_002", [{"l2_event_id": "L2E01"}])
    assert first[0]["l2_event_id"] != second[0]["l2_event_id"]


def test_canonicalize_l3_maps_legacy_ids_by_source_shot_evidence() -> None:
    events = normalize_l2_event_ids(
        "seq_001",
        [{"l2_event_id": "legacy_event", "source_shot_ids": ["shot_001"]}],
    )
    scenes = canonicalize_l3_scenes(
        [{
            "l2_event_ids": ["legacy_event"],
            "evidence_trace": {"source_l1_ids": ["shot_001"]},
        }],
        events,
    )
    assert scenes[0]["l2_event_ids"] == ["seq_001_l2_01"]


def test_validate_l3_accepts_complete_window_outputs() -> None:
    result = validate_l3_collection(
        ["seq_001_l2_01", "seq_002_l2_01"],
        [{"l2_event_ids": ["seq_001_l2_01"]}, {"l2_event_ids": ["seq_002_l2_01"]}],
    )
    assert result["complete"] is True


def test_validate_l3_rejects_window_that_drops_an_event() -> None:
    result = validate_l3_collection(
        ["seq_001_l2_01", "seq_001_l2_02"],
        [{"l2_event_ids": ["seq_001_l2_01"]}],
    )
    assert result["complete"] is False
    assert result["missing_ids"] == ["seq_001_l2_02"]


def test_l3_collection_requires_exactly_once_event_assignment() -> None:
    result = validate_l3_collection(
        ["event_001", "event_002"],
        [{"l2_event_ids": ["event_001"]}, {"l2_event_ids": ["event_001", "event_002"]}],
    )
    assert result["complete"] is False
    assert result["duplicated_ids"] == ["event_001"]


def test_track_b_rejects_withheld_detail_leakage() -> None:
    assert validate_track_b_no_leakage(
        {
            "short_storyline_L": "A family faces tension.",
            "withheld_event_ids": ["event_002"],
            "short_storyline_source_event_ids": ["event_001"],
            "withheld_events": [{"event_id": "event_002", "description": "private detail"}],
        }
    )["leakage"] is False

    leaked = validate_track_b_no_leakage(
        {
            "short_storyline_L": "A family faces tension and private detail.",
            "withheld_event_ids": ["event_002"],
            "short_storyline_source_event_ids": ["event_001", "event_002"],
            "withheld_events": [{"event_id": "event_002", "description": "private detail"}],
        }
    )
    assert leaked["leakage"] is True


def test_track_b_visible_source_ids_are_l2_event_ids() -> None:
    _, track_b = package_track_a_and_b(
        movie_id="movie",
        story_arcs=[
            {
                "l4_story_arc_id": "arc_1",
                "l3_scene_ids": ["scene_1"],
                "l2_event_ids": ["event_1", "event_2", "event_3"],
                "l1_shot_ids": [],
                "short_storyline_L": "A family faces tension.",
            }
        ],
        l1_by_id={},
    )
    sample = track_b["samples"][0]
    assert sample["short_storyline_source_event_ids"] == ["event_1"]
    assert sample["withheld_event_ids"] == ["event_2", "event_3"]
    assert sample["leakage_check"]["leakage"] is False


def test_prompts_require_json_and_evidence_trace() -> None:
    l1_system, l1_user = build_l1_prompt(
        {"file_name": "shot.json", "Plot": "A person walks."},
        shot_id="shot_001",
        movie_id="movie",
    )
    l2_system, l2_user = build_l2_prompt(
        {"movie_id": "movie", "sequence_id": "seq", "shots": []}
    )

    combined = "\n".join((l1_system, l1_user, l2_system, l2_user))
    assert "Return valid JSON only" in combined
    assert "evidence" in combined.lower()
    assert "Every input shot must be assigned exactly once" in combined


def test_compact_projections_keep_lineage_and_remove_repeated_payloads() -> None:
    shot = {
        "l1_shot_id": "shot_001",
        "movie_id": "movie",
        "time_range": {"start_ms": 1, "end_ms": 2},
        "characters": [{"name": "A", "visible_action": "walks"}],
        "location_mentions": [{"name": "hall"}],
        "item_mentions": [{"name": "key"}],
        "actions": [{"actor": "A", "action": "walks"}],
        "observable_facts": ["A walks"],
        "interpretive_facts": ["A is worried"],
        "source_evidence": {"Plot": "long repeated source text"},
    }
    compact_shot = compact_l1_shot_for_l2(shot)
    assert compact_shot["l1_shot_id"] == "shot_001"
    assert compact_shot["observable_facts"] == ["A walks"]
    assert "interpretive_facts" not in compact_shot
    assert "source_evidence" not in compact_shot

    event = {
        "l2_event_id": "event_001",
        "source_shot_ids": ["shot_001"],
        "time_range": {"start_ms": 1, "end_ms": 2},
        "event_title": "Walk",
        "event_description": "A walks through the hall.",
        "main_characters": ["A"],
        "setting": "hall",
        "locations": ["hall"],
        "items": ["key"],
        "actions": ["walks"],
        "interactions": [],
        "state_changes": ["position changes"],
        "causal_preconditions": ["long causal text"],
        "camera_and_visual_style": ["long camera text"],
        "evidence_trace": {"source_l1_ids": ["shot_001"]},
    }
    compact_event = compact_l2_event_for_l3(event)
    assert compact_event["l2_event_id"] == "event_001"
    assert compact_event["source_shot_ids"] == ["shot_001"]
    assert "causal_preconditions" not in compact_event
    assert "camera_and_visual_style" not in compact_event


def test_l4_projections_keep_final_evaluation_fields() -> None:
    scene = {
        "l3_scene_id": "scene_001",
        "l2_event_ids": ["event_001"],
        "time_range": {"start_ms": 1, "end_ms": 2},
        "scene_setting": "hall",
        "active_characters": ["A"],
        "scene_purpose": "movement",
        "scene_narrative": "A crosses the hall.",
        "narrative_state_transition": {"entry_state": {}, "exit_state": {}},
        "location_continuity": {"same_location": True},
        "causal_links": ["long causal explanation"],
        "evidence_trace": {"source_l2_ids": ["event_001"]},
        "quality": {"unsupported_claims": []},
    }
    compact_scene = compact_l3_scene_for_l4(scene)
    assert compact_scene["l3_scene_id"] == "scene_001"
    assert compact_scene["l2_event_ids"] == ["event_001"]
    assert "quality" not in compact_scene

    bible = {
        "movie_id": "movie",
        "coverage": {"scene_count": 1},
        "movie_synopsis": {"short_synopsis": "short"},
        "global_character_profiles": [{"name": "A", "profile": "long"}],
        "recurring_locations": ["hall"],
        "recurring_objects": ["key"],
        "global_constraints_supported_by_evidence": ["constraint"],
        "unresolved_questions": ["question"],
        "evidence_trace": {"source_l3_ids": ["scene_001"]},
        "very_large_unused_field": "x" * 1000,
    }
    compact_bible = compact_movie_bible_for_arc(bible)
    assert compact_bible["movie_id"] == "movie"
    assert compact_bible["recurring_locations"] == ["hall"]
    assert "very_large_unused_field" not in compact_bible

    arc = {
        "movie_id": "movie",
        "story_arcs": [{
            "l4_story_arc_id": "arc_001",
            "l3_scene_ids": ["scene_001"],
            "l2_event_ids": ["event_001"],
            "l1_shot_ids": ["shot_001"],
            "full_story_S": "full story",
            "short_storyline_L": "short story",
            "essential_events": [{"event_id": "event_001"}],
            "fine_grained_units": [{"unit_id": "fine_001"}],
            "evidence_trace": {"source_l3_ids": ["scene_001"]},
            "quality": {"unsupported_claims": []},
        }],
    }
    compact_arc = compact_arc_fragment_for_consolidation(arc)
    assert compact_arc["story_arcs"][0]["full_story_S"] == "full story"
    assert compact_arc["story_arcs"][0]["fine_grained_units"] == [{"unit_id": "fine_001"}]
    assert "quality" not in compact_arc["story_arcs"][0]


def test_run_unit_recomputes_when_prompt_hash_changes(tmp_path: Path) -> None:
    class Runner:
        model = "test-model"

        def __init__(self) -> None:
            self.calls = 0

        def call(self, system: str, user: str):
            self.calls += 1
            return {"value": self.calls}, {
                "status": "done",
                "model": self.model,
                "attempts": 1,
                "prompt_sha256": __import__("hashlib").sha256(
                    (system + "\n" + user).encode("utf-8")
                ).hexdigest(),
            }

    runner = Runner()
    output = tmp_path / "unit.json"
    run_unit(
        output_path=output,
        prompt_builder=lambda: ("system", "first"),
        runner=runner,
        unit_id="unit",
        stage="test",
    )
    run_unit(
        output_path=output,
        prompt_builder=lambda: ("system", "second"),
        runner=runner,
        unit_id="unit",
        stage="test",
    )
    assert runner.calls == 2


def test_hierarchical_batches_are_bounded_and_ordered() -> None:
    from scripts.build_moviebench_cineworldsim_pilot import _batches

    assert _batches(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_pilot_script_imports_when_called_by_file_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_moviebench_cineworldsim_pilot.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Build L1-L4 CineWorldSim artifacts" in result.stdout
