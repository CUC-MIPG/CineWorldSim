import json
from pathlib import Path

import pytest

from scripts.restructure_cineworldsim_hierarchy import (
    build_l4_generation_prompt,
    restructure_dataset,
    validate_generated_l4,
)


def _write_unit(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unit_id": "unit", "stage": "test", "status": "ok", "result": result, "meta": {}}, indent=2))


def _source_tree(root: Path) -> None:
    _write_unit(root / "l4_story/movie_bible.json", {"movie_id": "film_01", "movie_synopsis": "whole film"})
    _write_unit(root / "l4_story/story_arcs.json", {
        "movie_id": "film_01",
        "story_arcs": [{
            "l4_story_arc_id": "film_01_arc_01",
            "arc_title": "A local arc",
            "l3_scene_ids": ["scene_01"],
            "l2_event_ids": ["event_01"],
            "l1_shot_ids": ["shot_01"],
            "essential_events": [{"event": "A person enters", "source_l3_ids": ["scene_01"], "source_l2_ids": ["event_01"]}],
        }],
    })
    _write_unit(root / "l3_scenes/final.json", {"movie_id": "film_01", "l3_scenes": [{"l3_scene_id": "scene_01", "l2_event_ids": ["event_01"]}]})
    _write_unit(root / "l2_events/units/events.json", {"l2_events": [{"l2_event_id": "event_01", "source_shot_ids": ["shot_01"]}]})
    _write_unit(root / "l1_shots/units/shot_01.json", {"l1_shot_id": "shot_01"})
    (root / "track_a.json").write_text(json.dumps({"track": "A", "samples": []}))
    (root / "track_b.json").write_text(json.dumps({"track": "B", "samples": []}))


def test_restructure_promotes_movie_bible_and_demotes_arcs_without_mutating_sources(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)

    report = restructure_dataset(source, target)

    global_story = json.loads((target / "l4_global/movie_bible.json").read_text())
    arcs_doc = json.loads((target / "l3_arcs/story_arcs.json").read_text())
    arc = arcs_doc["result"]["story_arcs"][0]
    assert global_story["result"]["movie_synopsis"] == "whole film"
    assert arc["l3_story_arc_id"] == "film_01_L3ARC_001"
    assert arc["source_l4_story_arc_id"] == "film_01_arc_01"
    assert arc["source_scene_occurrence_ids"] == ["scene_01"]
    assert arc["l2_event_ids"] == ["event_01"]
    assert arc["l1_shot_ids"] == ["shot_01"]
    assert report["validation"]["unresolved_l2_event_ids"] == []
    assert json.loads((source / "l4_story/story_arcs.json").read_text())["result"]["story_arcs"][0]["l4_story_arc_id"] == "film_01_arc_01"
    assert (target / "l2_events/units/events.json").exists()
    assert (target / "l1_shots/units/shot_01.json").exists()


def test_restructure_rejects_unresolved_event_references(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)
    arcs_path = source / "l4_story/story_arcs.json"
    payload = json.loads(arcs_path.read_text())
    payload["result"]["story_arcs"][0]["l2_event_ids"] = ["missing_event"]
    arcs_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unresolved_l2_event_ids"):
        restructure_dataset(source, target)


def test_restructure_refuses_existing_output_directory(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)
    target.mkdir()

    with pytest.raises(FileExistsError):
        restructure_dataset(source, target)


def test_allow_unresolved_keeps_source_ids_and_reports_them(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)
    path = source / "l4_story/story_arcs.json"
    payload = json.loads(path.read_text())
    payload["result"]["story_arcs"][0]["l1_shot_ids"] = ["missing_shot"]
    path.write_text(json.dumps(payload))

    report = restructure_dataset(source, target, allow_unresolved_references=True)

    converted = json.loads((target / "l3_arcs/story_arcs.json").read_text())
    assert converted["result"]["story_arcs"][0]["l1_shot_ids"] == ["missing_shot"]
    assert report["validation"]["status"] == "incomplete"
    assert report["validation"]["unresolved_l1_shot_ids"] == ["missing_shot"]


def test_l4_prompt_contains_prior_and_compact_scene_event_ids() -> None:
    system, user = build_l4_generation_prompt(
        "film_01",
        {"movie_id": "film_01", "movie_synopsis": {"short_synopsis": "prior"}},
        [{"l3_scene_id": "scene_01", "l2_event_ids": ["event_01"], "scene_narrative": "scene"}],
        [{"l2_event_id": "event_01", "source_shot_ids": ["shot_01"], "event_description": "event"}],
    )
    assert "film_01" in user
    assert "scene_01" in user
    assert "event_01" in user
    assert "shot_01" in user
    assert "movie_synopsis" in user
    assert "Return valid JSON only" in system


def test_l4_prompt_bounds_large_evidence_and_keeps_causal_event_fields() -> None:
    scenes = [
        {
            "l3_scene_id": f"scene_{index:03d}",
            "l2_event_ids": [f"event_{index:03d}_{event_index:02d}" for event_index in range(20)],
            "scene_narrative": "scene narrative " + ("x" * 600),
            "event_progression": ["progression " + ("y" * 300)] * 10,
            "narrative_state_transition": {"before": "b" * 300, "after": "a" * 300},
        }
        for index in range(143)
    ]
    events = [
        {
            "l2_event_id": f"event_{index:04d}",
            "source_shot_ids": [f"shot_{index:04d}_{shot_index:02d}" for shot_index in range(5)],
            "event_title": "event title " + ("t" * 200),
            "event_description": "event description " + ("d" * 1000),
            "main_characters": ["character " + ("c" * 50)] * 5,
            "setting": "setting " + ("s" * 300),
            "actions": [{"actor": "actor", "action": "q" * 300}] * 5,
            "state_changes": [{"before": "b" * 300, "after": "a" * 300}] * 3,
        }
        for index in range(598)
    ]

    _, user = build_l4_generation_prompt(
        "film_01",
        {"movie_id": "film_01", "movie_synopsis": "whole film"},
        scenes,
        events,
    )

    assert len(user) < 250_000
    assert "actions, state_changes" in user


def test_validate_generated_l4_requires_global_storyline_and_synopsis() -> None:
    valid = validate_generated_l4(
        {
            "movie_id": "film_01",
            "movie_synopsis": {"short_synopsis": "whole film"},
            "global_storyline": "A complete global storyline.",
            "evidence_trace": {"source_l3_ids": [], "source_l2_ids": [], "source_l1_ids": []},
        },
        "film_01",
    )
    assert valid["movie_id"] == "film_01"
    with pytest.raises(ValueError, match="global_storyline"):
        validate_generated_l4({"movie_id": "film_01", "movie_synopsis": "x"}, "film_01")


def test_restructure_generate_l4_writes_llm_bible_and_audits(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)
    response = {
        "movie_id": "film_01",
        "movie_synopsis": {"short_synopsis": "generated whole film"},
        "global_storyline": "Generated global storyline.",
        "evidence_trace": {"source_l3_ids": ["scene_01"], "source_l2_ids": ["event_01"], "source_l1_ids": ["shot_01"]},
    }

    class FakeTool:
        def __init__(self, **kwargs):
            self.model = "fake-model"

        def run_json(self, **kwargs):
            return response

    monkeypatch.setattr("scripts.restructure_cineworldsim_hierarchy.LLMJsonTool", FakeTool)
    report = restructure_dataset(source, target, generate_l4=True)
    generated = json.loads((target / "l4_global/movie_bible.json").read_text())
    assert generated["result"]["global_storyline"] == "Generated global storyline."
    assert generated["result"]["generation_method"] == "llm_compaction"
    assert (target / "l4_global/l4_generation_request.json").exists()
    assert (target / "l4_global/l4_generation_response.json").exists()
    assert report["l4_generation"]["method"] == "llm_compaction"


def test_default_restructure_does_not_construct_llm_tool(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)

    def fail_if_constructed(**kwargs):
        raise AssertionError("LLM tool must not be constructed in default mode")

    monkeypatch.setattr("scripts.restructure_cineworldsim_hierarchy.LLMJsonTool", fail_if_constructed)
    report = restructure_dataset(source, target)

    assert report["l4_generation"]["method"] == "source_copy"
    bible = json.loads((target / "l4_global/movie_bible.json").read_text())
    assert bible["result"]["movie_synopsis"] == "whole film"
    assert not (target / "l4_global/l4_generation_request.json").exists()


def test_generate_l4_explicit_fallback_records_failure(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)

    class FailedTool:
        model = "fake-model"

        def __init__(self, **kwargs):
            pass

        def run_json(self, **kwargs):
            raise RuntimeError("API unavailable")

    monkeypatch.setattr("scripts.restructure_cineworldsim_hierarchy.LLMJsonTool", FailedTool)
    report = restructure_dataset(source, target, generate_l4=True, fallback_existing_bible=True)

    bible = json.loads((target / "l4_global/movie_bible.json").read_text())
    response = json.loads((target / "l4_global/l4_generation_response.json").read_text())
    assert bible["result"]["generation_method"] == "source_fallback"
    assert report["l4_generation"]["fallback_used"] is True
    assert response["status"] == "fallback"
    assert "API unavailable" in response["error"]
