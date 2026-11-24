import json
from pathlib import Path

from build_macro_events import build_macro_events, audit_hierarchy


def wrapped(path: Path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"result": result}), encoding="utf-8")


def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "movie"
    wrapped(root / "l2_events/units/seq.json", {
        "l2_events": [
            {"l2_event_id": "e1", "source_shot_ids": ["s1"], "event_title": "walk", "event_description": "walk", "main_characters": ["A"], "setting": "room", "locations": ["room"], "actions": ["walk"], "state_changes": [], "time_range": {"start_ms": 0, "end_ms": 2500, "duration_sec": 2.5}},
            {"l2_event_id": "e2", "source_shot_ids": ["s2"], "event_title": "talk", "event_description": "talk", "main_characters": ["A"], "setting": "room", "locations": ["room"], "actions": ["talk"], "state_changes": [], "time_range": {"start_ms": 2600, "end_ms": 5000, "duration_sec": 2.4}},
            {"l2_event_id": "e3", "source_shot_ids": ["s3"], "event_title": "leave", "event_description": "leave", "main_characters": ["B"], "setting": "street", "locations": ["street"], "actions": ["leave"], "state_changes": ["A leaves"], "time_range": {"start_ms": 10000, "end_ms": 18000, "duration_sec": 8.0}},
        ]
    })
    wrapped(root / "l1_shots/units/s1.json", {"l1_shot_id": "s1", "time_range": {"start_ms": 0, "end_ms": 2500}})
    wrapped(root / "l1_shots/units/s2.json", {"l1_shot_id": "s2", "time_range": {"start_ms": 2600, "end_ms": 5000}})
    wrapped(root / "l1_shots/units/s3.json", {"l1_shot_id": "s3", "time_range": {"start_ms": 10000, "end_ms": 18000}})
    wrapped(root / "source_context/l3_scene_occurrences.json", {"l3_scenes": [
        {"l3_scene_id": "scene", "l2_event_ids": ["e1", "e2"], "time_range": {"start_ms": 0, "end_ms": 5000}, "scene_setting": "room", "active_characters": ["A"], "scene_narrative": "A acts in a room", "evidence_trace": {"source_l2_ids": ["e1", "e2"], "source_l1_ids": ["s1", "s2"]}},
        {"l3_scene_id": "scene", "l2_event_ids": ["e3", "missing"], "time_range": {"start_ms": 10000, "end_ms": 18000}, "scene_setting": "street", "active_characters": ["B"], "scene_narrative": "B leaves", "evidence_trace": {"source_l2_ids": ["e3", "missing"], "source_l1_ids": ["s3", "missing-shot"]}},
    ]})
    wrapped(root / "l3_arcs/story_arcs.json", {"story_arcs": [{"l3_story_arc_id": "arc1", "l2_event_ids": ["e1", "e2", "e3"], "l1_shot_ids": ["s1", "s2", "s3"], "source_scene_occurrence_ids": ["scene"]}]})
    wrapped(root / "l4_global/movie_bible.json", {"movie_id": "movie", "evidence_trace": {"source_l2_ids": ["e1", "e3"], "source_l1_ids": ["s1", "missing-shot"]}})
    return root


def test_audit_uses_composite_occurrence_keys_and_reports_unresolved(tmp_path):
    root = fixture_root(tmp_path)
    report = audit_hierarchy(root)
    assert report["counts"] == {"l4": 1, "l3_arcs": 1, "l2": 3, "l1": 3, "source_occurrences": 2}
    assert report["unresolved_references"]["l2"] == ["missing"]
    assert report["unresolved_references"]["l1"] == ["missing-shot"]
    assert report["duplicate_occurrence_ids"] == ["scene"]


def test_build_macro_events_merges_continuous_events_and_preserves_break(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path / "out"
    result = build_macro_events(root, output, max_duration_sec=15, merge_gap_sec=1.0)
    payload = json.loads((output / "macro_events.json").read_text())
    assert len(payload["macro_events"]) == 2
    assert payload["macro_events"][0]["source_l2_ids"] == ["e1", "e2"]
    assert payload["macro_events"][1]["source_l2_ids"] == ["e3"]
    assert result["audit"]["unresolved_references"]["l1"] == ["missing-shot"]


def test_max_duration_splits_continuous_bundle(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path / "out"
    build_macro_events(root, output, max_duration_sec=3.0, merge_gap_sec=1.0)
    payload = json.loads((output / "macro_events.json").read_text())
    assert len(payload["macro_events"]) == 3


def test_summary_counts_macro_events_without_arc_mapping(tmp_path):
    root = fixture_root(tmp_path)
    arc_path = root / "l3_arcs/story_arcs.json"
    arc_payload = json.loads(arc_path.read_text())
    arc_payload["result"]["story_arcs"][0]["l2_event_ids"] = ["e1", "e2"]
    arc_path.write_text(json.dumps(arc_payload), encoding="utf-8")

    output = tmp_path / "out"
    build_macro_events(root, output, max_duration_sec=15, merge_gap_sec=1.0)

    macro_events = json.loads((output / "macro_events.json").read_text())["macro_events"]
    summary = json.loads((output / "macro_event_summary.json").read_text())
    unmapped = [row for row in macro_events if not row["l3_story_arc_ids"]]
    assert [row["source_l2_ids"] for row in unmapped] == [["e3"]]
    assert summary["macro_events_without_l3_arc_mapping"] == 1
