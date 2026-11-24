from __future__ import annotations

from pathlib import Path

from scripts.run_moviebench_arc_video_eval import build_arc_nodes, run_all_arcs


def test_build_arc_nodes_uses_l4_events_and_resolves_l3_context() -> None:
    arc = {
        "l4_story_arc_id": "movie_arc_01",
        "short_storyline_L": "A short arc.",
        "key_objects": ["Bed", "Blinds"],
        "essential_events": [
            {
                "event": "Jane sits up and looks toward the camera.",
                "source_l3_ids": ["scene_01"],
                "source_l2_ids": ["event_01"],
            }
        ],
    }
    l3_scenes = {
        "scene_01": [
            {
                "l3_scene_id": "scene_01",
                "scene_setting": "Later bedroom.",
                "scene_narrative": "Lester changes expression.",
                "active_characters": ["Lester"],
                "evidence_trace": {"source_l2_ids": ["event_99"]},
            },
            {
                "l3_scene_id": "scene_01",
                "scene_setting": "Dim bedroom with closed blinds.",
                "scene_narrative": "Jane lies down on the bed, then sits up with blinds closed.",
                "active_characters": ["Jane"],
                "evidence_trace": {"source_l2_ids": ["event_01"]},
                "narrative_state_transition": {
                    "key_objects_and_states": ["Bed present.", "Blinds remain closed."]
                },
            },
        ]
    }

    nodes, storyline = build_arc_nodes(arc, l3_scenes)

    assert storyline == "A short arc."
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "movie_arc_01_scene_001"
    assert nodes[0]["event_summary"] == "Jane lies down on the bed, then sits up with blinds closed."
    assert nodes[0]["characters"] == ["Jane"]
    assert nodes[0]["locations"] == ["Dim bedroom with closed blinds."]
    assert nodes[0]["items"] == ["Bed", "Blinds"]
    assert nodes[0]["source_l3_ids"] == ["scene_01"]


def test_build_arc_nodes_expands_l4_events_to_distinct_l3_scenes_in_time_order() -> None:
    arc = {
        "l4_story_arc_id": "movie_arc_01",
        "key_objects": ["Alarm clock", "Rose", "Clippers"],
        "essential_events": [
            {"event": "A combined sequence.", "source_l3_ids": ["scene_02", "scene_01"], "source_l2_ids": ["l2a", "l2b"]}
        ],
    }
    l3_scenes = {
        "scene_01": [{
            "l3_scene_id": "scene_01",
            "time_range": {"start_ms": 100},
            "scene_narrative": "An alarm clock rings in the bedroom.",
            "scene_setting": "Bedroom.",
            "evidence_trace": {"source_l2_ids": ["l2a"]},
        }],
        "scene_02": [{
            "l3_scene_id": "scene_02",
            "time_range": {"start_ms": 200},
            "scene_narrative": "A rose is cut with clippers.",
            "scene_setting": "Garden.",
            "evidence_trace": {"source_l2_ids": ["l2b"]},
        }],
    }

    nodes, _ = build_arc_nodes(arc, l3_scenes)

    assert [node["event_summary"] for node in nodes] == [
        "An alarm clock rings in the bedroom.",
        "A rose is cut with clippers.",
    ]
    assert nodes[0]["items"] == ["Alarm clock"]
    assert nodes[1]["items"] == ["Rose", "Clippers"]
    assert nodes[0]["source_l4_event_ids"] == ["movie_arc_01_event_001"]


def test_build_arc_nodes_fails_on_unresolved_l3_reference() -> None:
    arc = {
        "l4_story_arc_id": "movie_arc_01",
        "essential_events": [{"event": "An event.", "source_l3_ids": ["missing"]}],
    }

    try:
        build_arc_nodes(arc, {})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unresolved L3 references must fail fast")


def test_build_arc_nodes_fails_when_duplicate_l3_id_cannot_be_disambiguated() -> None:
    arc = {
        "l4_story_arc_id": "movie_arc_01",
        "essential_events": [{"event": "An event.", "source_l3_ids": ["scene_01"], "source_l2_ids": ["missing"]}],
    }
    l3_scenes = {
        "scene_01": [
            {"l3_scene_id": "scene_01", "evidence_trace": {"source_l2_ids": ["event_01"]}},
            {"l3_scene_id": "scene_01", "evidence_trace": {"source_l2_ids": ["event_02"]}},
        ]
    }

    try:
        build_arc_nodes(arc, l3_scenes)
    except ValueError as exc:
        assert "no source L1 shot evidence" in str(exc)
    else:
        raise AssertionError("ambiguous duplicate L3 IDs must fail fast")


def test_build_arc_nodes_resolves_stale_l3_aliases_through_l1_evidence() -> None:
    arc = {
        "l4_story_arc_id": "movie_arc_01",
        "essential_events": [{
            "event": "A visitor enters the room.",
            "source_l3_ids": ["stale_scene_alias"],
            "source_l2_ids": ["event_01"],
        }],
    }
    l3_scenes = {
        "current_scene": [{
            "l3_scene_id": "current_scene",
            "scene_narrative": "A visitor enters the room.",
            "scene_setting": "A room.",
            "evidence_trace": {
                "source_l2_ids": ["different_l2_segmentation"],
                "source_l1_ids": ["shot_01"],
            },
        }]
    }

    nodes, _ = build_arc_nodes(
        arc,
        l3_scenes,
        source_l1_by_l2={"event_01": ["shot_01"]},
    )

    assert len(nodes) == 1
    assert nodes[0]["source_l3_ids"] == ["current_scene"]
    assert nodes[0]["source_l4_event_ids"] == ["movie_arc_01_event_001"]
    assert nodes[0]["l3_resolution"] == [{
        "method": "shared_l1_shot_evidence",
        "source_l4_l3_references": ["stale_scene_alias"],
        "shared_l1_shot_ids": ["shot_01"],
    }]


def test_run_all_arcs_creates_isolated_outputs_and_aggregate_summary(tmp_path, monkeypatch) -> None:
    arcs_path = tmp_path / "arcs.json"
    scenes_path = tmp_path / "scenes.json"
    arcs_path.write_text(
        '{"result":{"story_arcs":['
        '{"l4_story_arc_id":"movie_arc_01","essential_events":[{"event":"A.","source_l3_ids":["s1"],"source_l2_ids":["e1"]}]},'
        '{"l4_story_arc_id":"movie_arc_02","essential_events":[{"event":"B.","source_l3_ids":["s2"],"source_l2_ids":["e2"]}]}'
        ']}}',
        encoding="utf-8",
    )
    scenes_path.write_text(
        '{"result":{"l3_scenes":['
        '{"l3_scene_id":"s1","scene_narrative":"A.","evidence_trace":{"source_l2_ids":["e1"]}},'
        '{"l3_scene_id":"s2","scene_narrative":"B.","evidence_trace":{"source_l2_ids":["e2"]}}'
        ']}}',
        encoding="utf-8",
    )

    def fake_run(args):
        output = Path(args.output_dir)
        output.mkdir(parents=True)
        (output / "moviebench_arc_video_eval_summary.json").write_text(
            '{"status":"dry_run","event_count":1,"planned_video_jobs":2}', encoding="utf-8"
        )
        return {"status": "dry_run", "event_count": 1, "planned_video_jobs": 2}

    monkeypatch.setattr("scripts.run_moviebench_arc_video_eval.run", fake_run)
    args = type("Args", (), {
        "l4_arcs": arcs_path,
        "l3_scenes": scenes_path,
        "output_dir": tmp_path / "all",
        "all_arcs_output_dir": tmp_path / "all",
        "movie_id": "movie",
        "run": False,
    })()

    summary = run_all_arcs(args)

    assert summary["arc_count"] == 2
    assert summary["event_count"] == 2
    assert summary["planned_video_jobs"] == 4
    assert (tmp_path / "all" / "movie_arc_01" / "moviebench_arc_video_eval_summary.json").is_file()
    assert (tmp_path / "all" / "movie_arc_02" / "moviebench_arc_video_eval_summary.json").is_file()
