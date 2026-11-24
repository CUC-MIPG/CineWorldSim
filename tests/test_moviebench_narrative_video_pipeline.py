from __future__ import annotations

from scripts.run_moviebench_narrative_video_pipeline import (
    build_track_a_l1_nodes,
    build_track_b_nodes,
    track_b_generation_context,
    _global_story_and_characters,
    _index_units,
    _merge_manifest_track_runs,
)


def test_global_story_loader_reads_l4_story_object_and_character_field(tmp_path):
    l4_dir = tmp_path / "l4_global"
    l4_dir.mkdir()
    (l4_dir / "movie_bible.json").write_text(
        '{"result":{"global_storyline":{"story":"Global story"},'
        '"global_character_profiles":[{"character":"Jane"}]}}',
        encoding="utf-8",
    )

    story, characters = _global_story_and_characters(tmp_path)

    assert story == "Global story"
    assert characters == ["Jane"]


def test_unit_index_reads_nested_l1_shot_payload(tmp_path):
    (tmp_path / "shot.json").write_text(
        '{"unit_id":"shot","result":{"l1_shot_id":"SHOT_1","order_index":1}}',
        encoding="utf-8",
    )

    assert _index_units(tmp_path) == {"SHOT_1": {"l1_shot_id": "SHOT_1", "order_index": 1}}


def test_track_a_expands_l3_anchor_to_l2_l1_video_nodes_in_source_order():
    arc = {
        "l3_story_arc_id": "ARC_01",
        "arc_title": "Opening",
        "full_story_S": "Jane wakes and a rose is cut.",
        "l2_event_ids": ["E2", "E1"],
        "l1_shot_ids": ["S2", "S1"],
    }
    l2 = {
        "E1": {"l2_event_id": "E1", "event_description": "Jane lies still.", "source_shot_ids": ["S1"]},
        "E2": {"l2_event_id": "E2", "event_description": "Jane sits up.", "source_shot_ids": ["S2"]},
    }
    l1 = {
        "S1": {"l1_shot_id": "S1", "order_index": 1, "observable_facts": ["Jane lies down"]},
        "S2": {"l1_shot_id": "S2", "order_index": 2, "observable_facts": ["Jane sits up"]},
    }

    nodes = build_track_a_l1_nodes(arc, l2, l1, ordered_l2_ids=["E2", "E1"])

    assert [node["node_id"] for node in nodes] == ["S2", "S1"]
    assert all(node["anchor_l3_id"] == "ARC_01" for node in nodes)
    assert [node["source_l2_ids"] for node in nodes] == [["E2"], ["E1"]]
    assert nodes[0]["event_summary"] == "Jane sits up."


def test_track_a_nodes_inherit_shared_character_location_and_item_assets():
    arc = {
        "l3_story_arc_id": "ARC_01",
        "l2_event_ids": ["E1"],
        "l1_shot_ids": ["S1"],
    }
    l2 = {"E1": {"event_description": "Jane looks through blinds", "source_shot_ids": ["S1"]}}
    l1 = {
        "S1": {
            "l1_shot_id": "S1",
            "characters": [{"name": "Jane Burnham"}],
            "location_mentions": [{"name": "Burnham Kitchen"}],
            "item_mentions": [{"name": "Blinds"}],
        }
    }
    shared_assets = {
        "character_reference_images": {"Jane_Burnham": "/assets/jane.jpg"},
        "location_images": {"Burnham_kitchen": "/assets/kitchen.jpg"},
        "item_images": {"Blinds": "/assets/blinds.jpg"},
    }
    shared_network_nodes = [
        {
            "node_id": "N1",
            "event_summary": "Jane lies on her side in a dim room with closed blinds.",
            "characters": ["Jane Burnham"],
            "visual_assets": {
                "character_reference_images": ["/assets/jane-network.jpg"],
                "location_images": ["/assets/room-network.jpg"],
                "item_images": ["/assets/blinds-network.jpg"],
            },
        }
    ]

    nodes = build_track_a_l1_nodes(
        arc, l2, l1, shared_assets=shared_assets,
        shared_network_nodes=shared_network_nodes,
    )

    assets = nodes[0]["visual_assets"]
    assert assets["character_reference_images"] == ["/assets/jane-network.jpg", "/assets/jane.jpg"]
    assert assets["location_images"] == ["/assets/room-network.jpg", "/assets/kitchen.jpg"]
    assert assets["item_images"] == ["/assets/blinds-network.jpg", "/assets/blinds.jpg"]


def test_track_b_nodes_use_only_simulation_nodes_and_do_not_ingest_truth_fields():
    payload = {
        "selected_nodes": [
            {"node_id": "N1", "event_summary": "A character enters the room.", "characters": ["Jane"]},
        ],
        "withheld_event_ids": ["SECRET"],
        "reference_story": "SECRET reference ending",
    }

    nodes = build_track_b_nodes(payload)
    generation_context = track_b_generation_context("Short setup only.", ["Jane"], "Jane enters a room.")

    assert [node["node_id"] for node in nodes] == ["N1"]
    assert generation_context == {
        "storyline": "Short setup only.",
        "known_characters": ["Jane"],
        "known_facts": "Jane enters a room.",
    }
    assert "SECRET" not in repr(nodes) + repr(generation_context)


def test_resume_manifest_replaces_only_selected_track_rows():
    previous = {
        "dataset_root": "/dataset",
        "track_runs": [
            {"track": "A", "samples": [{"old": True}]},
            {"track": "B", "samples": [{"preserve": True}]},
        ],
    }
    updated = _merge_manifest_track_runs(
        previous, [{"track": "A", "samples": [{"new": True}]}]
    )

    assert updated["dataset_root"] == "/dataset"
    assert updated["track_runs"] == [
        {"track": "B", "samples": [{"preserve": True}]},
        {"track": "A", "samples": [{"new": True}]},
    ]
