from __future__ import annotations

from pathlib import Path

from PIL import Image


def _img(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 9), (12, 34, 56)).save(path)
    return str(path)


def test_event_reference_selector_uses_only_role_location_and_mentioned_items(tmp_path: Path) -> None:
    from narrative.asset_reference_selector import select_event_reference_images

    mara = _img(tmp_path / "portraits" / "Mara_portrait.png")
    ivo = _img(tmp_path / "portraits" / "Ivo_portrait.png")
    archive = _img(tmp_path / "locations" / "Flooded_Archive" / "medium.png")
    radio_room = _img(tmp_path / "locations" / "Radio_Room" / "wide.png")
    letter = _img(tmp_path / "items" / "Sealed_Letter" / "initial.png")
    compass = _img(tmp_path / "items" / "Compass" / "initial.png")
    state = {
        "locations": [
            {
                "id": "Flooded_Archive",
                "name": "Flooded Archive",
                "last_image_path": archive,
                "views": {"medium": {"last_image_path": archive}},
            },
            {"id": "Radio_Room", "name": "Radio Room", "last_image_path": radio_room},
        ],
        "items": [
            {"id": "Sealed_Letter", "name": "Sealed Letter", "last_image_path": letter},
            {"id": "Compass", "name": "Compass", "last_image_path": compass},
        ],
    }
    event = {
        "characters": ["Mara"],
        "locations": ["Flooded Archive"],
        "items": ["Sealed Letter", "Compass"],
        "story_description": "Mara opens the sealed letter in the flooded archive.",
    }

    bundle = select_event_reference_images(
        event=event,
        current_state=state,
        character_portraits={"Mara": mara, "Ivo": ivo},
        max_reference_images=4,
    )

    assert bundle["reference_image_paths"] == [mara, archive, letter]
    assert bundle["character_reference_paths"] == [mara]
    assert bundle["location_reference_paths"] == [archive]
    assert bundle["item_reference_paths"] == [letter]
    assert compass in bundle["excluded_assets"]
    assert ivo in bundle["excluded_assets"]


def test_event_reference_selector_allows_multiple_characters_only_for_interaction(tmp_path: Path) -> None:
    from narrative.asset_reference_selector import select_event_reference_images

    mara = _img(tmp_path / "Mara_portrait.png")
    ivo = _img(tmp_path / "Ivo_portrait.png")
    archive = _img(tmp_path / "archive.png")
    state = {
        "locations": [{"id": "Flooded_Archive", "name": "Flooded Archive", "last_image_path": archive}],
        "items": [],
    }
    event = {
        "characters": ["Mara", "Ivo"],
        "locations": ["Flooded Archive"],
        "event_summary": "Mara confronts Ivo and asks him to open the hidden gate.",
        "node_granularity": "multi_role_interaction",
    }

    bundle = select_event_reference_images(
        event=event,
        current_state=state,
        character_portraits={"Mara": mara, "Ivo": ivo},
        max_reference_images=4,
    )

    assert bundle["character_reference_paths"] == [mara, ivo]
    assert bundle["reference_image_paths"] == [mara, ivo, archive]
