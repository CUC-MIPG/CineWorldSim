from scripts.run_vimax_track_ab_pipeline import (
    build_track_a_nodes,
    load_vimax_case,
    resolve_track_b_characters,
    split_character_names,
)


def test_vimax_case_is_mapped_to_track_a_shot_nodes(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(
        '{"story_overview":"A story.","consistency_type":"Type A",'
        '"scenes":[{"scene_num":2,"shots":[{"shot_id":7,'
        '"first_frame":"A first frame.","video_prompt":"A video."}]}]}',
        encoding="utf-8",
    )

    payload = load_vimax_case(case)
    nodes = build_track_a_nodes(payload)

    assert nodes[0]["node_id"] == "scene_002_shot_007"
    assert nodes[0]["anchor_l3_id"] == "scene_002"
    assert nodes[0]["first_frame_prompt"] == "A first frame."
    assert nodes[0]["video_description"] == "A video."
    assert nodes[0]["visual_assets"]["event_keyframes"] == []


def test_character_names_have_stable_cli_parsing():
    assert split_character_names(" Artist,  Subject,Artist ,, ") == ["Artist", "Subject"]


class FakeCharacterTool:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def run_json(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_track_b_discovers_and_deduplicates_case_characters():
    case = {
        "story_overview": "A painter completes a landscape.",
        "scenes": [{"shots": [{
            "first_frame": "The artist paints beside the easel.",
            "video_prompt": "A friend brings her water.",
        }]}],
    }
    tool = FakeCharacterTool({"characters": ["The Artist", "Friend", "The Artist"]})

    names, source = resolve_track_b_characters(case, None, tool=tool)

    assert names == ["The Artist", "Friend"]
    assert source == "llm_extracted"
    prompt = tool.kwargs["user_prompt"]
    assert "A painter completes a landscape." in prompt
    assert "The artist paints beside the easel." in prompt
    assert "A friend brings her water." in prompt


def test_track_b_rejects_empty_character_extraction():
    tool = FakeCharacterTool({"characters": []})

    try:
        resolve_track_b_characters({"story_overview": "A story.", "scenes": []}, None, tool=tool)
    except ValueError as exc:
        assert "no character names" in str(exc).lower()
    else:
        raise AssertionError("empty extraction must fail")


def test_track_b_cli_character_override_skips_llm_extraction():
    names, source = resolve_track_b_characters(
        {"story_overview": "A story."}, " Artist, Friend,Artist ", tool=None
    )

    assert names == ["Artist", "Friend"]
    assert source == "cli_override"
