from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

from agents.video_agent import VideoAgent
from agents.skills.adapter import SkillAdapter


def _img(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (30, 50, 70)).save(path)
    return str(path)


def _attach_shot_planner(agent: VideoAgent, count: int, durations: list[float] | None = None) -> None:
    class FakeShotPlanner:
        def generate(self, *, node, **kwargs):
            del kwargs
            weights = durations or [1.0] * count
            return {
                "status": "done",
                "duration_sec": 10,
                "num_frames": 243,
                "shots": [
                    {
                        "shot_index": index,
                        "duration_sec": weight,
                        "video_description": f"Planned beat {index}: {node.get('event_summary', '')}",
                        "cinematography": "controlled camera movement",
                        "dialogue": node.get("dialogue") or "none",
                        "bgm": "subtle score",
                    }
                    for index, weight in enumerate(weights, start=1)
                ],
            }

    agent.shot_planning_tool = FakeShotPlanner()


def test_process_orchestrated_events_collects_character_and_world_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A test story.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "A character reaches a glowing room.",
                        "event_type": "information_reveal",
                        "story_phase": "development",
                        "source_steps": [1],
                        "source_node_ids": ["step::1::Hero"],
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {"storyline": "A test story.", "ordered_node_ids": ["NN_001"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
    )

    assert summary["total"] == 1
    assert summary["success"] == 1
    task_json = out_dir / "event_video_tasks.json"
    tasks = json.loads(task_json.read_text(encoding="utf-8"))
    assert tasks[0]["node_id"] == "NN_001"
    assert tasks[0]["reference_images"] == [char_img, world_img]
    assert "character identity" in tasks[0]["prompt"].lower()
    assert tasks[0]["skill_name"] == "cameraatist-event-video-prompt"
    assert tasks[0]["prompt_style"] == "compact"
    timeline_path = Path(tasks[0]["timeline_path"])
    assert timeline_path.name == "event_video_001_NN_001.timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert timeline["video_filename"] == "event_video_001_NN_001.mp4"
    assert timeline["segments"] == []


def test_node_reference_images_resolves_repaired_image_extension(tmp_path):
    repaired = _img(tmp_path / "step_001" / "Hero_event_keyframe.jpg")
    stale = str(tmp_path / "step_001" / "Hero_event_keyframe.png")

    refs = VideoAgent._node_reference_images(
        {
            "visual_assets": {
                "character_keyframes": [stale],
                "event_keyframes": [stale],
            }
        },
        include_character_keyframes=True,
        include_world_images=True,
        max_reference_images=4,
    )

    assert refs == [repaired]


def test_node_reference_images_caps_character_refs_without_multi_person_flag(tmp_path):
    hero = _img(tmp_path / "hero.png")
    guard = _img(tmp_path / "guard.png")
    world = _img(tmp_path / "world.png")
    refs = VideoAgent._node_reference_images(
        {
            "visual_assets": {
                "character_keyframes": [hero, guard],
                "character_reference_images": [hero, guard],
                "world_update_images": [world],
                "multi_person_reference_allowed": False,
            }
        },
        include_character_keyframes=True,
        include_world_images=True,
        max_reference_images=4,
    )

    assert refs[0] == hero
    assert guard not in refs
    assert world in refs


def test_node_reference_images_allows_bounded_multi_person_refs(tmp_path):
    hero = _img(tmp_path / "hero.png")
    guard = _img(tmp_path / "guard.png")
    world = _img(tmp_path / "world.png")
    refs = VideoAgent._node_reference_images(
        {
            "visual_assets": {
                "character_keyframes": [hero, guard],
                "character_reference_images": [hero, guard],
                "world_update_images": [world],
                "multi_person_reference_allowed": True,
            }
        },
        include_character_keyframes=True,
        include_world_images=True,
        max_reference_images=4,
    )

    assert hero in refs
    assert guard in refs
    assert world in refs


def test_node_reference_images_only_includes_mentioned_item_refs(tmp_path):
    hero = _img(tmp_path / "Hero_event_keyframe.png")
    room = _img(tmp_path / "world_updates" / "locations" / "Archive_Room" / "step_001.png")
    compass = _img(tmp_path / "world_updates" / "items" / "Cracked_Compass" / "step_001.png")
    radio = _img(tmp_path / "world_updates" / "items" / "Old_Radio_Set" / "step_001.png")

    refs = VideoAgent._node_reference_images(
        {
            "event_summary": "Hero studies the cracked compass beside the archive table.",
            "characters": ["Hero"],
            "locations": ["Archive Room"],
            "items": ["Cracked Compass", "Old Radio Set"],
            "visual_assets": {
                "character_keyframes": [hero],
                "location_images": [room],
                "item_images": [compass, radio],
                "multi_person_reference_allowed": False,
            },
        },
        include_character_keyframes=True,
        include_world_images=True,
        max_reference_images=4,
    )

    assert refs == [hero, room, compass]


def test_node_reference_images_does_not_fill_with_unmentioned_world_items(tmp_path):
    hero = _img(tmp_path / "Hero_event_keyframe.png")
    room = _img(tmp_path / "world_updates" / "locations" / "Archive_Room" / "step_001.png")
    compass = _img(tmp_path / "world_updates" / "items" / "Cracked_Compass" / "step_001.png")
    radio = _img(tmp_path / "world_updates" / "items" / "Old_Radio_Set" / "step_001.png")

    refs = VideoAgent._node_reference_images(
        {
            "event_summary": "Hero crosses the archive room and listens near the sealed door.",
            "characters": ["Hero"],
            "locations": ["Archive Room"],
            "items": ["Cracked Compass", "Old Radio Set"],
            "visual_assets": {
                "character_keyframes": [hero],
                "location_images": [room],
                "item_images": [compass, radio],
                "multi_person_reference_allowed": False,
            },
        },
        include_character_keyframes=True,
        include_world_images=True,
        max_reference_images=4,
    )

    assert refs == [hero, room]


def test_event_video_prompt_can_use_h3_official_ref_style(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    prompt = VideoAgent._event_video_prompt(
        {
            "story_description": "Hero enters a glowing room and lifts a lantern toward the sealed door.",
            "event_summary": "Hero enters a glowing room and lifts a lantern toward the sealed door.",
            "video_description": "Hero enters the room, then pauses to examine the sealed door.",
            "event_type": "information_reveal",
            "story_phase": "development",
            "characters": ["Hero"],
            "locations": ["Glowing Room"],
            "items": ["Lantern"],
        },
        storyline="A test story.",
    )
    assert "subject_definitions:" in prompt
    assert "summary:" in prompt
    assert "retention_analysis:" in prompt
    assert "detailed_description:" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert "<Subject 1>" in prompt
    assert "[reference generation]" in prompt
    assert "story_description:" in prompt
    assert "video_description:" not in prompt
    assert "cinematography:" in prompt
    assert "dialogue:" in prompt
    assert "bgm:" in prompt
    assert "visual_constraints:" in prompt.lower()
    assert "people_policy:" in prompt.lower()
    assert "pedestrians" in prompt.lower()
    assert "unrelated people" in prompt.lower()

def test_event_video_prompt_filters_unmentioned_item_subjects(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    prompt = VideoAgent._event_video_prompt(
        {
            "story_description": "Hero crosses the archive room and listens near the sealed door.",
            "event_summary": "Hero crosses the archive room and listens near the sealed door.",
            "video_description": "Hero moves through the archive room without using any handheld prop.",
            "event_type": "suspense_action",
            "story_phase": "development",
            "characters": ["Hero"],
            "locations": ["Archive Room"],
            "items": ["Cracked Compass", "Old Radio Set"],
        },
        storyline="A test story.",
    )

    assert "Cracked Compass" not in prompt
    assert "Old Radio Set" not in prompt
    assert "do not fill unused reference slots" in prompt.lower()


def test_event_video_prompt_includes_only_mentioned_item_subject(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    prompt = VideoAgent._event_video_prompt(
        {
            "story_description": "Hero kneels beside the old radio and adjusts its tuning knob.",
            "event_summary": "Hero uses the old radio to catch a hidden signal.",
            "video_description": "Hero reaches for the radio and reacts to the signal.",
            "event_type": "information_reveal",
            "story_phase": "development",
            "characters": ["Hero"],
            "locations": ["Archive Room"],
            "items": ["Cracked Compass", "Old Radio Set"],
        },
        storyline="A test story.",
    )

    assert "Old Radio Set" in prompt
    assert "Cracked Compass" not in prompt
    assert "do not fill unused reference slots" in prompt.lower()


def test_process_orchestrated_events_records_official_ref_skill_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A skill metadata test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero enters a glowing room and lifts a lantern.",
                        "event_type": "information_reveal",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Glowing Room"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {"storyline": "A skill metadata test.", "ordered_node_ids": ["NN_001"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
    )

    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[0]["skill_name"] == "h3-prompt-writing"
    assert tasks[0]["skill_kind"]
    assert tasks[0]["skill_source"]
    assert tasks[0]["prompt_style"] == "official_ref"


def test_process_orchestrated_events_persists_story_description_without_overwriting_video_description(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A story description backfill test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero reaches the door.",
                        "video_description": "Hero walks to the door and listens.",
                        "event_type": "information_reveal",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Hallway"],
                        "items": ["Door"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {"storyline": "A story description backfill test.", "ordered_node_ids": ["NN_001"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeStoryTool:
        def generate(self, **kwargs):
            del kwargs
            return {
                "status": "done",
                "story_description": "Hero steps toward the door and listens for movement inside.",
                "artifact_path": "",
                "meta_path": "",
                "error": "",
            }

    agent = VideoAgent.__new__(VideoAgent)
    agent.story_tool = FakeStoryTool()
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
    )

    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[0]["video_description"] == "Hero walks to the door and listens."
    assert tasks[0]["story_description"] == "Hero steps toward the door and listens for movement inside."
    assert "Hero steps toward the door and listens for movement inside." in tasks[0]["prompt"]


def test_story_backfill_can_be_disabled_for_offline_dry_run(tmp_path):
    nodes_json = tmp_path / "nodes.json"
    plan_json = tmp_path / "plan.json"
    nodes_json.write_text(json.dumps({"selected_nodes": [{
        "node_id": "NN_001", "event_summary": "A character waits.",
        "story_description": "A character waits by the door.",
        "visual_assets": {"event_keyframes": [str(_img(tmp_path / "frame.png"))]},
    }]}), encoding="utf-8")
    plan_json.write_text(json.dumps({"ordered_node_ids": ["NN_001"]}), encoding="utf-8")

    class ForbiddenStoryTool:
        def generate(self, **kwargs):
            raise AssertionError("story backfill must not run when disabled")

    agent = VideoAgent.__new__(VideoAgent)
    agent.story_tool = ForbiddenStoryTool()
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json, path_plan_json=plan_json,
        output_dir=tmp_path / "videos", dry_run=True, story_backfill=False,
        dialogue_backfill=False,
    )

    assert summary["total"] == 1
    task = json.loads((tmp_path / "videos/event_video_tasks.json").read_text())[0]
    assert task["story_backfill"]["status"] == "disabled"


def test_process_orchestrated_events_uses_previous_tail_frame_and_audio_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A two-event continuity test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero sees a lantern flicker in the room.",
                        "event_type": "information_reveal",
                        "story_phase": "setup",
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    },
                    {
                        "node_id": "NN_002",
                        "event_id": "EVT_002",
                        "event_summary": "Hero follows the lantern into a hidden corridor.",
                        "event_type": "world_state_change",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Corridor"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {
                "storyline": "A two-event continuity test.",
                "ordered_node_ids": ["NN_001", "NN_002"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_send(self, **kwargs):
        calls.append(kwargs)
        Path(kwargs["video_path"]).write_bytes(b"mp4")
        return True

    def fake_extract(video_path, tail_frame_path, audio_path):
        Path(tail_frame_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tail_frame_path).write_bytes(b"jpg")
        Path(audio_path).write_bytes(b"wav")
        return {"tail_frame_path": tail_frame_path, "audio_path": audio_path}

    monkeypatch.setattr(VideoAgent, "send_generation_request", fake_send)
    monkeypatch.setattr(VideoAgent, "_extract_video_continuity_assets", staticmethod(fake_extract))
    monkeypatch.setattr(
        VideoAgent,
        "_judge_previous_tail_frame",
        lambda self, previous_node, current_node: {
            "use_previous_tail_frame": True,
            "continuity_score": 0.92,
            "reason": "The same character and lantern continue into the next location.",
        },
    )

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=False,
        use_previous_tail_frame=True,
        use_previous_audio=True,
        upload_previous_audio=False,
    )

    assert summary["success"] == 2
    assert len(calls) == 2
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[0]["continuity_reference"]["used_previous_tail_frame"] is False
    assert tasks[1]["continuity_reference"]["used_previous_tail_frame"] is True
    assert Path(tasks[1]["continuity_reference"]["previous_tail_frame_path"]).exists()
    assert Path(tasks[1]["continuity_reference"]["previous_audio_path"]).exists()
    assert tasks[1]["reference_images"][0] == tasks[1]["continuity_reference"]["previous_tail_frame_path"]
    assert calls[1]["reference_audio_paths"] == []
    assert tasks[1]["continuity_reference"]["continuity_score"] == 0.92
    assert "same character" in tasks[1]["continuity_reference"]["continuity_reason"]


def test_previous_tail_frame_is_not_used_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"
    nodes = [
        {
            "node_id": "NN_001",
            "event_id": "EVT_001",
            "event_summary": "Hero waits beside the lantern.",
            "characters": ["Hero"],
            "locations": ["Room"],
            "items": ["Lantern"],
            "visual_assets": {"character_keyframes": [char_img], "world_update_images": [world_img]},
        },
        {
            "node_id": "NN_002",
            "event_id": "EVT_002",
            "event_summary": "Hero carries the lantern into the corridor.",
            "characters": ["Hero"],
            "locations": ["Corridor"],
            "items": ["Lantern"],
            "visual_assets": {"character_keyframes": [char_img], "world_update_images": [world_img]},
        },
    ]
    nodes_json.write_text(json.dumps({"storyline": "Default continuity behavior.", "selected_nodes": nodes}), encoding="utf-8")
    plan_json.write_text(json.dumps({"ordered_node_ids": ["NN_001", "NN_002"]}), encoding="utf-8")

    def fake_send(self, **kwargs):
        Path(kwargs["video_path"]).write_bytes(b"mp4")
        return True

    def fake_extract(video_path, tail_frame_path, audio_path):
        Path(tail_frame_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tail_frame_path).write_bytes(b"jpg")
        return {"tail_frame_path": tail_frame_path, "audio_path": ""}

    def unexpected_judge(*args, **kwargs):
        raise AssertionError("continuity judge must not run when the option is disabled")

    monkeypatch.setattr(VideoAgent, "send_generation_request", fake_send)
    monkeypatch.setattr(VideoAgent, "_extract_video_continuity_assets", staticmethod(fake_extract))
    monkeypatch.setattr(VideoAgent, "_judge_previous_tail_frame", unexpected_judge)

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=False,
        use_previous_tail_frame=False,
    )

    assert summary["success"] == 2
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[1]["continuity_reference"]["used_previous_tail_frame"] is False
    assert tasks[1]["reference_images"][0] != tasks[1]["continuity_reference"]["previous_tail_frame_path"]


def test_judge_previous_tail_frame_parses_structured_llm_decision(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "use_previous_tail_frame": True,
                                    "continuity_score": 0.88,
                                    "reason": "The same character continues holding the same object.",
                                }
                            )
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    agent = VideoAgent.__new__(VideoAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model = "gpt-5.2"

    decision = agent._judge_previous_tail_frame(
        {
            "event_summary": "Hero finds the lantern.",
            "characters": ["Hero"],
            "locations": ["Room"],
            "items": ["Lantern"],
        },
        {
            "event_summary": "Hero carries the lantern into the corridor.",
            "characters": ["Hero"],
            "locations": ["Corridor"],
            "items": ["Lantern"],
        },
    )

    assert decision["use_previous_tail_frame"] is True
    assert decision["continuity_score"] == 0.88
    assert "same character" in decision["reason"]
    assert "previous event" in completions.calls[0]["messages"][0]["content"].lower()
    assert "current event" in completions.calls[0]["messages"][0]["content"].lower()
    assert "default decision is no" in completions.calls[0]["messages"][0]["content"].lower()
    assert "character identity" in completions.calls[0]["messages"][0]["content"].lower()
    assert "location" in completions.calls[0]["messages"][0]["content"].lower()


def test_judge_previous_tail_frame_fails_closed_on_invalid_llm_response(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("temporary LLM failure")

    agent = VideoAgent.__new__(VideoAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    agent.model = "gpt-5.2"

    decision = agent._judge_previous_tail_frame({"event_summary": "A"}, {"event_summary": "B"})

    assert decision["use_previous_tail_frame"] is False
    assert decision["continuity_score"] == 0.0
    assert decision["reason"]


def test_judge_previous_tail_frame_fails_closed_on_malformed_json():
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

    agent = VideoAgent.__new__(VideoAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    agent.model = "gpt-5.2"

    decision = agent._judge_previous_tail_frame({"event_summary": "A"}, {"event_summary": "B"})

    assert decision["use_previous_tail_frame"] is False
    assert decision["continuity_score"] == 0.0


def test_process_orchestrated_events_skips_existing_event_videos_when_resuming(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "compact")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"
    existing_video = out_dir / "event_video_001_NN_001.mp4"
    existing_video.parent.mkdir(parents=True, exist_ok=True)
    existing_video.write_bytes(b"existing-mp4")

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A resume test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero sees the first clue.",
                        "event_type": "information_reveal",
                        "story_phase": "setup",
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Clue"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    },
                    {
                        "node_id": "NN_002",
                        "event_id": "EVT_002",
                        "event_summary": "Hero follows the clue into a corridor.",
                        "event_type": "world_state_change",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Corridor"],
                        "items": ["Clue"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps({"storyline": "A resume test.", "ordered_node_ids": ["NN_001", "NN_002"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    calls = []

    def fake_send(self, **kwargs):
        calls.append(kwargs)
        Path(kwargs["video_path"]).write_bytes(b"new-mp4")
        return True

    def fake_extract(video_path, tail_frame_path, audio_path):
        Path(tail_frame_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tail_frame_path).write_bytes(b"jpg")
        Path(audio_path).write_bytes(b"wav")
        return {"tail_frame_path": tail_frame_path, "audio_path": audio_path}

    monkeypatch.setattr(VideoAgent, "send_generation_request", fake_send)
    monkeypatch.setattr(VideoAgent, "_extract_video_continuity_assets", staticmethod(fake_extract))
    monkeypatch.setattr(
        VideoAgent,
        "_judge_previous_tail_frame",
        lambda self, previous_node, current_node: {
            "use_previous_tail_frame": True,
            "continuity_score": 0.9,
            "reason": "The clue and protagonist continue into the corridor.",
        },
    )

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=False,
        use_previous_tail_frame=True,
        use_previous_audio=True,
    )

    assert summary["success"] == 2
    assert summary["failed"] == 0
    assert len(calls) == 1
    assert calls[0]["video_path"].endswith("event_video_002_NN_002.mp4")
    assert existing_video.read_bytes() == b"existing-mp4"
    assert summary["generated"][0]["status"] == "done"
    assert summary["generated"][0]["resume_existing"] is True
    assert summary["generated"][1]["status"] == "done"
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    assert tasks[1]["continuity_reference"]["used_previous_tail_frame"] is True


def test_extract_video_continuity_assets_uses_reverse_filter_for_tail_frame(tmp_path, monkeypatch):
    source = tmp_path / "event.mp4"
    source.write_bytes(b"mp4")
    tail = tmp_path / "tail.jpg"
    audio = tmp_path / "audio.wav"
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        output = Path(cmd[-1])
        output.write_bytes(b"asset")

    monkeypatch.setattr("agents.video_agent.subprocess.run", fake_run)

    result = VideoAgent._extract_video_continuity_assets(str(source), str(tail), str(audio))

    assert result["tail_frame_path"] == str(tail)
    assert result["audio_path"] == str(audio)
    assert "-vf" in commands[0]
    assert commands[0][commands[0].index("-vf") + 1] == "reverse"


def test_process_orchestrated_events_can_expand_event_into_multi_shot_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A cinematic event split into multiple shots.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero discovers the lantern and realizes the room is moving.",
                        "event_type": "information_reveal",
                        "story_phase": "escalation",
                        "characters": ["Hero"],
                        "locations": ["Moving Room"],
                        "items": ["Lantern"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps(
            {"storyline": "A cinematic event split into multiple shots.", "ordered_node_ids": ["NN_001"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    agent = VideoAgent.__new__(VideoAgent)
    _attach_shot_planner(agent, 4, [2.5] * 4)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        multi_shot_events=True,
        event_shot_count=4,
        event_duration_sec=8.0,
    )

    assert summary["total"] == 1
    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    task = tasks[0]
    assert task["mode"] == "multi_shot_reference"
    assert len(task["event_shot_plan"]) == 4
    assert len(task["shot_video_tasks"]) == 4
    assert task["duration_sec"] == 10
    assert task["num_frames"] == 243
    assert task["event_shot_plan"][1]["start_timecode"] == "00:02.500"
    assert task["event_shot_plan"][1]["duration_sec"] == 2.5
    assert "integrated_multimodal description:" in task["shot_video_tasks"][0]["prompt"]
    assert "[Shot 1]" in task["shot_video_tasks"][0]["prompt"]
    assert "At 00:02.500" in task["shot_video_tasks"][1]["prompt"]
    assert "video_description:" in task["shot_video_tasks"][0]["prompt"]
    assert "cinematography:" in task["shot_video_tasks"][0]["prompt"]
    assert "dialogue:" in task["shot_video_tasks"][0]["prompt"]
    assert "bgm:" in task["shot_video_tasks"][0]["prompt"]


def test_multi_shot_event_generation_calls_video_backend_once_with_combined_shot_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    monkeypatch.setenv("MINIMAX_H3_NUM_FRAMES", "243")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A two-shot event generation test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero opens the door and sees rain entering the room.",
                        "event_type": "world_state_change",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Door"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps({"storyline": "A two-shot event generation test.", "ordered_node_ids": ["NN_001"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    calls = []

    def fake_send(self, **kwargs):
        calls.append(kwargs)
        Path(kwargs["video_path"]).write_bytes(b"mp4")
        return True

    def fake_concat(self, video_paths, output_path):
        Path(output_path).write_bytes(b"event")
        return True

    def fake_extract(video_path, tail_frame_path, audio_path):
        Path(tail_frame_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tail_frame_path).write_bytes(b"jpg")
        Path(audio_path).write_bytes(b"wav")
        return {"tail_frame_path": tail_frame_path, "audio_path": audio_path}

    monkeypatch.setattr(VideoAgent, "send_generation_request", fake_send)
    monkeypatch.setattr(VideoAgent, "concat_videos", fake_concat)
    monkeypatch.setattr(VideoAgent, "_extract_video_continuity_assets", staticmethod(fake_extract))

    agent = VideoAgent.__new__(VideoAgent)
    _attach_shot_planner(agent, 2, [4, 6])
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=False,
        multi_shot_events=True,
        event_shot_count=2,
        event_duration_sec=6.0,
    )

    assert summary["success"] == 1
    assert len(calls) == 1
    assert calls[0]["duration_sec"] == 10
    assert calls[0]["num_frames"] == 243
    assert "[Shot 1]" in calls[0]["prompt"]
    assert "[Shot 2]" in calls[0]["prompt"]
    assert "At 00:04.000" in calls[0]["prompt"]
    assert Path(out_dir / "event_video_001_NN_001.mp4").read_bytes() == b"mp4"


def test_multi_shot_event_generation_routes_final_prompt_through_skill_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A multi-shot adapter routing test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero opens the door and sees rain entering the room.",
                        "event_type": "world_state_change",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Door"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps({"storyline": "A multi-shot adapter routing test.", "ordered_node_ids": ["NN_001"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    modes: list[str] = []
    original_build = SkillAdapter.build

    def wrapped_build(self, agent_name, skill, context):
        modes.append(str(context.get("mode") or ""))
        return original_build(self, agent_name, skill, context)

    monkeypatch.setattr(SkillAdapter, "build", wrapped_build)

    agent = VideoAgent.__new__(VideoAgent)
    _attach_shot_planner(agent, 2)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        multi_shot_events=True,
        event_shot_count=2,
        event_duration_sec=6.0,
    )

    assert summary["success"] == 1
    assert modes.count("multi_shot_segment") == 2
    assert "multi_shot_event" in modes
    assert modes[-1] == "multi_shot_event"


def test_multi_shot_event_generation_uses_h3_ref_multishot_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A H3 multi-shot prompt test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Hero opens the door and sees rain entering the room.",
                        "event_type": "world_state_change",
                        "story_phase": "development",
                        "characters": ["Hero"],
                        "locations": ["Room"],
                        "items": ["Door"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps({"storyline": "A H3 multi-shot prompt test.", "ordered_node_ids": ["NN_001"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    agent = VideoAgent.__new__(VideoAgent)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        multi_shot_events=True,
        event_shot_count=3,
        event_duration_sec=9.0,
    )

    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    prompt = tasks[0]["prompt"]
    assert tasks[0]["skill_name"] == "h3-prompt-writing"
    assert "subject_definitions:" in prompt
    assert "summary:" in prompt
    assert "retention_analysis:" in prompt
    assert "detailed_description:" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert prompt.count("[Shot ") >= 3


def test_process_orchestrated_events_backfills_dialogue_from_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    char_img = _img(tmp_path / "char.png")
    world_img = _img(tmp_path / "world.png")
    nodes_json = tmp_path / "narrative_nodes.json"
    plan_json = tmp_path / "narrative_path_plan.json"
    out_dir = tmp_path / "videos"

    nodes_json.write_text(
        json.dumps(
            {
                "storyline": "A dialogue backfill test.",
                "selected_nodes": [
                    {
                        "node_id": "NN_001",
                        "event_id": "EVT_001",
                        "event_summary": "Chef asks the radio host to broadcast the demolition notice.",
                        "event_type": "collaboration",
                        "story_phase": "development",
                        "characters": ["LateShiftChef", "RadioHost"],
                        "locations": ["Rooftop Diner"],
                        "items": ["Demolition Notice"],
                        "visual_assets": {
                            "character_keyframes": [char_img],
                            "world_update_images": [world_img],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan_json.write_text(
        json.dumps({"storyline": "A dialogue backfill test.", "ordered_node_ids": ["NN_001"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    class FakeDialogueTool:
        def generate(self, *, node, storyline, output_dir, rank, node_id, base_prompt_context):
            del node, storyline, base_prompt_context
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact = output_dir / f"event_dialogue_{rank:03d}_{node_id}.json"
            artifact.write_text(
                json.dumps({"Character1": "We need to put this on air.", "Character2": "none"}, ensure_ascii=False),
                encoding="utf-8",
            )
            return {
                "status": "done",
                "dialogue": {"Character1": "We need to put this on air.", "Character2": "none"},
                "slot_mapping": {"Character1": "LateShiftChef", "Character2": "RadioHost"},
                "rendered_dialogue": "Character1: We need to put this on air.",
                "artifact_path": str(artifact),
                "meta_path": "",
                "error": "",
            }

    agent = VideoAgent.__new__(VideoAgent)
    _attach_shot_planner(agent, 2)
    agent.dialogue_tool = FakeDialogueTool()
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=nodes_json,
        path_plan_json=plan_json,
        output_dir=out_dir,
        dry_run=True,
        multi_shot_events=True,
        event_shot_count=2,
        event_duration_sec=6.0,
    )

    assert summary["success"] == 1
    tasks = json.loads((out_dir / "event_video_tasks.json").read_text(encoding="utf-8"))
    task = tasks[0]
    assert task["dialogue_backfill"]["status"] == "done"
    assert Path(task["dialogue_backfill"]["artifact_path"]).exists()
    assert task["dialogue_backfill"]["rendered_dialogue"] == "Character1: We need to put this on air."
    assert "Character1: We need to put this on air." in task["shot_video_tasks"][0]["prompt"]
    assert "<d>[English] We need to put this on air.</d>" in task["prompt"]
    assert "Character2: none" not in task["prompt"]


def test_orchestrated_video_cli_exposes_redesign_flags():
    import scripts.run_orchestrated_event_video_generation as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert 'default=None' in source[source.index('"--event-duration-sec"'):source.index('"--dialogue-backfill"')]
    assert '"--insert-auxiliary-videos"' in source
    assert '"--video-reference-mode"' in source
    assert 'insert_auxiliary_videos=args.insert_auxiliary_videos' in source
    assert 'video_reference_mode=args.video_reference_mode' in source
