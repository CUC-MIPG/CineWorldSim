from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.event_keyframe_renderer import EventKeyframeRenderer


def test_event_keyframe_renderer_dry_run_creates_role_keyframe(tmp_path: Path) -> None:
    renderer = EventKeyframeRenderer(backend_mode="all_banana", dry_run=True)
    role_log = {
        "role_name": "Journalist",
        "next_action": {
            "action": "opens the evidence box",
            "location": "Archive",
            "reason": "to verify the conspiracy",
        },
    }

    result = renderer.render_event_keyframe(step=2, role_log=role_log, output_dir=tmp_path)

    assert result["ok"] is True
    assert result["tool"] == "nano_banana"
    assert Path(result["output_path"]).exists()
    assert "Journalist" in result["prompt"]
    assert "people_policy:" in result["prompt"].lower()
    assert "pedestrians" in result["prompt"].lower()
    assert "unrelated people" in result["prompt"].lower()


def test_event_keyframe_renderer_uses_reference_images_when_provided(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_generate_image_nano_banana(**kwargs):
        calls.update(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"image")
        return str(kwargs["output_path"])

    monkeypatch.setattr("narrative.event_keyframe_renderer.generate_image_nano_banana", fake_generate_image_nano_banana)
    monkeypatch.setattr(
        "narrative.event_keyframe_renderer.nano_banana_settings_from_env",
        lambda: {"api_key": "x", "base_url": "http://example.com", "model_name": "model", "timeout_sec": 1},
    )

    portrait = tmp_path / "portrait.png"
    world = tmp_path / "world.png"
    portrait.write_bytes(b"portrait")
    world.write_bytes(b"world")

    renderer = EventKeyframeRenderer(backend_mode="all_banana", dry_run=False)
    role_log = {
        "role_name": "Journalist",
        "next_action": {
            "action": "opens the evidence box",
            "location": "Archive",
            "reason": "to verify the conspiracy",
        },
    }

    result = renderer.render_event_keyframe(
        step=2,
        role_log=role_log,
        output_dir=tmp_path,
        reference_image_paths=[str(portrait), str(world)],
    )

    assert result["ok"] is True
    assert calls["reference_image_paths"] == [str(portrait), str(world)]


def test_event_keyframe_renderer_reuses_existing_keyframe(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "step_002" / "Journalist_event_keyframe.png"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"existing keyframe")

    def fail_generate(**kwargs):
        raise AssertionError("backend should not be called for an existing keyframe")

    monkeypatch.setattr("narrative.event_keyframe_renderer.generate_image_nano_banana", fail_generate)
    monkeypatch.setattr(
        "narrative.event_keyframe_renderer.nano_banana_settings_from_env",
        lambda: {"api_key": "x", "base_url": "http://example.com", "model_name": "model", "timeout_sec": 1},
    )

    renderer = EventKeyframeRenderer(backend_mode="all_banana", dry_run=False)
    result = renderer.render_event_keyframe(
        step=2,
        role_log={"role_name": "Journalist", "next_action": {"action": "opens the evidence box"}},
        output_dir=tmp_path,
        reference_image_paths=[str(tmp_path / "world.png")],
    )

    assert result["ok"] is True
    assert result["output_path"] == str(out_path)
    assert result["reused_existing"] is True


def test_event_keyframe_renderer_retries_reference_generation(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    def flaky_generate_image_nano_banana(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary SSL failure")
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"image")
        return str(kwargs["output_path"])

    monkeypatch.setenv("NARRATIVE_IMAGE_RENDER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("NARRATIVE_IMAGE_RENDER_RETRY_SLEEP_SEC", "0")
    monkeypatch.setattr("narrative.event_keyframe_renderer.generate_image_nano_banana", flaky_generate_image_nano_banana)
    monkeypatch.setattr(
        "narrative.event_keyframe_renderer.nano_banana_settings_from_env",
        lambda: {"api_key": "x", "base_url": "http://example.com", "model_name": "model", "timeout_sec": 1},
    )
    ref = tmp_path / "world.png"
    ref.write_bytes(b"world")

    renderer = EventKeyframeRenderer(backend_mode="all_banana", dry_run=False)
    result = renderer.render_event_keyframe(
        step=2,
        role_log={"role_name": "Journalist", "next_action": {"action": "opens the evidence box"}},
        output_dir=tmp_path,
        reference_image_paths=[str(ref)],
    )

    assert result["ok"] is True
    assert result["attempts"] == 2
    assert calls["count"] == 2


def test_event_keyframe_prompt_prefers_story_description() -> None:
    prompt = EventKeyframeRenderer.build_prompt(
        step=3,
        role_log={"role_name": "Mara", "next_action": {"action": "opens the sealed letter", "location": "Flooded Archive"}},
        event_context={
            "story_description": "Mara kneels in the flooded archive and opens the sealed letter while water drips from the shelves.",
            "items": ["Sealed Letter"],
            "locations": ["Flooded Archive"],
        },
    )

    assert "Mara kneels in the flooded archive" in prompt
    assert "Sealed Letter" in prompt
    assert "Flooded Archive" in prompt
    assert "event keyframe" in prompt.lower()
