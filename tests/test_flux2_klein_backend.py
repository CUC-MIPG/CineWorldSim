import base64
from pathlib import Path

from PIL import Image


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeResponse:
    status_code = 200
    content = PNG_BYTES
    headers = {"content-type": "image/png"}
    text = ""

    def raise_for_status(self) -> None:
        return None


def test_flux2_klein_sends_repeated_images_and_writes_png(monkeypatch, tmp_path: Path) -> None:
    from agents import generation_backends as backends

    refs = []
    for name in ("portrait.jpg", "location.png"):
        path = tmp_path / name
        path.write_bytes(b"reference")
        refs.append(str(path))
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(backends, "requests_post", fake_post)
    output = tmp_path / "result.png"
    result = backends.generate_image_flux2_klein(
        prompt="event keyframe",
        output_path=output,
        api_url="http://127.0.0.1:8000/generate",
        width=1536,
        height=864,
        num_inference_steps=4,
        seed=42,
        reference_image_paths=refs,
    )

    assert result == str(output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert captured["url"] == "http://127.0.0.1:8000/generate"
    assert captured["data"] == {
        "prompt": "event keyframe",
        "width": "1536",
        "height": "864",
        "num_inference_steps": "4",
        "seed": "42",
    }
    assert [field for field, _ in captured["files"]] == ["images", "images"]


def test_flux2_klein_settings_from_env(monkeypatch) -> None:
    from agents.generation_backends import flux2_klein_settings_from_env

    monkeypatch.setenv("FLUX2_KLEIN_API_URL", "http://example/generate")
    monkeypatch.setenv("FLUX2_KLEIN_HEALTH_URL", "http://example/health")
    monkeypatch.setenv("FLUX2_KLEIN_WIDTH", "1536")
    monkeypatch.setenv("FLUX2_KLEIN_HEIGHT", "864")
    monkeypatch.setenv("FLUX2_KLEIN_STEPS", "4")
    monkeypatch.setenv("FLUX2_KLEIN_SEED", "42")

    assert flux2_klein_settings_from_env() == {
        "api_url": "http://example/generate",
        "health_url": "http://example/health",
        "width": 1536,
        "height": 864,
        "num_inference_steps": 4,
        "seed": 42,
        "timeout_sec": 1200,
    }


def test_world_renderer_maps_flux2_klein_and_event_renderer_uses_it(monkeypatch, tmp_path: Path) -> None:
    from narrative.event_keyframe_renderer import EventKeyframeRenderer
    from narrative.world_renderer import WorldRenderer, backend_mode_to_tool

    assert backend_mode_to_tool("all_flux2_klein") == "flux2_klein"
    renderer = WorldRenderer(backend_mode="all_flux2_klein", dry_run=True)
    assert renderer.tool_name == "flux2_klein"

    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(PNG_BYTES)
        return str(kwargs["output_path"])

    monkeypatch.setattr("narrative.event_keyframe_renderer.generate_image_flux2_klein", fake_generate)
    monkeypatch.setattr(
        "narrative.event_keyframe_renderer.flux2_klein_settings_from_env",
        lambda: {
            "api_url": "http://127.0.0.1:8000/generate",
            "health_url": "http://127.0.0.1:8000/health",
            "width": 1536,
            "height": 864,
            "num_inference_steps": 4,
            "seed": 42,
            "timeout_sec": 1200,
        },
    )
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"reference")
    result = EventKeyframeRenderer(backend_mode="all_flux2_klein").render_event_keyframe(
        step=1,
        role_log={"role_name": "Mara", "next_action": {"action": "open the map"}},
        output_dir=tmp_path / "keyframes",
        reference_image_paths=[str(ref)],
    )

    assert result["ok"] is True
    assert calls["reference_image_paths"] == [str(ref)]


def test_vistorybench_flux_backend_generates_shot_png(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_vistorybench_pipeline as runner

    called = {}

    def fake_generate(**kwargs):
        called.update(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(PNG_BYTES)
        return str(kwargs["output_path"])

    monkeypatch.setattr(runner, "generate_image_flux2_klein", fake_generate)
    monkeypatch.setattr(
        runner,
        "flux2_klein_settings_from_env",
        lambda: {
            "api_url": "http://127.0.0.1:8000/generate",
            "health_url": "http://127.0.0.1:8000/health",
            "width": 1536,
            "height": 864,
            "num_inference_steps": 4,
            "seed": 42,
            "timeout_sec": 1200,
        },
    )
    output = tmp_path / "shot.png"
    result = runner._generate_vistorybench_image(
        image_backend="flux2_klein",
        prompt="benchmark shot",
        output_path=output,
        reference_image_paths=["reference.jpg"],
    )

    assert result == str(output)
    assert called["prompt"] == "benchmark shot"
    assert called["reference_image_paths"] == ["reference.jpg"]
