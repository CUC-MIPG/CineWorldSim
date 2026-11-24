from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_renderer import WorldRenderer, backend_mode_to_tool


def test_backend_mode_to_tool() -> None:
    assert backend_mode_to_tool("all_flux") == "flux_local"
    assert backend_mode_to_tool("all_banana") == "nano_banana"


def test_world_renderer_dry_run_writes_placeholder(tmp_path: Path) -> None:
    renderer = WorldRenderer(backend_mode="all_flux", dry_run=True)
    out = renderer.render_world_asset(
        prompt="storm lighthouse",
        output_path=tmp_path / "Lighthouse" / "flux_local.png",
    )

    assert out["ok"] is True
    assert Path(out["output_path"]).exists()
    assert out["tool"] == "flux_local"


def test_world_renderer_reuses_existing_output_without_calling_backend(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "Lighthouse" / "nano_banana.png"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"existing image")

    def fail_render(*args, **kwargs):
        raise AssertionError("backend should not be called for an existing image")

    monkeypatch.setattr("narrative.simulation_scaffold.WorldModelAgent._render_with_tool", fail_render)

    renderer = WorldRenderer(backend_mode="all_banana", dry_run=False)
    out = renderer.render_world_asset(prompt="storm lighthouse", output_path=out_path)

    assert out["ok"] is True
    assert out["output_path"] == str(out_path)
    assert out["reused_existing"] is True
