from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_world_model_preview_html import generate_preview_html


def test_generate_preview_html_contains_story_assets_and_comparison(tmp_path: Path) -> None:
    result_path = tmp_path / "world_model_result.json"
    manifest_path = tmp_path / "comparison_manifest.json"
    output_path = tmp_path / "preview" / "index.html"

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ["flux.png", "nano.png", "board.png"]:
        (images_dir / name).write_bytes(b"stub")

    result_path.write_text(
        (
            '{"storyline":"Storm story","world_model":{"summary":"World summary",'
            '"locations":[{"name":"Lighthouse","description":"Tall tower","evidence":"Main clue",'
            '"image_prompt":"storm lighthouse","images":[{"tool":"flux_local","ok":true,'
            '"output_path":"images/flux.png","error":""},{"tool":"nano_banana","ok":true,'
            '"output_path":"images/nano.png","error":""}]}],'
            '"items":[{"name":"Compass","description":"Old compass","evidence":"Needed clue",'
            '"image_prompt":"old compass","images":[{"tool":"flux_local","ok":true,'
            '"output_path":"images/flux.png","error":""},{"tool":"nano_banana","ok":true,'
            '"output_path":"images/nano.png","error":""}]}]}}'
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        (
            '[{"group":"location","name":"Lighthouse","flux_path":"images/flux.png",'
            '"nano_path":"images/nano.png","nano_source":"world_model_run",'
            '"board_path":"images/board.png"}]'
        ),
        encoding="utf-8",
    )

    generate_preview_html(result_path=result_path, manifest_path=manifest_path, output_path=output_path)
    html_text = output_path.read_text(encoding="utf-8")

    assert "Storyline" in html_text
    assert "Planned Assets" in html_text
    assert "Method Comparison" in html_text
    assert "Storm story" in html_text
    assert "Lighthouse" in html_text
    assert "Compass" in html_text
    assert "flux_local" in html_text
    assert "nano_banana" in html_text
