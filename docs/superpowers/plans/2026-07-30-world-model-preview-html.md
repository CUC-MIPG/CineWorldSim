# World Model Preview HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static HTML preview page that shows the story line, the world assets the agent planned to create, and side-by-side image comparisons for `flux_local` and `nano_banana`.

**Architecture:** Add one small generator script that reads `world_model_result.json` plus the existing comparison manifest and emits a self-contained static HTML file under the output directory. Add one regression test that feeds a tiny synthetic payload into the generator and asserts the required sections and labels exist in the generated HTML.

**Tech Stack:** Python 3, stdlib `json`/`html`/`pathlib`, existing project output JSON files, `pytest`

---

### Task 1: Add HTML preview regression test

**Files:**
- Create: `tests/test_generate_world_model_preview_html.py`
- Test: `tests/test_generate_world_model_preview_html.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.generate_world_model_preview_html import generate_preview_html


def test_generate_preview_html_contains_story_assets_and_comparison(tmp_path: Path) -> None:
    result_path = tmp_path / "world_model_result.json"
    manifest_path = tmp_path / "comparison_manifest.json"
    output_path = tmp_path / "preview" / "index.html"
    (tmp_path / "images").mkdir()
    for name in ["flux.png", "nano.png", "board.png"]:
        (tmp_path / "images" / name).write_bytes(b"stub")

    result_path.write_text(
        "{\"storyline\":\"Storm story\",\"world_model\":{\"summary\":\"World summary\",\"locations\":[{\"name\":\"Lighthouse\",\"description\":\"Tall tower\",\"evidence\":\"Main clue\",\"image_prompt\":\"storm lighthouse\",\"images\":[{\"tool\":\"flux_local\",\"ok\":true,\"output_path\":\"images/flux.png\",\"error\":\"\"},{\"tool\":\"nano_banana\",\"ok\":true,\"output_path\":\"images/nano.png\",\"error\":\"\"}]}],\"items\":[{\"name\":\"Compass\",\"description\":\"Old compass\",\"evidence\":\"Needed clue\",\"image_prompt\":\"old compass\",\"images\":[{\"tool\":\"flux_local\",\"ok\":true,\"output_path\":\"images/flux.png\",\"error\":\"\"},{\"tool\":\"nano_banana\",\"ok\":true,\"output_path\":\"images/nano.png\",\"error\":\"\"}]}]}}",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "[{\"group\":\"location\",\"name\":\"Lighthouse\",\"flux_path\":\"images/flux.png\",\"nano_path\":\"images/nano.png\",\"nano_source\":\"world_model_run\",\"board_path\":\"images/board.png\"}]",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_world_model_preview_html.py -q`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `scripts/generate_world_model_preview_html.py` does not exist yet.

### Task 2: Implement the preview generator

**Files:**
- Create: `scripts/generate_world_model_preview_html.py`
- Modify: `scripts/generate_world_model_preview_html.py`

- [ ] **Step 1: Write minimal implementation**

```python
def generate_preview_html(*, result_path: Path, manifest_path: Path, output_path: Path) -> Path:
    ...
```

Implementation requirements:
- Read `world_model_result.json`
- Read `comparison_manifest.json`
- Render three sections: `Storyline`, `Planned Assets`, `Method Comparison`
- Split assets into `Locations` and `Items`
- Render side-by-side `flux_local` and `nano_banana` images with relative paths
- Write `index.html` under the requested output path

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_generate_world_model_preview_html.py -q`
Expected: PASS

### Task 3: Generate the real preview page

**Files:**
- Use: `outputs/world_model_black_tidemark_flux2_vs_banana_20260730/world_model_result.json`
- Use: `outputs/world_model_black_tidemark_flux2_vs_banana_20260730/comparison_boards/comparison_manifest.json`
- Create: `outputs/world_model_black_tidemark_flux2_vs_banana_20260730/preview/index.html`

- [ ] **Step 1: Run the generator on the real output**

Run:

```bash
python scripts/generate_world_model_preview_html.py \
  --result-json outputs/world_model_black_tidemark_flux2_vs_banana_20260730/world_model_result.json \
  --comparison-manifest outputs/world_model_black_tidemark_flux2_vs_banana_20260730/comparison_boards/comparison_manifest.json \
  --output-html outputs/world_model_black_tidemark_flux2_vs_banana_20260730/preview/index.html
```

Expected: prints the generated HTML path and creates the preview file.

- [ ] **Step 2: Smoke check the generated HTML**

Run:

```bash
rg -n "Storyline|Planned Assets|Method Comparison|Abandoned Lighthouse|Cracked Brass Compass" \
  outputs/world_model_black_tidemark_flux2_vs_banana_20260730/preview/index.html
```

Expected: matching lines for the required sections and representative assets.
