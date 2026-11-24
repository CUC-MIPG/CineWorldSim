# Dual-Backend Narrative World Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable pipeline that runs the same custom story through two complete narrative-world loops: one using FLUX for all images and one using nano_banana for all images.

**Architecture:** Keep the existing `OrchestratorAgent`, `WorldModelAgent`, and `StateAnalysisMachine` as the narrative core. Add small focused modules for structured world state, world delta judgment, backend rendering, event keyframes, and combined visualization, then expose everything through one CLI script.

**Tech Stack:** Python 3, dataclasses, stdlib JSON/HTML/pathlib, existing `agents.generation_backends`, pytest, existing `mifage` conda runtime for full integration runs

---

## File Structure

- Create `narrative/world_state_tracker.py`
  - Owns structured location/item state snapshots and serialization.
- Create `narrative/world_delta_judge.py`
  - Compares world snapshots and per-step events to decide whether world images need refresh.
- Create `narrative/world_renderer.py`
  - Provides one rendering interface for `all_flux` and `all_banana`.
- Create `narrative/event_keyframe_renderer.py`
  - Renders each role event keyframe with the selected backend.
- Create `scripts/generate_narrative_world_pipeline_html.py`
  - Renders final temporal and spatial overview HTML.
- Create `scripts/run_dual_backend_narrative_world_pipeline.py`
  - Runs one backend or both backends from the same story input.
- Create tests:
  - `tests/test_world_state_tracker.py`
  - `tests/test_world_delta_judge.py`
  - `tests/test_world_renderer.py`
  - `tests/test_event_keyframe_renderer.py`
  - `tests/test_generate_narrative_world_pipeline_html.py`
  - `tests/test_dual_backend_pipeline_cli.py`

---

### Task 1: World State Tracker

**Files:**
- Create: `narrative/world_state_tracker.py`
- Test: `tests/test_world_state_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_state_tracker import build_initial_world_state, update_world_state_from_delta


def test_build_initial_world_state_from_world_model() -> None:
    world_model = {
        "summary": "A storm island world.",
        "locations": [
            {
                "asset_id": "Lighthouse",
                "name": "Lighthouse",
                "description": "A broken coastal tower.",
                "images": [{"tool": "flux_local", "ok": True, "output_path": "/tmp/lighthouse.png"}],
            }
        ],
        "items": [
            {
                "asset_id": "Compass",
                "name": "Compass",
                "description": "A cracked compass.",
                "images": [{"tool": "flux_local", "ok": True, "output_path": "/tmp/compass.png"}],
            }
        ],
    }

    state = build_initial_world_state(world_model=world_model, backend_tool="flux_local")

    assert state["step"] == 0
    assert state["summary"] == "A storm island world."
    assert state["locations"][0]["id"] == "Lighthouse"
    assert state["locations"][0]["current_state_text"] == "A broken coastal tower."
    assert state["locations"][0]["last_image_path"] == "/tmp/lighthouse.png"
    assert state["items"][0]["id"] == "Compass"
    assert state["items"][0]["last_image_path"] == "/tmp/compass.png"


def test_update_world_state_from_delta_marks_changed_elements() -> None:
    previous = {
        "step": 0,
        "summary": "A storm island world.",
        "locations": [
            {
                "id": "Lighthouse",
                "name": "Lighthouse",
                "description": "A broken coastal tower.",
                "current_state_text": "Dark and locked.",
                "last_image_path": "/tmp/lighthouse.png",
                "changed_this_step": False,
            }
        ],
        "items": [],
    }
    delta = {
        "step": 1,
        "summary": "The lighthouse has been opened.",
        "location_updates": [
            {
                "id": "Lighthouse",
                "new_state_text": "The door is open and rain enters the stairwell.",
                "image_refresh_required": True,
                "reason": "door opened",
            }
        ],
        "item_updates": [],
    }

    state = update_world_state_from_delta(previous_state=previous, delta=delta)

    assert state["step"] == 1
    assert state["summary"] == "The lighthouse has been opened."
    assert state["locations"][0]["current_state_text"] == "The door is open and rain enters the stairwell."
    assert state["locations"][0]["changed_this_step"] is True
    assert state["locations"][0]["change_reason"] == "door opened"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_state_tracker.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'narrative.world_state_tracker'`.

- [ ] **Step 3: Implement `narrative/world_state_tracker.py`**

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _image_for_backend(images: list[dict[str, Any]], backend_tool: str) -> str:
    for image in images or []:
        if image.get("tool") == backend_tool and image.get("ok"):
            return str(image.get("output_path", "") or "")
    for image in images or []:
        if image.get("ok"):
            return str(image.get("output_path", "") or "")
    return ""


def _asset_row(asset: dict[str, Any], backend_tool: str) -> dict[str, Any]:
    description = str(asset.get("description", "") or "")
    return {
        "id": str(asset.get("asset_id") or asset.get("id") or asset.get("name") or ""),
        "name": str(asset.get("name", "") or ""),
        "description": description,
        "current_state_text": str(asset.get("current_state_text") or description),
        "last_image_path": _image_for_backend(asset.get("images", []), backend_tool),
        "changed_this_step": False,
        "change_reason": "",
        "image_history": [
            {
                "step": 0,
                "path": _image_for_backend(asset.get("images", []), backend_tool),
                "reason": "initial_world_model",
            }
        ],
    }


def build_initial_world_state(*, world_model: dict[str, Any], backend_tool: str) -> dict[str, Any]:
    return {
        "step": 0,
        "summary": str(world_model.get("summary", "") or ""),
        "locations": [_asset_row(asset, backend_tool) for asset in world_model.get("locations", [])],
        "items": [_asset_row(asset, backend_tool) for asset in world_model.get("items", [])],
    }


def _apply_updates(rows: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(update.get("id", "")): update for update in updates or []}
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = deepcopy(row)
        new_row["changed_this_step"] = False
        new_row["change_reason"] = ""
        update = by_id.get(str(row.get("id", "")))
        if update:
            new_row["current_state_text"] = str(update.get("new_state_text") or row.get("current_state_text") or "")
            new_row["changed_this_step"] = bool(update.get("image_refresh_required", False))
            new_row["change_reason"] = str(update.get("reason", "") or "")
            if update.get("new_image_path"):
                new_row["last_image_path"] = str(update["new_image_path"])
                history = list(new_row.get("image_history", []))
                history.append(
                    {
                        "step": int(update.get("step", 0) or 0),
                        "path": str(update["new_image_path"]),
                        "reason": str(update.get("reason", "") or "world_delta_refresh"),
                    }
                )
                new_row["image_history"] = history
        updated_rows.append(new_row)
    return updated_rows


def update_world_state_from_delta(*, previous_state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(previous_state)
    state["step"] = int(delta.get("step", previous_state.get("step", 0)))
    state["summary"] = str(delta.get("summary") or previous_state.get("summary") or "")
    state["locations"] = _apply_updates(state.get("locations", []), delta.get("location_updates", []))
    state["items"] = _apply_updates(state.get("items", []), delta.get("item_updates", []))
    return state


def write_world_state(path: Path, state: dict[str, Any]) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_world_state_tracker.py -q`

Expected: `2 passed`.

---

### Task 2: World Delta Judge

**Files:**
- Create: `narrative/world_delta_judge.py`
- Test: `tests/test_world_delta_judge.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_delta_judge import WorldDeltaJudge


def test_world_delta_judge_detects_location_and_item_changes() -> None:
    judge = WorldDeltaJudge(model_config=None)
    previous_state = {
        "step": 0,
        "locations": [{"id": "Lighthouse", "name": "Lighthouse", "current_state_text": "Locked door."}],
        "items": [{"id": "Compass", "name": "Compass", "current_state_text": "Hidden in a drawer."}],
    }
    step_log = {
        "step": 1,
        "role_logs": [
            {
                "role_name": "Journalist",
                "next_action": {
                    "action": "opens the lighthouse door and takes the compass",
                    "location": "Lighthouse",
                    "reason": "The compass is needed as evidence.",
                },
            }
        ],
        "sam_summary": "The lighthouse door is opened and the compass is taken.",
    }

    delta = judge.judge_step_delta(previous_state=previous_state, step_log=step_log)

    assert delta["step"] == 1
    assert delta["location_updates"][0]["id"] == "Lighthouse"
    assert delta["location_updates"][0]["image_refresh_required"] is True
    assert delta["item_updates"][0]["id"] == "Compass"
    assert delta["item_updates"][0]["image_refresh_required"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_delta_judge.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'narrative.world_delta_judge'`.

- [ ] **Step 3: Implement deterministic first-pass judge**

```python
from __future__ import annotations

from typing import Any


class WorldDeltaJudge:
    def __init__(self, model_config: Any = None) -> None:
        self.model_config = model_config

    @staticmethod
    def _step_text(step_log: dict[str, Any]) -> str:
        parts = [str(step_log.get("sam_summary", "") or "")]
        for role in step_log.get("role_logs", []) or []:
            action = role.get("next_action", {}) or {}
            parts.extend(str(action.get(key, "") or "") for key in ["action", "action_type", "location", "reason", "monologue"])
            parts.append(str(role.get("role_output_text", "") or ""))
        return " ".join(parts).lower()

    @staticmethod
    def _changed(name: str, text: str) -> bool:
        name_tokens = [part.lower() for part in str(name).replace("_", " ").split() if part]
        if not any(token in text for token in name_tokens):
            return False
        change_words = [
            "open",
            "opened",
            "break",
            "broken",
            "take",
            "taken",
            "steal",
            "stolen",
            "move",
            "moved",
            "discover",
            "discovered",
            "flood",
            "flooded",
            "burn",
            "burned",
            "destroy",
            "destroyed",
            "unlock",
            "unlocked",
        ]
        return any(word in text for word in change_words)

    def judge_step_delta(self, *, previous_state: dict[str, Any], step_log: dict[str, Any]) -> dict[str, Any]:
        step = int(step_log.get("step", previous_state.get("step", 0)) or 0)
        text = self._step_text(step_log)
        location_updates = []
        for loc in previous_state.get("locations", []) or []:
            if self._changed(str(loc.get("name") or loc.get("id")), text):
                location_updates.append(
                    {
                        "step": step,
                        "id": str(loc.get("id", "")),
                        "new_state_text": str(step_log.get("sam_summary") or loc.get("current_state_text") or ""),
                        "image_refresh_required": True,
                        "reason": "step_event_mentions_location_change",
                    }
                )
        item_updates = []
        for item in previous_state.get("items", []) or []:
            if self._changed(str(item.get("name") or item.get("id")), text):
                item_updates.append(
                    {
                        "step": step,
                        "id": str(item.get("id", "")),
                        "new_state_text": str(step_log.get("sam_summary") or item.get("current_state_text") or ""),
                        "image_refresh_required": True,
                        "reason": "step_event_mentions_item_change",
                    }
                )
        return {
            "step": step,
            "summary": str(step_log.get("sam_summary") or previous_state.get("summary") or ""),
            "location_updates": location_updates,
            "item_updates": item_updates,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_world_delta_judge.py -q`

Expected: `1 passed`.

---

### Task 3: Unified World Renderer

**Files:**
- Create: `narrative/world_renderer.py`
- Test: `tests/test_world_renderer.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative.world_renderer import backend_mode_to_tool, WorldRenderer


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_renderer.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'narrative.world_renderer'`.

- [ ] **Step 3: Implement `narrative/world_renderer.py`**

```python
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def backend_mode_to_tool(backend_mode: str) -> str:
    if backend_mode == "all_flux":
        return "flux_local"
    if backend_mode == "all_banana":
        return "nano_banana"
    raise ValueError(f"Unsupported backend mode: {backend_mode}")


class WorldRenderer:
    def __init__(self, *, backend_mode: str, dry_run: bool = False) -> None:
        self.backend_mode = backend_mode
        self.tool_name = backend_mode_to_tool(backend_mode)
        self.dry_run = dry_run

    def render_world_asset(self, *, prompt: str, output_path: Path) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            output_path.write_bytes(_ONE_PIXEL_PNG)
            return {"tool": self.tool_name, "ok": True, "output_path": str(output_path), "error": ""}

        from narrative.simulation_scaffold import WorldModelAgent

        result = WorldModelAgent()._render_with_tool(
            tool_name=self.tool_name,
            prompt=prompt,
            output_path=output_path,
        )
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_world_renderer.py -q`

Expected: `2 passed`.

---

### Task 4: Event Keyframe Renderer

**Files:**
- Create: `narrative/event_keyframe_renderer.py`
- Test: `tests/test_event_keyframe_renderer.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_keyframe_renderer.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'narrative.event_keyframe_renderer'`.

- [ ] **Step 3: Implement `narrative/event_keyframe_renderer.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from narrative.simulation_scaffold import _safe_name
from narrative.world_renderer import WorldRenderer


class EventKeyframeRenderer:
    def __init__(self, *, backend_mode: str, dry_run: bool = False) -> None:
        self.world_renderer = WorldRenderer(backend_mode=backend_mode, dry_run=dry_run)

    @staticmethod
    def build_prompt(*, step: int, role_log: dict[str, Any]) -> str:
        action = role_log.get("next_action", {}) or {}
        role_name = str(role_log.get("role_name", "") or "Character")
        return (
            f"Cinematic narrative keyframe for step {step}, character {role_name}. "
            f"Action: {action.get('action') or action.get('action_type') or ''}. "
            f"Location: {action.get('location') or ''}. "
            f"Reason: {action.get('reason') or ''}. "
            "16:9 film still, clear event readability, consistent character identity, high detail."
        )

    def render_event_keyframe(self, *, step: int, role_log: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        role_name = str(role_log.get("role_name", "") or "Character")
        out_path = output_dir / f"step_{step:03d}" / f"{_safe_name(role_name)}_event_keyframe.png"
        prompt = self.build_prompt(step=step, role_log=role_log)
        result = self.world_renderer.render_world_asset(prompt=prompt, output_path=out_path)
        result["prompt"] = prompt
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_keyframe_renderer.py -q`

Expected: `1 passed`.

---

### Task 5: Pipeline HTML Visualizer

**Files:**
- Create: `scripts/generate_narrative_world_pipeline_html.py`
- Test: `tests/test_generate_narrative_world_pipeline_html.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_narrative_world_pipeline_html import generate_pipeline_html


def test_generate_pipeline_html_contains_time_and_space_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "all_flux"
    run_dir.mkdir()
    payload = {
        "backend_mode": "all_flux",
        "story": "A storm mystery.",
        "characters": ["Journalist"],
        "world_states": [{"step": 0, "locations": [], "items": []}],
        "step_logs": [{"step": 1, "role_logs": [], "sam_summary": "A door opens."}],
        "event_keyframes": [],
        "world_deltas": [],
    }
    (run_dir / "pipeline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    output_html = tmp_path / "combined_overview.html"

    generate_pipeline_html(run_dirs=[run_dir], output_html=output_html)

    text = output_html.read_text(encoding="utf-8")
    assert "Narrative Time" in text
    assert "World Space" in text
    assert "Backend Comparison" in text
    assert "all_flux" in text
    assert "A storm mystery." in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_narrative_world_pipeline_html.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_narrative_world_pipeline_html'`.

- [ ] **Step 3: Implement static HTML generator**

```python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "pipeline_result.json").read_text(encoding="utf-8"))


def _render_run(payload: dict[str, Any]) -> str:
    backend = html.escape(str(payload.get("backend_mode", "")))
    story = html.escape(str(payload.get("story", "")))
    step_rows = []
    for step in payload.get("step_logs", []) or []:
        step_rows.append(
            f"<article class='row'><h3>Step {html.escape(str(step.get('step', '')))}</h3>"
            f"<p>{html.escape(str(step.get('sam_summary', '') or ''))}</p></article>"
        )
    state_rows = []
    for state in payload.get("world_states", []) or []:
        state_rows.append(
            f"<article class='row'><h3>World Step {html.escape(str(state.get('step', '')))}</h3>"
            f"<p>Locations: {len(state.get('locations', []) or [])} | Items: {len(state.get('items', []) or [])}</p></article>"
        )
    return (
        f"<section class='run'><h2>{backend}</h2><p>{story}</p>"
        "<h3>Narrative Time</h3>"
        + "".join(step_rows)
        + "<h3>World Space</h3>"
        + "".join(state_rows)
        + "</section>"
    )


def generate_pipeline_html(*, run_dirs: list[Path], output_html: Path) -> Path:
    runs = [_load_run(path) for path in run_dirs]
    body = "".join(_render_run(payload) for payload in runs)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Narrative World Pipeline</title>"
        "<style>body{font-family:Georgia,serif;margin:28px;background:#f7f7f2;color:#191919}"
        ".run{border:1px solid #ccc;padding:20px;margin:18px 0;background:white}.row{border-top:1px solid #ddd;padding:10px 0}</style>"
        "</head><body><h1>Backend Comparison</h1>"
        + body
        + "</body></html>",
        encoding="utf-8",
    )
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    print(generate_pipeline_html(run_dirs=args.run_dir, output_html=args.output_html))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_narrative_world_pipeline_html.py -q`

Expected: `1 passed`.

---

### Task 6: Pipeline Runner CLI

**Files:**
- Create: `scripts/run_dual_backend_narrative_world_pipeline.py`
- Test: `tests/test_dual_backend_pipeline_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import subprocess
import sys


def test_dual_backend_pipeline_dry_run(tmp_path: Path) -> None:
    story_file = tmp_path / "story.txt"
    story_file.write_text(
        "A journalist enters a lighthouse, opens a box, and discovers a compass.",
        encoding="utf-8",
    )
    output_dir = tmp_path / "pipeline"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_dual_backend_narrative_world_pipeline.py",
            "--story-file",
            str(story_file),
            "--run-both",
            "--max-steps",
            "1",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "all_flux" / "pipeline_result.json").exists()
    assert (output_dir / "all_banana" / "pipeline_result.json").exists()
    assert (output_dir / "comparison_overview.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dual_backend_pipeline_cli.py -q`

Expected: FAIL because `scripts/run_dual_backend_narrative_world_pipeline.py` does not exist.

- [ ] **Step 3: Implement the CLI with dry-run support**

Implementation requirements:

- Args:
  - `--story-file`
  - `--storyline`
  - `--characters`
  - `--backend-mode all_flux|all_banana`
  - `--run-both`
  - `--max-steps`
  - `--output-dir`
  - `--dry-run`
  - `--log-level`
- If `--run-both`, execute `all_flux` and `all_banana`.
- In `--dry-run`, avoid LLM and real image calls:
  - use provided characters or `Journalist,Watcher`
  - create a minimal world model with one location and one item
  - write placeholder images through `WorldRenderer(dry_run=True)`
  - create synthetic step logs
- In real mode:
  - use `OrchestratorAgent`
  - use `WorldModelAgent.extract_world_blueprint`
  - render initial world assets with selected backend through `WorldRenderer`
  - build initial state with `build_initial_world_state`
  - run `StateAnalysisMachine.run_narrative_deduction`
  - judge deltas for each step with `WorldDeltaJudge`
  - refresh changed world images with `WorldRenderer`
  - render event keyframes with `EventKeyframeRenderer`
- Always write:
  - `pipeline_result.json`
  - `world_state_step0.json`
  - `world_state_stepN.json`
  - `world_deltas/world_delta_stepN.json`
  - `visualizations/combined_overview.html`
- When `--run-both`, write root `comparison_overview.html`.

- [ ] **Step 4: Run dry-run test to verify it passes**

Run: `pytest tests/test_dual_backend_pipeline_cli.py -q`

Expected: `1 passed`.

---

### Task 7: Integration Verification

**Files:**
- Use: `scripts/run_dual_backend_narrative_world_pipeline.py`
- Use: `.env`
- Use output directory: `outputs/narrative_world_pipeline_dual_backend_test`

- [ ] **Step 1: Run dry-run integration**

Run:

```bash
python scripts/run_dual_backend_narrative_world_pipeline.py \
  --storyline "On a storm island, a journalist enters an abandoned lighthouse, discovers a cracked compass, and forces the hidden witness to reveal what happened in the flooded archive." \
  --characters Journalist,HiddenWitness,Caretaker \
  --run-both \
  --max-steps 3 \
  --output-dir outputs/narrative_world_pipeline_dual_backend_test_dry \
  --dry-run \
  --log-level INFO
```

Expected:

- `outputs/narrative_world_pipeline_dual_backend_test_dry/all_flux/pipeline_result.json`
- `outputs/narrative_world_pipeline_dual_backend_test_dry/all_banana/pipeline_result.json`
- `outputs/narrative_world_pipeline_dual_backend_test_dry/comparison_overview.html`

- [ ] **Step 2: Run real all-banana smoke test**

Run outside sandbox if network is restricted:

```bash
conda run -n mifage python scripts/run_dual_backend_narrative_world_pipeline.py \
  --storyline "On a storm island, a journalist enters an abandoned lighthouse, discovers a cracked compass, and forces the hidden witness to reveal what happened in the flooded archive." \
  --characters Journalist,HiddenWitness,Caretaker \
  --backend-mode all_banana \
  --max-steps 3 \
  --output-dir outputs/narrative_world_pipeline_all_banana_real \
  --log-level INFO
```

Expected:

- `pipeline_result.json`
- initial world images
- event keyframes
- at least one `world_state_stepN.json`
- `visualizations/combined_overview.html`

- [ ] **Step 3: Run real all-flux smoke test**

Run outside sandbox if GPU access is restricted:

```bash
conda run -n mifage python scripts/run_dual_backend_narrative_world_pipeline.py \
  --storyline "On a storm island, a journalist enters an abandoned lighthouse, discovers a cracked compass, and forces the hidden witness to reveal what happened in the flooded archive." \
  --characters Journalist,HiddenWitness,Caretaker \
  --backend-mode all_flux \
  --max-steps 3 \
  --output-dir outputs/narrative_world_pipeline_all_flux_real \
  --log-level INFO
```

Expected:

- `pipeline_result.json`
- FLUX-generated initial world images
- FLUX-generated event keyframes
- world delta snapshots
- `visualizations/combined_overview.html`

- [ ] **Step 4: Run both real modes**

Run outside sandbox:

```bash
conda run -n mifage python scripts/run_dual_backend_narrative_world_pipeline.py \
  --storyline "On a storm island, a journalist enters an abandoned lighthouse, discovers a cracked compass, and forces the hidden witness to reveal what happened in the flooded archive." \
  --characters Journalist,HiddenWitness,Caretaker \
  --run-both \
  --max-steps 3 \
  --output-dir outputs/narrative_world_pipeline_dual_backend_real \
  --log-level INFO
```

Expected:

- `outputs/narrative_world_pipeline_dual_backend_real/all_flux/pipeline_result.json`
- `outputs/narrative_world_pipeline_dual_backend_real/all_banana/pipeline_result.json`
- `outputs/narrative_world_pipeline_dual_backend_real/comparison_overview.html`

---

## Self-Review

Spec coverage:

- Same story dual backend: covered by Task 6 and Task 7.
- `all_flux` and `all_banana`: covered by Task 3, Task 4, Task 6.
- 3-5 step default scope: covered by Task 6 and Task 7.
- World state snapshots: covered by Task 1 and Task 6.
- Significant world delta checks: covered by Task 2 and Task 6.
- Event keyframes: covered by Task 4 and Task 6.
- Temporal/spatial visualization: covered by Task 5 and Task 6.

Placeholder scan:

- No implementation step contains `TBD`, `TODO`, or unspecified file targets.
- CLI args, output files, and verification commands are concrete.

Type consistency:

- `backend_mode` maps to `flux_local` or `nano_banana` through `backend_mode_to_tool`.
- `WorldRenderer.render_world_asset` returns the same dictionary shape expected by `WorldStateTracker` and `EventKeyframeRenderer`.
- `WorldDeltaJudge.judge_step_delta` returns `location_updates` and `item_updates`, which `update_world_state_from_delta` consumes.
