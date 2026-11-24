#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relpath(target: str | Path, base_dir: Path) -> str:
    path = Path(target)
    if path.is_absolute():
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix() if str(Path.cwd().resolve()) in str(Path(path).resolve()) else path.as_posix()
    try:
        return Path(path).resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _to_output_relative(target: str | Path, output_path: Path) -> str:
    path = Path(target)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return Path(path).relative_to(output_path.parent.resolve()).as_posix() if str(output_path.parent.resolve()) in str(path) else str(Path("../") / path.name if path.exists() else path.as_posix())


def _safe_rel_to_output(target: str | Path, output_path: Path) -> str:
    path = Path(target)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        return path.relative_to(output_path.parent.resolve()).as_posix()
    except Exception:
        return Path(
            *Path(
                Path(
                    __import__("os").path.relpath(str(path), str(output_path.parent.resolve()))
                )
            ).parts
        ).as_posix()


def _img_map(images: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in images or []:
        tool = str(item.get("tool", "")).strip()
        if tool:
            out[tool] = item
    return out


def _asset_card(asset: dict[str, Any], group_label: str, output_path: Path) -> str:
    images = _img_map(asset.get("images", []))
    flux = images.get("flux_local", {})
    nano = images.get("nano_banana", {})
    flux_path = _safe_rel_to_output(flux.get("output_path", ""), output_path) if flux.get("ok") else ""
    nano_path = _safe_rel_to_output(nano.get("output_path", ""), output_path) if nano.get("ok") else ""
    prompt = str(asset.get("image_prompt", "")).strip()
    description = str(asset.get("description", "")).strip()
    evidence = str(asset.get("evidence", "")).strip()
    name = str(asset.get("name", "")).strip()
    flux_html = (
        f'<img src="{html.escape(flux_path)}" alt="{html.escape(name + " flux_local")}">'
        if flux_path
        else "<div class='missing'>missing</div>"
    )
    nano_html = (
        f'<img src="{html.escape(nano_path)}" alt="{html.escape(name + " nano_banana")}">'
        if nano_path
        else "<div class='missing'>missing</div>"
    )

    return f"""
    <article class="asset-card">
      <div class="asset-meta">
        <div class="asset-kind">{html.escape(group_label)}</div>
        <h3>{html.escape(name)}</h3>
        <p class="asset-description">{html.escape(description)}</p>
        <p><strong>Why it matters:</strong> {html.escape(evidence)}</p>
        <details>
          <summary>Prompt</summary>
          <pre>{html.escape(prompt)}</pre>
        </details>
      </div>
      <div class="asset-images">
        <div class="image-slot">
          <div class="slot-label">flux_local</div>
          {flux_html}
        </div>
        <div class="image-slot">
          <div class="slot-label">nano_banana</div>
          {nano_html}
        </div>
      </div>
    </article>
    """


def _comparison_card(entry: dict[str, Any], output_path: Path) -> str:
    board_path = _safe_rel_to_output(entry.get("board_path", ""), output_path)
    flux_path = _safe_rel_to_output(entry.get("flux_path", ""), output_path)
    nano_path = _safe_rel_to_output(entry.get("nano_path", ""), output_path)
    name = str(entry.get("name", "")).strip()
    group_name = str(entry.get("group", "")).strip()
    nano_source = str(entry.get("nano_source", "")).strip()
    retry_badge = "<span class='badge warn'>nano source: single_retry</span>" if nano_source == "single_retry" else "<span class='badge ok'>batch run</span>"
    return f"""
    <article class="compare-card">
      <div class="compare-head">
        <div>
          <div class="asset-kind">{html.escape(group_name)}</div>
          <h3>{html.escape(name)}</h3>
        </div>
        {retry_badge}
      </div>
      <div class="compare-grid">
        <div class="compare-col">
          <div class="slot-label">flux_local</div>
          <img src="{html.escape(flux_path)}" alt="{html.escape(name + ' flux_local')}">
        </div>
        <div class="compare-col">
          <div class="slot-label">nano_banana</div>
          <img src="{html.escape(nano_path)}" alt="{html.escape(name + ' nano_banana')}">
        </div>
      </div>
      <details>
        <summary>Combined board</summary>
        <img class="board-image" src="{html.escape(board_path)}" alt="{html.escape(name + ' combined board')}">
      </details>
    </article>
    """


def build_preview_html(*, result_payload: dict[str, Any], manifest_payload: list[dict[str, Any]], output_path: Path) -> str:
    world = result_payload.get("world_model", {}) or {}
    storyline = str(result_payload.get("storyline", "")).strip()
    summary = str(world.get("summary", "")).strip()
    locations = world.get("locations", []) or []
    items = world.get("items", []) or []

    location_cards = "\n".join(_asset_card(asset, "Location", output_path) for asset in locations)
    item_cards = "\n".join(_asset_card(asset, "Item", output_path) for asset in items)
    compare_cards = "\n".join(_comparison_card(entry, output_path) for entry in manifest_payload)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Model Preview</title>
  <style>
    :root {{
      --bg: #f4efe4;
      --panel: #fffaf1;
      --ink: #1f1a17;
      --muted: #6f655d;
      --line: #d7ccbc;
      --accent: #8f3b2e;
      --accent-2: #24465a;
      --warn: #9e5b16;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(143,59,46,0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(36,70,90,0.14), transparent 28%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto 64px;
    }}
    .hero, .section {{
      background: rgba(255,250,241,0.92);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 12px 40px rgba(50, 35, 22, 0.08);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 20px;
    }}
    .eyebrow {{
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 10px;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 700;
    }}
    h1 {{ font-size: clamp(34px, 4vw, 54px); line-height: 0.96; max-width: 12ch; }}
    h2 {{ font-size: 30px; margin-bottom: 14px; }}
    h3 {{ font-size: 22px; margin-bottom: 10px; }}
    .lede {{
      margin-top: 18px;
      font-size: 18px;
      line-height: 1.7;
      color: var(--ink);
    }}
    .summary {{
      margin-top: 16px;
      padding: 16px 18px;
      background: rgba(36,70,90,0.06);
      border-left: 4px solid var(--accent-2);
      border-radius: 12px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .section {{
      padding: 24px;
      margin-top: 20px;
    }}
    .subhead {{
      margin: 22px 0 12px;
      font-size: 20px;
      color: var(--accent-2);
    }}
    .asset-list, .compare-list {{
      display: grid;
      gap: 18px;
    }}
    .asset-card {{
      display: grid;
      grid-template-columns: minmax(260px, 0.95fr) minmax(360px, 1.05fr);
      gap: 18px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
    }}
    .asset-kind {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .asset-description, .section p, li {{
      line-height: 1.65;
    }}
    .asset-images, .compare-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .image-slot, .compare-col {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
    }}
    .slot-label {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.06);
      background: #ece8e1;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 140px;
      border-radius: 10px;
      background: #efe7da;
      color: var(--warn);
      font-style: italic;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent-2);
      font-weight: 700;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Courier New", monospace;
      font-size: 13px;
      line-height: 1.55;
      color: #3d342d;
      background: #f6eee2;
      border-radius: 12px;
      padding: 12px;
      border: 1px solid var(--line);
    }}
    .compare-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 18px;
    }}
    .compare-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .badge.ok {{ background: rgba(36,70,90,0.10); color: var(--accent-2); }}
    .badge.warn {{ background: rgba(158,91,22,0.14); color: var(--warn); }}
    .board-image {{
      margin-top: 12px;
    }}
    @media (max-width: 960px) {{
      .asset-card {{
        grid-template-columns: 1fr;
      }}
      .asset-images, .compare-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">World Model Preview</div>
      <h1>Storyline</h1>
      <p class="lede">{html.escape(storyline)}</p>
      <div class="summary">{html.escape(summary)}</div>
    </section>
    <section class="section">
      <div class="eyebrow">Agent Plan</div>
      <h2>Planned Assets</h2>
      <p>This section lists the spaces and clue objects that the WorldModelAgent decided to create for downstream narrative and visual generation.</p>
      <div class="subhead">Locations</div>
      <div class="asset-list">
        {location_cards}
      </div>
      <div class="subhead">Items</div>
      <div class="asset-list">
        {item_cards}
      </div>
    </section>
    <section class="section">
      <div class="eyebrow">Visual Comparison</div>
      <h2>Method Comparison</h2>
      <p>Each asset is rendered with the local FLUX tool and the remote nano_banana tool so the visual direction can be compared side by side.</p>
      <div class="compare-list">
        {compare_cards}
      </div>
    </section>
  </main>
</body>
</html>
"""


def generate_preview_html(*, result_path: Path, manifest_path: Path, output_path: Path) -> Path:
    result_payload = _load_json(result_path)
    manifest_payload = _load_json(manifest_path)
    if not isinstance(result_payload, dict):
        raise ValueError("result JSON root must be an object")
    if not isinstance(manifest_payload, list):
        raise ValueError("comparison manifest root must be a list")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_preview_html(
        result_payload=result_payload,
        manifest_payload=manifest_payload,
        output_path=output_path,
    )
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static HTML preview for world model outputs.")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    out = generate_preview_html(
        result_path=args.result_json,
        manifest_path=args.comparison_manifest,
        output_path=args.output_html,
    )
    print(out)


if __name__ == "__main__":
    main()
