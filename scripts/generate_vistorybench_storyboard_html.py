#!/usr/bin/env python3
"""Generate a shot-aligned ViStoryBench storyboard visualization.

The page is intentionally limited to one ViStoryBench generation batch. Every
storyboard shot is paired with its generated PNG, and missing pairs are shown
explicitly instead of being silently omitted.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def short(value: Any, limit: int = 360) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def rel(path: Path, output_html: Path) -> str:
    return os.path.relpath(path.resolve(), output_html.parent.resolve()).replace(os.sep, "/")


def link(path: Path, label: str, output_html: Path) -> str:
    if not path.exists():
        return f"<span class='missing'>{esc(label)} (missing)</span>"
    return f"<a href='{esc(rel(path, output_html))}'>{esc(label)}</a>"


def image_tag(path: Path, label: str, output_html: Path) -> str:
    if not path.exists():
        return "<div class='image missing'>PNG missing</div>"
    src = esc(rel(path, output_html))
    return f"<a href='{src}' target='_blank'><img loading='lazy' src='{src}' alt='{esc(label)}'></a>"


def field(label: str, value: Any) -> str:
    if isinstance(value, list):
        rendered = ", ".join(esc(item) for item in value)
    else:
        rendered = esc(value)
    return f"<div class='field'><b>{esc(label)}</b><div>{rendered or '<span class=muted>none</span>'}</div></div>"


def story_data(dataset_root: Path, story_id: str) -> dict[str, Any]:
    payload = load_json(dataset_root / story_id / "story.json", {})
    shots = payload.get("Shots", []) if isinstance(payload, dict) else []
    by_index = {}
    for shot in shots:
        if isinstance(shot, dict):
            by_index[int(shot.get("index", 0))] = shot
    return {"payload": payload, "shots": by_index}


def render_shot(
    *,
    shot: dict[str, Any],
    node: dict[str, Any],
    generated: dict[str, Any],
    dataset_shot: dict[str, Any],
    output_html: Path,
) -> str:
    anchor = node.get("benchmark_anchor", {}) or {}
    image_path = Path(str(generated.get("output_path", "")))
    if not image_path.is_absolute():
        image_path = (output_html.parent / image_path).resolve()
    image_exists = image_path.exists()
    status = str(generated.get("status", "unknown"))
    status_class = "ok" if status in {"success", "reused"} and image_exists else "warn"
    refs = generated.get("references", []) or []
    refs_html = "".join(f"<li>{esc(ref)}</li>" for ref in refs) or "<li>none</li>"
    prompt = generated.get("prompt", "")
    return f"""
    <article class='shot-card' data-search='{esc(json.dumps({"shot": shot, "anchor": anchor, "prompt": prompt}, ensure_ascii=False))}'>
      <div class='shot-head'>
        <div><span class='shot-number'>SHOT {int(shot):03d}</span>
        <span class='status {status_class}'>{esc(status.upper())}{" / PNG MISSING" if not image_exists else ""}</span></div>
        <span class='node-id'>{esc(node.get("node_id", ""))}</span>
      </div>
      <div class='shot-grid'>
        <figure class='shot-image'>
          {image_tag(image_path, f"story shot {int(shot):03d}", output_html)}
          <figcaption>{link(image_path, image_path.name, output_html)}</figcaption>
        </figure>
        <div class='storyboard'>
          <h3>Storyboard</h3>
          {field("Camera", anchor.get("camera") or dataset_shot.get("Shot Perspective Design", {}).get("en"))}
          {field("Scene", anchor.get("scene") or dataset_shot.get("Setting Description", {}).get("en"))}
          {field("Action", anchor.get("action") or dataset_shot.get("Static Shot Description", {}).get("en"))}
          {field("Plot", anchor.get("plot") or dataset_shot.get("Plot Correspondence", {}).get("en"))}
          {field("Characters", anchor.get("characters") or dataset_shot.get("Characters Appearing", {}).get("en"))}
          <details><summary>Generation prompt</summary><pre>{esc(prompt)}</pre></details>
          <details><summary>Reference images ({len(refs)})</summary><ul class='refs'>{refs_html}</ul></details>
        </div>
      </div>
    </article>"""


def render_story(
    *, story_id: str, story_root: Path, manifest_story: dict[str, Any], dataset_root: Path, output_html: Path
) -> tuple[str, int, int]:
    nodes_payload = load_json(story_root / "nodes.json", {})
    nodes = {int(row.get("shot_index", 0)): row for row in nodes_payload.get("nodes", []) if isinstance(row, dict)}
    dataset = story_data(dataset_root, story_id)
    generated = {
        int(row.get("shot_index", 0)): row
        for row in manifest_story.get("generated", [])
        if isinstance(row, dict)
    }
    expected = int(manifest_story.get("expected_shots", 0) or 0)
    indices = sorted(set(nodes) | set(generated) | set(dataset["shots"]))
    cards = []
    matched = 0
    for index in indices:
        row = generated.get(index, {})
        path = Path(str(row.get("output_path", "")))
        if not path.is_absolute():
            path = (story_root / path).resolve()
        if path.exists() and index in nodes:
            matched += 1
        cards.append(
            render_shot(
                shot=index,
                node=nodes.get(index, {}),
                generated=row,
                dataset_shot=dataset["shots"].get(index, {}),
                output_html=output_html,
            )
        )
    story_manifest_link = link(story_root / "generation_manifest.json", "story manifest", output_html)
    nodes_link = link(story_root / "nodes.json", "nodes.json", output_html)
    prompts_link = link(story_root / "prompts.json", "prompts.json", output_html)
    title = dataset["payload"].get("Story_type", {}).get("en", "") if isinstance(dataset["payload"], dict) else ""
    return (
        f"""<section class='story' id='story-{esc(story_id)}'>
          <div class='story-head'><div><span class='eyebrow'>STORY {esc(story_id)}</span><h2>{esc(title or 'ViStoryBench story')}</h2></div>
          <div class='story-links'>{story_manifest_link} · {nodes_link} · {prompts_link}</div></div>
          <p class='story-summary'>{len(indices)} storyboard shots · {matched}/{expected or len(indices)} image/storyboard pairs with existing PNG</p>
          <div class='shots'>{''.join(cards)}</div>
        </section>""",
        len(indices),
        matched,
    )


def build_page(batch_dir: Path, dataset_root: Path, output_html: Path) -> tuple[str, dict[str, int]]:
    root_manifest_path = batch_dir / "generation_manifest.json"
    root_manifest = load_json(root_manifest_path, {})
    stories = root_manifest.get("stories", []) if isinstance(root_manifest, dict) else []
    story_sections = []
    total_shots = total_pairs = total_images = 0
    for manifest_story in stories:
        story_id = str(manifest_story.get("story_id", ""))
        if not story_id:
            continue
        section, shots, pairs = render_story(
            story_id=story_id,
            story_root=batch_dir / story_id,
            manifest_story=manifest_story,
            dataset_root=dataset_root,
            output_html=output_html,
        )
        story_sections.append(section)
        total_shots += shots
        total_pairs += pairs
    total_images = sum(1 for path in batch_dir.rglob("*.png") if path.is_file())
    missing_pairs = total_shots - total_pairs
    meta = {
        "stories": len(story_sections),
        "shots": total_shots,
        "images": total_images,
        "matched_pairs": total_pairs,
        "missing_pairs": missing_pairs,
    }
    status = "COMPLETE" if missing_pairs == 0 and total_images == total_shots else "PARTIAL"
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ViStoryBench Storyboard Visualization · {esc(batch_dir.name)}</title>
<style>
:root {{ --ink:#18202d; --muted:#637083; --line:#dce3ec; --paper:#f4f7fb; --card:#fff; --blue:#185adb; --green:#137333; --amber:#a15c00; --navy:#10243e; }}
* {{ box-sizing:border-box }} body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
a {{ color:var(--blue); text-decoration:none }} a:hover {{ text-decoration:underline }} .hero {{ background:linear-gradient(135deg,#10243e,#145b70); color:#fff; padding:34px 24px }}
.wrap {{ max-width:1540px; margin:auto; padding:0 24px }} .hero h1 {{ margin:5px 0 8px; font-size:clamp(25px,3.5vw,46px); letter-spacing:-.04em }}
.hero p {{ color:#dcefff; max-width:980px }} .eyebrow {{ color:#5c8ef2; font-size:11px; font-weight:800; letter-spacing:.12em }}
.hero .eyebrow {{ color:#a8d8ff }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:22px 0 }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:15px 17px }} .metric strong {{ display:block; font-size:28px }} .metric span {{ color:var(--muted) }}
.toolbar {{ position:sticky; top:0; z-index:5; background:rgba(244,247,251,.96); border-bottom:1px solid var(--line); padding:12px 0; backdrop-filter:blur(8px) }}
.toolbar-inner {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }} input,select {{ border:1px solid var(--line); border-radius:8px; padding:9px 11px; background:#fff }} input {{ min-width:280px; flex:1 }}
.story {{ margin:32px 0 60px }} .story-head {{ display:flex; justify-content:space-between; gap:20px; align-items:end; border-bottom:2px solid var(--navy); padding-bottom:10px }}
.story-head h2 {{ margin:4px 0; font-size:26px }} .story-links {{ color:var(--muted); font-size:12px }} .story-summary {{ color:var(--muted) }}
.shots {{ display:grid; gap:18px }} .shot-card {{ background:var(--card); border:1px solid var(--line); border-radius:15px; overflow:hidden; box-shadow:0 5px 18px rgba(16,36,62,.05) }}
.shot-head {{ display:flex; justify-content:space-between; gap:15px; align-items:center; padding:11px 15px; background:#f8fafc; border-bottom:1px solid var(--line) }} .shot-number {{ font-weight:850; letter-spacing:.05em }} .node-id {{ color:var(--muted); font-family:ui-monospace,monospace; font-size:12px }}
.status {{ display:inline-block; margin-left:10px; padding:3px 8px; border-radius:99px; font-size:10px; font-weight:850 }} .status.ok {{ color:var(--green); background:#dcfce7 }} .status.warn {{ color:var(--amber); background:#fef3c7 }}
.shot-grid {{ display:grid; grid-template-columns:minmax(260px,38%) 1fr; gap:20px; padding:15px }} .shot-image {{ margin:0 }} .shot-image img {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#e8edf4; border-radius:9px }} figcaption {{ margin-top:7px; font-size:12px; word-break:break-all }}
.storyboard h3 {{ margin:0 0 8px; font-size:20px }} .field {{ display:grid; grid-template-columns:105px 1fr; gap:10px; border-top:1px solid var(--line); padding:8px 0 }} .field b {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em }}
details {{ margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:8px 10px }} summary {{ cursor:pointer; font-weight:700 }} pre {{ white-space:pre-wrap; word-break:break-word; max-height:360px; overflow:auto; background:#10243e; color:#dcefff; border-radius:7px; padding:11px; font-size:12px }}
.refs {{ margin:8px 0 0; padding-left:20px; color:var(--muted); word-break:break-all; font-size:12px }} .muted {{ color:var(--muted) }} .missing {{ color:#a33; background:#fee2e2; padding:4px 7px; border-radius:5px }}
.note {{ margin:22px 0; border-left:4px solid var(--blue); padding:12px 15px; background:#eaf2ff; color:#153b75; border-radius:6px }} .hidden {{ display:none !important }}
@media (max-width:780px) {{ .wrap {{ padding:0 14px }} .shot-grid {{ grid-template-columns:1fr }} .story-head {{ display:block }} .story-links {{ margin-top:8px }} input {{ min-width:190px }} }}
</style></head><body>
<header class='hero'><div class='wrap'><div class='eyebrow'>VISTORYBENCH GENERATION ONLY</div><h1>Storyboard and Image Alignment</h1>
<p>Batch <b>{esc(batch_dir.name)}</b> · backend <b>{esc(root_manifest.get('image_backend', ''))}</b> · mode <b>{esc(root_manifest.get('mode', ''))}</b> · context <b>{esc(root_manifest.get('context_mode', ''))}</b></p>
<p>Every displayed shot is aligned with its storyboard description and generated PNG. This page reads existing files only and does not rerun generation or benchmark evaluation.</p></div></header>
<div class='toolbar'><div class='wrap toolbar-inner'><input id='filter' placeholder='Filter stories, shots, actions, characters, prompts...'><select id='storyFilter'><option value=''>All stories</option>{''.join(f"<option value='story-{esc(str(s.get('story_id','')))}'>Story {esc(str(s.get('story_id','')))}</option>" for s in stories)}</select><a href='{esc(rel(root_manifest_path, output_html))}'>root manifest</a></div></div>
<main class='wrap'><div class='metrics'><div class='metric'><strong>{meta['stories']}</strong><span>stories</span></div><div class='metric'><strong>{meta['shots']}</strong><span>storyboard shots</span></div><div class='metric'><strong>{meta['images']}</strong><span>PNG images</span></div><div class='metric'><strong>{meta['matched_pairs']}</strong><span>matched image/storyboard pairs</span></div><div class='metric'><strong>{meta['missing_pairs']}</strong><span>missing pairs</span></div></div>
<div class='note'><b>{status}</b>: pairing is counted only when the shot has both a storyboard node and an existing generated PNG. Images and descriptions are shown in shot order; use the search box or story selector to inspect the full batch.</div>
{''.join(story_sections)}</main>
<script>
const filter=document.getElementById('filter'), storyFilter=document.getElementById('storyFilter');
function applyFilter() {{ const q=(filter.value||'').toLowerCase(), story=storyFilter.value; document.querySelectorAll('.story').forEach(s=>{{ const storyOk=!story||s.id===story; let visible=0; s.querySelectorAll('.shot-card').forEach(c=>{{ const ok=(!q||c.innerText.toLowerCase().includes(q))&&storyOk; c.classList.toggle('hidden',!ok); if(ok) visible++; }}); s.classList.toggle('hidden',!storyOk||visible===0); }}); }}
filter.addEventListener('input',applyFilter); storyFilter.addEventListener('change',applyFilter);
</script></body></html>""", meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/vistorybench/cameramaster/vistorybench_anchor_only/en/20261010_163808"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/data1/AgentG/hhb/vistorybench-main/data/dataset/ViStory"))
    parser.add_argument("--output-html", type=Path, default=None)
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_html = (args.output_html or batch_dir / "vistorybench_storyboard_visualization.html").resolve()
    if not (batch_dir / "generation_manifest.json").exists():
        raise SystemExit(f"generation manifest not found: {batch_dir / 'generation_manifest.json'}")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    page, meta = build_page(batch_dir, dataset_root, output_html)
    output_html.write_text(page, encoding="utf-8")
    print(output_html)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
