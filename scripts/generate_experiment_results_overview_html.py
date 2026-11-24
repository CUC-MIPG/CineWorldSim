#!/usr/bin/env python3
"""Build a static overview page for the existing CameraMaster experiments."""
from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def short(value: Any, limit: int = 180) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def rel_path(path_value: str | Path, output_html: Path, project_root: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    return os.path.relpath(path, output_html.parent.resolve()).replace(os.sep, "/")


def link(path_value: str | Path, label: str, output_html: Path, project_root: Path) -> str:
    path = Path(path_value)
    exists = path.exists() if path.is_absolute() else (project_root / path).exists()
    if not exists:
        return f"<span class='missing'>{esc(label)}</span>"
    return f"<a href='{esc(rel_path(path, output_html, project_root))}'>{esc(label)}</a>"


def media(path_value: str | Path, output_html: Path, project_root: Path, alt: str = "") -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.exists():
        return "<div class='media missing'>missing asset</div>"
    src = esc(rel_path(path, output_html, project_root))
    if path.suffix.lower() in VIDEO_EXTS:
        return f"<video controls preload='metadata' src='{src}'></video>"
    return f"<img loading='lazy' src='{src}' alt='{esc(alt)}'>"


def file_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def batch_record(manifest_path: Path, project_root: Path) -> dict[str, Any] | None:
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("stories"), list):
        return None
    generated = [row for story in payload["stories"] if isinstance(story, dict) for row in story.get("generated", [])]
    status_counts = Counter(str(row.get("status", "unknown")) for row in generated if isinstance(row, dict))
    expected = sum(int(story.get("expected_shots", 0) or 0) for story in payload["stories"] if isinstance(story, dict))
    png_count = sum(
        1
        for path in manifest_path.parent.rglob("*.png")
        if path.is_file()
    )
    successful = status_counts.get("success", 0) + status_counts.get("reused", 0)
    return {
        "path": manifest_path,
        "timestamp": payload.get("timestamp", manifest_path.parent.name),
        "method": payload.get("method", ""),
        "mode": payload.get("mode", ""),
        "backend": payload.get("image_backend", ""),
        "language": payload.get("language", ""),
        "context": payload.get("context_mode", ""),
        "stories": len(payload["stories"]),
        "expected": expected,
        "generated": len(generated),
        "successful": successful,
        "png": png_count,
        "failed": sum(value for key, value in status_counts.items() if key not in {"success", "reused"}),
        "complete": expected == successful == png_count and expected > 0,
    }


def pipeline_record(result_path: Path) -> dict[str, Any]:
    payload = load_json(result_path, {})
    run_dir = result_path.parent
    video_files = [path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS]
    html_files = [path for path in run_dir.rglob("*.html") if path.is_file()]
    nodes_path = run_dir / "path_orchestration" / "narrative_nodes.json"
    nodes_payload = load_json(nodes_path, {})
    selected_nodes = nodes_payload.get("selected_nodes", []) if isinstance(nodes_payload, dict) else []
    return {
        "path": result_path,
        "name": run_dir.name,
        "parent": run_dir.parent.name,
        "backend": payload.get("backend_mode", ""),
        "tool": payload.get("backend_tool", ""),
        "characters": len(payload.get("characters", []) or []),
        "steps": len(payload.get("step_logs", []) or []),
        "world_states": len(payload.get("world_states", []) or []),
        "keyframes": len(payload.get("event_keyframes", []) or []),
        "nodes": len(selected_nodes),
        "videos": len(video_files),
        "visualizations": len(html_files),
        "dry_run": bool(payload.get("dry_run", False)),
        "story": short(payload.get("story", ""), 150),
    }


def visual_record(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "topology" if path.name == "topology_graph.html" else "pipeline",
        "size": file_size(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "refs": len([x for x in path.read_text(errors="replace").split("<img") if x]) - 1,
    }


def html_page(
    *,
    output_html: Path,
    project_root: Path,
    batches: list[dict[str, Any]],
    pipelines: list[dict[str, Any]],
    visualizations: list[dict[str, Any]],
    videos: list[Path],
    gallery: list[tuple[str, Path]],
) -> str:
    latest = max(batches, key=lambda row: str(row["timestamp"]), default=None)
    total_png = sum(row["png"] for row in batches)
    total_videos = len(videos)
    complete_batches = sum(1 for row in batches if row["complete"])
    complete_pipelines = sum(1 for row in pipelines if not row["dry_run"] and row["videos"] > 0)

    batch_rows = []
    for row in sorted(batches, key=lambda item: str(item["timestamp"]), reverse=True):
        status = "COMPLETE" if row["complete"] else "PARTIAL"
        batch_rows.append(
            "<tr>"
            f"<td>{link(row['path'], row['timestamp'], output_html, project_root)}</td>"
            f"<td>{esc(row['mode'])}</td><td>{esc(row['backend'])}</td><td>{esc(row['context'])}</td>"
            f"<td>{row['stories']}</td><td>{row['expected']}</td><td>{row['successful']}</td><td>{row['png']}</td>"
            f"<td><span class='status {'ok' if row['complete'] else 'warn'}'>{status}</span></td>"
            "</tr>"
        )

    pipeline_rows = []
    for row in sorted(pipelines, key=lambda item: str(item["path"]), reverse=True):
        status = "DRY RUN" if row["dry_run"] else ("VIDEO" if row["videos"] else "IMAGE/STATE")
        pipeline_rows.append(
            "<tr>"
            f"<td>{link(row['path'], row['parent'] + '/' + row['name'], output_html, project_root)}</td>"
            f"<td>{esc(row['backend'])}</td><td>{esc(row['tool'])}</td><td>{row['characters']}</td>"
            f"<td>{row['steps']}</td><td>{row['world_states']}</td><td>{row['keyframes']}</td>"
            f"<td>{row['nodes']}</td><td>{row['videos']}</td><td><span class='status {'warn' if row['dry_run'] else 'ok'}'>{status}</span></td>"
            "</tr>"
        )

    viz_cards = []
    for row in sorted(visualizations, key=lambda item: str(item["path"])):
        viz_cards.append(
            "<article class='viz-card'>"
            f"<div class='eyebrow'>{esc(row['kind'])}</div>"
            f"<h3>{link(row['path'], row['path'].name, output_html, project_root)}</h3>"
            f"<p>{esc(row['path'].parent)}</p>"
            f"<small>{esc(row['size'])} · modified {esc(row['modified'])}</small>"
            "</article>"
        )

    gallery_cards = []
    for label, path in gallery:
        gallery_cards.append(
            "<figure class='gallery-card'>"
            + media(path, output_html, project_root, label)
            + f"<figcaption>{esc(label)}<br>{link(path, path.name, output_html, project_root)}</figcaption>"
            + "</figure>"
        )

    video_cards = []
    for path in videos[:48]:
        video_cards.append(
            "<article class='video-card'>"
            + media(path, output_html, project_root, path.name)
            + f"<p>{link(path, path.name, output_html, project_root)}</p>"
            + "</article>"
        )

    latest_text = "No batch manifest found."
    if latest:
        latest_text = (
            f"Latest batch {latest['timestamp']}: {latest['successful']}/{latest['expected']} shots, "
            f"{latest['stories']} stories, backend {latest['backend']}."
        )

    topology = [row for row in visualizations if row["kind"] == "topology"]
    topology_links = "".join(
        f"<li>{link(row['path'], str(row['path']), output_html, project_root)}</li>" for row in topology
    ) or "<li>No topology HTML found.</li>"

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>CameraMaster Experiment Results</title>
<style>
:root {{ --ink:#172033; --muted:#64748b; --line:#dbe3ef; --paper:#f5f7fb; --card:#fff; --blue:#2563eb; --green:#15803d; --amber:#b45309; --dark:#0f172a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
.hero {{ padding:42px max(24px,calc((100vw - 1380px)/2)); background:linear-gradient(135deg,#0f172a 0%,#173b68 58%,#0f766e 100%); color:#fff; }}
.hero h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,52px); letter-spacing:-.04em; }} .hero p {{ max-width:850px; color:#dbeafe; margin:8px 0; }}
.wrap {{ max-width:1380px; margin:0 auto; padding:24px; }} .nav {{ position:sticky;top:0;z-index:3; padding:10px 24px; background:rgba(245,247,251,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
.nav a {{ margin-right:18px; font-weight:700; }} .metrics {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0 34px; }}
.metric {{ background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(15,23,42,.05); }} .metric strong {{ display:block;font-size:30px;letter-spacing:-.04em; }} .metric span {{ color:var(--muted); }}
section {{ margin:36px 0 52px; }} h2 {{ margin:0 0 8px;font-size:28px;letter-spacing:-.03em; }} h3 {{ margin:6px 0 8px; }} .lede {{ color:var(--muted);max-width:900px; }}
.panel {{ background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;overflow:auto; }} table {{ width:100%;border-collapse:collapse;min-width:850px; }} th,td {{ text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top; }} th {{ color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em; }} tr:hover {{ background:#f8fafc; }}
.status {{ display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.04em; }} .status.ok {{ color:#166534;background:#dcfce7; }} .status.warn {{ color:#92400e;background:#fef3c7; }} .missing {{ color:#b91c1c;background:#fee2e2;padding:8px;border-radius:8px; }}
.gallery {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px; }} .gallery-card,.video-card,.viz-card {{ margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px;overflow:hidden; }} .gallery-card img,.video-card video {{ width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:9px;background:#e2e8f0; }} figcaption,.video-card p {{ font-size:12px;color:var(--muted);margin:8px 2px 2px;word-break:break-word; }}
.viz-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px; }} .viz-card p {{ color:var(--muted);font-size:12px;word-break:break-word; }} .eyebrow {{ color:var(--blue);font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:800; }}
.video-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px; }} .video-card video {{ display:block; }} details {{ background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:10px 0; }} summary {{ cursor:pointer;font-weight:700; }} pre {{ white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#dbeafe;padding:14px;border-radius:10px;overflow:auto; }} .note {{ border-left:4px solid var(--blue);padding:12px 16px;background:#eff6ff;border-radius:8px;color:#1e3a8a; }}
@media (max-width:700px) {{ .wrap {{ padding:16px; }} .hero {{ padding:30px 16px; }} .nav {{ padding:10px 16px;overflow:auto;white-space:nowrap; }} }}
</style></head>
<body>
<header class='hero'><div class='wrap'>
<div class='eyebrow' style='color:#93c5fd'>CameraMaster / Experiment Archive</div>
<h1>Experiment Results Overview</h1>
<p>Static aggregation of existing narrative simulation, world-state rendering, ViStoryBench image generation, event-video generation, and topology visualizations. This page does not rerun any model or alter source results.</p>
<p><b>Generated:</b> {esc(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} · <b>Latest:</b> {esc(latest_text)}</p>
</div></header>
<nav class='nav'><div class='wrap' style='padding-top:0;padding-bottom:0'><a href='#summary'>Summary</a><a href='#vistory'>ViStoryBench</a><a href='#pipelines'>Pipelines</a><a href='#gallery'>Gallery</a><a href='#visualizations'>HTML assets</a><a href='#videos'>Videos</a></div></nav>
<main class='wrap'>
<section id='summary'><h2>Summary</h2><p class='lede'>The archive distinguishes generated artifacts from official benchmark metrics. A complete generation batch means every expected shot has a successful PNG; it does not by itself mean that official ViStoryBench metrics have been evaluated.</p>
<div class='metrics'>
<div class='metric'><strong>{len(batches)}</strong><span>ViStoryBench batch manifests</span></div>
<div class='metric'><strong>{complete_batches}</strong><span>complete batches</span></div>
<div class='metric'><strong>{total_png}</strong><span>PNG artifacts across batches</span></div>
<div class='metric'><strong>{len(pipelines)}</strong><span>pipeline result runs</span></div>
<div class='metric'><strong>{complete_pipelines}</strong><span>runs with video artifacts</span></div>
<div class='metric'><strong>{total_videos}</strong><span>video files discovered</span></div>
</div><div class='note'>Latest verified ViStoryBench full run: 80 stories and 1317/1317 shots generated with FLUX.2 Klein 9B. Official benchmark metric output is not inferred here unless a separate result file exists.</div></section>

<section id='vistory'><h2>ViStoryBench Generation</h2><p class='lede'>Batch-level completeness and reproducibility records from all discovered ViStoryBench generation manifests.</p><div class='panel'><table><thead><tr><th>Manifest</th><th>Mode</th><th>Backend</th><th>Context</th><th>Stories</th><th>Expected shots</th><th>Success</th><th>PNG</th><th>Status</th></tr></thead><tbody>{''.join(batch_rows) or '<tr><td colspan=9>No manifests found.</td></tr>'}</tbody></table></div>
{('<h3>Latest batch sample shots</h3><div class="gallery">' + ''.join(gallery_cards) + '</div>') if gallery_cards else ''}</section>

<section id='pipelines'><h2>CameraMaster Pipeline Runs</h2><p class='lede'>Existing pipeline_result.json files summarized by backend, simulation length, keyframes, selected nodes, and discovered video artifacts. Dry runs are explicitly marked.</p><div class='panel'><input id='pipelineFilter' placeholder='Filter pipeline rows...' style='width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;margin-bottom:12px'><table id='pipelineTable'><thead><tr><th>Run</th><th>Backend</th><th>Tool</th><th>Characters</th><th>Steps</th><th>World states</th><th>Keyframes</th><th>Nodes</th><th>Videos</th><th>Type</th></tr></thead><tbody>{''.join(pipeline_rows) or '<tr><td colspan=10>No pipeline results found.</td></tr>'}</tbody></table></div></section>

<section id='gallery'><h2>Representative Image Results</h2><p class='lede'>Sample shots from the latest ViStoryBench batch. The complete batch remains available through its manifest and story directories above.</p><div class='gallery'>{''.join(gallery_cards) or '<p>No gallery images found.</p>'}</div></section>

<section id='visualizations'><h2>Existing HTML Visualizations</h2><p class='lede'>Links to previously generated visualizations, including full pipeline views, Banana/Flux comparisons, world-model previews, and early node topology graphs.</p><div class='viz-grid'>{''.join(viz_cards) or '<p>No HTML assets found.</p>'}</div><details><summary>Topology graph links</summary><ul>{topology_links}</ul></details></section>

<section id='videos'><h2>Existing Video Results</h2><p class='lede'>The first 48 discovered video files are embedded for browsing. Full paths remain available in the corresponding pipeline output directories.</p><div class='video-grid'>{''.join(video_cards) or '<p>No video files found.</p>'}</div></section>

<section><h2>Interpretation Boundary</h2><div class='panel'><ul><li>Generated PNG/video artifacts indicate execution output, not semantic quality.</li><li>ViStoryBench official scores require a separate <code>bench_run.py</code> evaluation result.</li><li>Historical Banana failures, dry runs, and successful FLUX.2 Klein runs are listed together but remain labeled by backend and run type.</li><li>All links are relative to this HTML file and are intended for local file access or a local HTTP server.</li></ul></div></section>
</main>
<script>
const filter = document.getElementById('pipelineFilter');
if (filter) filter.addEventListener('input', () => {{
  const q = filter.value.toLowerCase();
  document.querySelectorAll('#pipelineTable tbody tr').forEach(row => {{ row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none'; }});
}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-html", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_html = (args.output_html or project_root / "outputs" / "cameramaster_experiment_results_overview.html").resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    batches = []
    for path in (project_root / "outputs" / "vistorybench").rglob("generation_manifest.json"):
        record = batch_record(path, project_root)
        if record:
            batches.append(record)

    pipelines = [
        pipeline_record(path)
        for path in (project_root / "outputs").rglob("pipeline_result.json")
        if path.is_file()
    ]

    viz_names = {"combined_overview.html", "full_stage_pipeline_visualization.html", "full_pipeline_visualization.html", "comparison_overview.html", "topology_graph.html", "index.html"}
    visualizations = [visual_record(path, project_root) for path in (project_root / "outputs").rglob("*.html") if path.name in viz_names]
    videos = sorted(
        [path for path in (project_root / "outputs").rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS],
        key=lambda path: str(path),
    )

    gallery: list[tuple[str, Path]] = []
    latest = max(batches, key=lambda row: str(row["timestamp"]), default=None)
    if latest:
        batch_root = latest["path"].parent
        for story_dir in sorted(path for path in batch_root.iterdir() if path.is_dir() and path.name.isdigit())[:12]:
            shots = sorted((story_dir / "shots").glob("*.png"))
            if shots:
                gallery.append((f"story {story_dir.name} / first shot", shots[0]))

    output_html.write_text(
        html_page(
            output_html=output_html,
            project_root=project_root,
            batches=batches,
            pipelines=pipelines,
            visualizations=visualizations,
            videos=videos,
            gallery=gallery,
        ),
        encoding="utf-8",
    )
    print(output_html)
    print(json.dumps({"batches": len(batches), "pipelines": len(pipelines), "visualizations": len(visualizations), "videos": len(videos)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
