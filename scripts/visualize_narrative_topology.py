#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError(f"JSON root must be object: {path}")


def _normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    # Accept both result.json and narrative_checkpoint.json
    if "result" in data and isinstance(data["result"], dict):
        result = data["result"]
        plan = data.get("plan", {})
        out = {
            "story_outline": {
                "story": data.get("story", ""),
                "max_steps": data.get("completed_steps", 0),
            },
            "character_configs": plan.get("character_configs", []),
            "character_portraits": result.get("character_portraits", {}),
            "step_logs": result.get("step_logs", []),
            "final_world_state": result.get("final_world_state", {}),
        }
        return out
    return data


def _build_graph(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_logs = payload.get("step_logs", []) or []
    character_cfgs = payload.get("character_configs", []) or []
    portraits = payload.get("character_portraits", {}) or {}

    character_names: list[str] = []
    for c in character_cfgs:
        name = str(c.get("name", "")).strip()
        if name and name not in character_names:
            character_names.append(name)
    if not character_names:
        seen: set[str] = set()
        for step in step_logs:
            for role in step.get("role_logs", []) or []:
                name = str(role.get("role_name", "")).strip()
                if name and name not in seen:
                    character_names.append(name)
                    seen.add(name)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    step_to_roles: dict[int, dict[str, str]] = {}
    step_numbers: list[int] = []

    for name in character_names:
        nodes.append(
            {
                "id": f"init::{name}",
                "kind": "character_init",
                "label": f"Init/{name}",
                "role_name": name,
                "image_path": portraits.get(name, ""),
            }
        )

    for step_item in step_logs:
        step_id = int(step_item.get("step", 0))
        if step_id <= 0:
            continue
        if step_id not in step_numbers:
            step_numbers.append(step_id)
        step_to_roles.setdefault(step_id, {})
        step_summary = str(step_item.get("sam_summary", "") or "")
        step_anchor = str(step_item.get("sam_story_anchor", "") or "")
        for role in step_item.get("role_logs", []) or []:
            role_name = str(role.get("role_name", "")).strip()
            if not role_name:
                continue
            node_id = f"step::{step_id}::{role_name}"
            next_action = role.get("next_action", {}) or {}
            kf_path = str(role.get("action_keyframe_path", "") or "").strip()
            image_path = kf_path or str(portraits.get(role_name, "") or "")
            nodes.append(
                {
                    "id": node_id,
                    "kind": "role_step",
                    "step": step_id,
                    "role_name": role_name,
                    "label": f"S{step_id:03d}/{role_name}",
                    "step_summary": step_summary,
                    "step_anchor": step_anchor,
                    "action_type": str(next_action.get("action_type", "") or ""),
                    "target_character": str(next_action.get("target_character", "") or ""),
                    "reason": str(next_action.get("reason", "") or ""),
                    "monologue": str(next_action.get("monologue", "") or ""),
                    "location": str(next_action.get("location", "") or ""),
                    "confidence": str(next_action.get("confidence", "") or ""),
                    "text": str(role.get("role_output_text", "") or ""),
                    "image_path": image_path,
                }
            )
            step_to_roles[step_id][role_name] = node_id

    step_numbers = sorted(step_numbers)
    nodes.append(
        {
            "id": "story_end",
            "kind": "story_end",
            "label": "Story End",
            "text": str(payload.get("story_outline", {}).get("story", "") or ""),
        }
    )

    if step_numbers:
        first_step = step_numbers[0]
        for role_name in character_names:
            first_node = step_to_roles.get(first_step, {}).get(role_name, "")
            if first_node:
                edges.append({"source": f"init::{role_name}", "target": first_node, "relation": "start"})

    for idx, step_id in enumerate(step_numbers):
        next_step = step_numbers[idx + 1] if idx + 1 < len(step_numbers) else None
        curr_map = step_to_roles.get(step_id, {})
        for role_name, curr_node in curr_map.items():
            if next_step is not None:
                next_node = step_to_roles.get(next_step, {}).get(role_name, "")
                if next_node:
                    edges.append({"source": curr_node, "target": next_node, "relation": "next_step"})
            else:
                edges.append({"source": curr_node, "target": "story_end", "relation": "converge"})

            # In-step interaction edge: point to target character in same step if present.
            node_obj = next((n for n in nodes if n.get("id") == curr_node), None)
            if node_obj is None:
                continue
            target = str(node_obj.get("target_character", "") or "").strip()
            if target and target.lower() != "none":
                tnode = curr_map.get(target, "")
                if tnode:
                    edges.append({"source": curr_node, "target": tnode, "relation": "targets"})

    # Dedup nodes by id (keep first)
    uniq_nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for n in nodes:
        nid = str(n.get("id", ""))
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        uniq_nodes.append(n)
    return uniq_nodes, edges


def _parse_keyframe_version(path: Path) -> int:
    m = re.search(r"_v(\d+)$", path.stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def _try_resolve_candidate(path_str: str, input_dir: Path) -> Path | None:
    raw = Path(path_str)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((input_dir / raw).resolve())
        candidates.append(raw.resolve())

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _resolve_keyframe_fallback(path_str: str, input_dir: Path) -> Path | None:
    raw = Path(path_str)
    role = raw.stem.split("_keyframe", 1)[0]
    if not role:
        return None

    step_dir = raw.parent
    step_candidates: list[Path] = []
    if step_dir.is_absolute():
        step_candidates.append(step_dir)
    else:
        step_candidates.append((input_dir / step_dir).resolve())
        step_candidates.append(step_dir.resolve())

    matches: list[Path] = []
    for sdir in step_candidates:
        if sdir.exists() and sdir.is_dir():
            matches.extend(sdir.glob(f"{role}_keyframe*.png"))
            matches.extend(sdir.glob(f"{role}_keyframe*.jpg"))
            matches.extend(sdir.glob(f"{role}_keyframe*.jpeg"))

    if not matches:
        # broad fallback in the narrative artifacts tree
        base = input_dir / "narrative_artifacts" / "keyframes"
        if base.exists():
            matches.extend(base.rglob(f"{role}_keyframe*.png"))
            matches.extend(base.rglob(f"{role}_keyframe*.jpg"))
            matches.extend(base.rglob(f"{role}_keyframe*.jpeg"))

    if not matches:
        return None

    matches = sorted(matches, key=lambda p: (_parse_keyframe_version(p), p.stat().st_mtime), reverse=True)
    return matches[0]


def _resolve_image_file(path_str: str, input_dir: Path) -> Path | None:
    if not path_str:
        return None
    direct = _try_resolve_candidate(path_str, input_dir)
    if direct is not None:
        return direct
    if "_keyframe" in Path(path_str).stem:
        return _resolve_keyframe_fallback(path_str, input_dir)
    return None


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "node"


def _localize_node_images(nodes: list[dict[str, Any]], output_dir: Path, input_dir: Path) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for idx, node in enumerate(nodes):
        img = str(node.get("image_path", "") or "").strip()
        if not img:
            continue
        src = _resolve_image_file(img, input_dir)
        if src is None:
            node["missing_image_path"] = img
            node["image_path"] = ""
            continue
        kind = str(node.get("kind", "misc"))
        target_dir = assets_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        nid = _safe_name(str(node.get("id", f"node_{idx}")))
        target = target_dir / f"{nid}{src.suffix.lower()}"
        # avoid redundant copies
        if not target.exists():
            shutil.copy2(src, target)
        node["image_path"] = target.relative_to(output_dir).as_posix()
        node["source_image_path"] = str(src)


def _layout_nodes(nodes: list[dict[str, Any]]) -> None:
    import math

    init_nodes = [n for n in nodes if n.get("kind") == "character_init"]
    step_nodes = [n for n in nodes if n.get("kind") == "role_step"]
    end_nodes = [n for n in nodes if n.get("kind") == "story_end"]
    role_names = sorted({str(n.get("role_name", "")).strip() for n in init_nodes if str(n.get("role_name", "")).strip()})
    if not role_names:
        role_names = sorted({str(n.get("role_name", "")).strip() for n in step_nodes if str(n.get("role_name", "")).strip()})
    step_numbers = sorted({int(n.get("step", 0)) for n in step_nodes if int(n.get("step", 0)) > 0})

    role_count = max(1, len(role_names))
    step_count = max(1, len(step_numbers))
    canvas = max(1500, 700 + role_count * 180 + step_count * 150)
    cx, cy = canvas // 2, canvas // 2
    max_radius = int(canvas * 0.40)
    ring_gap = max(90, int(max_radius / (step_count + 2)))
    init_radius = max_radius

    role_angles: dict[str, float] = {}
    for i, role_name in enumerate(role_names):
        role_angles[role_name] = (2.0 * math.pi * i / role_count) - (math.pi / 2.0)

    for n in init_nodes:
        role_name = str(n.get("role_name", "")).strip()
        angle = role_angles.get(role_name, -math.pi / 2.0)
        n["x"] = int(cx + init_radius * math.cos(angle))
        n["y"] = int(cy + init_radius * math.sin(angle))
        n["radius"] = 16
        n["color"] = "#334155"

    step_rank = {s: i for i, s in enumerate(step_numbers, start=1)}
    for n in step_nodes:
        role_name = str(n.get("role_name", "")).strip()
        step_id = int(n.get("step", 0))
        angle = role_angles.get(role_name, -math.pi / 2.0)
        rank = step_rank.get(step_id, 1)
        radius = max(120, init_radius - rank * ring_gap)
        n["x"] = int(cx + radius * math.cos(angle))
        n["y"] = int(cy + radius * math.sin(angle))
        n["radius"] = 13
        hue = (rank * 53) % 360
        n["color"] = f"hsl({hue} 72% 45%)"

    for n in end_nodes:
        n["x"] = cx
        n["y"] = cy
        n["radius"] = 22
        n["color"] = "#dc2626"


def _build_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str) -> str:
    node_json = json.dumps(nodes, ensure_ascii=False)
    edge_json = json.dumps(edges, ensure_ascii=False)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#f5f7fb; color:#111; }}
    .top {{ padding:12px 16px; background:#111827; color:#fff; }}
    .wrap {{ display:flex; height: calc(100vh - 52px); }}
    .left {{ flex: 1; overflow:auto; padding: 10px; }}
    .right {{ width: 380px; border-left:1px solid #d1d5db; background:#fff; padding:12px; overflow:auto; }}
    svg {{ background: #fff; border:1px solid #d1d5db; border-radius:8px; }}
    .legend {{ font-size:12px; margin-bottom:8px; color:#374151; }}
    .chip {{ display:inline-block; padding:2px 8px; border-radius:999px; margin-right:6px; margin-bottom:6px; color:#fff; font-size:11px; }}
    .node-label {{ font-size:11px; text-anchor:middle; fill:#111; pointer-events:none; }}
    .panel-title {{ font-size:16px; margin:0 0 8px 0; }}
    .meta {{ font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; }}
    img.preview {{ width: 100%; border:1px solid #ddd; border-radius:6px; margin-top:8px; }}
  </style>
</head>
<body>
  <div class="top"><strong>{safe_title}</strong></div>
  <div class="wrap">
    <div class="left">
      <div class="legend">
        <span class="chip" style="background:#334155">character init</span>
        <span class="chip" style="background:#0f766e">step role nodes (same step same color)</span>
        <span class="chip" style="background:#dc2626">story end</span>
      </div>
      <svg id="graph" width="2200" height="2200"></svg>
    </div>
    <div class="right">
      <h3 class="panel-title">Node Detail</h3>
      <div id="meta" class="meta">Click a node to inspect metadata.</div>
      <img id="preview" class="preview" style="display:none;" />
    </div>
  </div>

  <script>
    const nodes = {node_json};
    const edges = {edge_json};
    const svg = document.getElementById("graph");
    const ns = "http://www.w3.org/2000/svg";
    const meta = document.getElementById("meta");
    const preview = document.getElementById("preview");

    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
    const width = Math.max(...nodes.map(n => n.x || 0), 2200) + 120;
    const height = Math.max(...nodes.map(n => n.y || 0), 2200) + 120;
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);

    // defs arrow
    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML = `<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#9ca3af"></path></marker>`;
    svg.appendChild(defs);

    // draw edges
    edges.forEach(e => {{
      const s = nodeById[e.source];
      const t = nodeById[e.target];
      if (!s || !t) return;
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", s.x);
      line.setAttribute("y1", s.y);
      line.setAttribute("x2", t.x);
      line.setAttribute("y2", t.y);
      line.setAttribute("stroke", "#9ca3af");
      line.setAttribute("stroke-width", "1.4");
      line.setAttribute("marker-end", "url(#arrow)");
      svg.appendChild(line);
    }});

    function nodeDetail(n) {{
      const clone = JSON.parse(JSON.stringify(n));
      delete clone.x;
      delete clone.y;
      meta.textContent = JSON.stringify(clone, null, 2);
      const imgPath = n.image_path || n.portrait_path || "";
      if (imgPath) {{
        preview.src = imgPath;
        preview.style.display = "block";
      }} else {{
        preview.style.display = "none";
        preview.removeAttribute("src");
      }}
    }}

    // draw nodes
    nodes.forEach(n => {{
      const g = document.createElementNS(ns, "g");
      g.style.cursor = "pointer";
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", n.x);
      c.setAttribute("cy", n.y);
      c.setAttribute("r", n.radius || 14);
      c.setAttribute("fill", n.color || "#6b7280");
      c.setAttribute("stroke", "#111827");
      c.setAttribute("stroke-width", "1");
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", n.x);
      label.setAttribute("y", n.y + 34);
      label.setAttribute("class", "node-label");
      label.textContent = n.label || n.id;
      if (n.image_path) {{
        const thumbBox = document.createElementNS(ns, "rect");
        thumbBox.setAttribute("x", n.x + 16);
        thumbBox.setAttribute("y", n.y - 34);
        thumbBox.setAttribute("width", 56);
        thumbBox.setAttribute("height", 56);
        thumbBox.setAttribute("rx", 6);
        thumbBox.setAttribute("fill", "#ffffff");
        thumbBox.setAttribute("stroke", "#d1d5db");
        thumbBox.setAttribute("stroke-width", "1");
        const thumb = document.createElementNS(ns, "image");
        thumb.setAttribute("href", n.image_path);
        thumb.setAttribute("x", n.x + 18);
        thumb.setAttribute("y", n.y - 32);
        thumb.setAttribute("width", 52);
        thumb.setAttribute("height", 52);
        thumb.setAttribute("preserveAspectRatio", "xMidYMid slice");
        g.appendChild(thumbBox);
        g.appendChild(thumb);
      }}
      g.appendChild(c);
      g.appendChild(label);
      g.addEventListener("click", () => nodeDetail(n));
      svg.appendChild(g);
    }});
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize narrative topology graph with keyframes and all nodes.")
    parser.add_argument("--input-json", type=Path, required=True, help="result.json or narrative_checkpoint.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Narrative Topology Graph")
    args = parser.parse_args()

    data = _load_json(args.input_json)
    payload = _normalize_payload(data)
    nodes, edges = _build_graph(payload)
    _localize_node_images(nodes, output_dir=(args.output_dir or (args.input_json.parent / "narrative_topology")), input_dir=args.input_json.parent)
    _layout_nodes(nodes)

    output_dir = args.output_dir or (args.input_json.parent / "narrative_topology")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_json_path = output_dir / "topology_graph.json"
    graph_html_path = output_dir / "topology_graph.html"
    summary_path = output_dir / "topology_summary.json"

    graph_json_path.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph_html_path.write_text(_build_html(nodes, edges, args.title), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "input_json": str(args.input_json),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "output_json": str(graph_json_path),
                "output_html": str(graph_html_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(graph_html_path)


if __name__ == "__main__":
    main()
