#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ImagePick:
    path: Path
    version: int
    mtime: float


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def _extract_version(path: Path) -> int:
    m = re.search(r"_v(\d+)$", path.stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def _pick_better(curr: ImagePick | None, cand: Path) -> ImagePick:
    c = ImagePick(path=cand, version=_extract_version(cand), mtime=cand.stat().st_mtime)
    if curr is None:
        return c
    if c.version != curr.version:
        return c if c.version > curr.version else curr
    if c.mtime != curr.mtime:
        return c if c.mtime > curr.mtime else curr
    return c if str(c.path) > str(curr.path) else curr


def _scan_portraits(portraits_dir: Path) -> dict[str, Path]:
    out: dict[str, ImagePick] = {}
    if not portraits_dir.exists():
        return {}
    for p in portraits_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.match(r"(?P<role>.+?)_portrait(?:_v\d+)?$", p.stem)
        if not m:
            continue
        role = m.group("role").strip()
        if not role:
            continue
        out[role] = _pick_better(out.get(role), p)
    return {k: v.path for k, v in out.items()}


def _parse_keyframe_from_step_dir(path: Path) -> tuple[int, str] | None:
    # step_001/Guard_keyframe_v2.png
    if path.parent is None:
        return None
    dm = re.match(r"step_(\d+)$", path.parent.name)
    fm = re.match(r"(?P<role>.+?)_keyframe(?:_v\d+)?$", path.stem)
    if not dm or not fm:
        return None
    step = int(dm.group(1))
    role = fm.group("role").strip()
    if step <= 0 or not role:
        return None
    return step, role


def _parse_keyframe_from_flat_name(path: Path) -> tuple[int, str] | None:
    # step_001_Guard_keyframe.png
    m = re.match(r"step_(\d+)_(?P<role>.+?)_keyframe(?:_v\d+)?$", path.stem)
    if not m:
        return None
    step = int(m.group(1))
    role = m.group("role").strip()
    if step <= 0 or not role:
        return None
    return step, role


def _scan_keyframes(keyframes_dir: Path) -> dict[tuple[int, str], Path]:
    out: dict[tuple[int, str], ImagePick] = {}
    if not keyframes_dir.exists():
        return {}
    for p in keyframes_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = _parse_keyframe_from_step_dir(p) or _parse_keyframe_from_flat_name(p)
        if not parsed:
            continue
        step, role = parsed
        out[(step, role)] = _pick_better(out.get((step, role)), p)
    return {k: v.path for k, v in out.items()}


def _copy_image_to_assets(src: Path, output_dir: Path, group: str, file_id: str) -> str:
    target_dir = output_dir / "assets" / group
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_safe_name(file_id)}{src.suffix.lower()}"
    if not target.exists():
        shutil.copy2(src, target)
    return target.relative_to(output_dir).as_posix()


def _truncate_text(text: str, max_len: int = 56) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rstrip() + "…"


def _load_step_logs_from_context(context_json: Path | None) -> list[dict[str, Any]]:
    if context_json is None or not context_json.exists():
        return []
    try:
        data = json.loads(context_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("step_logs"), list):
        return [x for x in data["step_logs"] if isinstance(x, dict)]
    result = data.get("result", {})
    if isinstance(result, dict) and isinstance(result.get("step_logs"), list):
        return [x for x in result["step_logs"] if isinstance(x, dict)]
    return []


def _build_text_maps(context_json: Path | None) -> tuple[dict[tuple[int, str], str], dict[int, str]]:
    step_logs = _load_step_logs_from_context(context_json)
    role_text_map: dict[tuple[int, str], str] = {}
    step_summary_map: dict[int, str] = {}
    for step_item in step_logs:
        step = int(step_item.get("step", 0) or 0)
        if step <= 0:
            continue
        step_summary_map[step] = str(step_item.get("sam_summary", "") or "").strip()
        role_logs = step_item.get("role_logs", [])
        if not isinstance(role_logs, list):
            continue
        for role in role_logs:
            if not isinstance(role, dict):
                continue
            role_name = str(role.get("role_name", "") or "").strip()
            if not role_name:
                continue
            txt = str(role.get("role_output_text", "") or "").strip()
            if not txt:
                nxt = role.get("next_action", {})
                if isinstance(nxt, dict):
                    txt = str(nxt.get("action_type", "") or "").strip()
            role_text_map[(step, role_name)] = txt
    return role_text_map, step_summary_map


def _load_path_plan(path_plan_json: Path | None) -> tuple[list[str], dict[str, int], float | None]:
    if path_plan_json is None or not path_plan_json.exists():
        return [], {}, None
    try:
        data = json.loads(path_plan_json.read_text(encoding="utf-8"))
    except Exception:
        return [], {}, None
    if not isinstance(data, dict):
        return [], {}, None
    ordered = data.get("ordered_node_ids", [])
    if not isinstance(ordered, list):
        ordered = []
    ordered_ids = [str(x) for x in ordered if str(x).strip()]
    rank_map_raw = data.get("node_rank_map", {})
    rank_map: dict[str, int] = {}
    if isinstance(rank_map_raw, dict):
        for k, v in rank_map_raw.items():
            try:
                rank_map[str(k)] = int(v)
            except Exception:
                continue
    if not rank_map:
        rank_map = {nid: i + 1 for i, nid in enumerate(ordered_ids)}
    score = None
    vb = data.get("value_breakdown", {})
    if isinstance(vb, dict):
        try:
            score = float(vb.get("total_score"))
        except Exception:
            score = None
    if score is None:
        try:
            score = float(data.get("score_total"))
        except Exception:
            score = None
    return ordered_ids, rank_map, score


def _path_label(default_name: str, score: float | None, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    if score is None:
        return default_name
    return f"{default_name} ({score:.2f})"


def _build_nodes_edges(
    *,
    roles: list[str],
    steps: list[int],
    portraits: dict[str, Path],
    keyframes: dict[tuple[int, str], Path],
    ordered_node_ids: list[str] | None = None,
    node_rank_map: dict[str, int] | None = None,
    ordered_node_ids_2: list[str] | None = None,
    node_rank_map_2: dict[str, int] | None = None,
    role_text_map: dict[tuple[int, str], str] | None = None,
    step_summary_map: dict[int, str] | None = None,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    role_text_map = role_text_map or {}
    step_summary_map = step_summary_map or {}
    ordered_node_ids = ordered_node_ids or []
    node_rank_map = node_rank_map or {}
    ordered_node_ids_2 = ordered_node_ids_2 or []
    node_rank_map_2 = node_rank_map_2 or {}

    for role in roles:
        img = portraits.get(role)
        rel_img = _copy_image_to_assets(img, output_dir, "portraits", f"init_{role}") if img else ""
        nodes.append(
            {
                "id": f"init::{role}",
                "kind": "character_init",
                "label": f"Init/{role}",
                "role_name": role,
                "image_path": rel_img,
            }
        )

    for step in steps:
        hue = (step * 53) % 360
        step_color = f"hsl({hue} 72% 45%)"
        for role in roles:
            kf = keyframes.get((step, role))
            if kf is None:
                continue
            rel_img = _copy_image_to_assets(kf, output_dir, "keyframes", f"s{step:03d}_{role}")
            nid = f"step::{step}::{role}"
            nodes.append(
                {
                    "id": nid,
                    "kind": "role_step",
                    "step": step,
                    "role_name": role,
                    "label": f"S{step:03d}/{role}",
                    "color": step_color,
                    "image_path": rel_img,
                    "plot_text": role_text_map.get((step, role), ""),
                    "short_plot": _truncate_text(role_text_map.get((step, role), ""), 52),
                    "step_summary": step_summary_map.get(step, ""),
                    "path_rank": int(node_rank_map.get(nid, 0) or 0),
                    "path_rank_2": int(node_rank_map_2.get(nid, 0) or 0),
                }
            )

    nodes.append(
        {
            "id": "story_end",
            "kind": "story_end",
            "label": "Story End",
            "image_path": "",
        }
    )

    node_ids = {str(n["id"]) for n in nodes}

    for role in roles:
        role_step_nodes = [f"step::{s}::{role}" for s in steps if f"step::{s}::{role}" in node_ids]
        if not role_step_nodes:
            continue
        edges.append({"source": f"init::{role}", "target": role_step_nodes[0], "relation": "start"})
        for i in range(len(role_step_nodes) - 1):
            edges.append({"source": role_step_nodes[i], "target": role_step_nodes[i + 1], "relation": "next"})
        edges.append({"source": role_step_nodes[-1], "target": "story_end", "relation": "converge"})

    # Overlay path orchestrator sequence edges.
    if ordered_node_ids:
        node_id_set = {str(n["id"]) for n in nodes}
        seq = [nid for nid in ordered_node_ids if nid in node_id_set]
        for i in range(len(seq) - 1):
            edges.append(
                {
                    "source": seq[i],
                    "target": seq[i + 1],
                    "relation": "path_next",
                    "path_rank": i + 1,
                    "path_set": 1,
                }
            )
    if ordered_node_ids_2:
        node_id_set = {str(n["id"]) for n in nodes}
        seq2 = [nid for nid in ordered_node_ids_2 if nid in node_id_set]
        for i in range(len(seq2) - 1):
            edges.append(
                {
                    "source": seq2[i],
                    "target": seq2[i + 1],
                    "relation": "path_next_2",
                    "path_rank": i + 1,
                    "path_set": 2,
                }
            )

    return nodes, edges


def _allocate_bin_counts(total: int, bins: int) -> list[int]:
    if bins <= 0:
        return []
    if total <= 0:
        return [0] * bins
    weights = [bins - i for i in range(bins)]  # outer -> inner
    wsum = float(sum(weights))
    raw = [total * (w / wsum) for w in weights]
    base = [int(x) for x in raw]
    remain = total - sum(base)
    frac_order = sorted(range(bins), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in frac_order[:remain]:
        base[i] += 1
    return base


def _layout(nodes: list[dict[str, Any]], radial_bins: int = 0) -> None:
    init_nodes = [n for n in nodes if n.get("kind") == "character_init"]
    step_nodes = [n for n in nodes if n.get("kind") == "role_step"]
    end_nodes = [n for n in nodes if n.get("kind") == "story_end"]

    role_names = sorted({str(n.get("role_name", "")).strip() for n in init_nodes if str(n.get("role_name", "")).strip()})
    if not role_names:
        role_names = sorted({str(n.get("role_name", "")).strip() for n in step_nodes if str(n.get("role_name", "")).strip()})
    step_numbers = sorted({int(n.get("step", 0)) for n in step_nodes if int(n.get("step", 0)) > 0})

    role_count = max(1, len(role_names))
    step_count = max(1, len(step_numbers))
    rank = {s: i for i, s in enumerate(step_numbers, start=1)}

    # Card region for node+thumbnail+label.
    card_w = 172
    card_h = 114

    # Use larger radii so 20-step cards can stay non-overlapping inside each sector.
    outer_radius = max(980.0, 520.0 + step_count * 62.0)
    inner_radius = 180.0
    canvas = int((outer_radius + 420.0) * 2.0)
    cx = canvas // 2
    cy = canvas // 2

    sector_width = (2.0 * math.pi) / role_count
    # Keep margin from sector boundary so card area stays inside its sector.
    sector_margin = min(0.22, sector_width * 0.16)
    bins = max(1, int(radial_bins or step_count))
    bin_span = (outer_radius - inner_radius) / max(1, bins)

    for n in init_nodes:
        role = str(n.get("role_name", "")).strip()
        ridx = role_names.index(role) if role in role_names else 0
        base = (ridx * sector_width) - (math.pi / 2.0)
        r = outer_radius + 170
        n["x"] = int(cx + r * math.cos(base))
        n["y"] = int(cy + r * math.sin(base))
        n["radius"] = 13
        n["color"] = "#334155"
        n["card_w"] = card_w
        n["card_h"] = card_h

    # Per-role sector-constrained annular bins:
    # distance from init->end is split into N equal radial bins;
    # closer bins hold fewer images (outer bins hold more).
    role_nodes: dict[str, list[dict[str, Any]]] = {}
    for n in step_nodes:
        role_nodes.setdefault(str(n.get("role_name", "")).strip(), []).append(n)
    for role, items in role_nodes.items():
        items.sort(key=lambda x: int(x.get("step", 0)))
        role_idx = role_names.index(role) if role in role_names else 0
        sector_center = (role_idx * sector_width) - (math.pi / 2.0)
        sector_start = sector_center - (sector_width / 2.0) + sector_margin
        sector_end = sector_center + (sector_width / 2.0) - sector_margin
        counts = _allocate_bin_counts(total=len(items), bins=bins)
        pointer = 0
        for bidx in range(bins):
            count = counts[bidx]
            if count <= 0:
                continue
            bin_nodes = items[pointer : pointer + count]
            pointer += count

            r_outer = outer_radius - bidx * bin_span
            r_inner = max(inner_radius, r_outer - bin_span)
            r = (r_outer + r_inner) / 2.0 - role_idx * 3.0

            # Serpentine ordering across layers so consecutive steps stay close:
            # bin1: start->end, bin2: end->start, bin3: start->end ...
            forward = (bidx % 2 == 0)
            for j, n in enumerate(bin_nodes, start=1):
                if count == 1:
                    theta = (sector_start + sector_end) / 2.0
                else:
                    t = j / (count + 1.0)
                    if forward:
                        theta = sector_start + t * (sector_end - sector_start)
                    else:
                        theta = sector_end - t * (sector_end - sector_start)
                n["x"] = int(cx + r * math.cos(theta))
                n["y"] = int(cy + r * math.sin(theta))
                step = int(n.get("step", 0))
                _ = rank.get(step, 1)
                n["radius"] = 11
                if not n.get("color"):
                    hue = (step * 53) % 360
                    n["color"] = f"hsl({hue} 72% 45%)"
                n["card_w"] = card_w
                n["card_h"] = card_h
                n["radial_bin"] = bidx + 1

        # fallback if any leftover due unexpected count mismatch
        if pointer < len(items):
            leftovers = items[pointer:]
            for n in leftovers:
                n["x"] = int(cx + inner_radius * math.cos((sector_start + sector_end) / 2.0))
                n["y"] = int(cy + inner_radius * math.sin((sector_start + sector_end) / 2.0))
                n["radius"] = 11
                n["card_w"] = card_w
                n["card_h"] = card_h
                n["radial_bin"] = bins

    for n in end_nodes:
        n["x"] = cx
        n["y"] = cy
        n["radius"] = 18
        n["color"] = "#dc2626"
        n["card_w"] = 140
        n["card_h"] = 84


def _build_html(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    title: str,
    path1_label: str,
    path2_label: str,
) -> str:
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
    .left {{ flex: 1; overflow:auto; padding: 10px; display:flex; flex-direction:column; gap:12px; }}
    .right {{ width: 380px; border-left:1px solid #d1d5db; background:#fff; padding:12px; overflow:auto; }}
    .board {{ background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:10px; }}
    .board h3 {{ margin:2px 0 8px 0; font-size:15px; }}
    svg {{ background: #fff; border:1px solid #d1d5db; border-radius:8px; width:100%; }}
    .legend {{ font-size:12px; margin-bottom:8px; color:#374151; }}
    .chip {{ display:inline-block; padding:2px 8px; border-radius:999px; margin-right:6px; margin-bottom:6px; color:#fff; font-size:11px; }}
    .node-label {{ font-size:14px; font-weight:700; text-anchor:middle; fill:#111; pointer-events:none; }}
    .node-desc {{ font-size:11px; text-anchor:middle; fill:#374151; pointer-events:none; }}
    .path-rank {{ font-size:11px; font-weight:700; text-anchor:middle; fill:#7c2d12; pointer-events:none; }}
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
        <span class="chip" style="background:#f59e0b">{html.escape(path1_label)}</span>
        <span class="chip" style="background:#2563eb">{html.escape(path2_label)}</span>
      </div>
      <div class="board">
        <h3>Narrative Topology</h3>
        <svg id="topologyGraph" width="1400" height="900"></svg>
      </div>
      <div class="board">
        <h3>{html.escape(path1_label)}</h3>
        <svg id="pathGraph1" width="1400" height="520"></svg>
      </div>
      <div class="board">
        <h3>{html.escape(path2_label)}</h3>
        <svg id="pathGraph2" width="1400" height="520"></svg>
      </div>
    </div>
    <div class="right">
      <div id="meta" class="meta">Click a node to inspect metadata.</div>
      <img id="preview" class="preview" style="display:none;" />
    </div>
  </div>

  <script>
    const nodes = {node_json};
    const edges = {edge_json};
    const topoSvg = document.getElementById("topologyGraph");
    const pathSvg1 = document.getElementById("pathGraph1");
    const pathSvg2 = document.getElementById("pathGraph2");
    const ns = "http://www.w3.org/2000/svg";
    const meta = document.getElementById("meta");
    const preview = document.getElementById("preview");
    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
    const topoEdges = edges.filter(e => e.relation !== "path_next" && e.relation !== "path_next_2");
    const pathEdges1 = edges.filter(e => e.relation === "path_next");
    const pathEdges2 = edges.filter(e => e.relation === "path_next_2");
    const pathNodes1 = nodes
      .filter(n => Number(n.path_rank || 0) > 0)
      .sort((a, b) => Number(a.path_rank || 0) - Number(b.path_rank || 0));
    const pathNodes2 = nodes
      .filter(n => Number(n.path_rank_2 || 0) > 0)
      .sort((a, b) => Number(a.path_rank_2 || 0) - Number(b.path_rank_2 || 0));

    function nodeDetail(n) {{
      const clone = JSON.parse(JSON.stringify(n));
      delete clone.x;
      delete clone.y;
      meta.textContent = JSON.stringify(clone, null, 2);
      const imgPath = n.image_path || "";
      if (imgPath) {{
        preview.src = imgPath;
        preview.style.display = "block";
      }} else {{
        preview.style.display = "none";
        preview.removeAttribute("src");
      }}
    }}

    function addArrowDef(svg, id, color) {{
      const defs = document.createElementNS(ns, "defs");
      defs.innerHTML = `<marker id="${{id}}" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="${{color}}"></path></marker>`;
      svg.appendChild(defs);
    }}

    function drawNode(svg, n, showRank=true) {{
      const g = document.createElementNS(ns, "g");
      g.style.cursor = "pointer";
      const cardW = n.card_w || 172;
      const cardH = n.card_h || 114;
      const cardX = n.x - cardW / 2;
      const cardY = n.y - cardH / 2;

      const card = document.createElementNS(ns, "rect");
      card.setAttribute("x", cardX);
      card.setAttribute("y", cardY);
      card.setAttribute("width", cardW);
      card.setAttribute("height", cardH);
      card.setAttribute("rx", 10);
      card.setAttribute("fill", "#ffffff");
      card.setAttribute("stroke", "#d1d5db");
      card.setAttribute("stroke-width", "1");

      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", n.x - 58);
      c.setAttribute("cy", n.y - 14);
      c.setAttribute("r", n.radius || 14);
      c.setAttribute("fill", n.color || "#6b7280");
      c.setAttribute("stroke", "#111827");
      c.setAttribute("stroke-width", "1");

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", n.x);
      label.setAttribute("y", n.y + 40);
      label.setAttribute("class", "node-label");
      label.textContent = n.label || n.id;
      const desc = document.createElementNS(ns, "text");
      desc.setAttribute("x", n.x);
      desc.setAttribute("y", n.y + 56);
      desc.setAttribute("class", "node-desc");
      desc.textContent = (n.short_plot || "");
      const rank = document.createElementNS(ns, "text");
      rank.setAttribute("x", n.x - 58);
      rank.setAttribute("y", n.y + 20);
      rank.setAttribute("class", "path-rank");
      rank.textContent = (n.path_rank && n.path_rank > 0) ? `#${{n.path_rank}}` : "";

      if (n.image_path) {{
        const thumbBox = document.createElementNS(ns, "rect");
        thumbBox.setAttribute("x", n.x + 2);
        thumbBox.setAttribute("y", n.y - 46);
        thumbBox.setAttribute("width", 78);
        thumbBox.setAttribute("height", 78);
        thumbBox.setAttribute("rx", 6);
        thumbBox.setAttribute("fill", "#ffffff");
        thumbBox.setAttribute("stroke", "#d1d5db");
        thumbBox.setAttribute("stroke-width", "1");
        const thumb = document.createElementNS(ns, "image");
        thumb.setAttribute("href", n.image_path);
        thumb.setAttribute("x", n.x + 5);
        thumb.setAttribute("y", n.y - 43);
        thumb.setAttribute("width", 72);
        thumb.setAttribute("height", 72);
        thumb.setAttribute("preserveAspectRatio", "xMidYMid slice");
        g.appendChild(card);
        g.appendChild(thumbBox);
        g.appendChild(thumb);
      }} else {{
        g.appendChild(card);
      }}

      g.appendChild(c);
      g.appendChild(label);
      if (showRank && rank.textContent) g.appendChild(rank);
      if (n.short_plot) g.appendChild(desc);
      g.addEventListener("click", () => nodeDetail(n));
      svg.appendChild(g);
    }}

    function renderTopologyBoard() {{
      const xs = nodes.map(n => n.x || 0);
      const ys = nodes.map(n => n.y || 0);
      const pad = 130;
      const minX = Math.min(...xs) - pad;
      const minY = Math.min(...ys) - pad;
      const maxX = Math.max(...xs) + pad;
      const maxY = Math.max(...ys) + pad;
      const width = Math.max(900, maxX - minX);
      const height = Math.max(900, maxY - minY);
      topoSvg.setAttribute("width", width);
      topoSvg.setAttribute("height", height);
      topoSvg.setAttribute("viewBox", `${{minX}} ${{minY}} ${{width}} ${{height}}`);
      addArrowDef(topoSvg, "topoArrow", "#9ca3af");

      topoEdges.forEach(e => {{
        const s = nodeById[e.source];
        const t = nodeById[e.target];
        if (!s || !t) return;
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", s.x);
        line.setAttribute("y1", s.y);
        line.setAttribute("x2", t.x);
        line.setAttribute("y2", t.y);
        line.setAttribute("stroke", "#9ca3af");
        line.setAttribute("stroke-width", "1.6");
        line.setAttribute("marker-end", "url(#topoArrow)");
        topoSvg.appendChild(line);
      }});

      nodes.forEach(n => drawNode(topoSvg, n, false));
    }}

    function renderPathBoard(pathSvg, pathNodes, pathEdges, edgeColor, edgeId, rankField) {{
      if (!pathNodes.length) {{
        pathSvg.setAttribute("viewBox", "0 0 1200 300");
        const t = document.createElementNS(ns, "text");
        t.setAttribute("x", 40);
        t.setAttribute("y", 80);
        t.setAttribute("fill", "#6b7280");
        t.setAttribute("font-size", "18");
        t.textContent = "No path sequence found.";
        pathSvg.appendChild(t);
        return;
      }}

      const cols = Math.min(6, Math.max(3, Math.ceil(Math.sqrt(pathNodes.length))));
      const gapX = 240;
      const gapY = 170;
      const startX = 140;
      const startY = 120;
      const pNodes = pathNodes.map((n, i) => {{
        const row = Math.floor(i / cols);
        const col = i % cols;
        const x = startX + col * gapX;
        const y = startY + row * gapY;
        return {{ ...n, x, y }};
      }});
      const pMap = Object.fromEntries(pNodes.map(n => [n.id, n]));
      const pWidth = Math.max(1200, startX * 2 + (cols - 1) * gapX + 200);
      const rows = Math.max(1, Math.ceil(pNodes.length / cols));
      const pHeight = Math.max(420, startY * 2 + (rows - 1) * gapY + 120);
      pathSvg.setAttribute("width", pWidth);
      pathSvg.setAttribute("height", pHeight);
      pathSvg.setAttribute("viewBox", `0 0 ${{pWidth}} ${{pHeight}}`);
      addArrowDef(pathSvg, edgeId, edgeColor);

      pathEdges.forEach(e => {{
        const s = pMap[e.source];
        const t = pMap[e.target];
        if (!s || !t) return;
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", s.x);
        line.setAttribute("y1", s.y);
        line.setAttribute("x2", t.x);
        line.setAttribute("y2", t.y);
        line.setAttribute("stroke", edgeColor);
        line.setAttribute("stroke-width", "2.6");
        line.setAttribute("stroke-dasharray", "7 4");
        line.setAttribute("marker-end", `url(#${{edgeId}})`);
        pathSvg.appendChild(line);
      }});

      pNodes.forEach(n => {{
        const n2 = {{...n, path_rank: Number(n[rankField] || 0)}};
        drawNode(pathSvg, n2, true);
      }});
    }}

    renderTopologyBoard();
    renderPathBoard(pathSvg1, pathNodes1, pathEdges1, "#f59e0b", "pathArrow1", "path_rank");
    renderPathBoard(pathSvg2, pathNodes2, pathEdges2, "#2563eb", "pathArrow2", "path_rank_2");
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build topology visualization from character portraits + keyframes only.")
    parser.add_argument("--assets-dir", type=Path, required=True, help="Directory containing portraits/ and keyframes/ (e.g. narrative_artifacts)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Narrative Topology From Images")
    parser.add_argument("--characters", type=str, default="", help="Optional comma-separated role names")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max step cap. 0 means no cap")
    parser.add_argument(
        "--context-json",
        type=Path,
        default=None,
        help="Optional result/checkpoint json for plot text (role_output_text) enrichment.",
    )
    parser.add_argument(
        "--path-plan-json",
        type=Path,
        default=None,
        help="Optional narrative path plan json (ordered_node_ids) to overlay path sequence.",
    )
    parser.add_argument(
        "--path-plan-json-2",
        type=Path,
        default=None,
        help="Optional second path plan json for comparison overlay.",
    )
    parser.add_argument(
        "--path-label-1",
        type=str,
        default="",
        help="Legend/title label for first path.",
    )
    parser.add_argument(
        "--path-label-2",
        type=str,
        default="",
        help="Legend/title label for second path.",
    )
    parser.add_argument(
        "--radial-bins",
        type=int,
        default=0,
        help="Split init->end distance into N equal radial bins. 0 means use step count.",
    )
    args = parser.parse_args()

    assets_dir = args.assets_dir
    portraits_dir = assets_dir / "portraits"
    keyframes_dir = assets_dir / "keyframes"
    output_dir = args.output_dir or (assets_dir.parent / "narrative_topology_from_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    portraits = _scan_portraits(portraits_dir)
    keyframes = _scan_keyframes(keyframes_dir)
    role_text_map, step_summary_map = _build_text_maps(args.context_json)
    ordered_node_ids, node_rank_map, path_score_1 = _load_path_plan(args.path_plan_json)
    ordered_node_ids_2, node_rank_map_2, path_score_2 = _load_path_plan(args.path_plan_json_2)
    path_label_1 = _path_label("Path #1", path_score_1, args.path_label_1)
    path_label_2 = _path_label("Path #2", path_score_2, args.path_label_2)

    cli_roles = [x.strip() for x in args.characters.split(",") if x.strip()]
    roles_set = set(cli_roles)
    roles_set.update(portraits.keys())
    roles_set.update(role for (_step, role) in keyframes.keys())
    roles = sorted(roles_set)

    steps = sorted({step for (step, _role) in keyframes.keys()})
    if args.max_steps > 0:
        steps = [s for s in steps if s <= args.max_steps]

    # Keep only selected steps if max step cap is set.
    if args.max_steps > 0:
        keyframes = {(s, r): p for (s, r), p in keyframes.items() if s <= args.max_steps}

    nodes, edges = _build_nodes_edges(
        roles=roles,
        steps=steps,
        portraits=portraits,
        keyframes=keyframes,
        ordered_node_ids=ordered_node_ids,
        node_rank_map=node_rank_map,
        ordered_node_ids_2=ordered_node_ids_2,
        node_rank_map_2=node_rank_map_2,
        role_text_map=role_text_map,
        step_summary_map=step_summary_map,
        output_dir=output_dir,
    )
    _layout(nodes, radial_bins=args.radial_bins)

    graph_json = output_dir / "topology_graph.json"
    graph_html = output_dir / "topology_graph.html"
    summary_json = output_dir / "topology_summary.json"

    graph_json.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph_html.write_text(
        _build_html(nodes, edges, args.title, path_label_1, path_label_2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(
            {
                "assets_dir": str(assets_dir),
                "roles": roles,
                "steps": steps,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "radial_bins": max(1, int(args.radial_bins or len(steps) or 1)),
                "context_json": str(args.context_json) if args.context_json else "",
                "path_plan_json": str(args.path_plan_json) if args.path_plan_json else "",
                "path_plan_json_2": str(args.path_plan_json_2) if args.path_plan_json_2 else "",
                "path_label_1": path_label_1,
                "path_label_2": path_label_2,
                "output_html": str(graph_html),
                "output_json": str(graph_json),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(graph_html)


if __name__ == "__main__":
    main()
