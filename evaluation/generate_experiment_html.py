#!/usr/bin/env python3
"""Build a standalone HTML experiment document from a completed comparison run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHOD_LABELS = {"CameraMaster": "CameraMaster", "ViMax": "ViMax"}
TRACK_DESCRIPTIONS = {
    "A": {
        "title": "剧情锚点给定：忠实可视化（L3 arc sequence）",
        "question": "既定剧情是否被正确、连贯地视觉化？",
        "evaluation": "采用基准锚点和生成 clip 边界；每个镜头具有可靠 scene_id，因此可计算 IC、CC、GC。",
        "reporting": "本例有 12 个镜头、3 个场景；该自定义适配样本用于 ViMax-Bench 风格评测，不等同于官方 ViMax-Bench 测试集结果。",
    },
    "B": {
        "title": "故事梗概给定：受约束扩展（global-story sequence）",
        "question": "在全局故事约束下，是否产生合理的剧情扩展？",
        "evaluation": "以全局故事序列评估；SSS v3 对 Track B 使用自动镜头检测。严格报告不提供正式 scene_id，因此只报告 GC。",
        "reporting": "Track B 是长序列诊断，镜头数和时长可能超出 ViMax-Bench 的常规样本范围；不得将其分数与 Track A 解释为一组受控 A/B 差值。",
    },
}

METRICS = [
    ("IC", "Intra-Consistency", "同一 scene 内所有不同镜头对的 ViCLIP 视觉 embedding 余弦相似度均值。衡量场景内视觉一致性。"),
    ("CC", "Cross-Consistency", "不同 scene 间所有镜头对的 ViCLIP 视觉 embedding 余弦相似度均值。衡量跨场景视觉一致性。"),
    ("GC", "Global-Consistency", "故事内全部不同镜头对的 ViCLIP 视觉 embedding 余弦相似度均值。衡量全局视觉一致性。"),
    ("CBC", "Character Behavior Consistency", "LLM 以 1–5 分判断角色身份、目标、行为和动作是否跨镜头连贯。"),
    ("NC", "Narrative Coherence", "LLM 以 1–5 分判断按顺序排列的镜头能否形成可理解的因果与时间推进。"),
    ("PPR", "Plot Pacing and Rhythm", "LLM 以 1–5 分判断镜头节奏、升级、停顿和粒度是否服务于叙事。"),
    ("VSU", "Visual Specificity and Shot Usability", "LLM 以 1–5 分判断主体、构图、环境、动作和摄影信息是否具体到可执行。"),
    ("STQ", "Scene Transition and Continuity Quality", "LLM 以 1–5 分判断空间、时间和状态变化在场景转换中的连续性。"),
    ("NF", "Narrative Faithfulness", "原始完整故事文本不可用时不计算。本次仅有 proxy reference story，故 NF 标为 N/A。"),
    ("SSS_emb", "Story Similarity Score (embedding)", "对 reference story 与 L4 重建故事分别取 text-embedding-3-small 向量，并计算原始余弦相似度；L1/L2/L3 仅用于生成 L4，不参与分层平均。"),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def fmt(value: Any, digits: int = 6) -> str:
    if value in (None, "", "None"):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def rel_link(path: str) -> str:
    return esc(path.replace("\\", "/"))


def read_rows(comparison_dir: Path) -> list[dict[str, str]]:
    path = comparison_dir / "comparison.csv"
    if not path.is_file():
        path = comparison_dir / "quantitative" / "comparison.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def planning_scores(comparison_dir: Path, method: str, track: str) -> tuple[dict[str, Any], int, str]:
    stem = f"{method.lower()}_track_{track}"
    payload = load_json(comparison_dir / "quantitative" / f"{stem}_planning.json")
    sample = payload["samples"][0]
    planning = sample["planning"]
    return planning["aggregate"]["scores"], int(planning["aggregate"]["batch_count"]), str(payload.get("model") or "")


def visual_metadata(comparison_dir: Path, method: str, track: str) -> dict[str, Any]:
    stem = f"{method.lower()}_track_{track}"
    payload = load_json(comparison_dir / "quantitative" / f"{stem}_visual.json")
    sample = payload["samples"][0]
    return {
        "encoder": payload.get("encoder"),
        "backend": payload.get("backend"),
        "sample_metrics": sample.get("metrics", {}),
        "official": payload.get("results_are_official_vimax_bench"),
        "note": payload.get("result_note"),
    }


def sss_result(comparison_dir: Path, method: str, track: str) -> dict[str, Any]:
    stem = f"{method.lower()}_track_{track}"
    return load_json(comparison_dir / "quantitative" / f"{stem}_sss.json")


def tag(text: str, kind: str = "neutral") -> str:
    return f'<span class="tag {kind}">{esc(text)}</span>'


def metric_table(rows: list[dict[str, str]], comparison_dir: Path) -> str:
    body: list[str] = []
    for row in rows:
        method, track = row["method"], row["track"]
        planning, batches, judge_model = planning_scores(comparison_dir, method, track)
        body.append("<tr>" + "".join([
            f"<td><strong>{esc(method)}</strong></td>",
            f"<td>{tag('Track ' + track, 'blue' if track == 'A' else 'orange')}</td>",
            f"<td>{esc(row.get('clip_count'))}</td>",
            *(f"<td>{fmt(row.get(key))}</td>" for key in ("ic", "cc", "gc")),
            *(f"<td>{fmt(planning.get(key), 1)}</td>" for key in ("CBC", "NC", "PPR", "VSU", "STQ")),
            f"<td>N/A</td>",
            f"<td>{fmt(row.get('sss_emb'))}</td>",
            f"<td class=\"muted\">{batches} batch · {esc(judge_model)}</td>",
        ]) + "</tr>")
    return """<div class="table-wrap"><table><caption>表 1. 量化结果（所有指标均为越高越好）</caption>
<thead><tr><th>方法</th><th>轨道</th><th>镜头数</th><th>IC</th><th>CC</th><th>GC</th><th>CBC</th><th>NC</th><th>PPR</th><th>VSU</th><th>STQ</th><th>NF</th><th>SSS_emb</th><th>规划评估</th></tr></thead>
<tbody>""" + "\n".join(body) + "</tbody></table></div>"


def low_score_section(comparison_dir: Path) -> str:
    analysis_dir = comparison_dir / "analysis"
    manifest = load_json(analysis_dir / "analysis_manifest.json")
    with (analysis_dir / "below_average_visual_pairs.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["track"], record["metric"])].append(record)
    blocks = [
        "<p>判定规则：在每个 <code>method × track × metric</code> 内，列出严格满足 <code>pair cosine &lt; 本组均值</code> 的镜头对。相同 pair 可能同时出现在 IC/CC 与 GC 中，因此总记录数并非独立 pair 数。</p>",
        f"<p>{tag('已定位 ' + str(manifest['below_average_visual_pair_count']) + ' 条低于均值记录', 'red')} "
        f"<a href=\"analysis/below_average_visual_pairs.csv\">下载完整 CSV（含视频路径与完整分镜文本）</a></p>",
    ]
    for (method, track, metric), items in sorted(grouped.items()):
        items.sort(key=lambda item: float(item["pair_cosine"]))
        lines = [f"<details><summary><strong>{esc(method)} · Track {esc(track)} · {esc(metric)}</strong> — 最低 5 条（共 {len(items)} 条）</summary>",
                 "<div class=\"table-wrap\"><table><thead><tr><th>cosine</th><th>距均值</th><th>镜头 A</th><th>镜头 B</th><th>分镜内容</th></tr></thead><tbody>"]
        for item in items[:5]:
            story = f"A: {item['shot_a_storyboard_description']}<br><br>B: {item['shot_b_storyboard_description']}"
            lines.append("<tr>" + "".join([
                f"<td>{fmt(item['pair_cosine'])}</td>", f"<td>{fmt(item['delta_to_mean'])}</td>",
                f"<td><code>{esc(item['shot_a_id'])}</code></td>", f"<td><code>{esc(item['shot_b_id'])}</code></td>",
                f"<td class=\"story-cell\">{esc(story).replace('&lt;br&gt;', '<br>')}</td>",
            ]) + "</tr>")
        lines.append("</tbody></table></div></details>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def sss_section(comparison_dir: Path) -> str:
    records = load_json(comparison_dir / "analysis" / "sss_emb_inputs.json")
    parts = [
        "<p><code>SSS_emb = cosine(embedding(reference story), embedding(normalized L4 reconstructed story))</code>。本次 embedding 模型为 <code>text-embedding-3-small</code>，两个向量维度均为 1536。</p>",
        "<p><a href=\"analysis/sss_emb_inputs.json\">下载机器可读的完整输入 JSON</a>；下列文本即本次实际送入 embedding 的故事输入。</p>",
    ]
    for record in records:
        parts.append(f"<details><summary><strong>{esc(record['method'])} · Track {esc(record['track'])}</strong> — SSS_emb = {fmt(record['sss_emb'])}</summary>"
                     "<div class=\"story-grid\">"
                     f"<div><h4>Reference story input</h4><pre>{esc(record['reference_story_input'])}</pre></div>"
                     f"<div><h4>L4 reconstructed story input</h4><pre>{esc(record['reconstructed_l4_story_input'])}</pre></div>"
                     "</div></details>")
    return "\n".join(parts)


def html_document(comparison_dir: Path) -> str:
    rows = read_rows(comparison_dir)
    visual = visual_metadata(comparison_dir, "CameraMaster", "A")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric_rows = "\n".join(
        f"<tr><td><code>{esc(code)}</code></td><td><strong>{esc(name)}</strong></td><td>{esc(desc)}</td></tr>"
        for code, name, desc in METRICS
    )
    track_cards = "\n".join(
        f"<section class=\"track-card\"><div class=\"track-top\">{tag('Track ' + track, 'blue' if track == 'A' else 'orange')}<h3>{esc(info['title'])}</h3></div>"
        f"<p><strong>核心问题：</strong>{esc(info['question'])}</p><p><strong>评测处理：</strong>{esc(info['evaluation'])}</p>"
        f"<p class=\"note\"><strong>报告边界：</strong>{esc(info['reporting'])}</p></section>"
        for track, info in TRACK_DESCRIPTIONS.items()
    )
    official_note = esc(str(visual.get("note") or ""))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraMaster 与 ViMax 对比实验文档</title>
<style>
:root {{ --ink:#17212b; --muted:#5c6875; --line:#d9e0e7; --panel:#f7f9fb; --blue:#175f9d; --blue-bg:#e9f3fb; --orange:#9a5412; --orange-bg:#fff1e4; --red:#9c2633; --red-bg:#fbecee; --green:#126b4f; --green-bg:#eaf7f1; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,"Microsoft YaHei",Arial,sans-serif; line-height:1.58; }}
main {{ max-width:1320px; margin:0 auto; padding:42px 28px 72px; }} h1 {{ margin:0 0 8px; font-size:30px; letter-spacing:-.35px; }} h2 {{ margin:44px 0 14px; padding-top:8px; border-top:1px solid var(--line); font-size:21px; }} h3 {{ margin:0; font-size:17px; }} h4 {{ margin:0 0 8px; font-size:14px; }} p {{ margin:8px 0; }} a {{ color:var(--blue); }} code {{ font-family:"Cascadia Code",Consolas,monospace; font-size:.88em; }}
.subtitle,.muted {{ color:var(--muted); }} .hero {{ display:grid; grid-template-columns:1fr auto; gap:20px; align-items:end; padding-bottom:22px; border-bottom:3px solid var(--ink); }} .meta {{ text-align:right; color:var(--muted); font-size:13px; }}
.callout {{ margin:20px 0; padding:14px 16px; border-left:4px solid var(--orange); background:var(--orange-bg); }} .note {{ color:var(--muted); font-size:14px; }} .tag {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:700; white-space:nowrap; }} .blue {{ background:var(--blue-bg); color:var(--blue); }} .orange {{ background:var(--orange-bg); color:var(--orange); }} .red {{ background:var(--red-bg); color:var(--red); }} .neutral {{ background:var(--panel); color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .track-card {{ border:1px solid var(--line); padding:18px; background:var(--panel); }} .track-top {{ display:flex; gap:10px; align-items:center; margin-bottom:12px; }}
.table-wrap {{ width:100%; overflow-x:auto; border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} caption {{ text-align:left; padding:11px 13px; background:var(--panel); font-weight:700; font-size:14px; }} th,td {{ padding:9px 10px; border-top:1px solid var(--line); vertical-align:top; text-align:left; }} th {{ background:#fbfcfd; font-weight:700; white-space:nowrap; }} tbody tr:nth-child(even) {{ background:#fcfdfe; }} .story-cell {{ min-width:440px; max-width:650px; font-size:12px; }}
details {{ margin:10px 0; border:1px solid var(--line); background:#fff; }} summary {{ cursor:pointer; padding:10px 12px; background:var(--panel); }} details > div, details > p {{ padding:0 12px 12px; }} details .table-wrap {{ margin:12px; width:calc(100% - 24px); }} .story-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:14px; }} pre {{ margin:0; padding:12px; white-space:pre-wrap; word-break:break-word; background:var(--panel); border:1px solid var(--line); font:12px/1.55 "Cascadia Code",Consolas,monospace; }}
.footer {{ margin-top:46px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }}
@media (max-width:760px) {{ main {{ padding:28px 16px 48px; }} .hero,.grid,.story-grid {{ grid-template-columns:1fr; }} .meta {{ text-align:left; }} h1 {{ font-size:25px; }} }}
</style></head><body><main>
<header class="hero"><div><h1>CameraMaster 与 ViMax：定量与定性对比实验文档</h1><p class="subtitle">案例：<code>artist_extreme_weather_typeA</code> · 双轨道生成与评估汇总</p></div><div class="meta">生成时间：{generated}<br>结果目录：<code>{esc(str(comparison_dir))}</code></div></header>
<section class="callout"><strong>结果解释边界。</strong> 这是基于自定义案例的 ViMax-Bench 风格适配评测，不是官方 ViMax-Bench 测试集成绩。Track A 与 Track B 代表不同任务粒度，须分别报告，不应作为同一受控条件下的直接差值。{official_note}</section>

<h2>1. 实验输入故事</h2>
<p>以下 reference story 用作 SSS_emb 的参考故事输入。本案例的完整输入与各方法的 L4 重建故事可在第 6 节核查。</p>
<pre>{esc((comparison_dir / 'reference_story.txt').read_text(encoding='utf-8').strip())}</pre>

<h2>2. Track A / Track B 定义</h2><div class="grid">{track_cards}</div>

<h2>3. 评估指标与读数方式</h2>
<div class="table-wrap"><table><caption>表 2. 指标含义与计算边界</caption><thead><tr><th>指标</th><th>名称</th><th>定义</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p class="note">视觉指标由 {esc(visual.get('encoder') or 'ViCLIP')} 片段级 embedding 的两两余弦相似度聚合得到；本次评估的 IC/CC/GC 不表示文本-视频语义对齐，也不直接代表剧情是否正确。规划指标为 LLM 1–5 分；各结果当前均只有 1 个 batch，因此其数值为该批直接评分，非跨 batch 平均。</p>

<h2>4. 定量结果</h2>{metric_table(rows, comparison_dir)}
<p class="note">Track B 的 IC/CC 记为 N/A：严格协议未提供正式 scene_id，因而无法定义同场景或跨场景镜头对；GC 不依赖 scene_id，仍可计算。NF 为 N/A：本次仅使用 proxy reference story，不将其作为完整原始叙事忠实度的依据。</p>
<p><a href="quantitative/comparison.csv">下载结构化结果 CSV</a> · <a href="quantitative/comparison.md">查看原始 Markdown 表</a> · <a href="qualitative/index.html">打开按方法/Track/分镜组织的定性比较页面</a></p>

<h2>5. IC / CC / GC 低于均值的镜头内容</h2>{low_score_section(comparison_dir)}

<h2>6. SSS_emb 的实际输入与分数</h2>{sss_section(comparison_dir)}

<h2>7. 复现与审计入口</h2><ul><li>量化汇总：<a href="quantitative/comparison.csv">quantitative/comparison.csv</a></li><li>低分镜头对（含完整视频路径、分镜文本、参考首帧与提示词）：<a href="analysis/below_average_visual_pairs.csv">below_average_visual_pairs.csv</a></li><li>SSS_emb 的原始输入：<a href="analysis/sss_emb_inputs.json">sss_emb_inputs.json</a></li><li>定性比较媒体：<a href="qualitative/index.html">qualitative/index.html</a></li><li>原始评估 JSON：<code>quantitative/*_visual.json</code>、<code>quantitative/*_planning.json</code>、<code>quantitative/*_sss.json</code></li></ul>
<footer class="footer">本文档由 <code>evaluation/generate_experiment_html.py</code> 从已完成的比较结果生成；生成过程不重跑视频、视觉 embedding、LLM 规划评估或 SSS。</footer>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir.resolve()
    output = (args.output or comparison_dir / "experiment_report.html").resolve()
    output.write_text(html_document(comparison_dir), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
