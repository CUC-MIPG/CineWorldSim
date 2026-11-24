#!/usr/bin/env python3
"""Export audit-friendly low-visual-score pairs and SSS_emb inputs.

This utility consumes an existing CameraMaster-vs-ViMax comparison directory.
It never invokes an evaluator or changes any evaluator result.  For each
method/track/metric it defines "below average" as a pair cosine strictly lower
than that metric's per-sample mean, and writes both machine-readable CSV/JSON
and a compact Markdown review report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("cameramaster", "vimax")
TRACKS = ("A", "B")
VISUAL_METRICS = ("IC", "CC", "GC")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _first_sample(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], dict):
        raise ValueError(f"Expected non-empty samples list in {path}")
    return samples[0]


def _short(text: str, limit: int = 150) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _shot_index(normalized_path: Path) -> dict[str, dict[str, str]]:
    sample = _first_sample(_read_json(normalized_path), normalized_path)
    shots = sample.get("shots")
    if not isinstance(shots, list):
        raise ValueError(f"Expected shots list in {normalized_path}")
    result: dict[str, dict[str, str]] = {}
    for raw in shots:
        if not isinstance(raw, dict):
            continue
        shot_id = str(raw.get("shot_id") or "")
        if not shot_id:
            continue
        result[shot_id] = {
            "shot_id": shot_id,
            "scene_id": str(raw.get("scene_id") or ""),
            "video_path": str(raw.get("video_path") or ""),
            "storyboard_description": str(raw.get("storyboard_description") or ""),
            "reference_first_frame": str(raw.get("reference_first_frame") or ""),
            "reference_video_prompt": str(raw.get("reference_video_prompt") or ""),
        }
    return result


def _shot_details(index: dict[str, dict[str, str]], shot_id: Any) -> dict[str, str]:
    identifier = str(shot_id or "")
    found = index.get(identifier)
    if found:
        return found
    return {
        "shot_id": identifier,
        "scene_id": "",
        "video_path": "",
        "storyboard_description": "",
        "reference_first_frame": "",
        "reference_video_prompt": "",
    }


def _collect_visual(comparison_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for method in METHODS:
        for track in TRACKS:
            stem = f"{method}_track_{track}"
            visual_path = comparison_dir / "quantitative" / f"{stem}_visual.json"
            normalized_path = comparison_dir / "normalized" / f"{stem}.json"
            if not visual_path.is_file() or not normalized_path.is_file():
                continue
            visual_sample = _first_sample(_read_json(visual_path), visual_path)
            normalized_sample = _first_sample(_read_json(normalized_path), normalized_path)
            sample_id = str(normalized_sample.get("sample_id") or stem)
            index = _shot_index(normalized_path)
            metrics = visual_sample.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for metric in VISUAL_METRICS:
                metric_value = metrics.get(metric)
                if not isinstance(metric_value, dict):
                    continue
                mean = _number(metric_value.get("mean"))
                pairs = metric_value.get("pairs")
                if mean is None or not isinstance(pairs, list):
                    continue
                valid_pairs = [pair for pair in pairs if isinstance(pair, dict) and _number(pair.get("cosine")) is not None]
                below = [pair for pair in valid_pairs if float(pair["cosine"]) < mean]
                groups.append(
                    {
                        "method": method,
                        "track": track,
                        "sample_id": sample_id,
                        "metric": metric,
                        "mean_score": mean,
                        "pair_count": len(valid_pairs),
                        "below_average_pair_count": len(below),
                    }
                )
                for pair in below:
                    shot_a = _shot_details(index, pair.get("shot_a"))
                    shot_b = _shot_details(index, pair.get("shot_b"))
                    score = float(pair["cosine"])
                    rows.append(
                        {
                            "method": method,
                            "track": track,
                            "sample_id": sample_id,
                            "metric": metric,
                            "metric_mean": mean,
                            "pair_cosine": score,
                            "delta_to_mean": score - mean,
                            "shot_a_id": shot_a["shot_id"],
                            "shot_a_scene_id": shot_a["scene_id"],
                            "shot_a_video_path": shot_a["video_path"],
                            "shot_a_storyboard_description": shot_a["storyboard_description"],
                            "shot_a_reference_first_frame": shot_a["reference_first_frame"],
                            "shot_a_reference_video_prompt": shot_a["reference_video_prompt"],
                            "shot_b_id": shot_b["shot_id"],
                            "shot_b_scene_id": shot_b["scene_id"],
                            "shot_b_video_path": shot_b["video_path"],
                            "shot_b_storyboard_description": shot_b["storyboard_description"],
                            "shot_b_reference_first_frame": shot_b["reference_first_frame"],
                            "shot_b_reference_video_prompt": shot_b["reference_video_prompt"],
                            "mapping_complete": bool(shot_a["video_path"] and shot_b["video_path"]),
                        }
                    )
    rows.sort(key=lambda row: (row["method"], row["track"], row["metric"], row["pair_cosine"]))
    return rows, groups


def _collect_sss_inputs(comparison_dir: Path) -> list[dict[str, Any]]:
    reference_path = comparison_dir / "reference_story.txt"
    reference_story = reference_path.read_text(encoding="utf-8").strip() if reference_path.is_file() else ""
    records: list[dict[str, Any]] = []
    for method in METHODS:
        for track in TRACKS:
            stem = f"{method}_track_{track}"
            normalized_path = comparison_dir / "normalized" / f"{stem}.json"
            result_path = comparison_dir / "quantitative" / f"{stem}_sss.json"
            reconstructed_path = comparison_dir / "quantitative" / "sss" / stem / "reconstructed_story.json"
            if not result_path.is_file() or not reconstructed_path.is_file():
                continue
            normalized_sample = _first_sample(_read_json(normalized_path), normalized_path) if normalized_path.is_file() else {}
            result = _read_json(result_path)
            primary = result.get("primary") if isinstance(result, dict) else None
            l4 = _read_json(reconstructed_path).get("L4", {})
            if not isinstance(l4, dict):
                l4 = {}
            records.append(
                {
                    "method": method,
                    "track": track,
                    "sample_id": str(normalized_sample.get("sample_id") or stem),
                    "metric": primary.get("metric") if isinstance(primary, dict) else None,
                    "sss_emb": primary.get("score") if isinstance(primary, dict) else None,
                    "embedding_model": primary.get("model") if isinstance(primary, dict) else None,
                    "reference_story_input": reference_story,
                    "reconstructed_l4_story_input": str(l4.get("story") or ""),
                    "reference_story_path": str(reference_path),
                    "reconstructed_story_path": str(reconstructed_path),
                }
            )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method", "track", "sample_id", "metric", "metric_mean", "pair_cosine", "delta_to_mean",
        "shot_a_id", "shot_a_scene_id", "shot_a_video_path", "shot_a_storyboard_description",
        "shot_a_reference_first_frame", "shot_a_reference_video_prompt",
        "shot_b_id", "shot_b_scene_id", "shot_b_video_path", "shot_b_storyboard_description",
        "shot_b_reference_first_frame", "shot_b_reference_video_prompt", "mapping_complete",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]], groups: list[dict[str, Any]], sss_inputs: list[dict[str, Any]], top_k: int) -> str:
    lines = [
        "# 低于均值的视觉一致性对与 SSS_emb 输入",
        "",
        "判定规则：对每个 `method × track × metric`，仅列出 `pair_cosine < 该指标该样本的 mean` 的镜头对；相等不计入。完整分镜、视频路径和文本见 CSV。",
        "",
        "## 汇总",
        "",
        "| 方法 | Track | 指标 | 均值 | 全部 pair | 低于均值 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['method']} | {group['track']} | {group['metric']} | {group['mean_score']:.6f} | "
            f"{group['pair_count']} | {group['below_average_pair_count']} |"
        )
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["method"], row["track"], row["metric"])].append(row)
    lines.extend(["", "## 低分镜头对（每组最低的前 %d 项）" % top_k])
    for key, items in grouped_rows.items():
        method, track, metric = key
        lines.extend(["", f"### {method} · Track {track} · {metric}", "", "| cosine | 距均值 | 镜头 A | 镜头 B |", "|---:|---:|---|---|"])
        for row in items[:top_k]:
            a = f"`{row['shot_a_id']}` — {_short(row['shot_a_storyboard_description']) or '未映射文本'}"
            b = f"`{row['shot_b_id']}` — {_short(row['shot_b_storyboard_description']) or '未映射文本'}"
            lines.append(f"| {row['pair_cosine']:.6f} | {row['delta_to_mean']:.6f} | {a} | {b} |")
    lines.extend(["", "## SSS_emb 的实际 embedding 输入", "", "SSS_emb 仅计算下列两段文本向量的余弦相似度：`reference_story_input` 与对应样本的 `reconstructed_l4_story_input`。L1/L2/L3 用于生成 L4，不直接参与该余弦平均。"])
    for record in sss_inputs:
        lines.extend([
            "", f"### {record['method']} · Track {record['track']} · SSS_emb={record['sss_emb']}",
            "", "**Reference story input**", "", record["reference_story_input"],
            "", "**L4 reconstructed story input**", "", record["reconstructed_l4_story_input"],
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir.resolve()
    if not comparison_dir.is_dir():
        raise SystemExit(f"Comparison directory does not exist: {comparison_dir}")
    output_dir = (args.output_dir or comparison_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, groups = _collect_visual(comparison_dir)
    sss_inputs = _collect_sss_inputs(comparison_dir)
    csv_path = output_dir / "below_average_visual_pairs.csv"
    json_path = output_dir / "sss_emb_inputs.json"
    markdown_path = output_dir / "low_score_visual_and_sss_inputs.md"
    _write_csv(csv_path, rows)
    _write_json(json_path, sss_inputs)
    markdown_path.write_text(_markdown(rows, groups, sss_inputs, args.top_k), encoding="utf-8")
    manifest = {
        "comparison_dir": str(comparison_dir),
        "definition": "below average means pair_cosine < per-sample metric mean",
        "below_average_visual_pair_count": len(rows),
        "visual_groups": groups,
        "sss_input_record_count": len(sss_inputs),
        "files": {"below_average_visual_pairs_csv": str(csv_path), "sss_emb_inputs_json": str(json_path), "review_markdown": str(markdown_path)},
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
