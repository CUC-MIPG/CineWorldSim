#!/usr/bin/env python3
"""Build a fair CameraMaster-versus-ViMax comparison package for one ViMax case.

The script is intentionally split into preparation and execution.  Preparation
normalizes existing generation outputs and creates qualitative artifacts without
calling any model.  ``--run-visual``, ``--run-planning`` and ``--run-sss`` opt
into the corresponding, potentially costly evaluators.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


METRICS = ("ic", "cc", "gc", "cbc", "nc", "ppr", "vsu", "stq", "nf", "sss_emb")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _has_valid_result(path: Path) -> bool:
    """Return whether an evaluator has already produced a usable JSON object."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return bool(_load(path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _report_existing_result(kind: str, sample: "Sample", path: Path) -> None:
    print(json.dumps({
        "status": "skipped_existing_result",
        "evaluator": kind,
        "method": sample.method,
        "track": sample.track,
        "output": str(path),
    }, ensure_ascii=False))


def _natural_key(path: Path | str) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(path))]


def _existing_video(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty video: {path}")
    return path.resolve()


def _source_shots(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in sorted(case.get("scenes", []), key=lambda row: int(row.get("scene_num", 0))):
        scene_num = int(scene.get("scene_num", 0))
        for shot in sorted(scene.get("shots", []), key=lambda row: int(row.get("shot_id", 0))):
            shot_num = int(shot.get("shot_id", 0))
            rows.append({
                "shot_id": f"scene_{scene_num:03d}_shot_{shot_num:03d}",
                "scene_id": f"scene_{scene_num:03d}",
                "first_frame": str(shot.get("first_frame", "")).strip(),
                "video_prompt": str(shot.get("video_prompt", "")).strip(),
            })
    if not rows:
        raise ValueError("benchmark case has no source shots")
    return rows


def _describe_storyboard(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return path.read_text(encoding="utf-8", errors="replace")[:1000]
    for key in ("video_prompt", "description", "storyboard_description", "prompt", "summary"):
        if isinstance(value.get(key), str) and value[key].strip():
            return value[key].strip()
    return json.dumps(value, ensure_ascii=False)[:1000]


@dataclass
class Shot:
    shot_id: str
    scene_id: str | None
    video_path: str
    storyboard_description: str
    reference_first_frame: str = ""
    reference_video_prompt: str = ""


@dataclass
class Sample:
    method: str
    track: str
    sample_id: str
    prompt: str
    source_root: str
    shots: list[Shot]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cameramaster-vimax-comparison-v1",
            "method": self.method,
            "samples": [{
                "sample_id": self.sample_id,
                "story_id": self.sample_id.rsplit("_track_", 1)[0],
                "track": self.track,
                "prompt": self.prompt,
                "shots": [asdict(shot) for shot in self.shots],
                "source_output_root": self.source_root,
            }],
        }


def _camera_sample(root: Path, case_name: str, track: str, case: dict[str, Any]) -> Sample:
    track_dir = root / f"track_{track}"
    nodes_payload = _load(track_dir / "narrative_nodes.json")
    plan = _load(track_dir / "path_plan.json")
    nodes = {str(row.get("node_id") or row.get("event_id")): row for row in nodes_payload.get("selected_nodes", [])}
    ordered = [str(item) for item in plan.get("ordered_node_ids", [])]
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError(f"CameraMaster Track {track}: invalid path_plan ordering")
    source = _source_shots(case)
    if track == "A" and len(source) != len(ordered):
        raise ValueError(f"CameraMaster Track A has {len(ordered)} clips but benchmark has {len(source)} source shots")
    shots: list[Shot] = []
    for index, node_id in enumerate(ordered, start=1):
        node = nodes.get(node_id)
        if node is None:
            raise ValueError(f"CameraMaster Track {track}: node absent from narrative_nodes: {node_id}")
        matches = sorted((track_dir / "video").glob(f"event_video_{index:03d}_*.mp4"), key=_natural_key)
        if len(matches) != 1:
            raise ValueError(f"CameraMaster Track {track}: expected one video at rank {index}, found {len(matches)}")
        reference = source[index - 1] if track == "A" else {}
        shots.append(Shot(
            shot_id=node_id,
            scene_id=str(reference.get("scene_id")) if reference else None,
            video_path=str(_existing_video(matches[0])),
            storyboard_description=str(node.get("video_description") or node.get("event_summary") or node.get("story_description") or "").strip(),
            reference_first_frame=str(reference.get("first_frame") or ""),
            reference_video_prompt=str(reference.get("video_prompt") or ""),
        ))
    return Sample("CameraMaster", track, f"{case_name}_track_{track}", str(nodes_payload.get("storyline") or case.get("story_overview") or ""), str(root.resolve()), shots)


def _vimax_sample(root: Path, case_name: str, track: str, case: dict[str, Any]) -> Sample:
    track_dir = root / f"track_{track}" / "vimax_pipeline"
    source = _source_shots(case)
    shots: list[Shot] = []
    if track == "A":
        videos = sorted(track_dir.glob("shots/*/video.mp4"), key=_natural_key)
        if len(videos) != len(source):
            raise ValueError(f"ViMax Track A has {len(videos)} clips but benchmark has {len(source)} source shots")
        for reference, video in zip(source, videos, strict=True):
            shots.append(Shot(reference["shot_id"], reference["scene_id"], str(_existing_video(video)), reference["video_prompt"], reference["first_frame"], reference["video_prompt"]))
    else:
        videos = sorted(track_dir.glob("scene_*/shots/*/video.mp4"), key=_natural_key)
        if not videos:
            raise ValueError(f"ViMax Track B has no shot videos beneath {track_dir}")
        for index, video in enumerate(videos, start=1):
            scene_match = re.search(r"scene_(\d+)", str(video.parent.parent))
            storyboard = video.parent.parent / "storyboard.json"
            shots.append(Shot(
                shot_id=f"generated_scene_{scene_match.group(1) if scene_match else 'unknown'}_shot_{index:03d}",
                scene_id=None,
                video_path=str(_existing_video(video)),
                storyboard_description=_describe_storyboard(storyboard) or str(case.get("story_overview") or ""),
            ))
    prompt = str(case.get("story_overview") or "").strip()
    return Sample("ViMax", track, f"{case_name}_track_{track}", prompt, str(root.resolve()), shots)


def _link_media(samples: Iterable[Sample], media_dir: Path) -> dict[str, str]:
    media_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for sample in samples:
        for index, shot in enumerate(sample.shots, start=1):
            target = Path(shot.video_path)
            name = f"{sample.method.lower()}_{sample.track}_{index:03d}_{target.name}"
            link = media_dir / name
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(target)
            except OSError:
                # Windows without Developer Mode cannot create symlinks.  A copy
                # keeps the report usable there; Linux normally takes this path.
                shutil.copy2(target, link)
            links[shot.video_path] = str(link)
    return links


def _render_html(samples: list[Sample], out: Path, links: dict[str, str]) -> None:
    grouped = {(sample.method, sample.track): sample for sample in samples}
    rows: list[str] = []
    for track in ("A", "B"):
        left, right = grouped.get(("CameraMaster", track)), grouped.get(("ViMax", track))
        if not left or not right:
            continue
        count = max(len(left.shots), len(right.shots))
        for index in range(count):
            a = left.shots[index] if index < len(left.shots) else None
            b = right.shots[index] if index < len(right.shots) else None
            def cell(shot: Shot | None) -> str:
                if shot is None:
                    return "<td>missing</td>"
                href = html.escape(os.path.relpath(links[shot.video_path], out.parent))
                return "<td><video controls preload='metadata' src='%s'></video><details><summary>text</summary><pre>%s</pre></details></td>" % (href, html.escape(shot.storyboard_description))
            reference = a or b
            reference_text = ""
            if track == "A" and reference:
                reference_text = "<details><summary>benchmark anchor</summary><pre>%s\n\n%s</pre></details>" % (html.escape(reference.reference_first_frame), html.escape(reference.reference_video_prompt))
            rows.append("<tr><td>%s-%03d%s</td>%s%s</tr>" % (track, index + 1, reference_text, cell(a), cell(b)))
    page = """<!doctype html><meta charset='utf-8'><title>CameraMaster vs ViMax</title>
<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse;width:100%%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top;width:33%%}video{width:100%%;max-width:420px}pre{white-space:pre-wrap}summary{cursor:pointer}</style>
<h1>CameraMaster vs ViMax qualitative comparison</h1><p>Track A uses benchmark anchors; Track B is overview-only. Videos are symlinked into <code>media/</code>.</p>
<table><thead><tr><th>Track / shot</th><th>CameraMaster</th><th>ViMax</th></tr></thead><tbody>%s</tbody></table>""" % "\n".join(rows)
    out.write_text(page, encoding="utf-8")


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _concat_for_sss(sample: Sample, output: Path) -> Path:
    """Build an ordered evaluation video without mutating either method's output."""
    target = output / "derived_media" / f"{sample.method.lower()}_track_{sample.track}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if len(sample.shots) == 1:
        source = Path(sample.shots[0].video_path)
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        return target
    listing = target.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    for shot in sample.shots:
        lines.append("file '" + shot.video_path.replace("'", "'\\\\''") + "'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    base = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    try:
        subprocess.run([*base, "-c", "copy", str(target)], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([*base, "-c:v", "libx264", "-c:a", "aac", str(target)], check=True)
    return target


def _sss_adapter(sample: Sample, video: Path, output: Path) -> Path:
    path = output / "sss_adapters" / f"{sample.method.lower()}_track_{sample.track}.json"
    _write(path, {
        "schema_version": "cameramaster-sss-track-adapter-v1",
        "samples": [{
            "track": sample.track,
            "video_path": str(video.resolve()),
            "source_clip_paths": [shot.video_path for shot in sample.shots],
            "ordered_node_ids": [shot.shot_id for shot in sample.shots],
        }],
        "protocol_notes": [
            "Reference story is deliberately excluded from this mapping.",
            "Track A uses generated clip boundaries; Track B uses automatic shot detection.",
        ],
    })
    return path


def _flatten_metrics(value: Any, result: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in METRICS and isinstance(item, (int, float)):
                result[normalized] = float(item)
            _flatten_metrics(item, result)
    elif isinstance(value, list):
        for item in value:
            _flatten_metrics(item, result)


def _extract_result_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Normalize evaluator-specific result schemas into the comparison columns."""
    values: dict[str, float] = {}
    _flatten_metrics(payload, values)

    # ViMax visual evaluator: aggregate IC/CC/GC are nested objects rather
    # than numeric leaves named directly after the comparison columns.
    aggregate = payload.get("dataset_aggregate")
    if isinstance(aggregate, dict):
        for metric in ("IC", "CC", "GC"):
            row = aggregate.get(metric)
            if not isinstance(row, dict):
                continue
            for key in ("story_macro_mean", "pair_weighted_mean", "mean", "score"):
                score = row.get(key)
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    values[metric.casefold()] = float(score)
                    break

    # SSS evaluator reports its primary score as ``primary.score`` and labels
    # the metric separately, so generic recursive extraction cannot identify
    # it as the sss_emb column.
    primary = payload.get("primary")
    if isinstance(primary, dict) and str(primary.get("metric") or "").casefold() == "sss_emb":
        score = primary.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            values["sss_emb"] = float(score)
    return values


def _table(samples: list[Sample], result_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for sample in samples:
        row = {"method": sample.method, "track": sample.track, "sample_id": sample.sample_id, "clip_count": str(len(sample.shots))}
        for kind in ("visual", "planning", "sss"):
            path = result_dir / f"{sample.method.lower()}_track_{sample.track}_{kind}.json"
            if path.is_file():
                values = _extract_result_metrics(_load(path))
                for key, value in values.items():
                    row[key] = f"{value:.6g}"
        rows.append(row)
    fields = ["method", "track", "sample_id", "clip_count", *METRICS]
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    markdown = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    markdown.extend("| " + " | ".join(row.get(field, "N/A") for field in fields) + " |" for row in rows)
    out.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-case", type=Path, required=True)
    parser.add_argument("--cameramaster-output-root", type=Path, required=True, help="case directory containing track_A/ and track_B/")
    parser.add_argument("--vimax-output-root", type=Path, required=True, help="case directory containing track_A/ and track_B/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--camera-master-root", type=Path, default=root)
    parser.add_argument("--embed-url", default="http://172.17.34.43:8890/embed")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--timeout", type=float, default=900, help="Per evaluator request timeout in seconds (default: 15 minutes).")
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    parser.add_argument("--run-visual", action="store_true")
    parser.add_argument("--run-planning", action="store_true")
    parser.add_argument("--run-sss", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip evaluator outputs that already contain a valid JSON object (default: enabled). Use --no-resume to force recomputation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_all:
        args.run_visual = args.run_planning = args.run_sss = True
    case = _load(args.benchmark_case.resolve())
    case_name = args.benchmark_case.stem
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for track in args.tracks:
        samples.extend([
            _camera_sample(args.cameramaster_output_root.resolve(), case_name, track, case),
            _vimax_sample(args.vimax_output_root.resolve(), case_name, track, case),
        ])
    normalized_dir, results_dir = output / "normalized", output / "quantitative"
    for sample in samples:
        _write(normalized_dir / f"{sample.method.lower()}_track_{sample.track}.json", sample.payload())
    reference_text = str(case.get("story_overview") or "").strip()
    (output / "reference_story.txt").write_text(reference_text + "\n", encoding="utf-8")
    _write(output / "reference_story.json", {"text": reference_text, "provenance": "benchmark_story_overview_proxy"})
    links = _link_media(samples, output / "qualitative" / "media")
    _render_html(samples, output / "qualitative" / "index.html", links)
    if args.run_visual or args.run_planning or args.run_sss:
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        except ModuleNotFoundError:
            pass
    for sample in samples:
        normalized = normalized_dir / f"{sample.method.lower()}_track_{sample.track}.json"
        base = f"{sample.method.lower()}_track_{sample.track}"
        visual_output = results_dir / f"{base}_visual.json"
        planning_output = results_dir / f"{base}_planning.json"
        sss_output = results_dir / f"{base}_sss.json"
        if args.run_visual:
            if args.resume and _has_valid_result(visual_output):
                _report_existing_result("visual", sample, visual_output)
            else:
                _run([sys.executable, str(args.camera_master_root / "evaluation" / "run_vimax_bench_eval.py"), "--inputs-json", str(normalized), "--output", str(visual_output), "--track", sample.track, "--backend", "http", "--embed-url", args.embed_url, "--num-frames", str(args.num_frames), "--resolution", str(args.resolution), "--timeout", str(args.timeout), "--cache-dir", str(results_dir / "viclip_cache")], args.camera_master_root)
        if args.run_planning:
            if args.resume and _has_valid_result(planning_output):
                _report_existing_result("planning", sample, planning_output)
            else:
                _run([sys.executable, str(args.camera_master_root / "evaluation" / "run_vimax_planning_judge.py"), "--samples-json", str(normalized), "--output", str(planning_output), "--metric", "planning", "--max-items", "20"], args.camera_master_root)
        if args.run_sss:
            sss_dir = output / "quantitative" / "sss" / f"{sample.method.lower()}_track_{sample.track}"
            result = sss_dir / "sss_result.json"
            if args.resume and _has_valid_result(sss_output):
                _report_existing_result("sss", sample, sss_output)
            elif args.resume and _has_valid_result(result):
                shutil.copy2(result, sss_output)
                _report_existing_result("sss_recovered_result", sample, sss_output)
            else:
                video = _concat_for_sss(sample, output)
                adapter = _sss_adapter(sample, video, output)
                command = [
                    sys.executable, str(args.camera_master_root / "evaluation" / "sss_smoke_test.py"),
                    "--video", str(video), "--summary-file", str(output / "reference_story.txt"),
                    "--summary-provenance", "proxy", "--allow-proxy-summary", "--track", sample.track,
                    "--mode", "v3" if sample.track == "A" else "v2", "--output-dir", str(sss_dir),
                    "--run-llm", "--timeout-sec", str(int(args.timeout)), "--overwrite",
                ]
                if sample.track == "A":
                    command.extend(["--track-adapter-inputs", str(adapter)])
                _run(command, args.camera_master_root)
                if not _has_valid_result(result):
                    raise RuntimeError(f"SSS completed without a valid sss_result.json: {sss_dir}")
                shutil.copy2(result, sss_output)
    _table(samples, results_dir, output / "quantitative" / "comparison.csv")
    _write(output / "comparison_manifest.json", {"case": str(args.benchmark_case.resolve()), "samples": [sample.payload() for sample in samples], "run_flags": {"visual": args.run_visual, "planning": args.run_planning, "sss": args.run_sss, "resume": args.resume}})
    print(output / "comparison_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
