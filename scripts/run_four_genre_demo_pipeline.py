#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SKILL_ENV_KEYS = [
    "CAMERAMASTER_SKILL_ROOTS",
    "CAMERAMASTER_SKILL_POLICY_PATH",
    "DIRECTOR_SKILL_NAME",
    "SHOT_SKILL_NAME",
    "SUPERVISOR_SKILL_NAME",
    "VIDEO_SKILL_NAME",
]


DEMOS: list[dict[str, Any]] = [
    {
        "index": 1,
        "genre": "suspense",
        "slug": "suspense_subway_archive",
        "title": "The Last Train Ledger",
        "characters": ["NightArchivist", "TransitDetective"],
        "storyline": (
            "At 01:17 in a deserted underground transfer station, a night archivist finds a wet ledger "
            "inside a locked maintenance cabinet. The transit detective arrives after every camera on the "
            "platform loops the same thirteen seconds. As the last train approaches with no driver, the two "
            "must determine whether the ledger lists future disappearances or confessions from the past."
        ),
    },
    {
        "index": 2,
        "genre": "adventure",
        "slug": "adventure_skyreef_compass",
        "title": "Compass Over the Sky Reef",
        "characters": ["CloudCartographer", "RunawayPrince", "ReefMechanic"],
        "storyline": (
            "Above a chain of floating coral islands, a cloud cartographer discovers that her antique compass "
            "points toward a storm-buried sky reef rather than north. A runaway prince and a reef mechanic join "
            "her on a patched glider to retrieve the lost wind bell before the seasonal monsoon tears the islands "
            "apart. Their route crosses whale-shaped clouds, broken rope bridges, and a temple that only appears "
            "when lightning strikes."
        ),
    },
    {
        "index": 3,
        "genre": "action",
        "slug": "action_harbor_convoy",
        "title": "Harbor Convoy Zero",
        "characters": ["RookieDriver", "ExSoldier", "DroneRunner", "HarborCommander"],
        "storyline": (
            "During a citywide blackout, a rookie driver must move a stolen medical battery across the industrial "
            "harbor before the hospital grid collapses. An ex-soldier guards the truck, a drone runner scouts "
            "rooftop ambushes, and the harbor commander secretly reroutes cranes to block a mercenary convoy. "
            "The mission becomes a rolling fight through container lanes, rain-slick ramps, and collapsing signal towers."
        ),
    },
    {
        "index": 4,
        "genre": "urban",
        "slug": "urban_midnight_rooftop_diner",
        "title": "Midnight Rooftop Diner",
        "characters": ["LateShiftChef", "RadioHost", "DeliveryRider", "StreetPainter", "ApartmentManager"],
        "storyline": (
            "In a neon apartment district, a late-shift chef opens a secret rooftop diner for neighbors who cannot "
            "sleep. A radio host broadcasts anonymous confessions from the kitchen window, a delivery rider brings "
            "misdelivered letters, a street painter documents the night on a billboard, and the apartment manager "
            "tries to shut the gathering down before sunrise. Their small arguments gradually reveal why the whole "
            "block is afraid of the demolition notice taped to the lobby door."
        ),
    },
]


def _bool_flag(enabled: bool, flag: str) -> str:
    return flag if enabled else f"--no-{flag[2:]}"


def _demo_root(output_root: Path, demo: dict[str, Any]) -> Path:
    return output_root / f"{int(demo['index']):02d}_{demo['slug']}"


def build_pipeline_command(
    *,
    demo: dict[str, Any],
    output_root: Path,
    backend_mode: str,
    max_steps: int,
    dry_run: bool,
    path_mode: str,
    node_llm_scoring: bool,
    strict_online: bool,
    retries: int,
    log_level: str,
    stop_on_complete: bool = True,
    node_selection_mode: str = "hybrid",
    node_score_threshold: float = 0.55,
    node_min_nodes: int = 3,
    node_max_nodes: int = 8,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_narrative_world_path_pipeline.py"),
        "--storyline",
        str(demo["storyline"]),
        "--characters",
        ",".join(str(x) for x in demo["characters"]),
        "--backend-mode",
        backend_mode,
        "--max-steps",
        str(max(1, int(max_steps))),
        _bool_flag(stop_on_complete, "--stop-on-complete"),
        "--output-dir",
        str(_demo_root(output_root, demo) / "pipeline"),
        "--path-mode",
        path_mode,
        "--node-selection-mode",
        node_selection_mode,
        "--node-score-threshold",
        str(float(node_score_threshold)),
        "--node-min-nodes",
        str(max(1, int(node_min_nodes))),
        "--node-max-nodes",
        str(max(1, int(node_max_nodes))),
        "--retries",
        str(max(1, int(retries))),
        "--log-level",
        log_level,
        _bool_flag(node_llm_scoring, "--node-llm-scoring"),
        _bool_flag(strict_online, "--strict-online"),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_resume_path_command(
    *,
    demo: dict[str, Any],
    pipeline_result_json: Path,
    output_dir: Path,
    path_mode: str,
    node_llm_scoring: bool,
    strict_online: bool,
    retries: int,
    log_level: str,
    node_selection_mode: str = "hybrid",
    node_score_threshold: float = 0.55,
    node_min_nodes: int = 3,
    node_max_nodes: int = 8,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_narrative_world_path_pipeline.py"),
        "--storyline",
        str(demo["storyline"]),
        "--characters",
        ",".join(str(x) for x in demo["characters"]),
        "--existing-pipeline-result-json",
        str(pipeline_result_json),
        "--output-dir",
        str(output_dir),
        "--path-mode",
        path_mode,
        "--node-selection-mode",
        node_selection_mode,
        "--node-score-threshold",
        str(float(node_score_threshold)),
        "--node-min-nodes",
        str(max(1, int(node_min_nodes))),
        "--node-max-nodes",
        str(max(1, int(node_max_nodes))),
        "--retries",
        str(max(1, int(retries))),
        "--log-level",
        log_level,
        _bool_flag(node_llm_scoring, "--node-llm-scoring"),
        _bool_flag(strict_online, "--strict-online"),
    ]


def build_video_command(
    *,
    run_dir: Path,
    output_dir: Path,
    dry_run: bool,
    max_reference_images: int,
    multi_shot_events: bool,
    event_shot_count: int,
    event_duration_sec: float,
    use_previous_tail_frame: bool,
    use_previous_audio: bool,
    upload_previous_audio: bool,
    cuda_visible_devices: str,
    log_level: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_orchestrated_event_video_generation.py"),
        "--pipeline-run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--max-reference-images",
        str(max(1, int(max_reference_images))),
        "--event-shot-count",
        str(max(1, int(event_shot_count))),
        "--event-duration-sec",
        str(float(event_duration_sec)),
        "--log-level",
        log_level,
        _bool_flag(dry_run, "--dry-run"),
        _bool_flag(multi_shot_events, "--multi-shot-events"),
        _bool_flag(use_previous_tail_frame, "--use-previous-tail-frame"),
        _bool_flag(use_previous_audio, "--use-previous-audio"),
        _bool_flag(upload_previous_audio, "--upload-previous-audio"),
    ]
    if cuda_visible_devices.strip():
        cmd.extend(["--cuda-visible-devices", cuda_visible_devices.strip()])
    return cmd


def build_visualization_command(*, run_dir: Path, video_dir: Path, output_html: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "generate_full_pipeline_stage_visualization.py"),
        "--run-dir",
        str(run_dir),
        "--video-dir",
        str(video_dir),
        "--output-html",
        str(output_html),
    ]


def build_asset_repair_command(
    *,
    pipeline_result_json: Path,
    backend_mode: str,
    dry_run: bool,
    log_level: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "repair_banana_pipeline_assets.py"),
        "--pipeline-result-json",
        str(pipeline_result_json),
        "--backend-mode",
        backend_mode,
        "--log-level",
        log_level,
        _bool_flag(dry_run, "--dry-run"),
    ]
    return cmd


def _recorded_output_missing(record: dict[str, Any]) -> str:
    output_path = str(record.get("output_path", "") or "").strip()
    if not bool(record.get("ok", False)):
        return output_path or str(record.get("error", "") or "record ok=false")
    if not output_path:
        return "record ok=true but output_path is empty"
    path = Path(output_path)
    if not path.exists() or path.stat().st_size <= 0:
        return output_path
    return ""


def collect_missing_pipeline_assets(pipeline_result_json: Path) -> list[str]:
    if not pipeline_result_json.exists():
        return []
    try:
        payload = json.loads(pipeline_result_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{pipeline_result_json}: unreadable JSON: {exc}"]
    missing: list[str] = []

    world_model = payload.get("world_model", {}) if isinstance(payload, dict) else {}
    for group_name in ["locations", "items"]:
        for asset in world_model.get(group_name, []) or []:
            if not isinstance(asset, dict):
                continue
            asset_name = str(asset.get("asset_id") or asset.get("name") or "asset")
            images = asset.get("images", []) or []
            if not images:
                missing.append(f"world_model.{group_name}.{asset_name}: no images")
                continue
            for image in images:
                if isinstance(image, dict):
                    reason = _recorded_output_missing(image)
                    if reason:
                        missing.append(f"world_model.{group_name}.{asset_name}: {reason}")

    for keyframe in payload.get("event_keyframes", []) or []:
        if not isinstance(keyframe, dict):
            continue
        reason = _recorded_output_missing(keyframe)
        if reason:
            step = keyframe.get("step", "")
            role = keyframe.get("role_name", "")
            missing.append(f"event_keyframes.step_{step}.{role}: {reason}")

    return missing


def collect_missing_video_assets(video_summary_json: Path) -> list[str]:
    if not video_summary_json.exists():
        return [f"{video_summary_json}: missing summary"]
    try:
        payload = json.loads(video_summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{video_summary_json}: unreadable JSON: {exc}"]

    missing: list[str] = []
    for item in payload.get("generated", []) or []:
        if not isinstance(item, dict):
            continue
        output_path = str(item.get("output_video_path", "") or "").strip()
        if not output_path:
            missing.append(f"generated.{item.get('node_id', '')}: empty output path")
            continue
        path = Path(output_path)
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(output_path)

    concat_output = str(payload.get("concat_output", "") or "").strip()
    if concat_output:
        path = Path(concat_output)
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(concat_output)
    return missing


def _append_log(log_file: Path | None, line: str) -> None:
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")


def prepare_child_env(*, base_env: dict[str, str] | None = None, call_log_jsonl: Path) -> dict[str, str]:
    env = dict(base_env or os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    env["CAMERAMASTER_CALL_LOG_JSONL"] = str(call_log_jsonl)
    env["CAMERAMASTER_RUN_LOG_DIR"] = str(call_log_jsonl.parent)
    for key in SKILL_ENV_KEYS:
        if key in env:
            continue
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_file: Path | None = None,
    stage_name: str = "",
) -> int:
    started = time.time()
    label = stage_name or Path(str(cmd[1] if len(cmd) > 1 else cmd[0])).name
    command_text = " ".join(str(x) for x in cmd)
    start_line = f"[STAGE START] {label} command={command_text}"
    print(start_line, flush=True)
    _append_log(log_file, start_line)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _pump(stream: Any, prefix: str) -> None:
        for line in iter(stream.readline, ""):
            rendered = f"{prefix}{line}"
            print(rendered, end="", flush=True)
            _append_log(log_file, rendered.rstrip("\n"))
        stream.close()

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, ""), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, "[stderr] "), daemon=True),
    ]
    for thread in threads:
        thread.start()
    rc = proc.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    done_line = f"[STAGE DONE] {label} rc={rc} elapsed_sec={time.time() - started:.1f}"
    print(done_line, flush=True)
    _append_log(log_file, done_line)
    return int(rc)


def write_demo_catalog(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "demo_catalog.json"
    path.write_text(json.dumps({"demos": DEMOS}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Run four genre-specific CineWorldSim demos end-to-end.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/four_genre_demos"))
    parser.add_argument(
        "--backend-mode",
        choices=["all_flux", "all_banana", "all_flux2_klein"],
        default="all_banana",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--stop-on-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path-mode", choices=["llm", "mcts", "both"], default="llm")
    parser.add_argument("--node-selection-mode", choices=["hybrid", "threshold", "top_k"], default="hybrid")
    parser.add_argument("--node-score-threshold", type=float, default=0.55)
    parser.add_argument("--node-min-nodes", type=int, default=3)
    parser.add_argument("--node-max-nodes", type=int, default=8)
    parser.add_argument("--node-llm-scoring", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing pipeline/path/video artifacts when present and continue from the next missing stage.",
    )
    parser.add_argument(
        "--repair-missing-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, rerun a demo pipeline if recorded image assets are missing or failed.",
    )
    parser.add_argument("--skip-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--multi-shot-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=8.0)
    parser.add_argument("--use-previous-tail-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-previous-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logs_dir = args.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runner_log = logs_dir / "four_genre_demo_runner.log"
    call_log_jsonl = logs_dir / "model_tool_calls.jsonl"
    env = prepare_child_env(base_env=os.environ.copy(), call_log_jsonl=call_log_jsonl)
    catalog = write_demo_catalog(args.output_root)
    runs: list[dict[str, Any]] = []

    header = (
        "[RUNNER START] "
        f"output_root={args.output_root} backend_mode={args.backend_mode} max_steps={args.max_steps} "
        f"video_dry_run={args.video_dry_run} call_log_jsonl={call_log_jsonl}"
    )
    print(header, flush=True)
    _append_log(runner_log, header)

    for demo in DEMOS:
        demo_root = _demo_root(args.output_root, demo)
        pipeline_root = demo_root / "pipeline"
        run_dir = pipeline_root / args.backend_mode
        video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
        html_path = run_dir / "visualizations" / "full_stage_pipeline_visualization.html"
        pipeline_result_json = run_dir / "pipeline_result.json"
        path_plan_json = run_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"
        video_summary_json = video_dir / "event_video_summary.json"
        demo_root.mkdir(parents=True, exist_ok=True)
        (demo_root / "story.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        demo_label = f"{int(demo['index']):02d}_{demo['genre']}"
        demo_start = f"[DEMO START] {demo_label} title={demo['title']} characters={len(demo['characters'])}"
        print(demo_start, flush=True)
        _append_log(runner_log, demo_start)

        missing_assets = collect_missing_pipeline_assets(pipeline_result_json) if args.repair_missing_assets else []
        pipeline_reran = False
        asset_repaired = False
        if args.resume_existing and pipeline_result_json.exists() and missing_assets and args.repair_missing_assets:
            repair_line = (
                f"[REPAIR] {demo_label} pipeline assets missing count={len(missing_assets)} "
                f"first={missing_assets[0]}"
            )
            print(repair_line, flush=True)
            _append_log(runner_log, repair_line)
            repair_cmd = build_asset_repair_command(
                pipeline_result_json=pipeline_result_json,
                backend_mode=args.backend_mode,
                dry_run=args.dry_run,
                log_level=args.log_level,
            )
            rc = run_command(
                repair_cmd,
                env=env,
                log_file=logs_dir / f"{demo_label}_asset_repair.log",
                stage_name=f"{demo_label}:asset_repair",
            )
            if rc != 0:
                raise SystemExit(rc)
            missing_assets = collect_missing_pipeline_assets(pipeline_result_json)
            repair_done_line = f"[REPAIR DONE] {demo_label} remaining_missing={len(missing_assets)}"
            print(repair_done_line, flush=True)
            _append_log(runner_log, repair_done_line)
            asset_repaired = True

        if args.resume_existing and pipeline_result_json.exists() and not missing_assets:
            skip_line = f"[SKIP] {demo_label} existing complete pipeline_result_json={pipeline_result_json}"
            print(skip_line, flush=True)
            _append_log(runner_log, skip_line)
            if path_plan_json.exists():
                skip_line = f"[SKIP] {demo_label} existing path_plan_json={path_plan_json}"
                print(skip_line, flush=True)
                _append_log(runner_log, skip_line)
            else:
                resume_cmd = build_resume_path_command(
                    demo=demo,
                    pipeline_result_json=pipeline_result_json,
                    output_dir=run_dir / "path_orchestration",
                    path_mode=args.path_mode,
                    node_selection_mode=args.node_selection_mode,
                    node_score_threshold=args.node_score_threshold,
                    node_min_nodes=args.node_min_nodes,
                    node_max_nodes=args.node_max_nodes,
                    node_llm_scoring=args.node_llm_scoring,
                    strict_online=args.strict_online,
                    retries=args.retries,
                    log_level=args.log_level,
                )
                rc = run_command(
                    resume_cmd,
                    env=env,
                    log_file=logs_dir / f"{demo_label}_resume_path.log",
                    stage_name=f"{demo_label}:resume_path",
                )
                if rc != 0:
                    raise SystemExit(rc)
        else:
            if args.resume_existing and pipeline_result_json.exists() and missing_assets:
                repair_line = (
                    f"[REPAIR] {demo_label} pipeline assets missing count={len(missing_assets)} "
                    f"first={missing_assets[0]}"
                )
                print(repair_line, flush=True)
                _append_log(runner_log, repair_line)
            pipeline_cmd = build_pipeline_command(
                demo=demo,
                output_root=args.output_root,
                backend_mode=args.backend_mode,
                max_steps=args.max_steps,
                dry_run=args.dry_run,
                stop_on_complete=args.stop_on_complete,
                path_mode=args.path_mode,
                node_selection_mode=args.node_selection_mode,
                node_score_threshold=args.node_score_threshold,
                node_min_nodes=args.node_min_nodes,
                node_max_nodes=args.node_max_nodes,
                node_llm_scoring=args.node_llm_scoring,
                strict_online=args.strict_online,
                retries=args.retries,
                log_level=args.log_level,
            )
            rc = run_command(
                pipeline_cmd,
                env=env,
                log_file=logs_dir / f"{demo_label}_pipeline.log",
                stage_name=f"{demo_label}:pipeline",
            )
            if rc != 0:
                raise SystemExit(rc)
            pipeline_reran = True

        video_summary = ""
        video_missing = collect_missing_video_assets(video_summary_json) if args.resume_existing else []
        video_needs_refresh = bool(asset_repaired or video_missing)
        if not args.skip_video:
            if args.resume_existing and video_summary_json.exists() and not pipeline_reran and not video_needs_refresh:
                skip_line = f"[SKIP] {demo_label} existing video_summary_json={video_summary_json}"
                print(skip_line, flush=True)
                _append_log(runner_log, skip_line)
                video_summary = str(video_summary_json)
            else:
                if args.resume_existing and video_summary_json.exists() and video_needs_refresh:
                    refresh_line = (
                        f"[REFRESH] {demo_label} video stage rerun asset_repaired={asset_repaired} "
                        f"missing_video_assets={len(video_missing)}"
                    )
                    print(refresh_line, flush=True)
                    _append_log(runner_log, refresh_line)
                video_cmd = build_video_command(
                    run_dir=run_dir,
                    output_dir=video_dir,
                    dry_run=args.video_dry_run,
                    max_reference_images=args.max_reference_images,
                    multi_shot_events=args.multi_shot_events,
                    event_shot_count=args.event_shot_count,
                    event_duration_sec=args.event_duration_sec,
                    use_previous_tail_frame=args.use_previous_tail_frame,
                    use_previous_audio=args.use_previous_audio,
                    upload_previous_audio=args.upload_previous_audio,
                    cuda_visible_devices=args.cuda_visible_devices,
                    log_level=args.log_level,
                )
                rc = run_command(
                    video_cmd,
                    env=env,
                    log_file=logs_dir / f"{demo_label}_video.log",
                    stage_name=f"{demo_label}:video",
                )
                if rc != 0:
                    raise SystemExit(rc)
                video_summary = str(video_summary_json)

        if video_dir.exists():
            visual_cmd = build_visualization_command(run_dir=run_dir, video_dir=video_dir, output_html=html_path)
            rc = run_command(
                visual_cmd,
                env=env,
                log_file=logs_dir / f"{demo_label}_visualization.log",
                stage_name=f"{demo_label}:visualization",
            )
            if rc != 0:
                raise SystemExit(rc)

        runs.append(
            {
                "genre": demo["genre"],
                "title": demo["title"],
                "characters": demo["characters"],
                "demo_root": str(demo_root),
                "pipeline_result_json": str(run_dir / "pipeline_result.json"),
                "path_plan_json": str(run_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"),
                "video_summary_json": video_summary,
                "visualization_html": str(html_path) if html_path.exists() else "",
            }
        )
        demo_done = f"[DEMO DONE] {demo_label} visualization={html_path if html_path.exists() else ''}"
        print(demo_done, flush=True)
        _append_log(runner_log, demo_done)

    manifest = {
        "status": "done",
        "catalog_json": str(catalog),
        "backend_mode": args.backend_mode,
        "dry_run": bool(args.dry_run),
        "video_dry_run": bool(args.video_dry_run),
        "runs": runs,
    }
    manifest_path = args.output_root / "four_genre_demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = f"[RUNNER DONE] manifest={manifest_path} call_log_jsonl={call_log_jsonl}"
    print(done, flush=True)
    _append_log(runner_log, done)


if __name__ == "__main__":
    main()
