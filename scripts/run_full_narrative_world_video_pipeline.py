#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent
from scripts.run_narrative_world_path_pipeline import _run_path_for_result, _run_world_stage, _read_story

LOGGER = logging.getLogger(__name__)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    story = _read_story(args)
    characters = [item.strip() for item in str(args.characters or "").split(",") if item.strip()]
    run_dirs = _run_world_stage(args, story, characters)
    video_agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path_output_dir = run_dir / "path_orchestration"
        path_manifest = _run_path_for_result(
            pipeline_result_json=run_dir / "pipeline_result.json",
            storyline=story,
            output_dir=path_output_dir,
            args=args,
        )
        artifacts = path_manifest.get("artifacts", {}) or {}
        narrative_nodes_json = Path(str(artifacts.get("narrative_nodes_json") or path_output_dir / "narrative_nodes.json"))
        path_plan_json = Path(str(artifacts.get("llm_path_json") or path_output_dir / "path_llm_direct" / "narrative_path_plan.json"))
        if not narrative_nodes_json.exists():
            raise FileNotFoundError(f"narrative nodes JSON not found: {narrative_nodes_json}")
        if not path_plan_json.exists():
            raise FileNotFoundError(f"path plan JSON not found: {path_plan_json}")

        video_output_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
        video_summary = video_agent.process_orchestrated_events(
            narrative_nodes_json=narrative_nodes_json,
            path_plan_json=path_plan_json,
            output_dir=video_output_dir,
            dry_run=args.video_dry_run,
            max_reference_images=args.max_reference_images,
            include_character_keyframes=True,
            include_world_images=True,
            use_previous_tail_frame=args.use_previous_tail_frame,
            use_previous_audio=args.use_previous_audio,
            upload_previous_audio=args.upload_previous_audio,
            multi_shot_events=args.multi_shot_events,
            event_shot_count=args.event_shot_count,
            event_duration_sec=args.event_duration_sec,
            dialogue_backfill=args.dialogue_backfill,
            insert_auxiliary_videos=bool(_arg(args, "insert_auxiliary_videos", True)),
            video_reference_mode=_arg(args, "video_reference_mode", None),
        )
        concat_output = video_output_dir / "final_orchestrated_events.mp4"
        concat_inputs = [
            str(Path(str(row.get("output_video_path") or "")))
            for row in video_summary.get("generated", [])
            if isinstance(row, dict)
            and row.get("status") == "done"
            and str(row.get("output_video_path") or "").strip()
            and Path(str(row.get("output_video_path"))).exists()
        ]
        concat_ok = False
        if not args.video_dry_run:
            concat_ok = video_agent.concat_videos(concat_inputs, concat_output)
        video_summary["concat_output"] = str(concat_output)
        video_summary["concat_inputs"] = concat_inputs
        video_summary["concat_ok"] = bool(concat_ok)
        if args.video_dry_run:
            video_summary["concat_skip_reason"] = "video_dry_run"
        runs.append(
            {
                "backend_mode": run_dir.name,
                "pipeline_result_json": str(run_dir / "pipeline_result.json"),
                "path_manifest_json": str(path_output_dir / "pipeline_manifest.json"),
                "video_summary_json": str(video_output_dir / "event_video_summary.json"),
                "video_task_json": str(video_output_dir / "event_video_tasks.json"),
                "video_output_dir": str(video_output_dir),
                "video_success": video_summary.get("success", 0),
                "video_total": video_summary.get("total", 0),
                "concat_ok": bool(concat_ok),
            }
        )
        _write_json(video_output_dir / "event_video_summary.json", video_summary)

    manifest = {
        "storyline": story,
        "characters": characters,
        "runs": runs,
    }
    _write_json(args.output_dir / "full_pipeline_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full narrative-world-video pipeline.")
    parser.add_argument("--story-file", type=Path, default=None)
    parser.add_argument("--storyline", type=str, default="")
    parser.add_argument("--characters", type=str, default="Journalist,HiddenWitness,Caretaker")
    parser.add_argument(
        "--backend-mode",
        type=str,
        choices=["all_flux", "all_banana", "all_flux2_klein"],
        default="all_flux",
    )
    parser.add_argument("--run-both", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--stop-on-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path-mode", choices=["llm", "mcts", "both"], default="llm")
    parser.add_argument("--enable-narrative-node-selector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--node-selection-mode", choices=["hybrid", "threshold", "top_k"], default="hybrid")
    parser.add_argument("--node-score-threshold", type=float, default=0.55)
    parser.add_argument("--node-min-nodes", type=int, default=3)
    parser.add_argument("--node-max-nodes", type=int, default=8)
    parser.add_argument("--node-llm-scoring", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--video-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-reference-images", type=int, default=4)
    parser.add_argument("--multi-shot-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-shot-count", type=int, default=4)
    parser.add_argument("--event-duration-sec", type=float, default=None)
    parser.add_argument("--use-previous-tail-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-previous-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dialogue-backfill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-reference-mode", choices=["event_keyframe_only", "semantic_assets", "all_allowed"], default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--mcts-model", type=str, default=None)
    parser.add_argument("--mcts-budget-iters", type=int, default=1)
    parser.add_argument("--mcts-topk-actions", type=int, default=8)
    parser.add_argument("--mcts-cpuct", type=float, default=1.2)
    parser.add_argument("--mcts-rollout-horizon", type=int, default=32)
    parser.add_argument("--mcts-rollout-topk", type=int, default=5)
    parser.add_argument("--mcts-rollout-epsilon", type=float, default=0.2)
    parser.add_argument("--mcts-pw-base", type=float, default=1.5)
    parser.add_argument("--mcts-pw-alpha", type=float, default=0.55)
    parser.add_argument("--pov-switch-penalty", type=float, default=0.05)
    parser.add_argument("--over-focus-threshold", type=int, default=4)
    parser.add_argument("--over-focus-penalty", type=float, default=0.08)
    parser.add_argument("--w-narrative-causality", type=float, default=0.4)
    parser.add_argument("--w-visual-transition", type=float, default=0.3)
    parser.add_argument("--w-rhythmic-pacing", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_pipeline(args)
    print(args.output_dir / "full_pipeline_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
