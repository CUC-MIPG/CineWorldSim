from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.video_agent import VideoAgent


def main() -> None:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Generate event-level videos from NarrativeNodeSelector + PathOrchestrator outputs. "
            "Each event automatically uses character keyframes and world-space images as multi-image references."
        )
    )
    parser.add_argument(
        "--pipeline-run-dir",
        type=Path,
        default=None,
        help="Run directory containing path_orchestration/. If set, JSON inputs are auto-resolved.",
    )
    parser.add_argument("--narrative-nodes-json", type=Path, default=None)
    parser.add_argument("--path-plan-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concat-output", type=Path, default=None)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="",
        help="Override MINIMAX_H3_CUDA_VISIBLE_DEVICES for this run. Use 4 for the fifth physical GPU.",
    )
    parser.add_argument("--minimax-parallel-mode", choices=["single_offload", "balanced"], default="")
    parser.add_argument("--minimax-device-map", choices=["auto", "balanced", "balanced_low_0", "sequential"], default="")
    parser.add_argument(
        "--minimax-max-memory",
        action="append",
        default=[],
        help="Per visible GPU memory, e.g. --minimax-max-memory 0:30GiB --minimax-max-memory 1:30GiB.",
    )
    parser.add_argument("--minimax-text-encoder-placement", choices=["cuda", "offload", "cpu"], default="")
    parser.add_argument("--minimax-offload-folder", type=str, default="")
    parser.add_argument("--minimax-runtime-device-index", type=int, default=None)
    parser.add_argument("--minimax-vae-parallel-mode", choices=["none", "tile"], default="")
    parser.add_argument("--minimax-vae-placement", choices=["cuda", "offload", "cpu"], default="")
    parser.add_argument("--minimax-vae-use-tiling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--minimax-vae-use-slicing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--minimax-reference-resize-width", type=int, default=0)
    parser.add_argument("--minimax-reference-resize-height", type=int, default=0)
    parser.add_argument("--max-reference-images", type=int, default=2)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--end-rank", type=int, default=None)
    parser.add_argument("--include-character-keyframes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-world-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-previous-tail-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract each completed event video's tail frame and prepend it as a reference for the next event.",
    )
    parser.add_argument(
        "--use-previous-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract each completed event video's audio and retain it as a continuity asset for the next event.",
    )
    parser.add_argument(
        "--upload-previous-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the prior event audio to MiniMax-H3 when audio continuity is enabled. Disabled by default for vLLM compatibility.",
    )
    parser.add_argument(
        "--multi-shot-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand each narrative event node into multiple generated shot clips, then concatenate them into one event video.",
    )
    parser.add_argument(
        "--event-shot-count",
        type=int,
        default=4,
        help="Number of shot clips generated for each event when --multi-shot-events is enabled.",
    )
    parser.add_argument(
        "--event-duration-sec",
        type=float,
        default=None,
        help=(
            "Optional total target duration per event when --multi-shot-events is enabled. "
            "When omitted, VideoAgent plans 5s/124-frame or 10s/243-frame output per event."
        ),
    )
    parser.add_argument("--insert-auxiliary-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-reference-mode",
        choices=["event_keyframe_only", "semantic_assets", "all_allowed"],
        default=None,
        help="Optional VideoAgent reference mode. Default uses each node's mode or all_allowed.",
    )
    parser.add_argument(
        "--dialogue-backfill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate event dialogue JSON and backfill it into prompts before skill adaptation. Default follows VIDEO_AGENT_DIALOGUE_BACKFILL, enabled when unset.",
    )
    parser.add_argument(
        "--disable-dialogue-backfill",
        dest="dialogue_backfill",
        action="store_false",
        help="Disable event dialogue JSON generation and preserve the legacy dialogue prompt path.",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    if args.cuda_visible_devices.strip():
        os.environ["MINIMAX_H3_CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()
    if args.minimax_parallel_mode:
        os.environ["MINIMAX_H3_PARALLEL_MODE"] = args.minimax_parallel_mode
    if args.minimax_device_map:
        os.environ["MINIMAX_H3_DEVICE_MAP"] = args.minimax_device_map
    if args.minimax_max_memory:
        os.environ["MINIMAX_H3_MAX_MEMORY"] = ",".join(args.minimax_max_memory)
    if args.minimax_text_encoder_placement:
        os.environ["MINIMAX_H3_TEXT_ENCODER_PLACEMENT"] = args.minimax_text_encoder_placement
    if args.minimax_offload_folder:
        os.environ["MINIMAX_H3_OFFLOAD_FOLDER"] = args.minimax_offload_folder
    if args.minimax_runtime_device_index is not None:
        os.environ["MINIMAX_H3_RUNTIME_DEVICE_INDEX"] = str(args.minimax_runtime_device_index)
    if args.minimax_vae_parallel_mode:
        os.environ["MINIMAX_H3_VAE_PARALLEL_MODE"] = args.minimax_vae_parallel_mode
    if args.minimax_vae_placement:
        os.environ["MINIMAX_H3_VAE_PLACEMENT"] = args.minimax_vae_placement
    if args.minimax_vae_use_tiling is not None:
        os.environ["MINIMAX_H3_VAE_USE_TILING"] = "true" if args.minimax_vae_use_tiling else "false"
    if args.minimax_vae_use_slicing is not None:
        os.environ["MINIMAX_H3_VAE_USE_SLICING"] = "true" if args.minimax_vae_use_slicing else "false"
    if args.minimax_reference_resize_width and args.minimax_reference_resize_height:
        os.environ["MINIMAX_H3_REFERENCE_RESIZE_WIDTH"] = str(args.minimax_reference_resize_width)
        os.environ["MINIMAX_H3_REFERENCE_RESIZE_HEIGHT"] = str(args.minimax_reference_resize_height)

    narrative_nodes_json = args.narrative_nodes_json
    path_plan_json = args.path_plan_json
    if args.pipeline_run_dir is not None:
        root = args.pipeline_run_dir
        narrative_nodes_json = narrative_nodes_json or root / "path_orchestration" / "narrative_nodes.json"
        path_plan_json = path_plan_json or root / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"
    if narrative_nodes_json is None or path_plan_json is None:
        raise SystemExit("--pipeline-run-dir or both --narrative-nodes-json and --path-plan-json are required")
    if not narrative_nodes_json.exists():
        raise FileNotFoundError(f"narrative nodes JSON not found: {narrative_nodes_json}")
    if not path_plan_json.exists():
        raise FileNotFoundError(f"path plan JSON not found: {path_plan_json}")

    agent = VideoAgent(strict_online=args.strict_online, retries=args.retries)
    summary = agent.process_orchestrated_events(
        narrative_nodes_json=narrative_nodes_json,
        path_plan_json=path_plan_json,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_reference_images=args.max_reference_images,
        include_character_keyframes=args.include_character_keyframes,
        include_world_images=args.include_world_images,
        start_rank=args.start_rank,
        end_rank=args.end_rank,
        use_previous_tail_frame=args.use_previous_tail_frame,
        use_previous_audio=args.use_previous_audio,
        upload_previous_audio=args.upload_previous_audio,
        multi_shot_events=args.multi_shot_events,
        event_shot_count=args.event_shot_count,
        event_duration_sec=args.event_duration_sec,
        dialogue_backfill=args.dialogue_backfill,
        insert_auxiliary_videos=args.insert_auxiliary_videos,
        video_reference_mode=args.video_reference_mode,
    )
    concat_output = args.concat_output or (args.output_dir / "final_orchestrated_events.mp4")
    concat_ok = False
    if args.concat and not args.dry_run:
        video_paths = [
            str(row.get("output_video_path", ""))
            for row in summary.get("generated", [])
            if row.get("status") == "done"
        ]
        concat_ok = agent.concat_videos(video_paths, concat_output)
        summary["concat_output"] = str(concat_output)
        summary["concat_ok"] = bool(concat_ok)
        (args.output_dir / "event_video_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("=== Orchestrated Event Video Generation Completed ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('MINIMAX_H3_CUDA_VISIBLE_DEVICES', '')}")
    print(f"MiniMax parallel mode: {os.environ.get('MINIMAX_H3_PARALLEL_MODE', '')}")
    print(f"MiniMax device map: {os.environ.get('MINIMAX_H3_DEVICE_MAP', '')}")
    print(f"MiniMax max memory: {os.environ.get('MINIMAX_H3_MAX_MEMORY', '')}")
    print(f"MiniMax text encoder placement: {os.environ.get('MINIMAX_H3_TEXT_ENCODER_PLACEMENT', '')}")
    print(f"MiniMax offload folder: {os.environ.get('MINIMAX_H3_OFFLOAD_FOLDER', '')}")
    print(f"MiniMax runtime device index: {os.environ.get('MINIMAX_H3_RUNTIME_DEVICE_INDEX', '')}")
    print(f"MiniMax VAE parallel mode: {os.environ.get('MINIMAX_H3_VAE_PARALLEL_MODE', '')}")
    print(f"MiniMax VAE placement: {os.environ.get('MINIMAX_H3_VAE_PLACEMENT', '')}")
    print(f"MiniMax VAE tiling: {os.environ.get('MINIMAX_H3_VAE_USE_TILING', '')}")
    print(
        "MiniMax reference resize: "
        f"{os.environ.get('MINIMAX_H3_REFERENCE_RESIZE_WIDTH', '')}x"
        f"{os.environ.get('MINIMAX_H3_REFERENCE_RESIZE_HEIGHT', '')}"
    )
    print(f"Multi-shot events: {args.multi_shot_events}")
    print(f"Event shot count: {args.event_shot_count}")
    print(f"Event duration sec: {args.event_duration_sec}")
    print(f"Insert auxiliary videos: {args.insert_auxiliary_videos}")
    print(f"Video reference mode: {args.video_reference_mode or 'node/default'}")
    print(f"Dialogue backfill: {args.dialogue_backfill if args.dialogue_backfill is not None else os.environ.get('VIDEO_AGENT_DIALOGUE_BACKFILL', 'true')}")
    print(f"Narrative nodes: {narrative_nodes_json}")
    print(f"Path plan: {path_plan_json}")
    print(f"Nodes: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Output: {args.output_dir}")
    print(f"Task JSON: {summary['task_json']}")
    if args.concat:
        print(f"Concat: {concat_ok}")
        print(f"Final video: {concat_output}")


if __name__ == "__main__":
    main()
