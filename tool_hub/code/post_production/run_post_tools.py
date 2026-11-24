from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.post_production_tools import (
    edit_master_video_pseudocode,
    generate_dubbing_pseudocode,
    generate_narration_pseudocode,
)


def _split_csv_paths(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _split_csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for tok in (raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            continue
    return out


def _audio_exists(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def run_audio(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if args.dubbing_provider:
        import os

        os.environ["POST_DUBBING_PROVIDER"] = args.dubbing_provider

    if args.mode in {"narration", "both"}:
        out = args.output_dir / f"{args.prefix}_narration.wav"
        results["narration"] = generate_narration_pseudocode(
            narration_text=args.text,
            narrator_profile=args.narrator_profile,
            output_audio_path=str(out),
            language=args.language,
        )

    if args.mode in {"dubbing", "both"}:
        out = args.output_dir / f"{args.prefix}_dubbing.wav"
        results["dubbing"] = generate_dubbing_pseudocode(
            script_text=args.text,
            voice_profile=args.voice_profile,
            output_audio_path=str(out),
            language=args.language,
        )

    failures: list[str] = []
    for name, res in results.items():
        out_path = str(res.get("output_audio_path", ""))
        if not _audio_exists(out_path):
            failures.append(f"{name}: missing output audio")
        if args.strict_service and res.get("mode") != "service":
            failures.append(f"{name}: mode={res.get('mode')} expected=service")

    report = {
        "task": "audio",
        "mode": args.mode,
        "strict_service": bool(args.strict_service),
        "results": results,
        "failures": failures,
    }
    report_path = args.output_dir / f"{args.prefix}_audio_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Tool Hub Audio Test ===")
    print(f"Output dir: {args.output_dir}")
    print(f"Report: {report_path}")
    for name, res in results.items():
        print(
            f"- {name}: ok={res.get('ok')} mode={res.get('mode')} "
            f"file={res.get('output_audio_path')} todo={res.get('todo_file', '')}"
        )
    if failures:
        print("FAILED:")
        for msg in failures:
            print(f"- {msg}")
        return 1
    return 0


def run_edit(args: argparse.Namespace) -> int:
    output_video = args.output_video
    output_video.parent.mkdir(parents=True, exist_ok=True)

    res = edit_master_video_pseudocode(
        input_video_paths=_split_csv_paths(args.input_videos),
        dubbing_paths=_split_csv_paths(args.dubbing_paths),
        narration_paths=_split_csv_paths(args.narration_paths),
        bgm_paths=_split_csv_paths(args.bgm_paths),
        output_video_path=str(output_video),
        edit_instructions=args.edit_instructions,
        dubbing_segment_durations=_split_csv_floats(args.dubbing_segment_durations),
        narration_segment_durations=_split_csv_floats(args.narration_segment_durations),
        bgm_segment_durations=_split_csv_floats(args.bgm_segment_durations),
        include_narration=args.include_narration,
    )

    report_path = output_video.parent / f"{output_video.stem}_edit_report.json"
    report_path.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Tool Hub Edit Test ===")
    print(f"Output video: {output_video}")
    print(f"Report: {report_path}")
    print(json.dumps(res, ensure_ascii=False, indent=2))

    return 0 if bool(res.get("ok")) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool Hub Post-Production test runner")
    parser.add_argument("--log-level", type=str, default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audio = sub.add_parser("audio", help="Test narration/dubbing generation")
    p_audio.add_argument("--mode", choices=["narration", "dubbing", "both"], default="both")
    p_audio.add_argument("--text", type=str, default="你好，这是工具中心测试音频。")
    p_audio.add_argument("--output-dir", type=Path, default=Path("outputs/tool_hub/post_audio_test"))
    p_audio.add_argument("--prefix", type=str, default="sample")
    p_audio.add_argument("--language", type=str, default="zh-CN")
    p_audio.add_argument("--voice-profile", type=str, default="default_character_voice")
    p_audio.add_argument("--narrator-profile", type=str, default="default_narrator")
    p_audio.add_argument(
        "--dubbing-provider",
        choices=["auto", "omnivoice", "service", "pseudo"],
        default="auto",
        help="Provider for dubbing generation only.",
    )
    p_audio.add_argument("--strict-service", action=argparse.BooleanOptionalAction, default=False)

    p_edit = sub.add_parser("edit", help="Test OpenCV+ffmpeg editing")
    p_edit.add_argument("--input-videos", type=str, required=True, help="Comma-separated input video paths")
    p_edit.add_argument("--dubbing-paths", type=str, default="", help="Comma-separated dubbing wav paths")
    p_edit.add_argument("--narration-paths", type=str, default="", help="Comma-separated narration wav paths")
    p_edit.add_argument("--bgm-paths", type=str, default="", help="Comma-separated bgm wav paths")
    p_edit.add_argument(
        "--dubbing-segment-durations",
        type=str,
        default="",
        help="Comma-separated segment durations(sec) for dubbing timeline alignment.",
    )
    p_edit.add_argument(
        "--narration-segment-durations",
        type=str,
        default="",
        help="Comma-separated segment durations(sec) for narration timeline alignment.",
    )
    p_edit.add_argument(
        "--bgm-segment-durations",
        type=str,
        default="",
        help="Comma-separated segment durations(sec) for bgm timeline alignment.",
    )
    p_edit.add_argument(
        "--include-narration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include narration track in final edit mix.",
    )
    p_edit.add_argument("--output-video", type=Path, required=True)
    p_edit.add_argument(
        "--edit-instructions",
        type=str,
        default="Mix dubbing/narration/bgm and render final master.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.cmd == "audio":
        raise SystemExit(run_audio(args))
    if args.cmd == "edit":
        raise SystemExit(run_edit(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
