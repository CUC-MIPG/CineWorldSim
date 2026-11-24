from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.post_production_tools import (
    generate_dubbing_pseudocode,
    generate_narration_pseudocode,
)


def _is_service_ok(res: dict) -> bool:
    return bool(res.get("ok")) and str(res.get("mode", "")) == "service"


def _audio_exists(res: dict) -> bool:
    path = str(res.get("output_audio_path", "")).strip()
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test audio generation tools used by PostProductionAgent (narration/dubbing)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["narration", "dubbing", "both"],
        default="narration",
        help="Which tool path to test.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好，这是一个后处理音频生成联调测试。",
        help="Input text for narration/dubbing service.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audio_generation_test"),
    )
    parser.add_argument("--prefix", type=str, default="sample")
    parser.add_argument("--language", type=str, default="zh-CN")
    parser.add_argument("--voice-profile", type=str, default="default_character_voice")
    parser.add_argument("--narrator-profile", type=str, default="default_narrator")
    parser.add_argument(
        "--dubbing-provider",
        type=str,
        choices=["auto", "omnivoice", "service", "pseudo"],
        default="auto",
        help="Provider for dubbing path.",
    )
    parser.add_argument(
        "--strict-service",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when result is not mode=service or output file missing.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["POST_DUBBING_PROVIDER"] = args.dubbing_provider

    results: dict[str, dict] = {}

    if args.mode in {"narration", "both"}:
        narration_out = args.output_dir / f"{args.prefix}_narration.wav"
        results["narration"] = generate_narration_pseudocode(
            narration_text=args.text,
            narrator_profile=args.narrator_profile,
            output_audio_path=str(narration_out),
            language=args.language,
        )

    if args.mode in {"dubbing", "both"}:
        dubbing_out = args.output_dir / f"{args.prefix}_dubbing.wav"
        results["dubbing"] = generate_dubbing_pseudocode(
            script_text=args.text,
            voice_profile=args.voice_profile,
            output_audio_path=str(dubbing_out),
            language=args.language,
        )

    failed: list[str] = []
    for name, res in results.items():
        if not _audio_exists(res):
            failed.append(f"{name}: output file missing/empty")
            continue
        if args.strict_service and not _is_service_ok(res):
            failed.append(f"{name}: mode={res.get('mode')} (expected service)")

    report_path = args.output_dir / f"{args.prefix}_audio_test_report.json"
    report = {
        "mode": args.mode,
        "strict_service": bool(args.strict_service),
        "results": results,
        "failed": failed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Audio Generation Test Completed ===")
    print(f"Output dir: {args.output_dir}")
    print(f"Report: {report_path}")
    for name, res in results.items():
        print(
            f"- {name}: ok={res.get('ok')} mode={res.get('mode')} "
            f"file={res.get('output_audio_path')} todo={res.get('todo_file', '')}"
        )

    if failed:
        print("FAILED CHECKS:")
        for msg in failed:
            print(f"- {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
