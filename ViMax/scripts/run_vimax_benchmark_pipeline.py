#!/usr/bin/env python3
"""Run ViMax Track A/B experiments on the local ``vimax_benchmark`` cases.

Track A is source-shot grounded: it includes the benchmark's shot-level
``first_frame`` and ``video_prompt`` fields and uses Script2VideoPipeline.
Track B is overview-only: it includes only ``story_overview`` and uses
Idea2VideoPipeline. The benchmark's shot fields are never passed to Track B.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline

LOGGER = logging.getLogger("vimax_benchmark_pipeline")
DEFAULT_BENCHMARK_ROOT = ROOT / "vimax_benchmark"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "vimax_benchmark"
DEFAULT_REQUIREMENT = (
    "Create a coherent cinematic video. Preserve the supplied character identities, "
    "locations, actions, and continuity constraints. Do not add unsupported plot facts."
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace(" ", "")
    if normalized in {"a", "typea"}:
        return "Type A"
    if normalized in {"b", "typeb"}:
        return "Type B"
    if normalized in {"c", "typec"}:
        return "Type C"
    raise ValueError(f"Unsupported benchmark type {value!r}; use A, B, or C")


def load_cases(root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    """Load and validate all cases listed by benchmark_index.json."""
    index_path = root / "benchmark_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing benchmark index: {index_path}")
    index = _read_json(index_path)
    rows = index.get("stories", []) if isinstance(index, dict) else []
    if not isinstance(rows, list):
        raise ValueError("benchmark_index.json: stories must be a list")

    cases: list[tuple[str, Path, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("file"):
            raise ValueError("Each benchmark index row must contain file")
        source = root / str(row["file"])
        if not source.exists():
            raise FileNotFoundError(f"Benchmark case listed in index does not exist: {source}")
        case = _read_json(source)
        _validate_case(case, source)
        theme = str(row.get("theme") or case.get("metadata", {}).get("theme_key") or source.stem)
        cases.append((theme, source, case))
    return cases


def _validate_case(case: dict[str, Any], source: Path) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{source}: case must be an object")
    for key in ("story_overview", "consistency_type", "scenes"):
        if not case.get(key):
            raise ValueError(f"{source}: missing required field {key}")
    if not isinstance(case["scenes"], list) or not case["scenes"]:
        raise ValueError(f"{source}: scenes must be a non-empty list")
    for scene in case["scenes"]:
        if not isinstance(scene, dict) or not isinstance(scene.get("shots"), list):
            raise ValueError(f"{source}: every scene must contain a shots list")
        for shot in scene["shots"]:
            if not isinstance(shot, dict) or not shot.get("first_frame") or not shot.get("video_prompt"):
                raise ValueError(f"{source}: every shot needs first_frame and video_prompt")


def select_cases(
    cases: Iterable[tuple[str, Path, dict[str, Any]]],
    selectors: list[str] | None = None,
    case_type: str | None = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    selectors_lower = {item.lower().removesuffix(".json") for item in (selectors or [])}
    wanted_type = _normalize_type(case_type)
    selected = []
    for theme, source, case in cases:
        aliases = {theme.lower(), source.name.lower(), source.stem.lower()}
        row_type = str(case["consistency_type"])
        if selectors_lower and not aliases.intersection(selectors_lower):
            continue
        if wanted_type and row_type != wanted_type:
            continue
        selected.append((theme, source, case))
    return selected


def flatten_shots(case: dict[str, Any], max_shots: int = 0) -> list[tuple[int, int, dict[str, Any]]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for scene in sorted(case["scenes"], key=lambda item: int(item.get("scene_num", 0))):
        scene_num = int(scene.get("scene_num", 0))
        for shot in sorted(scene["shots"], key=lambda item: int(item.get("shot_id", 0))):
            rows.append((scene_num, int(shot.get("shot_id", 0)), shot))
    return rows[:max_shots] if max_shots > 0 else rows


def build_track_a_script(case: dict[str, Any], max_shots: int = 0) -> str:
    """Build a grounded script while preserving every selected source shot."""
    lines = [
        "VIMAX BENCHMARK TRACK A: SOURCE-SHOT GROUNDED SCRIPT",
        f"Consistency type: {case['consistency_type']}",
        f"Story overview: {case['story_overview']}",
        "Adapt the following source shot plan in order. Preserve identity, spatial continuity, and the stated actions.",
    ]
    for scene_num, shot_id, shot in flatten_shots(case, max_shots):
        lines.extend([
            f"\nSCENE {scene_num} / SOURCE SHOT {shot_id}",
            "First-frame specification:",
            str(shot["first_frame"]).strip(),
            "Motion and video specification:",
            str(shot["video_prompt"]).strip(),
        ])
    return "\n".join(lines).strip() + "\n"


def build_track_b_idea(case: dict[str, Any]) -> str:
    """Return the only benchmark field allowed as Track B content."""
    return str(case["story_overview"]).strip() + "\n"


def _write_track_config(source_path: Path, track_dir: Path) -> Path:
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    config["working_dir"] = str(track_dir / "vimax_pipeline")
    target = track_dir / "vimax_config.yaml"
    _write_yaml(target, config)
    return target


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _case_manifest(theme: str, source: Path, case: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running" if args.run else "dry_run",
        "case": theme,
        "source": str(source.resolve()),
        "consistency_type": case["consistency_type"],
        "scene_count": len(case["scenes"]),
        "source_shot_count": len(flatten_shots(case)),
        "max_shots": args.max_shots,
        "track_policy": {
            "A": "story_overview + selected first_frame/video_prompt shot fields",
            "B": "story_overview only; shot-level fields excluded",
        },
        "output_dir": str(output_dir.resolve()),
        "tracks": [],
    }


async def _run_case_online(
    case: dict[str, Any],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    from interfaces import CharacterInScene

    for spec in specs:
        track_dir = Path(spec["track_dir"])
        if spec["track"] == "A":
            config_path = _write_track_config(Path(args.script_config), track_dir)
            pipeline = Script2VideoPipeline.init_from_config(str(config_path))
            final_path = await pipeline(
                script=Path(spec["input_path"]).read_text(encoding="utf-8"),
                user_requirement=args.user_requirement,
                style=args.style,
            )
        else:
            config_path = _write_track_config(Path(args.idea_config), track_dir)
            pipeline = Idea2VideoPipeline.init_from_config(str(config_path))
            final_path = await pipeline(
                idea=Path(spec["input_path"]).read_text(encoding="utf-8"),
                user_requirement=args.user_requirement,
                style=args.style,
            )
        spec["resolved_config"] = str(config_path.resolve())
        spec["status"] = "success"
        spec["final_video"] = str(Path(final_path).resolve())


def run_case(theme: str, source: Path, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case_output = Path(args.output_dir).resolve() / theme
    if case_output.exists() and any(case_output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty; use a fresh path or --resume: {case_output}")
    case_output.mkdir(parents=True, exist_ok=True)

    manifest = _case_manifest(theme, source, case, case_output, args)
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    specs: list[dict[str, Any]] = []
    for track in tracks:
        track_dir = case_output / f"track_{track}"
        track_dir.mkdir(parents=True, exist_ok=True)
        if track == "A":
            input_path = track_dir / "input_script.txt"
            input_path.write_text(build_track_a_script(case, args.max_shots), encoding="utf-8")
            policy = "source-shot grounded; includes first_frame and video_prompt"
            pipeline_name = "Script2VideoPipeline"
            config_source = Path(args.script_config).resolve()
        else:
            input_path = track_dir / "input_idea.txt"
            input_path.write_text(build_track_b_idea(case), encoding="utf-8")
            policy = "overview-only; excludes scenes and shot-level fields"
            pipeline_name = "Idea2VideoPipeline"
            config_source = Path(args.idea_config).resolve()
        specs.append({
            "track": track,
            "pipeline": pipeline_name,
            "input_policy": policy,
            "input_path": str(input_path.resolve()),
            "config_source": str(config_source),
            "track_dir": str(track_dir.resolve()),
            "status": "pending",
        })
    manifest["tracks"] = specs
    manifest_path = case_output / "case_manifest.json"
    _write_json(manifest_path, manifest)

    if args.run:
        try:
            load_dotenv(ROOT / ".env", override=False)
            asyncio.run(_run_case_online(case, specs, args))
            manifest["status"] = "complete"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--case", action="append", help="Case theme, filename, or stem; repeatable")
    parser.add_argument("--type", dest="case_type", choices=("A", "B", "C", "Type A", "Type B", "Type C"))
    parser.add_argument("--all", action="store_true", help="Run all cases matching --type")
    parser.add_argument("--track", choices=("A", "B", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--script-config", type=Path, default=ROOT / "configs" / "script2video.yaml")
    parser.add_argument("--idea-config", type=Path, default=ROOT / "configs" / "idea2video.yaml")
    parser.add_argument("--max-shots", type=int, default=0, help="Cap Track A source shots; 0 uses all shots")
    parser.add_argument("--style", default="cinematic realistic")
    parser.add_argument("--user-requirement", default=DEFAULT_REQUIREMENT)
    parser.add_argument("--run", action="store_true", help="Execute online generation; default is dry-run")
    parser.add_argument("--resume", action="store_true", help="Allow reuse of an existing case output directory")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.max_shots < 0:
        raise ValueError("--max-shots cannot be negative")
    if not args.all and not args.case:
        raise ValueError("Specify --case NAME or --all")

    cases = select_cases(load_cases(args.benchmark_root), args.case, args.case_type)
    if not cases:
        raise ValueError("No benchmark cases matched the requested selector")
    if not args.all and len(cases) > 1:
        raise ValueError("Multiple cases matched; use --all or select one exact --case")

    summaries = []
    for theme, source, case in cases:
        LOGGER.info("Preparing %s (%s, %d source shots)", theme, case["consistency_type"], len(flatten_shots(case)))
        summaries.append(run_case(theme, source, case, args))

    summary_path = Path(args.output_dir).resolve() / "benchmark_run_manifest.json"
    _write_json(summary_path, {"status": "complete" if not args.run else "online_run_finished", "cases": summaries})
    print(f"Wrote benchmark manifest: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
