#!/usr/bin/env python3
"""Run independent MovieBench Track A/B experiments through ViMax pipelines."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interfaces import CharacterInScene
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline

LOGGER = logging.getLogger("vimax_moviebench_pipeline")
DEFAULT_DATASET = Path("/data1/AgentG/hhb/CameraMaster/outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4")
DEFAULT_OUTPUT = ROOT / "outputs" / "vimax_moviebench" / "0001_American_Beauty" / "native_pipeline"
USER_REQUIREMENT = (
    "Create a coherent cinematic adaptation from the supplied MovieBench material. "
    "Preserve its supported events and character identities; do not invent unsupported plot facts."
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_unit_index(folder: Path, id_field: str, list_field: str | None = None) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        payload = _read_json(path)
        data = _unwrap(payload)
        rows = data.get(list_field, []) if list_field and isinstance(data, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get(id_field):
                    indexed[str(row[id_field])] = row
        if isinstance(data, dict) and data.get(id_field):
            indexed[str(data[id_field])] = data
    return indexed


def load_moviebench_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    required = [
        root / "l4_global" / "movie_bible.json",
        root / "l3_arcs" / "story_arcs.json",
        root / "l2_events" / "units",
        root / "l1_shots" / "units",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MovieBench dataset is missing required paths: " + ", ".join(missing))

    bible = _unwrap(_read_json(required[0]))
    if not isinstance(bible, dict):
        raise ValueError("l4_global/movie_bible.json must contain an object")
    arc_payload = _unwrap(_read_json(required[1]))
    arcs = arc_payload.get("story_arcs", []) if isinstance(arc_payload, dict) else []
    if not isinstance(arcs, list):
        raise ValueError("l3_arcs/story_arcs.json must contain a story_arcs list")
    return {
        "root": root,
        "movie_id": str(bible.get("movie_id") or root.name.removesuffix("_llm_l4")),
        "bible": bible,
        "arcs": [row for row in arcs if isinstance(row, dict)],
        "l2_by_id": _load_unit_index(required[2], "l2_event_id", "l2_events"),
        "l1_by_id": _load_unit_index(required[3], "l1_shot_id"),
    }


def _selected_arc(data: dict[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [
        arc for arc in data["arcs"]
        if not sample_id
        or str(arc.get("l3_story_arc_id") or "") == sample_id
        or str(arc.get("source_l4_story_arc_id") or "") == sample_id
    ]
    if not matches:
        raise ValueError(f"No L3 arc matches sample ID {sample_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Sample ID {sample_id!r} matches multiple L3 arcs; use the exact L3 arc ID")
    return matches[0]


def _time_key(row: dict[str, Any]) -> int:
    value = row.get("time_range", {})
    return int(value.get("start_ms", 0)) if isinstance(value, dict) else 0


def _arc_shots(data: dict[str, Any], arc: dict[str, Any], max_l1_nodes: int = 0) -> list[dict[str, Any]]:
    shot_ids = [str(value) for value in arc.get("l1_shot_ids", []) or []]
    shots = [data["l1_by_id"][shot_id] for shot_id in shot_ids if shot_id in data["l1_by_id"]]
    shots.sort(key=lambda row: (_time_key(row), int(row.get("order_index", 0) or 0), str(row.get("l1_shot_id", ""))))
    return shots[:max_l1_nodes] if max_l1_nodes > 0 else shots


def build_track_a_script(data: dict[str, Any], sample_id: str, max_l1_nodes: int = 0) -> str:
    arc = _selected_arc(data, sample_id)
    l2_ids = [str(value) for value in arc.get("l2_event_ids", []) or []]
    events = [data["l2_by_id"][event_id] for event_id in l2_ids if event_id in data["l2_by_id"]]
    events.sort(key=lambda row: (_time_key(row), str(row.get("l2_event_id", ""))))
    shots = _arc_shots(data, arc, max_l1_nodes)
    selected_shot_ids = {str(row.get("l1_shot_id", "")) for row in shots}
    event_by_shot: dict[str, list[str]] = {}
    for event in events:
        for shot_id in event.get("source_shot_ids", []) or []:
            if str(shot_id) in selected_shot_ids:
                event_by_shot.setdefault(str(shot_id), []).append(str(event.get("event_description") or ""))

    sections = [
        f"MOVIEBENCH L3 ARC: {arc.get('arc_title') or arc.get('l3_story_arc_id') or sample_id}",
        str(arc.get("full_story_S") or arc.get("short_storyline_L") or "").strip(),
        "Adapt the following observed shot sequence in order:",
    ]
    if events:
        sections.append("Linked L2 events:\n" + "\n".join(
            f"- {event.get('event_title') or event.get('l2_event_id')}: {event.get('event_description') or ''}".strip()
            for event in events
        ))
    for index, shot in enumerate(shots, 1):
        shot_id = str(shot.get("l1_shot_id") or "")
        time_range = shot.get("time_range", {})
        time_text = " - ".join(str(time_range.get(key, "")) for key in ("start_tc", "end_tc")) if isinstance(time_range, dict) else ""
        facts = [str(value).strip() for value in shot.get("observable_facts", []) or [] if str(value).strip()]
        action_rows = shot.get("actions", []) or []
        actions = [" ".join(str(item) for item in (row.get("actor"), row.get("action"), row.get("target")) if item) for row in action_rows if isinstance(row, dict)]
        locations = [str(row.get("name", "")).strip() for row in shot.get("location_mentions", []) or [] if isinstance(row, dict) and row.get("name")]
        event_context = [value for value in event_by_shot.get(shot_id, []) if value]
        detail = facts or actions or event_context
        if not detail:
            detail = [str(row.get("visible_action", "")).strip() for row in shot.get("characters", []) or [] if isinstance(row, dict) and row.get("visible_action")]
        lines = [f"Shot {index} [{time_text}] ({shot_id}):"]
        if locations:
            lines.append("Location: " + "; ".join(dict.fromkeys(locations)))
        lines.extend(detail)
        sections.append("\n".join(lines))
    script = "\n\n".join(section for section in sections if section)
    if not shots:
        raise ValueError(f"L3 arc {sample_id!r} has no resolvable L1 shots")
    return script


def build_track_a_characters(data: dict[str, Any], sample_id: str, max_l1_nodes: int = 0) -> list[dict[str, Any]]:
    arc = _selected_arc(data, sample_id)
    by_name: dict[str, dict[str, Any]] = {}
    for shot in _arc_shots(data, arc, max_l1_nodes):
        for row in shot.get("characters", []) or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            target = by_name.setdefault(name, {
                "idx": len(by_name), "identifier_in_scene": name, "is_visible": True,
                "static_features": " ".join(str(row.get(key) or "").strip() for key in ("visible_state", "appearance") if row.get(key)) or f"{name}, appearance unspecified in the selected shot annotations.",
                "dynamic_features": str(row.get("visible_action") or "").strip() or None,
            })
            action = str(row.get("visible_action") or "").strip()
            if action and action not in (target.get("dynamic_features") or ""):
                target["dynamic_features"] = "; ".join(filter(None, [target.get("dynamic_features"), action]))
    return list(by_name.values())


def build_track_b_idea(data: dict[str, Any]) -> str:
    bible = data["bible"]
    storyline = bible.get("global_storyline")
    story = str(storyline.get("story") or "").strip() if isinstance(storyline, dict) else str(storyline or "").strip()
    synopsis = bible.get("movie_synopsis")
    summary = str(synopsis.get("short_synopsis") or "").strip() if isinstance(synopsis, dict) else str(synopsis or "").strip()
    profiles = []
    for row in bible.get("global_character_profiles", []) or []:
        if isinstance(row, dict):
            name = str(row.get("character") or row.get("name") or row.get("character_name") or "").strip()
            profile_parts = [
                str(row.get(key) or "").strip()
                for key in ("observed_profile", "inferred_arc", "profile", "description", "static_features")
                if row.get(key)
            ]
            profile = " ".join(profile_parts)
            if name:
                profiles.append(f"{name}: {profile}" if profile else name)
    core = story or summary
    if not core:
        raise ValueError("L4 movie_bible.json has neither global_storyline nor movie_synopsis")
    lines = [core]
    if summary and summary != core:
        lines.append("Movie synopsis: " + summary)
    if profiles:
        lines.append("Global character profiles:\n" + "\n".join(profiles))
    return "\n\n".join(lines)


def _write_track_config(source_path: Path, track_dir: Path) -> Path:
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    config["working_dir"] = str(track_dir / "vimax_pipeline")
    target = track_dir / "vimax_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    if args.max_l1_nodes < 0:
        raise ValueError("--max-l1-nodes cannot be negative")
    dataset = load_moviebench_dataset(Path(args.dataset_root))
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty; select a fresh path or pass --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    arc = _selected_arc(dataset, args.sample_id) if "A" in tracks else None

    manifest: dict[str, Any] = {
        "status": "running" if args.run else "dry_run",
        "movie_id": dataset["movie_id"],
        "dataset_root": str(dataset["root"]),
        "execution": "ViMax-native pipelines",
        "tracks": [],
    }
    run_specs: list[dict[str, Any]] = []
    for track in tracks:
        track_dir = output_root / f"track_{track}"
        track_dir.mkdir(parents=True, exist_ok=True)
        if track == "A":
            script = build_track_a_script(dataset, args.sample_id, args.max_l1_nodes)
            characters = build_track_a_characters(dataset, args.sample_id, args.max_l1_nodes)
            input_path = track_dir / "input_script.txt"
            input_path.write_text(script + "\n", encoding="utf-8")
            character_path = track_dir / "input_characters.json"
            _write_json(character_path, characters)
            run_specs.append({
                "track": "A", "input_level": "L3+linked L2/L1", "anchor_l3_id": arc.get("l3_story_arc_id"),
                "l1_count": len(_arc_shots(dataset, arc, args.max_l1_nodes)),
                "script_path": str(input_path), "characters_path": str(character_path),
                "pipeline": "Script2VideoPipeline", "config_source": str(Path(args.script_config).resolve()),
                "track_dir": track_dir,
            })
        else:
            idea = build_track_b_idea(dataset)
            input_path = track_dir / "input_idea.txt"
            input_path.write_text(idea + "\n", encoding="utf-8")
            run_specs.append({
                "track": "B", "input_level": "L4 movie_bible only", "idea_path": str(input_path),
                "pipeline": "Idea2VideoPipeline", "config_source": str(Path(args.idea_config).resolve()),
                "track_dir": track_dir,
            })

    for spec in run_specs:
        spec.pop("track_dir")
    manifest["tracks"] = run_specs
    _write_json(output_root / "moviebench_vimax_manifest.json", manifest)
    if not args.run:
        return manifest

    async def execute() -> None:
        for spec in run_specs:
            track_dir = output_root / f"track_{spec['track']}"
            source = Path(spec["config_source"])
            config_path = _write_track_config(source, track_dir)
            spec["resolved_config"] = str(config_path)
            if spec["track"] == "A":
                pipeline = Script2VideoPipeline.init_from_config(str(config_path))
                characters = [CharacterInScene.model_validate(row) for row in _read_json(Path(spec["characters_path"]))]
                final_path = await pipeline(
                    script=Path(spec["script_path"]).read_text(encoding="utf-8"),
                    user_requirement=USER_REQUIREMENT,
                    style=args.style,
                    characters=characters,
                )
            else:
                pipeline = Idea2VideoPipeline.init_from_config(str(config_path))
                final_path = await pipeline(
                    idea=Path(spec["idea_path"]).read_text(encoding="utf-8"),
                    user_requirement=USER_REQUIREMENT,
                    style=args.style,
                )
            spec["status"] = "success"
            spec["final_video"] = str(final_path)
            _write_json(output_root / "moviebench_vimax_manifest.json", manifest)

    try:
        asyncio.run(execute())
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_root / "moviebench_vimax_manifest.json", manifest)
        raise
    _write_json(output_root / "moviebench_vimax_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--track", choices=("A", "B", "both"), default="both")
    parser.add_argument("--sample-id", default="", help="L3 arc ID for Track A; Track B remains movie-level.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--script-config", type=Path, default=ROOT / "configs" / "script2video.yaml")
    parser.add_argument("--idea-config", type=Path, default=ROOT / "configs" / "idea2video.yaml")
    parser.add_argument("--max-l1-nodes", type=int, default=0, help="Cap Track A source shots; 0 uses the entire selected arc.")
    parser.add_argument("--style", default="cinematic")
    parser.add_argument("--run", action="store_true", help="Execute online ViMax generation; default is offline input dry-run.")
    parser.add_argument("--resume", action="store_true", help="Allow use of an existing output directory; ViMax may reuse its cached artifacts.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
