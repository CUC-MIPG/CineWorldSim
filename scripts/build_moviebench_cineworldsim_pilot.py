from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative.moviebench_llm_builder import (
    LLMJsonRunner,
    atomic_write_json,
    build_l1_prompt,
    build_l2_prompt,
    build_l3_prompt,
    build_l4_movie_bible_prompt,
    build_l4_movie_bible_consolidation_prompt,
    build_l4_story_arcs_prompt,
    build_l4_story_arcs_consolidation_prompt,
    build_source_manifest,
    canonicalize_l3_scenes,
    compact_l1_shot_for_l2,
    normalize_l2_event_ids,
    package_track_a_and_b,
    run_unit,
    validate_l1_collection,
    validate_l2_collection,
    validate_l3_collection,
)


LOGGER = logging.getLogger("moviebench_cineworldsim_pilot")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _setup_logging(output_root: Path, level: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_root / "pipeline.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_continuous_sequences(continuous_root: Path, movie_id: str) -> list[dict[str, Any]]:
    movie_root = continuous_root / movie_id
    sequences = []
    for path in sorted(movie_root.glob("*.json")):
        payload = _read_json(path)
        payload["_source_file"] = str(path)
        sequences.append(payload)
    return sequences


def _write_stage_index(output_root: Path, stage: str, payload: dict[str, Any]) -> None:
    atomic_write_json(output_root / stage / "index.json", payload)


def _status_results(root: Path, pattern: str = "*.json") -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob(pattern)):
        try:
            results.append(_read_json(path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("invalid stage file path=%s error=%s", path, exc)
    return results


def _normalize_l2_unit(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace model-generated L2 IDs with stable source-sequence IDs in place."""
    if payload.get("status") != "done":
        return payload
    result = payload.get("result")
    if not isinstance(result, dict):
        return payload
    sequence_id = str(result.get("source_sequence_id") or payload.get("unit_id") or "")
    events = result.get("l2_events")
    if not sequence_id or not isinstance(events, list):
        return payload
    normalized = normalize_l2_event_ids(sequence_id, events)
    if [item.get("l2_event_id") for item in events] != [item.get("l2_event_id") for item in normalized]:
        result = dict(result)
        result["l2_events"] = normalized
        result["id_normalization"] = {
            "strategy": "source_sequence_id_plus_position",
            "normalized": True,
        }
        payload = dict(payload)
        payload["result"] = result
        meta = dict(payload.get("meta") or {})
        meta["event_ids_normalized"] = True
        payload["meta"] = meta
        atomic_write_json(path, payload)
    return payload


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return _chunks(items, size)


def _run_hierarchical_consolidation(
    *,
    output_root: Path,
    stage: str,
    fragments: list[dict[str, Any]],
    prompt_builder: Any,
    runner: Any,
    force: bool,
    final_name: str,
) -> dict[str, Any]:
    """Merge fragments in bounded pairs to avoid one oversized final prompt."""
    if not fragments:
        raise ValueError(f"no fragments available for {stage}")
    current = fragments
    round_index = 1
    while len(current) > 1:
        batches = _batches(current, 2)
        merged: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            is_final = len(batches) == 1
            path = (
                output_root / final_name
                if is_final
                else output_root / "consolidation" / stage
                / f"round_{round_index:02d}_batch_{batch_index:02d}.json"
            )
            unit_id = (
                stage
                if is_final
                else f"{stage}_round_{round_index:02d}_batch_{batch_index:02d}"
            )
            result = run_unit(
                output_path=path,
                prompt_builder=lambda batch=batch: prompt_builder(batch),
                runner=runner,
                unit_id=unit_id,
                stage=stage,
                force=force,
            )
            if result.get("status") != "done":
                return result
            merged.append(result.get("result", {}))
        current = merged
        round_index += 1
    return {
        "unit_id": stage,
        "stage": stage,
        "status": "done",
        "result": current[0],
        "meta": {"status": "done", "model": runner.model, "hierarchical": True},
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    continuous_root = Path(args.continuous_root).resolve()
    output_root = Path(args.output_root).resolve()
    _setup_logging(output_root, args.log_level)

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not continuous_root.is_dir():
        raise FileNotFoundError(f"continuous root does not exist: {continuous_root}")

    LOGGER.info("start movie=%s source=%s output=%s", args.movie_id, source_root, output_root)
    manifest = build_source_manifest(source_root, args.movie_id)
    sequences = _load_continuous_sequences(continuous_root, args.movie_id)
    audit = {
        "schema_version": "1.0",
        "movie_id": args.movie_id,
        "source_root": str(source_root),
        "continuous_root": str(continuous_root),
        "source_shot_count": manifest["shot_count"],
        "continuous_sequence_count": len(sequences),
        "continuous_clip_count": sum(int(item.get("clip_count", 0)) for item in sequences),
        "invalid_source_files": manifest["invalid_files"],
        "source_sha256": manifest["source_sha256"],
        "processing_policy": {
            "llm_only": True,
            "fallback": False,
            "source_is_read_only": True,
            "model_source": "OPENAI_MODEL from .env",
            "base_url_source": "OPENAI_BASE_URL from .env",
        },
    }
    atomic_write_json(output_root / "audit_manifest.json", audit)
    atomic_write_json(output_root / "source_manifest.json", manifest)

    runner = LLMJsonRunner(model=args.model, retries=args.retries, sleep_sec=args.retry_sleep_sec)

    # Stage 1: one LLM result per source shot.
    l1_root = output_root / "l1_shots" / "units"
    l1_results: list[dict[str, Any]] = []
    shots = manifest["shots"]
    if args.max_l1:
        shots = shots[: args.max_l1]
    for shot in shots:
        shot_id = str(shot["l1_shot_id"])
        path = l1_root / f"{_safe_name(shot_id)}.json"
        result = run_unit(
            output_path=path,
            prompt_builder=lambda shot=shot: build_l1_prompt(
                shot, shot_id=shot_id, movie_id=args.movie_id
            ),
            runner=runner,
            unit_id=shot_id,
            stage="l1",
            force=args.force,
        )
        l1_results.append(result)
        LOGGER.info("l1 unit=%s status=%s", shot_id, result.get("status"))
    l1_done = [item["result"] for item in l1_results if item.get("status") == "done"]
    l1_validation = validate_l1_collection([str(item["l1_shot_id"]) for item in shots], l1_done)
    _write_stage_index(
        output_root,
        "l1_shots",
        {"stage": "l1", "unit_count": len(shots), "done_count": len(l1_done), "coverage": l1_validation},
    )
    if not l1_validation["complete"]:
        LOGGER.error("l1 coverage incomplete: %s", l1_validation)
        return {"status": "blocked", "stage": "l1", "audit": audit, "coverage": l1_validation}

    l1_by_id = {item["l1_shot_id"]: item for item in l1_done}

    # Stage 2: one LLM segmentation result per physical continuous sequence.
    l2_root = output_root / "l2_events" / "units"
    l2_results: list[dict[str, Any]] = []
    selected_sequences = sequences
    for sequence in selected_sequences:
        sequence_id = str(sequence.get("sequence_id", ""))
        l1_shots = []
        for clip in sequence.get("clips", []):
            shot_id = Path(str(clip.get("file_name", ""))).stem
            if shot_id in l1_by_id:
                l1_shots.append(compact_l1_shot_for_l2(l1_by_id[shot_id]))
        if not l1_shots:
            LOGGER.warning("l2 sequence has no completed l1 shots sequence=%s", sequence_id)
            continue
        l2_input = {
            "movie_id": args.movie_id,
            "source_sequence_id": sequence_id,
            "source_sequence_file": sequence.get("_source_file", ""),
            "time_range": sequence.get("time_range", {}),
            "shots": l1_shots,
        }
        path = l2_root / f"{_safe_name(sequence_id)}.json"
        result = run_unit(
            output_path=path,
            prompt_builder=lambda l2_input=l2_input: build_l2_prompt(l2_input),
            runner=runner,
            unit_id=sequence_id,
            stage="l2",
            force=args.force,
        )
        result = _normalize_l2_unit(path, result)
        l2_results.append(result)
        LOGGER.info("l2 unit=%s status=%s", sequence_id, result.get("status"))

    l2_events: list[dict[str, Any]] = []
    l2_coverage: dict[str, Any] = {"complete": False, "missing_ids": [], "duplicated_ids": [], "unexpected_ids": []}
    for item in l2_results:
        if item.get("status") != "done":
            continue
        payload = item.get("result", {})
        l2_events.extend(payload.get("l2_events", []))
    source_l1_ids = [str(item["l1_shot_id"]) for item in shots]
    l2_coverage = validate_l2_collection(source_l1_ids, l2_events)
    _write_stage_index(
        output_root,
        "l2_events",
        {"stage": "l2", "sequence_count": len(l2_results), "event_count": len(l2_events), "coverage": l2_coverage},
    )
    if not l2_coverage["complete"]:
        LOGGER.error("l2 coverage incomplete: %s", l2_coverage)
        return {"status": "blocked", "stage": "l2", "audit": audit, "coverage": l2_coverage}

    # Stage 3: L3 local windows. A full-film prompt is intentionally avoided:
    # it can exceed the output budget and silently return an empty result.
    l3_root = output_root / "l3_scenes"
    l3_local_root = l3_root / "local_units"
    l2_events.sort(key=lambda item: (item.get("time_range", {}).get("start_ms", 0), item.get("l2_event_id", "")))
    local_scenes: list[dict[str, Any]] = []
    for start in range(0, len(l2_events), args.l3_window_events):
        window = l2_events[start : start + args.l3_window_events]
        window_id = f"window_{start // args.l3_window_events + 1:04d}"
        path = l3_local_root / f"{window_id}.json"
        result = run_unit(
            output_path=path,
            prompt_builder=lambda window=window: build_l3_prompt(args.movie_id, window),
            runner=runner,
            unit_id=window_id,
            stage="l3_local",
            force=args.force,
        )
        window_ids = [str(item.get("l2_event_id", "")) for item in window]
        window_scenes = result.get("result", {}).get("l3_scenes", []) if result.get("status") == "done" else []
        window_scenes = canonicalize_l3_scenes(window_scenes, window)
        window_coverage = validate_l3_collection(window_ids, window_scenes)
        if result.get("status") == "done" and not window_coverage["complete"]:
            LOGGER.warning("l3 local coverage invalid; rerunning window=%s coverage=%s", window_id, window_coverage)
            result = run_unit(
                output_path=path,
                prompt_builder=lambda window=window: build_l3_prompt(args.movie_id, window),
                runner=runner,
                unit_id=window_id,
                stage="l3_local",
                force=True,
            )
            window_scenes = result.get("result", {}).get("l3_scenes", []) if result.get("status") == "done" else []
            window_scenes = canonicalize_l3_scenes(window_scenes, window)
            window_coverage = validate_l3_collection(window_ids, window_scenes)
        if result.get("status") == "done" and window_coverage["complete"]:
            local_scenes.extend(window_scenes)
        else:
            LOGGER.error("l3 local coverage incomplete window=%s coverage=%s", window_id, window_coverage)
        LOGGER.info("l3 local=%s status=%s", window_id, result.get("status"))

    l3_scenes = local_scenes
    l3_ids = [str(item.get("l2_event_id")) for item in l2_events]
    l3_coverage = validate_l3_collection(l3_ids, l3_scenes)
    atomic_write_json(l3_root / "coverage.json", l3_coverage)
    if l3_coverage["complete"]:
        atomic_write_json(
            l3_root / "final.json",
            {
                "unit_id": "final",
                "stage": "l3",
                "status": "done",
                "result": {
                    "movie_id": args.movie_id,
                    "l3_scenes": l3_scenes,
                    "coverage_check": l3_coverage,
                    "consolidation": {
                        "method": "validated_non_overlapping_local_windows",
                        "window_events": args.l3_window_events,
                    },
                },
                "meta": {"status": "done", "model": runner.model},
            },
        )
    _write_stage_index(
        output_root,
        "l3_scenes",
        {
            "stage": "l3",
            "local_scene_count": len(local_scenes),
            "scene_count": len(l3_scenes),
            "window_events": args.l3_window_events,
            "coverage": l3_coverage,
        },
    )
    if not l3_coverage["complete"]:
        LOGGER.error("l3 coverage incomplete: %s", l3_coverage)
        return {"status": "blocked", "stage": "l3", "audit": audit, "coverage": l3_coverage}

    # Stage 4: L4 chunked movie bible and story arcs, followed by LLM consolidation.
    l4_root = output_root / "l4_story"
    scene_chunks = _chunks(l3_scenes, args.l4_window_scenes)
    bible_chunk_root = l4_root / "bible_chunks"
    bible_fragments: list[dict[str, Any]] = []
    for index, scene_chunk in enumerate(scene_chunks, start=1):
        chunk_id = f"chunk_{index:04d}"
        chunk_path = bible_chunk_root / f"{chunk_id}.json"
        result = run_unit(
            output_path=chunk_path,
            prompt_builder=lambda scene_chunk=scene_chunk: build_l4_movie_bible_prompt(args.movie_id, scene_chunk),
            runner=runner,
            unit_id=chunk_id,
            stage="l4_bible_chunk",
            force=args.force,
        )
        if result.get("status") == "done":
            bible_fragments.append(result.get("result", {}))
        LOGGER.info("l4 bible chunk=%s status=%s", chunk_id, result.get("status"))
    if len(bible_fragments) != len(scene_chunks):
        return {"status": "blocked", "stage": "l4_bible_chunks", "audit": audit}

    bible = _run_hierarchical_consolidation(
        output_root=l4_root,
        stage="l4_movie_bible",
        fragments=bible_fragments,
        prompt_builder=lambda batch: build_l4_movie_bible_consolidation_prompt(args.movie_id, batch),
        runner=runner,
        force=args.force,
        final_name="movie_bible.json",
    )
    if bible.get("status") != "done":
        return {"status": "blocked", "stage": "l4_movie_bible", "audit": audit}

    arc_chunk_root = l4_root / "arc_chunks"
    arc_fragments: list[dict[str, Any]] = []
    for index, scene_chunk in enumerate(scene_chunks, start=1):
        chunk_id = f"chunk_{index:04d}"
        chunk_path = arc_chunk_root / f"{chunk_id}.json"
        result = run_unit(
            output_path=chunk_path,
            prompt_builder=lambda scene_chunk=scene_chunk: build_l4_story_arcs_prompt(
                args.movie_id, bible.get("result", {}), scene_chunk
            ),
            runner=runner,
            unit_id=chunk_id,
            stage="l4_arc_chunk",
            force=args.force,
        )
        if result.get("status") == "done":
            arc_fragments.append(result.get("result", {}))
        LOGGER.info("l4 arc chunk=%s status=%s", chunk_id, result.get("status"))
    if len(arc_fragments) != len(scene_chunks):
        return {"status": "blocked", "stage": "l4_arc_chunks", "audit": audit}

    story_arcs = _run_hierarchical_consolidation(
        output_root=l4_root,
        stage="l4_story_arcs",
        fragments=arc_fragments,
        prompt_builder=lambda batch: build_l4_story_arcs_consolidation_prompt(args.movie_id, batch),
        runner=runner,
        force=args.force,
        final_name="story_arcs.json",
    )
    if story_arcs.get("status") != "done":
        return {"status": "blocked", "stage": "l4_story_arcs", "audit": audit}

    arcs = story_arcs.get("result", {}).get("story_arcs", [])
    track_a, track_b = package_track_a_and_b(movie_id=args.movie_id, story_arcs=arcs, l1_by_id=l1_by_id)
    if any(sample.get("leakage_check", {}).get("leakage") for sample in track_b.get("samples", [])):
        LOGGER.error("track B leakage detected")
        return {"status": "blocked", "stage": "track_b", "audit": audit}
    atomic_write_json(output_root / "track_a.json", track_a)
    atomic_write_json(output_root / "track_b.json", track_b)
    summary = {
        "status": "done",
        "movie_id": args.movie_id,
        "counts": {
            "l1_shots": len(l1_done),
            "l2_events": len(l2_events),
            "l3_scenes": len(l3_scenes),
            "l4_story_arcs": len(arcs),
            "track_a_samples": len(track_a["samples"]),
            "track_b_samples": len(track_b["samples"]),
            "l3_window_events": args.l3_window_events,
            "l4_window_scenes": args.l4_window_scenes,
        },
        "model": runner.model,
        "output_root": str(output_root),
    }
    atomic_write_json(output_root / "pipeline_summary.json", summary)
    LOGGER.info("pipeline complete summary=%s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L1-L4 CineWorldSim artifacts for one MovieBench movie.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--continuous-root", required=True)
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0)
    parser.add_argument("--l3-window-events", type=int, default=20)
    parser.add_argument("--l4-window-scenes", type=int, default=20)
    parser.add_argument("--max-l1", type=int, default=0)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    result = run_pipeline(args)
    if result.get("status") != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
