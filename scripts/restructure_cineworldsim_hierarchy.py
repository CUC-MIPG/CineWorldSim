#!/usr/bin/env python3
"""Create a non-destructive CineWorldSim hierarchy with movie-level L4 and arc-level L3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm_tool import LLMJsonTool


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_value(value: Any, limit: int) -> Any:
    """Bound verbose prompt values while retaining provenance ID fields."""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        result: list[Any] = []
        used = 0
        for item in value:
            remaining = limit - used
            if remaining <= 0:
                break
            compact_item = _compact_value(item, remaining)
            encoded = json.dumps(compact_item, ensure_ascii=False)
            if used + len(encoded) > limit and not isinstance(item, (str, int, float, bool)):
                compact_item = _compact_value(item, max(80, remaining // 2))
                encoded = json.dumps(compact_item, ensure_ascii=False)
            if used + len(encoded) > limit:
                break
            result.append(compact_item)
            used += len(encoded)
        return result
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        used = 0
        for key, item in value.items():
            remaining = limit - used - len(key) - 4
            if remaining <= 0 and not key.endswith(("_id", "_ids")):
                continue
            if key.endswith(("_id", "_ids")):
                compact_item = item
            else:
                compact_item = _compact_value(item, max(0, remaining))
            encoded = json.dumps(compact_item, ensure_ascii=False)
            if used + len(key) + len(encoded) > limit and not key.endswith(("_id", "_ids")):
                continue
            result[key] = compact_item
            used += len(key) + len(encoded)
        return result
    return value


def _compact_id_list(value: Any, max_items: int | None = None) -> list[str]:
    """Keep readable provenance IDs without allowing repeated lists to dominate a prompt."""
    if not isinstance(value, list):
        return []
    ids = [str(item) for item in value if str(item).strip()]
    if max_items is not None and len(ids) > max_items:
        ids = ids[:max_items] + [f"...+{len(ids) - max_items}_ids"]
    return ids


def _compact_records(value: Any, limit: int, max_items: int = 1) -> list[Any]:
    """Keep the first short action/state records in a predictable compact form."""
    if not isinstance(value, list):
        return []
    result: list[Any] = []
    used = 0
    preferred_keys = (
        "actor",
        "action",
        "target",
        "subject",
        "change",
        "before",
        "after",
        "from",
        "to",
        "certainty",
    )
    for item in value[:max_items]:
        if isinstance(item, dict):
            parts = [
                f"{key}={str(item[key]).strip()}"
                for key in preferred_keys
                if key in item and str(item[key]).strip()
            ]
            compact_item = _compact_value("; ".join(parts), max(1, limit - 2))
        else:
            compact_item = _compact_value(item, max(1, limit - 2))
        encoded = json.dumps(compact_item, ensure_ascii=False, separators=(",", ":"))
        if used + len(encoded) > limit:
            break
        result.append(compact_item)
        used += len(encoded)
    return result


def _compact_movie_bible(bible: dict[str, Any]) -> dict[str, Any]:
    """Keep global prior fields and bound repeated generated descriptions."""
    fields = (
        "movie_id",
        "coverage",
        "movie_synopsis",
        "global_character_profiles",
        "recurring_locations",
        "recurring_objects",
        "global_constraints_supported_by_evidence",
        "unresolved_questions",
        "evidence_trace",
    )
    compact = {key: bible[key] for key in fields if key in bible}
    return _compact_value(compact, 24000)


def _compact_l3_scene_for_l4(scene: dict[str, Any]) -> dict[str, Any]:
    """Keep one bounded narrative summary per L3 scene and its lineage."""
    return {
        "l3_scene_id": scene.get("l3_scene_id", ""),
        "l2_event_ids": _compact_id_list(scene.get("l2_event_ids", []), max_items=2),
        "l2_event_count": len(scene.get("l2_event_ids", []))
        if isinstance(scene.get("l2_event_ids", []), list)
        else 0,
        "time_range": _compact_value(scene.get("time_range", {}), 40),
        "scene_setting": _compact_value(scene.get("scene_setting", ""), 40),
        "active_characters": _compact_value(scene.get("active_characters", []), 40),
        "scene_purpose": _compact_value(scene.get("scene_purpose", ""), 60),
        "scene_narrative": _compact_value(scene.get("scene_narrative", ""), 60),
        "event_progression": _compact_records(scene.get("event_progression", []), 35, max_items=1),
        "narrative_state_transition": _compact_value(scene.get("narrative_state_transition", {}), 45),
    }


def _compact_l2_event_for_l4(event: dict[str, Any]) -> dict[str, Any]:
    """Keep event identity, visual evidence, and minimal causal action/state facts."""
    return {
        "l2_event_id": event.get("l2_event_id", ""),
        "source_shot_ids": _compact_id_list(event.get("source_shot_ids", [])),
        "event_title": _compact_value(event.get("event_title", ""), 20),
        "event_description": _compact_value(event.get("event_description", ""), 40),
        "actions": _compact_records(event.get("actions", []), 30),
        "state_changes": _compact_records(event.get("state_changes", []), 30),
    }


def build_l4_generation_prompt(
    movie_id: str,
    movie_bible: dict[str, Any],
    l3_scenes: list[dict[str, Any]],
    l2_events: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build a bounded full-film L4 synthesis prompt from existing evidence."""
    system = (
        "You are a strict full-film narrative synthesis assistant.\n"
        "Create one movie-level L4 global story from the existing Movie Bible and "
        "compact L3/L2 evidence. Do not invent unsupported facts, and distinguish "
        "observed, inferred, and uncertain information. Preserve source IDs for "
        "important claims. Return valid JSON only."
    )
    compact_scene_objects = [_compact_l3_scene_for_l4(scene) for scene in l3_scenes]
    compact_event_objects = [_compact_l2_event_for_l4(event) for event in l2_events]
    scene_keys = (
        "l3_scene_id",
        "l2_event_ids",
        "l2_event_count",
        "time_range",
        "scene_setting",
        "active_characters",
        "scene_purpose",
        "scene_narrative",
        "event_progression",
        "narrative_state_transition",
    )
    event_keys = (
        "l2_event_id",
        "source_shot_ids",
        "event_title",
        "event_description",
        "actions",
        "state_changes",
    )
    compact_scenes = [[scene.get(key) for key in scene_keys] for scene in compact_scene_objects]
    compact_events = [[event.get(key) for key in event_keys] for event in compact_event_objects]
    user = f"""Task: Regenerate the global L4 movie story for {movie_id}.

Use the existing Movie Bible as a prior, then correct, consolidate, and complete it
only where the supplied L3 scenes and L2 events provide evidence. L4 is the single
global story of the whole movie, not a local story arc.

Required output fields:
- movie_id: exactly {movie_id}
- movie_synopsis: an object with short_synopsis, chronological_synopsis,
  main_conflicts, and major_state_transitions
- global_storyline: one coherent full-film storyline, separate from local arcs
- global_character_profiles, recurring_locations, recurring_objects
- global_constraints_supported_by_evidence, unresolved_questions
- evidence_trace with source_l3_ids, source_l2_ids, and source_l1_ids arrays

Existing Movie Bible prior:
{json.dumps(_compact_movie_bible(movie_bible), ensure_ascii=False, separators=(",", ":"))}

Compact L3 scene evidence ({len(compact_scenes)} records):
Schema: [l3_scene_id, l2_event_ids_sample, l2_event_count, time_range,
scene_setting, active_characters, scene_purpose, scene_narrative,
event_progression, narrative_state_transition]
{json.dumps(compact_scenes, ensure_ascii=False, separators=(",", ":"))}

Compact L2 event evidence ({len(compact_events)} records):
Schema: [l2_event_id, source_shot_ids, event_title, event_description, actions, state_changes]
{json.dumps(compact_events, ensure_ascii=False, separators=(",", ":"))}

Return valid JSON only. Do not return Markdown fences or commentary."""
    return system, user


def validate_generated_l4(payload: Any, movie_id: str) -> dict[str, Any]:
    """Validate the minimum contract for a generated movie-level L4 story."""
    if not isinstance(payload, dict):
        raise ValueError("LLM L4 response must be a JSON object")
    result = dict(payload)
    if str(result.get("movie_id") or "") != movie_id:
        raise ValueError(f"LLM L4 movie_id must be {movie_id}")
    synopsis = result.get("movie_synopsis")
    if isinstance(synopsis, dict):
        if not any(str(value or "").strip() for value in synopsis.values()):
            raise ValueError("LLM L4 movie_synopsis must not be empty")
    elif not str(synopsis or "").strip():
        raise ValueError("LLM L4 movie_synopsis must not be empty")
    if not str(result.get("global_storyline") or "").strip():
        raise ValueError("LLM L4 global_storyline must not be empty")
    evidence = result.get("evidence_trace")
    if evidence is not None:
        if not isinstance(evidence, dict):
            raise ValueError("LLM L4 evidence_trace must be an object")
        for key in ("source_l3_ids", "source_l2_ids", "source_l1_ids"):
            if key in evidence and not isinstance(evidence[key], list):
                raise ValueError(f"LLM L4 evidence_trace.{key} must be a list")
    result["movie_id"] = movie_id
    return result


def _result(payload: Any) -> Any:
    return payload.get("result", payload) if isinstance(payload, dict) else payload


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    data = _result(payload)
    rows = data.get(key, []) if isinstance(data, dict) else data
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of objects for {key}")
    return rows


def _collect_l2_ids(l2_root: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    source_shots: set[str] = set()
    for path in sorted(l2_root.rglob("*.json")):
        for event in _rows(_read_json(path), "l2_events"):
            event_id = str(event.get("l2_event_id") or "").strip()
            if event_id:
                if event_id in ids:
                    raise ValueError(f"Duplicate L2 event ID: {event_id}")
                ids.add(event_id)
                source_shots.update(str(shot_id) for shot_id in event.get("source_shot_ids", []) if str(shot_id).strip())
    return ids, source_shots


def _collect_l1_ids(l1_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(l1_root.rglob("*.json")):
        payload = _read_json(path)
        data = _result(payload)
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict):
                shot_id = str(row.get("l1_shot_id") or "").strip()
                if shot_id:
                    ids.add(shot_id)
    return ids


def _collect_l2_rows(l2_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(l2_root.rglob("*.json")):
        rows.extend(_rows(_read_json(path), "l2_events"))
    return rows


def _validate_source(
    source: Path, allow_unresolved: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str], set[str], dict[str, list[str]]]:
    required = [
        source / "l4_story/movie_bible.json",
        source / "l4_story/story_arcs.json",
        source / "l3_scenes/final.json",
        source / "l2_events/units",
        source / "l1_shots/units",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source paths: " + ", ".join(missing))

    bible_payload = _read_json(required[0])
    bible_result = _result(bible_payload)
    if not isinstance(bible_result, dict) or not str(bible_result.get("movie_synopsis") or "").strip():
        raise ValueError("Movie Bible result must contain a non-empty movie_synopsis")
    arcs_payload = _read_json(required[1])
    arcs_result = _result(arcs_payload)
    if not isinstance(arcs_result, dict):
        raise ValueError("Story arcs result must be an object")
    arcs = _rows(arcs_result.get("story_arcs"), "story_arcs")
    scenes_payload = _read_json(required[2])
    scenes = _rows(scenes_payload, "l3_scenes")
    scene_ids = {str(scene.get("l3_scene_id") or "").strip() for scene in scenes}
    scene_ids.discard("")
    l2_ids, l2_source_shots = _collect_l2_ids(source / "l2_events/units")
    l1_ids = _collect_l1_ids(source / "l1_shots/units")

    unresolved = {"unresolved_l2_event_ids": [], "unresolved_l1_shot_ids": [], "unresolved_scene_occurrence_ids": []}
    unresolved["unresolved_l1_shot_ids"].extend(l2_source_shots - l1_ids)
    for index, arc in enumerate(arcs, start=1):
        if not str(arc.get("arc_title") or "").strip():
            raise ValueError(f"Story arc {index} has no arc_title")
        for scene_id in arc.get("l3_scene_ids", []) or []:
            if str(scene_id) not in scene_ids:
                unresolved["unresolved_scene_occurrence_ids"].append(str(scene_id))
        for event_id in arc.get("l2_event_ids", []) or []:
            if str(event_id) not in l2_ids:
                unresolved["unresolved_l2_event_ids"].append(str(event_id))
        for shot_id in arc.get("l1_shot_ids", []) or []:
            if str(shot_id) not in l1_ids:
                unresolved["unresolved_l1_shot_ids"].append(str(shot_id))
        for event_index, event in enumerate(arc.get("essential_events", []) or [], start=1):
            if not isinstance(event, dict):
                raise ValueError(f"Story arc {index} essential event {event_index} is not an object")
            for event_id in event.get("source_l2_ids", []) or []:
                if str(event_id) not in l2_ids:
                    unresolved["unresolved_l2_event_ids"].append(str(event_id))
            for scene_id in event.get("source_l3_ids", []) or []:
                if str(scene_id) not in scene_ids:
                    unresolved["unresolved_scene_occurrence_ids"].append(str(scene_id))
    for key in unresolved:
        unresolved[key] = sorted(set(unresolved[key]))
    if not allow_unresolved and any(unresolved.values()):
        details = "; ".join(f"{key}={len(values)}" for key, values in unresolved.items() if values)
        raise ValueError(f"Source references do not resolve ({details}); use --allow-unresolved-references to preserve and report them")
    return bible_payload, arcs_result, scenes_payload, l2_ids, l1_ids, unresolved


def restructure_dataset(
    input_root: Path | str,
    output_root: Path | str,
    allow_unresolved_references: bool = False,
    *,
    generate_l4: bool = False,
    llm_scope: str = "OPENAI",
    llm_model: str | None = None,
    llm_retries: int = 3,
    llm_timeout: float | None = None,
    fallback_existing_bible: bool = False,
) -> dict[str, Any]:
    """Build the corrected hierarchy in a new directory, retaining source evidence."""
    source = Path(input_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    if source == target or source in target.parents:
        raise ValueError("Output must not be the input directory or a child of it")
    if target.exists():
        raise FileExistsError(f"Output directory already exists: {target}")

    bible_payload, arcs_result, scenes_payload, l2_ids, l1_ids, unresolved = _validate_source(
        source, allow_unresolved=allow_unresolved_references
    )
    original_arcs = arcs_result["story_arcs"]
    movie_id = str(arcs_result.get("movie_id") or _result(bible_payload).get("movie_id") or "unknown_movie")
    source_bible = _result(bible_payload)
    l4_generation: dict[str, Any] = {
        "method": "source_copy",
        "status": "not_requested",
        "fallback_used": False,
    }
    l4_request: dict[str, Any] | None = None
    l4_response: dict[str, Any] | None = None

    if generate_l4:
        scenes = _rows(scenes_payload, "l3_scenes")
        events = _collect_l2_rows(source / "l2_events/units")
        system_prompt, user_prompt = build_l4_generation_prompt(movie_id, source_bible, scenes, events)
        selected_model = llm_model
        tool: LLMJsonTool | None = None
        try:
            tool = LLMJsonTool(scope=llm_scope, model=llm_model, retries=llm_retries)
            selected_model = tool.model
            generated = validate_generated_l4(
                tool.run_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
                    timeout=llm_timeout,
                ),
                movie_id,
            )
            generated["hierarchy_level"] = "L4_global_movie_story"
            generated["generation_method"] = "llm_compaction"
            generated["generation_metadata"] = {
                "scope": llm_scope,
                "model": selected_model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_movie_bible": str(source / "l4_story/movie_bible.json"),
                "source_l3_scenes": str(source / "l3_scenes/final.json"),
                "source_l2_events": str(source / "l2_events/units"),
                "l3_scene_count": len(scenes),
                "l2_event_count": len(events),
            }
            generated_bible_payload = dict(bible_payload)
            generated_bible_payload["result"] = generated
            l4_generation = {
                "method": "llm_compaction",
                "status": "done",
                "fallback_used": False,
                "scope": llm_scope,
                "model": selected_model,
                "l3_scene_count": len(scenes),
                "l2_event_count": len(events),
            }
            l4_response = {"status": "done", "model": selected_model, "result": generated}
        except Exception as exc:
            if not fallback_existing_bible:
                raise RuntimeError(f"L4 generation failed: {type(exc).__name__}: {exc}") from exc
            generated_bible_payload = dict(bible_payload)
            fallback_result = dict(source_bible)
            fallback_result["hierarchy_level"] = "L4_global_movie_story"
            fallback_result["generation_method"] = "source_fallback"
            generated_bible_payload["result"] = fallback_result
            l4_generation = {
                "method": "source_fallback",
                "status": "fallback",
                "fallback_used": True,
                "scope": llm_scope,
                "model": selected_model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            l4_response = {
                "status": "fallback",
                "model": selected_model,
                "error": f"{type(exc).__name__}: {exc}",
                "result": fallback_result,
            }

        l4_request = {
            "status": l4_generation["status"],
            "scope": llm_scope,
            "model": selected_model,
            "source_paths": {
                "movie_bible": str(source / "l4_story/movie_bible.json"),
                "l3_scenes": str(source / "l3_scenes/final.json"),
                "l2_events": str(source / "l2_events/units"),
            },
            "input_counts": {"l3_scenes": len(scenes), "l2_events": len(events)},
            "prompt_sha256": _sha256_text(system_prompt + "\n" + user_prompt),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    else:
        generated_bible_payload = dict(bible_payload)

    l3_arcs: list[dict[str, Any]] = []
    for index, source_arc in enumerate(original_arcs, start=1):
        arc = dict(source_arc)
        legacy_arc_id = str(arc.pop("l4_story_arc_id", "") or f"{movie_id}_arc_{index:02d}")
        scene_ids = arc.pop("l3_scene_ids", []) or []
        arc["l3_story_arc_id"] = f"{movie_id}_L3ARC_{index:03d}"
        arc["source_l4_story_arc_id"] = legacy_arc_id
        arc["source_scene_occurrence_ids"] = scene_ids
        for event in arc.get("essential_events", []) or []:
            if isinstance(event, dict) and "source_l3_ids" in event:
                event["source_scene_occurrence_ids"] = event.pop("source_l3_ids")
        l3_arcs.append(arc)

    track_arc_mapping = [
        {
            "source_arc_id": arc["source_l4_story_arc_id"],
            "l3_story_arc_id": arc["l3_story_arc_id"],
        }
        for arc in l3_arcs
    ]

    target.mkdir(parents=True)
    shutil.copytree(source / "l1_shots", target / "l1_shots")
    shutil.copytree(source / "l2_events", target / "l2_events")
    for filename in ("track_a.json", "track_b.json"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)

    global_bible = dict(generated_bible_payload)
    global_result = dict(_result(generated_bible_payload))
    global_result["hierarchy_level"] = "L4_global_movie_story"
    global_result["movie_id"] = movie_id
    global_bible["result"] = global_result
    _write_json(target / "l4_global/movie_bible.json", global_bible)
    if l4_request is not None:
        _write_json(target / "l4_global/l4_generation_request.json", l4_request)
    if l4_response is not None:
        _write_json(target / "l4_global/l4_generation_response.json", l4_response)

    source_arc_envelope = _read_json(source / "l4_story/story_arcs.json")
    arcs_envelope = dict(source_arc_envelope)
    arcs_envelope["stage"] = "L3_local_story_arcs"
    arcs_envelope["result"] = {"movie_id": movie_id, "story_arcs": l3_arcs}
    _write_json(target / "l3_arcs/story_arcs.json", arcs_envelope)

    _write_json(target / "source_context/l3_scene_occurrences.json", scenes_payload)
    report = {
        "movie_id": movie_id,
        "source_root": str(source),
        "output_root": str(target),
        "hierarchy": {"L4": "one global movie story", "L3": "local story arcs", "L2": "event units", "L1": "shot evidence"},
        "counts": {"l4_global_stories": 1, "l3_local_story_arcs": len(l3_arcs), "l2_events_available": len(l2_ids), "l1_shots_available": len(l1_ids)},
        "track_sample_arc_mapping": track_arc_mapping,
        "validation": {
            "status": "complete" if not any(unresolved.values()) else "incomplete",
            **unresolved,
        },
        "copied_unchanged": ["l1_shots/", "l2_events/", "track_a.json", "track_b.json"],
        "source_scene_occurrences": "source_context/l3_scene_occurrences.json",
        "l4_generation": l4_generation,
    }
    _write_json(target / "hierarchy_restructure_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-unresolved-references", action="store_true")
    parser.add_argument("--generate-l4", action="store_true")
    parser.add_argument("--llm-scope", default="OPENAI")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--fallback-existing-bible", action="store_true")
    args = parser.parse_args()
    report = restructure_dataset(
        args.input_root,
        args.output_root,
        args.allow_unresolved_references,
        generate_l4=args.generate_l4,
        llm_scope=args.llm_scope,
        llm_model=args.llm_model,
        llm_retries=args.llm_retries,
        llm_timeout=args.llm_timeout,
        fallback_existing_bible=args.fallback_existing_bible,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
