from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from agents.client import make_client, pick_model


LOGGER = logging.getLogger(__name__)
TC_RE = re.compile(
    r"(?P<start>\d{2}\.\d{2}\.\d{2}\.\d{3})-"
    r"(?P<end>\d{2}\.\d{2}\.\d{2}\.\d{3})"
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_text(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False)[:limit]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _pick_fields(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in fields if key in value}


def compact_l1_shot_for_l2(shot: dict[str, Any]) -> dict[str, Any]:
    """Keep observed shot evidence and IDs needed for temporal segmentation."""
    return _pick_fields(
        shot,
        (
            "l1_shot_id", "movie_id", "source_file", "time_range", "order_index",
            "characters", "location_mentions", "item_mentions", "actions",
            "interactions", "background_state", "camera", "style_elements",
            "observable_facts",
        ),
    )


def compact_l2_event_for_l3(event: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose L2 diagnostics while preserving scene grouping evidence."""
    return _pick_fields(
        event,
        (
            "l2_event_id", "source_shot_ids", "shot_order", "time_range",
            "event_title", "event_description", "main_characters", "setting",
            "locations", "items", "actions", "interactions", "state_changes",
        ),
    )


def compact_l3_scene_for_l4(scene: dict[str, Any]) -> dict[str, Any]:
    """Keep narrative and provenance fields used by L4 synthesis."""
    return _pick_fields(
        scene,
        (
            "l3_scene_id", "l2_event_ids", "time_range", "scene_setting",
            "active_characters", "scene_purpose", "scene_narrative",
            "event_progression", "narrative_state_transition",
            "location_continuity", "causal_links", "evidence_trace",
        ),
    )


def compact_movie_bible_for_arc(bible: dict[str, Any]) -> dict[str, Any]:
    """Provide arc extraction with a short global context, not a repeated full bible."""
    synopsis = bible.get("movie_synopsis", {})
    if isinstance(synopsis, dict):
        synopsis = _pick_fields(
            synopsis,
            ("short_synopsis", "main_conflicts", "major_state_transitions"),
        )
    characters = []
    for item in bible.get("global_character_profiles", []):
        if isinstance(item, dict):
            characters.append(_pick_fields(item, ("name", "character", "role", "profile", "traits", "evidence_scene_ids")))
        else:
            characters.append(item)
    return {
        "movie_id": bible.get("movie_id", ""),
        "movie_synopsis": synopsis,
        "global_character_profiles": characters,
        "recurring_locations": bible.get("recurring_locations", []),
        "recurring_objects": bible.get("recurring_objects", []),
        "global_constraints_supported_by_evidence": bible.get("global_constraints_supported_by_evidence", []),
        "unresolved_questions": bible.get("unresolved_questions", []),
    }


def compact_bible_fragment_for_consolidation(fragment: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant evidence traces before hierarchical bible consolidation."""
    compact = dict(fragment)
    evidence = compact.get("evidence_trace")
    if isinstance(evidence, dict):
        compact["evidence_trace"] = _pick_fields(
            evidence, ("source_l3_ids", "source_l2_ids", "source_l1_ids")
        )
    compact["movie_synopsis"] = _bounded_json_value(compact.get("movie_synopsis"), 9000)
    compact["global_character_profiles"] = _bounded_json_value(compact.get("global_character_profiles"), 6000)
    compact["recurring_locations"] = _bounded_json_value(compact.get("recurring_locations"), 2500)
    compact["recurring_objects"] = _bounded_json_value(compact.get("recurring_objects"), 2500)
    compact["global_constraints_supported_by_evidence"] = _bounded_json_value(
        compact.get("global_constraints_supported_by_evidence"), 2500
    )
    compact["unresolved_questions"] = _bounded_json_value(compact.get("unresolved_questions"), 2000)
    return compact


def _bounded_json_value(value: Any, max_chars: int) -> Any:
    """Bound verbose generated text while retaining IDs and object structure."""
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        result = []
        used = 0
        for item in value:
            encoded = json.dumps(item, ensure_ascii=False)
            if used + len(encoded) > max_chars:
                break
            result.append(item)
            used += len(encoded)
        return result
    if isinstance(value, dict):
        result = {}
        used = 0
        for key, item in value.items():
            encoded = json.dumps(item, ensure_ascii=False)
            if used + len(key) + len(encoded) > max_chars:
                if key.endswith("_ids"):
                    result[key] = item
                continue
            result[key] = item
            used += len(key) + len(encoded)
        return result
    return value


def compact_arc_fragment_for_consolidation(fragment: dict[str, Any]) -> dict[str, Any]:
    """Strip evaluation diagnostics but keep benchmark targets and source lineage."""
    arcs = []
    for arc in fragment.get("story_arcs", []):
        if not isinstance(arc, dict):
            continue
        compact_arc = dict(arc)
        compact_arc.pop("quality", None)
        evidence = compact_arc.get("evidence_trace")
        if isinstance(evidence, dict):
            compact_arc["evidence_trace"] = _pick_fields(
                evidence, ("source_l3_ids", "source_l2_ids", "source_l1_ids")
            )
        arcs.append(compact_arc)
    return {"movie_id": fragment.get("movie_id", ""), "story_arcs": arcs}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tc_to_ms(tc: str) -> int:
    hh, mm, ss, ms = (int(part) for part in tc.split("."))
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def _parse_source_file(path: Path, movie_id: str, order_index: int) -> dict[str, Any]:
    raw = _read_json(path)
    match = TC_RE.search(path.name)
    if match:
        start_tc = match.group("start")
        end_tc = match.group("end")
        start_ms = _tc_to_ms(start_tc)
        end_ms = _tc_to_ms(end_tc)
    else:
        start_tc = end_tc = ""
        start_ms = end_ms = 0
    shot_id = path.stem
    return {
        "l1_shot_id": shot_id,
        "movie_id": movie_id,
        "source_file": str(path),
        "file_name": path.name,
        "time_range": {
            "start_tc": start_tc,
            "end_tc": end_tc,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_sec": round(max(0, end_ms - start_ms) / 1000, 3),
        },
        "order_index": order_index,
        "raw": raw,
    }


def build_source_manifest(source_root: Path, movie_id: str) -> dict[str, Any]:
    files = sorted(source_root.glob("*.json"))
    shots: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        try:
            shots.append(_parse_source_file(path, movie_id, index))
        except Exception as exc:  # noqa: BLE001
            invalid.append({"path": str(path), "error": str(exc)})
    shots.sort(key=lambda item: (item["time_range"]["start_ms"], item["file_name"]))
    for index, shot in enumerate(shots, start=1):
        shot["order_index"] = index
    return {
        "movie_id": movie_id,
        "source_root": str(source_root),
        "shot_count": len(shots),
        "invalid_files": invalid,
        "shots": shots,
        "source_sha256": _hash_text(_json_text(shots)),
    }


def _l1_system() -> str:
    return (
        "You are a strict film-shot annotation normalizer.\n"
        "You receive exactly one MovieBench shot annotation. Convert it into a "
        "structured L1 Shot Unit.\n"
        "Use only information explicitly present in the input. Do not infer hidden "
        "plot, unstated character intentions, dialogue, causal relationships, or "
        "off-screen events.\n"
        "Preserve all observable visual facts, separate observed from inferred and "
        "uncertain information, preserve the source identity and time range, and "
        "return valid JSON only."
    )


def build_l1_prompt(shot: dict[str, Any], *, shot_id: str, movie_id: str) -> tuple[str, str]:
    raw = shot.get("raw", shot)
    user = f"""Task: Normalize the following MovieBench annotation into one L1 Shot Unit.

Rules:
- This is one shot only. Do not merge it with another shot.
- Do not add facts not present in the input.
- Use observed for facts directly stated by Characters, Plot, Background Description,
  Style Elements, or Camera Motion.
- Use inferred only for a cautious interpretation directly supported by the input.
- Use uncertain when the input is ambiguous.
- Extract location and item mentions conservatively.
- If a field is unsupported, return an empty list or empty string.
- Include source field names in source_evidence.

Identifiers:
movie_id={movie_id}
shot_id={shot_id}

Input shot:
{_json_text(raw)}

Return exactly:
{{
  "l1_shot_id": "{shot_id}",
  "movie_id": "{movie_id}",
  "source_file": "{shot.get('source_file', '')}",
  "time_range": { _json_text(shot.get('time_range', {})) },
  "order_index": {int(shot.get('order_index', 0))},
  "characters": [{{"name":"...","visible_action":"...","visible_state":"...","emotion_observation":"...","evidence_source":[],"certainty":"observed|inferred|uncertain"}}],
  "location_mentions": [{{"name":"...","description":"...","evidence_source":[],"certainty":"observed|inferred|uncertain"}}],
  "item_mentions": [{{"name":"...","state_or_action":"...","evidence_source":[],"certainty":"observed|inferred|uncertain"}}],
  "actions": [{{"actor":"...","action":"...","target":"...","certainty":"observed|inferred|uncertain"}}],
  "interactions": [],
  "background_state": [{{"element":"...","state":"...","certainty":"observed|inferred|uncertain"}}],
  "camera": {{"motion":"...","framing_or_view":"...","certainty":"observed|inferred|uncertain"}},
  "style_elements": [],
  "observable_facts": [],
  "interpretive_facts": [],
  "source_evidence": {{"Characters":...,"Style Elements":...,"Plot":...,"Background Description":...,"Camera Motion":...}},
  "annotation_quality": {{"missing_fields":[],"ambiguities":[],"confidence":0.0}}
}}"""
    return _l1_system(), user


def _l2_system() -> str:
    return (
        "You are a strict temporal event segmentation and reconstruction assistant.\n"
        "You receive ordered L1 shot units from one movie. Construct one or more "
        "L2 Micro-Events. Physical continuity does not guarantee semantic continuity.\n"
        "Split on location changes, main-character or target changes, hard visual "
        "transitions, completed actions followed by independent actions, substantial "
        "objective changes, or independent symbolic/object inserts.\n"
        "Use only supplied evidence. Every input shot must be assigned exactly once. "
        "Return valid JSON only."
    )


def build_l2_prompt(sequence: dict[str, Any]) -> tuple[str, str]:
    compact_sequence = dict(sequence)
    compact_sequence["shots"] = [
        compact_l1_shot_for_l2(shot)
        for shot in sequence.get("shots", [])
        if isinstance(shot, dict)
    ]
    user = f"""Task: Convert the ordered L1 shot units below into semantically coherent L2 Micro-Events.

Rules:
- Preserve chronological order.
- Every input shot must appear exactly once in source_shot_ids.
- A one-shot event is allowed when it is semantically self-contained.
- Do not merge unrelated actions merely because they are temporally adjacent.
- Record observable state changes only; mark interpretations as inferred or uncertain.
- Do not convert inferred motivation into confirmed fact.

Input:
{_json_text(compact_sequence)}

Return exactly:
{{
  "movie_id": "...",
  "source_sequence_id": "...",
  "segmentation_decision": {{"input_shot_count":0,"output_event_count":0,"was_split":false,"split_reasons":[],"confidence":0.0}},
  "l2_events": [{{
    "l2_event_id":"...",
    "source_shot_ids":[],
    "shot_order":[],
    "time_range":{{"start_tc":"...","end_tc":"...","start_ms":0,"end_ms":0,"duration_sec":0.0}},
    "event_title":"...",
    "event_description":"...",
    "main_characters":[],
    "setting":"...",
    "locations":[],
    "items":[],
    "actions":[],
    "interactions":[],
    "temporal_progression":[],
    "causal_preconditions":[],
    "causal_effects":[],
    "state_changes":[],
    "camera_and_visual_style":[],
    "shot_to_event_mapping":[],
    "evidence_trace":{{"source_l1_ids":[],"source_fields":[]}},
    "quality":{{"boundary_confidence":0.0,"coverage_check":"complete","unsupported_claims":[]}}
  }}],
  "coverage_check":{{"all_input_shots_assigned_once":true,"missing_shot_ids":[],"duplicated_shot_ids":[]}}
}}"""
    return _l2_system(), user


def _l3_system() -> str:
    return (
        "You are a strict scene-level narrative aggregation assistant.\n"
        "Group ordered L2 Micro-Events into L3 Scene Segments using spatial continuity, "
        "temporal proximity, active characters, and local dramatic purpose.\n"
        "Use only supplied L2 evidence. Do not fabricate dialogue, motives, objects, "
        "relationships, or outcomes. Every input L2 event must be assigned exactly once. "
        "Return valid JSON only."
    )


def build_l3_prompt(movie_id: str, l2_events: list[dict[str, Any]]) -> tuple[str, str]:
    compact_events = [compact_l2_event_for_l3(event) for event in l2_events]
    user = f"""Task: Aggregate these ordered L2 events into L3 Scene Segments for movie {movie_id}.

Rules:
- Preserve chronological order.
- Each L2 event belongs to exactly one L3 scene.
- Do not force all events into one scene.
- Same or connected subspaces may be grouped only when the local dramatic thread continues.
- Record observable state transitions separately from inferred interpretation.
- Include source L2 and L1 IDs for every major claim.

Input L2 events:
{_json_text(compact_events)}

Return exactly:
{{
  "movie_id":"{movie_id}",
  "l3_scenes":[{{
    "l3_scene_id":"...",
    "l2_event_ids":[],
    "time_range":{{"start_tc":"...","end_tc":"...","start_ms":0,"end_ms":0,"duration_sec":0.0}},
    "scene_setting":"...",
    "active_characters":[],
    "scene_purpose":"...",
    "scene_narrative":"...",
    "event_progression":[],
    "narrative_state_transition":{{"entry_state":{{"observable_facts":[],"inferred_context":[],"uncertain_facts":[]}},"key_actions_and_conflicts":[],"relationship_or_goal_changes":[],"key_objects_and_states":[],"exit_state":{{"observable_facts":[],"inferred_context":[],"uncertain_facts":[]}}}},
    "location_continuity":{{"same_location":true,"connected_subspaces":[],"location_changes":[]}},
    "causal_links":[],
    "evidence_trace":{{"source_l2_ids":[],"source_l1_ids":[]}},
    "quality":{{"boundary_confidence":0.0,"unsupported_claims":[],"ambiguities":[]}}
  }}],
  "coverage_check":{{"all_input_l2_events_assigned_once":true,"missing_l2_event_ids":[],"duplicated_l2_event_ids":[]}}
}}"""
    return _l3_system(), user


def build_l4_movie_bible_prompt(movie_id: str, l3_scenes: list[dict[str, Any]]) -> tuple[str, str]:
    compact_scenes = [compact_l3_scene_for_l4(scene) for scene in l3_scenes]
    system = (
        "You are a strict full-film narrative synthesis assistant.\n"
        "Build a movie-level bible from ordered L3 Scene Segments without inventing "
        "unsupported facts. Separate observed, inferred, and uncertain information. "
        "Preserve temporal gaps and provide evidence scene IDs. Return valid JSON only."
    )
    user = f"""Task: Build a movie-level narrative bible for {movie_id}.

Requirements:
- Cover the supplied scene range and explicitly list gaps.
- Identify recurring characters, locations, objects, conflicts, and state transitions.
- Do not treat inferred motives as confirmed facts.
- Every important claim must include evidence scene IDs.

Input L3 scenes:
{_json_text(compact_scenes)}

Return exactly:
{{
  "movie_id":"{movie_id}",
  "coverage":{{"first_scene_id":"...","last_scene_id":"...","scene_count":0,"covered_time_range_ms":[0,0],"uncovered_gaps":[]}},
  "movie_synopsis":{{"short_synopsis":"...","chronological_synopsis":[],"main_conflicts":[],"major_state_transitions":[]}},
  "global_character_profiles":[],
  "recurring_locations":[],
  "recurring_objects":[],
  "global_constraints_supported_by_evidence":[],
  "unresolved_questions":[],
  "evidence_trace":{{"source_l3_ids":[],"source_l2_ids":[],"source_l1_ids":[]}}
}}"""
    return system, user


def build_l4_story_arcs_prompt(movie_id: str, movie_bible: dict[str, Any], l3_scenes: list[dict[str, Any]]) -> tuple[str, str]:
    compact_bible = compact_movie_bible_for_arc(movie_bible)
    compact_scenes = [compact_l3_scene_for_l4(scene) for scene in l3_scenes]
    system = (
        "You are a strict multi-scene story-arc extraction assistant.\n"
        "Extract chronological L4 story arcs connected by character continuity, causal "
        "progression, recurring conflict, or state transfer. Do not invent missing "
        "transitions. Mark inferred connections and return valid JSON only."
    )
    user = f"""Task: Extract candidate multi-scene L4 Story Arcs for {movie_id}.

Requirements:
- Each arc contains at least two L3 scenes when possible.
- Prefer 2-5 L3 scenes and preserve chronological order.
- Produce a complete source story S and a compressed storyline L.
- The short storyline must omit intermediate event details and must not leak withheld events.
- Trace every essential event and fine-grained unit to source L1/L2/L3 IDs.

Movie bible:
{_json_text(compact_bible)}

L3 scenes:
{_json_text(compact_scenes)}

Return exactly:
{{
  "movie_id":"{movie_id}",
  "story_arcs":[{{
    "l4_story_arc_id":"...",
    "l3_scene_ids":[],
    "l2_event_ids":[],
    "l1_shot_ids":[],
    "time_range":{{"start_tc":"...","end_tc":"...","start_ms":0,"end_ms":0,"duration_sec":0.0}},
    "arc_title":"...",
    "full_story_S":"...",
    "short_storyline_L":"...",
    "arc_structure":{{"initial_state":"...","trigger_or_inciting_change":"...","development":[],"turning_points":[],"current_or_final_state":"...","unresolved_elements":[]}},
    "characters":[],"locations":[],"key_objects":[],
    "essential_events":[],
    "fine_grained_units":[],
    "cross_scene_state_transitions":[],
    "evidence_trace":{{"source_l3_ids":[],"source_l2_ids":[],"source_l1_ids":[]}},
    "quality":{{"arc_coherence":0.0,"evidence_coverage":0.0,"unsupported_claims":[],"leakage_check":{{"withheld_events_in_short_storyline":false,"intermediate_details_leaked":[]}}}}
  }}]
}}"""
    return system, user


def build_l4_movie_bible_consolidation_prompt(
    movie_id: str, bible_fragments: list[dict[str, Any]]
) -> tuple[str, str]:
    compact_fragments = [compact_bible_fragment_for_consolidation(item) for item in bible_fragments]
    system = (
        "You are a strict narrative-bible consolidation assistant. Combine L4 bible "
        "fragments into one movie bible without inventing facts. Preserve observed, "
        "inferred, and uncertain distinctions, retain all source scene/event/shot IDs, "
        "and return valid JSON only."
    )
    user = f"""Task: Consolidate the L4 movie-bible fragments for {movie_id}.

Rules:
- Fragments are ordered by film time and may have boundary overlap.
- Deduplicate repeated facts, but never discard source evidence IDs.
- Do not infer transitions that are not supported by a fragment.
- Preserve temporal gaps and unresolved questions.

Input fragments:
{_json_text(compact_fragments)}

Return exactly:
{{
  "movie_id":"{movie_id}",
  "coverage":{{"first_scene_id":"...","last_scene_id":"...","scene_count":0,"covered_time_range_ms":[0,0],"uncovered_gaps":[]}},
  "movie_synopsis":{{"short_synopsis":"...","chronological_synopsis":[],"main_conflicts":[],"major_state_transitions":[]}},
  "global_character_profiles":[],
  "recurring_locations":[],
  "recurring_objects":[],
  "global_constraints_supported_by_evidence":[],
  "unresolved_questions":[],
  "evidence_trace":{{"source_l3_ids":[],"source_l2_ids":[],"source_l1_ids":[]}}
}}"""
    return system, user


def build_l4_story_arcs_consolidation_prompt(
    movie_id: str, arc_fragments: list[dict[str, Any]]
) -> tuple[str, str]:
    compact_fragments = [compact_arc_fragment_for_consolidation(item) for item in arc_fragments]
    system = (
        "You are a strict story-arc consolidation assistant. Merge candidate arcs "
        "from ordered film windows into chronological multi-scene L4 arcs. Do not "
        "invent missing transitions. Preserve source L1/L2/L3 IDs and distinguish "
        "observed, inferred, and uncertain claims. Return valid JSON only."
    )
    user = f"""Task: Consolidate candidate L4 story arcs for {movie_id}.

Rules:
- Merge overlapping candidates when they describe the same continuing thread.
- Keep separate arcs when continuity is unsupported.
- Each output arc should contain at least two L3 scenes when the evidence permits.
- Preserve full_story_S and short_storyline_L distinction.
- The short storyline must not contain details whose source event IDs are withheld.
- Retain source evidence IDs for every essential event and fine-grained unit.

Input arc fragments:
{_json_text(compact_fragments)}

Return exactly:
{{
  "movie_id":"{movie_id}",
  "story_arcs":[{{
    "l4_story_arc_id":"{movie_id}_arc_01",
    "l3_scene_ids":[],"l2_event_ids":[],"l1_shot_ids":[],
    "time_range":{{"start_tc":"...","end_tc":"...","start_ms":0,"end_ms":0,"duration_sec":0.0}},
    "arc_title":"...","full_story_S":"...","short_storyline_L":"...",
    "arc_structure":{{"initial_state":"","trigger_or_inciting_change":"","development":[],"turning_points":[],"current_or_final_state":"","unresolved_elements":[]}},
    "characters":[],"locations":[],"key_objects":[],"essential_events":[],"fine_grained_units":[],"cross_scene_state_transitions":[],
    "evidence_trace":{{"source_l3_ids":[],"source_l2_ids":[],"source_l1_ids":[]}},
    "quality":{{"arc_coherence":0.0,"evidence_coverage":0.0,"unsupported_claims":[],"leakage_check":{{"withheld_events_in_short_storyline":false,"intermediate_details_leaked":[]}}}}
  }}]
}}"""
    return system, user


def validate_l1_collection(source_ids: list[str], results: list[dict[str, Any]]) -> dict[str, Any]:
    result_ids = [str(item.get("l1_shot_id", "")) for item in results]
    counts = Counter(result_ids)
    expected = set(source_ids)
    actual = set(result_ids)
    return {
        "complete": actual == expected and len(result_ids) == len(expected) and all(v == 1 for v in counts.values()),
        "missing_ids": sorted(expected - actual),
        "duplicated_ids": sorted(k for k, v in counts.items() if v > 1),
        "unexpected_ids": sorted(actual - expected),
    }


def validate_l2_collection(source_ids: list[str], events: list[dict[str, Any]]) -> dict[str, Any]:
    assigned = [str(shot_id) for event in events for shot_id in event.get("source_shot_ids", [])]
    counts = Counter(assigned)
    expected = set(source_ids)
    actual = set(assigned)
    return {
        "complete": actual == expected and len(assigned) == len(expected) and all(v == 1 for v in counts.values()),
        "missing_ids": sorted(expected - actual),
        "duplicated_ids": sorted(k for k, v in counts.items() if v > 1),
        "unexpected_ids": sorted(actual - expected),
    }


def normalize_l2_event_ids(sequence_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        item = dict(event)
        if item.get("l2_event_id") and item.get("l2_event_id") != f"{sequence_id}_l2_{index:02d}":
            item["model_l2_event_id"] = item["l2_event_id"]
        item["l2_event_id"] = f"{sequence_id}_l2_{index:02d}"
        normalized.append(item)
    return normalized


def canonicalize_l3_scenes(
    scenes: list[dict[str, Any]], l2_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Map model-copied legacy L2 IDs back to canonical IDs using shot evidence."""
    canonical_by_id = {str(event.get("l2_event_id")): event for event in l2_events}
    by_model_id: dict[str, list[str]] = {}
    by_shot_id: dict[str, list[str]] = {}
    for event in l2_events:
        canonical_id = str(event.get("l2_event_id", ""))
        model_id = str(event.get("model_l2_event_id", ""))
        if model_id:
            by_model_id.setdefault(model_id, []).append(canonical_id)
        for shot_id in event.get("source_shot_ids", []):
            by_shot_id.setdefault(str(shot_id), []).append(canonical_id)

    normalized_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        item = dict(scene)
        evidence = item.get("evidence_trace") or {}
        evidence_shots = {str(value) for value in evidence.get("source_l1_ids", [])}
        mapped: list[str] = []
        for event_id in item.get("l2_event_ids", []):
            event_id = str(event_id)
            if event_id in canonical_by_id:
                candidates = [event_id]
            else:
                candidates = list(by_model_id.get(event_id, []))
            if candidates and evidence_shots:
                evidence_candidates = {
                    candidate
                    for shot in evidence_shots
                    for candidate in by_shot_id.get(shot, [])
                }
                narrowed = [candidate for candidate in candidates if candidate in evidence_candidates]
                if narrowed:
                    candidates = narrowed
            if not candidates and evidence_shots:
                candidates = sorted({candidate for shot in evidence_shots for candidate in by_shot_id.get(shot, [])})
            if candidates:
                mapped.extend(candidates)
        item["l2_event_ids"] = list(dict.fromkeys(mapped))
        trace = dict(evidence)
        trace["source_l2_ids"] = item["l2_event_ids"]
        item["evidence_trace"] = trace
        normalized_scenes.append(item)
    return normalized_scenes


def validate_l3_collection(event_ids: list[str], scenes: list[dict[str, Any]]) -> dict[str, Any]:
    assigned = [str(event_id) for scene in scenes for event_id in scene.get("l2_event_ids", [])]
    counts = Counter(assigned)
    expected = set(event_ids)
    actual = set(assigned)
    return {
        "complete": actual == expected and len(assigned) == len(expected) and all(v == 1 for v in counts.values()),
        "missing_ids": sorted(expected - actual),
        "duplicated_ids": sorted(k for k, v in counts.items() if v > 1),
        "unexpected_ids": sorted(actual - expected),
    }


def validate_track_b_no_leakage(track_b: dict[str, Any]) -> dict[str, Any]:
    withheld = set(track_b.get("withheld_event_ids", []))
    visible = set(track_b.get("short_storyline_source_event_ids", []))
    leaked_ids = sorted(withheld & visible)
    return {"leakage": bool(leaked_ids), "leaked_event_ids": leaked_ids}


class LLMJsonRunner:
    def __init__(self, *, model: str | None = None, retries: int = 5, sleep_sec: float = 1.0) -> None:
        self.client = make_client(scope="MOVIEBENCH_CINEWORLDSIM")
        self.model = model or pick_model("gpt-5.5", scope="MOVIEBENCH_CINEWORLDSIM")
        self.retries = max(1, int(retries))
        self.sleep_sec = max(0.0, sleep_sec)

    def call(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_hash = _hash_text(system + "\n" + user)
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content if response.choices else "{}"
                payload = json.loads(str(content))
                return payload, {
                    "status": "done",
                    "model": self.model,
                    "attempts": attempt,
                    "prompt_sha256": prompt_hash,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                LOGGER.warning("LLM call failed attempt=%s/%s error=%s", attempt, self.retries, last_error)
                if attempt < self.retries:
                    time.sleep(self.sleep_sec * (2 ** (attempt - 1)))
        return {}, {
            "status": "failed",
            "model": self.model,
            "attempts": self.retries,
            "prompt_sha256": prompt_hash,
            "error": last_error,
        }


def _unit_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    return payload.get("status") == "done" and isinstance(payload.get("result"), dict)


def run_unit(
    *,
    output_path: Path,
    prompt_builder: Callable[[], tuple[str, str]],
    runner: LLMJsonRunner,
    unit_id: str,
    stage: str,
    force: bool = False,
) -> dict[str, Any]:
    system, user = prompt_builder()
    prompt_hash = _hash_text(system + "\n" + user)
    if not force and _unit_done(output_path):
        cached = _read_json(output_path)
        if cached.get("meta", {}).get("prompt_sha256") == prompt_hash:
            return cached
    result, meta = runner.call(system, user)
    payload = {
        "unit_id": unit_id,
        "stage": stage,
        "status": meta["status"],
        "result": result,
        "meta": meta,
    }
    atomic_write_json(output_path, payload)
    return payload


def package_track_a_and_b(
    *,
    movie_id: str,
    story_arcs: list[dict[str, Any]],
    l1_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    track_a: list[dict[str, Any]] = []
    track_b: list[dict[str, Any]] = []
    for arc in story_arcs:
        anchors = []
        for shot_id in arc.get("l1_shot_ids", []):
            shot = l1_by_id.get(shot_id, {})
            anchors.append(
                {
                    "shot_id": shot_id,
                    "surface_action": shot.get("observable_facts", []),
                    "camera": shot.get("camera", {}),
                    "source_l1_id": shot_id,
                }
            )
        track_a.append(
            {
                "sample_id": arc.get("l4_story_arc_id", ""),
                "movie_id": movie_id,
                "full_story_S": arc.get("full_story_S", ""),
                "storyboard_anchors_A": anchors,
                "essential_events_E_source": arc.get("essential_events", []),
                "fine_grained_units_FDC": arc.get("fine_grained_units", []),
                "source_trace": arc.get("evidence_trace", {}),
            }
        )
        visible_ids = list(arc.get("l2_event_ids", []))[:1]
        withheld_ids = list(arc.get("l2_event_ids", []))[1:]
        item = {
            "sample_id": arc.get("l4_story_arc_id", ""),
            "movie_id": movie_id,
            "short_storyline_L": arc.get("short_storyline_L", ""),
            "known_characters": arc.get("characters", []),
            "known_facts": arc.get("arc_structure", {}).get("initial_state", ""),
            "hard_constraints": {
                "character_constraints": [],
                "premise_and_setting_constraints": arc.get("locations", []),
                "core_conflict_constraints": arc.get("arc_structure", {}).get("trigger_or_inciting_change", ""),
                "outcome_boundary": arc.get("arc_structure", {}).get("current_or_final_state", ""),
            },
            "withheld_event_ids": withheld_ids,
            "short_storyline_source_event_ids": visible_ids,
            "reference_story": arc.get("full_story_S", ""),
            "allowed_expansion_space": [
                "character-consistent concrete actions",
                "causally coherent intermediate interactions",
                "environment or object details supported by the setting",
            ],
        }
        item["leakage_check"] = validate_track_b_no_leakage(item)
        track_b.append(item)
    return {
        "schema_version": "1.0",
        "track": "A",
        "movie_id": movie_id,
        "samples": track_a,
    }, {
        "schema_version": "1.0",
        "track": "B",
        "movie_id": movie_id,
        "samples": track_b,
    }
