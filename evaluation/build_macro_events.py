"""Aggregate MovieBench L2 events into continuity-aware macro-events and audit references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _result(path: Path) -> dict[str, Any]:
    payload = _load(path)
    return payload.get("result", payload)


def _unit_records(directory: Path, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((directory / "units").glob("*.json")):
        value = _result(path)
        rows = value.get(key)
        if isinstance(rows, list):
            records.extend(row for row in rows if isinstance(row, dict))
        elif isinstance(value, dict):
            records.append(value)
    return records


def _time(row: dict[str, Any]) -> tuple[int, int]:
    time_range = row.get("time_range") or {}
    start = time_range.get("start_ms")
    end = time_range.get("end_ms")
    if start is None:
        start = int(float(time_range.get("start_sec", 0)) * 1000)
    if end is None:
        duration = time_range.get("duration_sec", 0)
        end = int(start + float(duration) * 1000)
    return int(start), int(end)


def _occurrence_key(row: dict[str, Any]) -> str:
    start, end = _time(row)
    return f"{row.get('l3_scene_id', 'unknown')}@{start}-{end}"


def _parse_range_from_id(value: str) -> tuple[int, int] | None:
    try:
        tail = value.rsplit("_", 1)[-1]
        start_text, end_text = tail.split("-", 1)
        def tc_to_ms(text: str) -> int:
            parts = text.split(".")
            if len(parts) != 4:
                raise ValueError
            h, m, s, ms = (int(part) for part in parts)
            return ((h * 60 + m) * 60 + s) * 1000 + ms
        return tc_to_ms(start_text), tc_to_ms(end_text)
    except (ValueError, TypeError):
        return None


def _temporal_l1_candidates(value: str, l1_rows: list[dict[str, Any]]) -> list[str]:
    target = _parse_range_from_id(value)
    if target is None:
        return []
    start, end = target
    candidates: list[tuple[int, str]] = []
    for row in l1_rows:
        row_start, row_end = _time(row)
        overlap = max(0, min(end, row_end) - max(start, row_start))
        if overlap > 0:
            candidates.append((overlap, str(row["l1_shot_id"])))
    return [value for _, value in sorted(candidates, key=lambda item: (-item[0], item[1]))]


def _unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _sorted_unique(values: list[Any]) -> list[Any]:
    return sorted(_unique(values), key=lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def audit_hierarchy(root: Path) -> dict[str, Any]:
    root = Path(root)
    l1_rows = _unit_records(root / "l1_shots", "l1_shot_id")
    l2_rows = _unit_records(root / "l2_events", "l2_events")
    l1_ids = {str(row.get("l1_shot_id")) for row in l1_rows if row.get("l1_shot_id")}
    l2_ids = {str(row.get("l2_event_id")) for row in l2_rows if row.get("l2_event_id")}

    occurrence_payload = _result(root / "source_context/l3_scene_occurrences.json")
    occurrences = occurrence_payload.get("l3_scenes", [])
    occurrence_ids = [str(row.get("l3_scene_id")) for row in occurrences if row.get("l3_scene_id")]
    occurrence_id_counts: dict[str, int] = {}
    for occurrence_id in occurrence_ids:
        occurrence_id_counts[occurrence_id] = occurrence_id_counts.get(occurrence_id, 0) + 1

    arcs = _result(root / "l3_arcs/story_arcs.json").get("story_arcs", [])
    l4 = _result(root / "l4_global/movie_bible.json")

    def refs(rows: list[dict[str, Any]], field: str) -> list[str]:
        return [str(value) for row in rows for value in row.get(field, []) if value]

    arc_l2 = refs(arcs, "l2_event_ids")
    arc_l1 = refs(arcs, "l1_shot_ids")
    arc_occ = refs(arcs, "source_scene_occurrence_ids")
    occ_l2 = [str(value) for row in occurrences for value in row.get("evidence_trace", {}).get("source_l2_ids", []) if value]
    occ_l1 = [str(value) for row in occurrences for value in row.get("evidence_trace", {}).get("source_l1_ids", []) if value]
    l4_trace = l4.get("evidence_trace", {}) if isinstance(l4, dict) else {}
    l4_l2 = [str(value) for value in l4_trace.get("source_l2_ids", []) if value]
    l4_l1 = [str(value) for value in l4_trace.get("source_l1_ids", []) if value]

    occurrence_by_l2: dict[str, list[str]] = {}
    for occurrence in occurrences:
        key = _occurrence_key(occurrence)
        for event_id in occurrence.get("l2_event_ids", []):
            occurrence_by_l2.setdefault(str(event_id), []).append(key)
    arc_occurrence_reconciliation: list[dict[str, Any]] = []
    reconciled_arc_occurrence_ids: set[str] = set()
    unresolved_arc_occurrence_ids: set[str] = set()
    for arc in arcs:
        unresolved = [value for value in arc.get("source_scene_occurrence_ids", []) if value not in occurrence_ids]
        if not unresolved:
            continue
        candidates: dict[str, int] = {}
        for event_id in arc.get("l2_event_ids", []):
            for key in occurrence_by_l2.get(str(event_id), []):
                candidates[key] = candidates.get(key, 0) + 1
        arc_occurrence_reconciliation.append({
            "l3_story_arc_id": arc.get("l3_story_arc_id"),
            "unresolved_source_ids": unresolved,
            "candidate_occurrences_by_l2_overlap": [
                {"occurrence_key": key, "shared_l2_count": count}
                for key, count in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
            ],
            "resolution_status": "needs_record_level_review",
        })
        candidate_keys = set(candidates)
        if candidate_keys:
            reconciled_arc_occurrence_ids.update(str(value) for value in unresolved)
        else:
            unresolved_arc_occurrence_ids.update(str(value) for value in unresolved)

    exact_unresolved_l1 = sorted(set(arc_l1) - l1_ids)
    temporal_l1_matches = {value: _temporal_l1_candidates(value, l1_rows) for value in exact_unresolved_l1}
    reconciled_l1 = sorted(value for value, candidates in temporal_l1_matches.items() if candidates)
    report = {
        "status": "complete",
        "counts": {"l4": 1 if l4 else 0, "l3_arcs": len(arcs), "l2": len(l2_ids), "l1": len(l1_ids), "source_occurrences": len(occurrences)},
        "duplicate_ids": {
            "l1": sorted([key for key, count in _counts(l1_rows, "l1_shot_id").items() if count > 1]),
            "l2": sorted([key for key, count in _counts(l2_rows, "l2_event_id").items() if count > 1]),
            "occurrence": sorted([key for key, count in occurrence_id_counts.items() if count > 1]),
        },
        "unresolved_references": {
            "arc_l2": sorted(set(arc_l2) - l2_ids),
            "arc_l1": sorted(set(arc_l1) - l1_ids),
            "arc_occurrence": sorted(set(arc_occ) - set(occurrence_ids)),
            "occurrence_l2": sorted(set(occ_l2) - l2_ids),
            "occurrence_l1": sorted(set(occ_l1) - l1_ids),
            "l4_l2": sorted(set(l4_l2) - l2_ids),
            "l4_l1": sorted(set(l4_l1) - l1_ids),
        },
        "ambiguous_occurrence_ids": sorted([key for key, count in occurrence_id_counts.items() if count > 1]),
        "occurrence_identity": {
            "record_count": len(occurrences),
            "unique_composite_keys": len({_occurrence_key(row) for row in occurrences}),
            "reused_raw_scene_labels": len([key for key, count in occurrence_id_counts.items() if count > 1]),
            "composite_key_format": "l3_scene_id@start_ms-end_ms",
        },
        "arc_occurrence_reconciliation": arc_occurrence_reconciliation,
        "reconciliation": {
            "l1_temporal_candidates": temporal_l1_matches,
            "l1_reconciled_by_time": reconciled_l1,
            "l1_no_temporal_candidate": sorted(set(exact_unresolved_l1) - set(reconciled_l1)),
            "arc_occurrence_candidates": arc_occurrence_reconciliation,
        "arc_occurrence_ids_with_arc_level_candidates": sorted(reconciled_arc_occurrence_ids),
            "arc_occurrence_without_candidate": sorted(unresolved_arc_occurrence_ids),
        },
        "coverage": {
            "all_l2_referenced_by_occurrences": set(occ_l2) == l2_ids,
            "all_l1_referenced_by_occurrences": set(occ_l1) == l1_ids,
            "l2_referenced_by_arcs": len(set(arc_l2)),
            "l1_referenced_by_arcs": len(set(arc_l1)),
        },
    }
    report["unresolved_references"]["l2"] = sorted(set(report["unresolved_references"]["arc_l2"]) | set(report["unresolved_references"]["occurrence_l2"]) | set(report["unresolved_references"]["l4_l2"]))
    report["unresolved_references"]["l1"] = sorted(set(report["unresolved_references"]["arc_l1"]) | set(report["unresolved_references"]["occurrence_l1"]) | set(report["unresolved_references"]["l4_l1"]))
    report["unresolved_references"]["l1_exact_only"] = sorted(set(report["unresolved_references"]["l1"]) - set(reconciled_l1))
    report["duplicate_occurrence_ids"] = report["ambiguous_occurrence_ids"]
    unresolved = report["unresolved_references"]
    report["status"] = "incomplete" if any(report["unresolved_references"][key] for key in ("l2", "l1", "arc_occurrence")) else "complete"
    report["candidate_reconciliation_status"] = "review_required" if reconciled_arc_occurrence_ids or reconciled_l1 else "no_candidates"
    return report


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _can_merge(previous: dict[str, Any], current: dict[str, Any], current_duration: float, max_duration_sec: float, merge_gap_sec: float) -> bool:
    previous_start, previous_end = _time(previous)
    current_start, current_end = _time(current)
    gap = max(0, current_start - previous_end) / 1000.0
    current_event_duration = max(0, current_end - current_start) / 1000.0
    if gap > merge_gap_sec or current_duration + gap + current_event_duration > max_duration_sec:
        return False
    return True


def _make_macro(events: list[dict[str, Any]], occurrence: dict[str, Any], index: int) -> dict[str, Any]:
    start = min(_time(event)[0] for event in events)
    end = max(_time(event)[1] for event in events)
    return {
        "macro_event_id": f"{occurrence.get('l3_scene_id', 'scene')}_ME_{index:03d}",
        "source_l2_ids": [event["l2_event_id"] for event in events],
        "source_l1_ids": _unique([shot for event in events for shot in event.get("source_shot_ids", [])]),
        "source_scene_occurrence_ids": [_occurrence_key(occurrence)],
        "l3_scene_id": occurrence.get("l3_scene_id"),
        "time_range": {"start_ms": start, "end_ms": end, "duration_sec": (end - start) / 1000.0},
        "characters": _sorted_unique([value for event in events for value in event.get("main_characters", [])]),
        "locations": _sorted_unique([value for event in events for value in event.get("locations", [])]),
        "event_titles": [event.get("event_title", "") for event in events],
        "event_descriptions": [event.get("event_description", "") for event in events],
        "actions": _unique([action for event in events for action in event.get("actions", [])]),
        "state_changes": _unique([change for event in events for change in event.get("state_changes", [])]),
        "causal_preconditions": _unique([value for event in events for value in event.get("causal_preconditions", [])]),
        "causal_effects": _unique([value for event in events for value in event.get("causal_effects", [])]),
        "quality": {"review_status": "pending", "source_event_count": len(events)},
    }


def build_macro_events(root: Path, output_dir: Path, max_duration_sec: float = 90.0, merge_gap_sec: float = 30.0) -> dict[str, Any]:
    root, output_dir = Path(root), Path(output_dir)
    events = {str(row["l2_event_id"]): row for row in _unit_records(root / "l2_events", "l2_events") if row.get("l2_event_id")}
    occurrences = _result(root / "source_context/l3_scene_occurrences.json").get("l3_scenes", [])
    macro_events: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for occurrence in sorted(occurrences, key=lambda row: _time(row)[0]):
        ordered = [events[event_id] for event_id in occurrence.get("l2_event_ids", []) if event_id in events]
        ordered.sort(key=lambda event: _time(event)[0])
        current: list[dict[str, Any]] = []
        for event in ordered:
            current_duration = ((_time(current[-1])[1] - _time(current[0])[0]) / 1000.0) if current else 0.0
            if current and not _can_merge(current[-1], event, current_duration, max_duration_sec, merge_gap_sec):
                macro_events.append(_make_macro(current, occurrence, len(macro_events) + 1))
                current = []
            current.append(event)
            assigned.add(event["l2_event_id"])
        if current:
            macro_events.append(_make_macro(current, occurrence, len(macro_events) + 1))

    unassigned = sorted(set(events) - assigned, key=lambda event_id: _time(events[event_id])[0])
    for event_id in unassigned:
        event = events[event_id]
        occurrence = {"l3_scene_id": "unassigned", "time_range": event.get("time_range", {})}
        macro_events.append(_make_macro([event], occurrence, len(macro_events) + 1))
    macro_events.sort(key=lambda row: row["time_range"]["start_ms"])
    arc_rows = _result(root / "l3_arcs/story_arcs.json").get("story_arcs", [])
    event_to_arcs: dict[str, list[str]] = {}
    for arc in arc_rows:
        arc_id = str(arc.get("l3_story_arc_id", ""))
        for event_id in arc.get("l2_event_ids", []):
            event_to_arcs.setdefault(str(event_id), []).append(arc_id)
    for index, row in enumerate(macro_events, start=1):
        row["macro_event_id"] = f"macro_event_{index:04d}"
        row["l3_story_arc_ids"] = _sorted_unique([arc_id for event_id in row["source_l2_ids"] for arc_id in event_to_arcs.get(event_id, [])])

    audit = audit_hierarchy(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "macro_events.json", {"schema_version": "cameramaster-macro-events-v1", "source_root": str(root), "macro_events": macro_events})
    _write(output_dir / "hierarchy_reference_audit.json", audit)
    _write(output_dir / "macro_event_summary.json", {
        "macro_event_count": len(macro_events),
        "macro_events_without_l3_arc_mapping": sum(not row["l3_story_arc_ids"] for row in macro_events),
        "source_l2_count": len(events),
        "unassigned_l2_event_ids": unassigned,
        "audit_status": audit["status"],
        "parameters": {"max_duration_sec": max_duration_sec, "merge_gap_sec": merge_gap_sec},
    })
    return {"macro_event_count": len(macro_events), "audit": audit, "output_dir": str(output_dir)}


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-duration-sec", type=float, default=90.0)
    parser.add_argument("--merge-gap-sec", type=float, default=30.0)
    args = parser.parse_args()
    result = build_macro_events(args.input_root, args.output_dir, args.max_duration_sec, args.merge_gap_sec)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
