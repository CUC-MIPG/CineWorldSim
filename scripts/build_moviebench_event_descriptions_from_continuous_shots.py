from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.client import make_client, pick_model
from agents.retry import call_with_retries


REQUIRED_SHOT_FIELDS = [
    "Characters",
    "Style Elements",
    "Plot",
    "Background Description",
    "Camera Motion",
]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())[:120] or "event"


def _safe_text(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,; ") + "..."


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON root is not an object")
    return payload


def _load_sequence_entries(input_root: Path) -> list[dict[str, Any]]:
    index_path = input_root / "continuous_dataset_index.json"
    if index_path.exists():
        index = _load_json_object(index_path) or {}
        rows = index.get("sequences") or []
        return [row for row in rows if isinstance(row, dict) and row.get("file")]
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("*/*.json")):
        if path.name == "continuous_dataset_index.json":
            continue
        payload = _load_json_object(path)
        if payload is None:
            continue
        rows.append(
            {
                "movie_id": payload.get("movie_id", path.parent.name),
                "sequence_id": payload.get("sequence_id", path.stem),
                "clip_count": payload.get("clip_count", len(payload.get("clips") or [])),
                "file": str(path),
            }
        )
    return rows


def _required_shot_payload(clip: dict[str, Any]) -> dict[str, Any]:
    row = {field: clip.get(field) for field in REQUIRED_SHOT_FIELDS}
    row.update(
        {
            "file_name": clip.get("file_name", ""),
            "source_json": clip.get("source_json", ""),
            "clip_start_tc": clip.get("clip_start_tc", ""),
            "clip_end_tc": clip.get("clip_end_tc", ""),
            "duration_sec": clip.get("duration_sec", 0),
        }
    )
    return row


def _shots_for_prompt(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, shot in enumerate(shots, start=1):
        rows.append(
            {
                "shot_index": idx,
                "time": f"{shot.get('clip_start_tc', '')}-{shot.get('clip_end_tc', '')}",
                "duration_sec": shot.get("duration_sec", 0),
                "characters": _safe_text(shot.get("Characters"), 500),
                "style_elements": _safe_text(shot.get("Style Elements"), 500),
                "plot": _safe_text(shot.get("Plot"), 700),
                "background_description": _safe_text(shot.get("Background Description"), 500),
                "camera_motion": _safe_text(shot.get("Camera Motion"), 300),
            }
        )
    return rows


def _merged_shot_sentence(index: int, shot: dict[str, Any]) -> str:
    parts = [
        f"Shot {index} ({shot.get('clip_start_tc', '')}-{shot.get('clip_end_tc', '')})",
        f"characters: {_safe_text(shot.get('Characters'), 260)}" if _safe_text(shot.get("Characters"), 260) else "",
        f"style: {_safe_text(shot.get('Style Elements'), 220)}" if _safe_text(shot.get("Style Elements"), 220) else "",
        f"plot: {_safe_text(shot.get('Plot'), 500)}" if _safe_text(shot.get("Plot"), 500) else "",
        f"background: {_safe_text(shot.get('Background Description'), 320)}" if _safe_text(shot.get("Background Description"), 320) else "",
        f"camera: {_safe_text(shot.get('Camera Motion'), 220)}" if _safe_text(shot.get("Camera Motion"), 220) else "",
    ]
    return "; ".join(part for part in parts if part) + "."


def _fallback_event_description(sequence: dict[str, Any], shots: list[dict[str, Any]]) -> dict[str, Any]:
    backgrounds = [_safe_text(shot.get("Background Description"), 180) for shot in shots if _safe_text(shot.get("Background Description"), 180)]
    event_description = " ".join(
        _merged_shot_sentence(i, shot) for i, shot in enumerate(shots, start=1)
    ).strip()
    if not event_description:
        event_description = "A continuous multi-shot event reconstructed from adjacent MovieBench shot annotations."
    return {
        "event_title": str(sequence.get("sequence_id", "continuous_event")),
        "event_description": event_description,
        "main_characters": [],
        "setting": backgrounds[0] if backgrounds else "",
        "temporal_progression": "The event follows the chronological order of the source shots.",
        "shot_to_event_mapping": [
            {
                "shot_index": i,
                "role_in_event": _safe_text(shot.get("Plot"), 180),
            }
            for i, shot in enumerate(shots, start=1)
        ],
    }


def _call_llm_event_description(
    *,
    sequence: dict[str, Any],
    shots: list[dict[str, Any]],
    model: str | None,
    retries: int,
) -> dict[str, Any]:
    client = make_client(scope="MOVIEBENCH_EVENT")
    model_name = model or pick_model("gpt-5.5", scope="MOVIEBENCH_EVENT")
    prompt_shots = _shots_for_prompt(shots)
    system_prompt = (
        "You are a film event reconstruction assistant. "
        "You receive consecutive MovieBench shot annotations from one continuous video segment. "
        "Merge every shot-level description, in order, into one concrete continuous-event description. "
        "Do not compress the sequence into a vague summary and do not omit any shot. "
        "The output event should read like one continuous multi-shot event whose details can be decomposed back into the provided shots. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Task: Merge the following continuous shots into one total event description.\n"
        "Requirements:\n"
        "- The event_description must include the content of every shot in chronological order.\n"
        "- Treat the output as a continuous event description, not a high-level abstract summary.\n"
        "- Preserve characters, setting, action progression, visual style, background details, and camera motion.\n"
        "- If a shot is transitional, still include what it shows and how it connects to the next shot.\n\n"
        f"Sequence metadata:\n{json.dumps({k: sequence.get(k) for k in ['movie_id', 'sequence_id', 'clip_count', 'time_range']}, ensure_ascii=False, indent=2)}\n\n"
        f"Shots:\n{json.dumps(prompt_shots, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "event_title": "short title",\n'
        '  "event_description": "one chronological paragraph merging every shot description into a continuous event",\n'
        '  "main_characters": ["..."],\n'
        '  "setting": "where the event happens",\n'
        '  "temporal_progression": "how the event unfolds across shots",\n'
        '  "shot_to_event_mapping": [\n'
        '    {"shot_index": 1, "role_in_event": "what this shot contributes"}\n'
        "  ]\n"
        "}\n"
    )

    def _call() -> dict[str, Any]:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content if response.choices else "{}"
        return _extract_json_object(str(content))

    payload = call_with_retries(_call, attempts=max(1, int(retries)), delay_sec=1.0)
    payload["_model"] = model_name
    return payload


def _event_output_payload(
    *,
    sequence: dict[str, Any],
    event: dict[str, Any],
    shots: list[dict[str, Any]],
    llm_status: str,
    model: str,
) -> dict[str, Any]:
    return {
        "movie_id": sequence.get("movie_id", ""),
        "source_sequence_id": sequence.get("sequence_id", ""),
        "source_sequence_file": sequence.get("_source_file", ""),
        "shot_count": len(shots),
        "time_range": sequence.get("time_range", {}),
        "event_title": event.get("event_title", ""),
        "event_description": event.get("event_description", ""),
        "main_characters": event.get("main_characters", []),
        "setting": event.get("setting", ""),
        "temporal_progression": event.get("temporal_progression", ""),
        "shot_to_event_mapping": event.get("shot_to_event_mapping", []),
        "shots": shots,
        "llm": {
            "status": llm_status,
            "model": model,
        },
    }


def build_event_descriptions(
    *,
    input_root: Path,
    output_root: Path,
    min_shots: int = 4,
    dry_run: bool = True,
    model: str | None = None,
    retries: int = 3,
    limit: int = 0,
    strict_online: bool = True,
) -> dict[str, Any]:
    entries = _load_sequence_entries(input_root)
    candidates = [row for row in entries if int(row.get("clip_count", 0) or 0) >= int(min_shots)]
    if limit > 0:
        candidates = candidates[:limit]
    output_root.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    failed = 0
    written = 0
    skipped_invalid = 0
    resolved_model = model or pick_model("gpt-5.5", scope="MOVIEBENCH_EVENT")

    for row in candidates:
        sequence_path = Path(str(row.get("file", "")))
        sequence = _load_json_object(sequence_path)
        if sequence is None:
            skipped_invalid += 1
            continue
        sequence["_source_file"] = str(sequence_path)
        raw_clips = [clip for clip in sequence.get("clips", []) or [] if isinstance(clip, dict)]
        shots = [_required_shot_payload(clip) for clip in raw_clips]
        if len(shots) < int(min_shots):
            continue

        llm_status = "dry_run"
        try:
            if dry_run:
                event = _fallback_event_description(sequence, shots)
            else:
                event = _call_llm_event_description(
                    sequence=sequence,
                    shots=shots,
                    model=model,
                    retries=retries,
                )
                llm_status = "done"
                resolved_model = str(event.pop("_model", resolved_model))
        except Exception as exc:  # noqa: BLE001
            if strict_online:
                raise
            failed += 1
            event = _fallback_event_description(sequence, shots)
            event["error"] = str(exc)
            llm_status = "fallback_after_error"

        movie_id = str(sequence.get("movie_id") or row.get("movie_id") or "movie")
        sequence_id = str(sequence.get("sequence_id") or row.get("sequence_id") or sequence_path.stem)
        movie_dir = output_root / _safe_name(movie_id)
        movie_dir.mkdir(parents=True, exist_ok=True)
        payload = _event_output_payload(
            sequence=sequence,
            event=event,
            shots=shots,
            llm_status=llm_status,
            model=resolved_model,
        )
        out_file = movie_dir / f"{_safe_name(sequence_id)}_event.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
        index_rows.append(
            {
                "movie_id": movie_id,
                "source_sequence_id": sequence_id,
                "shot_count": len(shots),
                "event_title": payload.get("event_title", ""),
                "event_description": payload.get("event_description", ""),
                "file": str(out_file),
            }
        )

    index = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "min_shots": int(min_shots),
        "dry_run": bool(dry_run),
        "total_candidates": len(candidates),
        "written": written,
        "failed": failed,
        "skipped_invalid": skipped_invalid,
        "events": index_rows,
    }
    (output_root / "event_description_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter MovieBench continuous shot groups with more than 3 shots and merge each group "
            "into one LLM event description."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("MovieBench/Continuous_Clip_Groups_11.15_V2_160movies"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("MovieBench/Continuous_Event_Descriptions_11.15_V2_160movies"),
    )
    parser.add_argument("--min-shots", type=int, default=4, help="Keep sequences with at least this many shots.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of sequences to process.")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    index = build_event_descriptions(
        input_root=args.input_root,
        output_root=args.output_root,
        min_shots=max(1, int(args.min_shots)),
        dry_run=args.dry_run,
        model=args.model,
        retries=max(1, int(args.retries)),
        limit=max(0, int(args.limit)),
        strict_online=args.strict_online,
    )
    print("=== MovieBench Event Description Build Completed ===")
    print(f"input_root: {index['input_root']}")
    print(f"output_root: {index['output_root']}")
    print(f"min_shots: {index['min_shots']}")
    print(f"dry_run: {index['dry_run']}")
    print(f"total_candidates: {index['total_candidates']}")
    print(f"written: {index['written']}")
    print(f"failed: {index['failed']}")
    print(f"skipped_invalid: {index['skipped_invalid']}")
    print(f"index: {Path(index['output_root']) / 'event_description_index.json'}")


if __name__ == "__main__":
    main()
