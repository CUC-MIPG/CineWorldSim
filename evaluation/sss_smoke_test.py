"""Prepare or run CameraMaster SSS-v1/v2/v3 evaluation protocols."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

from sss_hierarchical import (
    build_l4_normalization_prompt,
    extract_segment_frames,
    resolve_segments,
    run_hierarchical_reconstruction,
    validate_l4_payload,
)
from sss_metrics import (
    build_event_coverage_prompt,
    build_grounding_prompt,
    build_reference_events_prompt,
    build_structured_alignment_prompt,
    calculate_caption_grounding,
    calculate_embedding_sss,
    calculate_event_coverage,
    calculate_structured_alignment,
    get_embedding,
    validate_event_matches,
    validate_grounding_payload,
    validate_reference_events,
)


DEFAULT_VIDEO = Path(
    "/data1/AgentG/hhb/CameraMaster/evaluation/adapted_cases/"
    "0001_American_Beauty_arc_001_real/directorbench/track_B.mp4"
)
ALLOWED_CATEGORIES = (
    "core_event",
    "character_relation",
    "conflict",
    "plot_direction",
    "setting_context",
)
STATUS_SCORES = {"supported": 1.0, "partial": 0.5, "absent": 0.0, "contradicted": 0.0}
ALLOWED_STATUSES = {*STATUS_SCORES, "not_applicable"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_summary(
    summary_file: Path | None,
    adapter_inputs: Path | None,
    l4_global: Path | None,
    track: str,
    l4_text: str = "short_synopsis",
) -> tuple[str, str, str]:
    if summary_file is not None:
        summary = summary_file.read_text(encoding="utf-8").strip()
        if not summary:
            raise ValueError("Input summary is empty")
        return summary, str(summary_file.resolve()), "file"
    if adapter_inputs is not None:
        payload = _read_json(adapter_inputs)
        samples = payload.get("samples", [])
        matches = [row for row in samples if str(row.get("track", "")).upper() == track]
        if len(matches) != 1 or not str(matches[0].get("prompt", "")).strip():
            raise ValueError(f"Expected exactly one non-empty Track {track} prompt in {adapter_inputs}")
        return str(matches[0]["prompt"]).strip(), str(adapter_inputs.resolve()), "proxy"
    if l4_global is not None:
        source = l4_global
        if source.is_dir():
            source = source / "movie_bible.json"
        payload = _read_json(source)
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise ValueError(f"L4 global result is not an object: {source}")
        if l4_text == "short_synopsis":
            summary = result.get("movie_synopsis", {}).get("short_synopsis")
        else:
            summary = result.get("global_storyline", {}).get("story")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"L4 global file has no non-empty {l4_text} text: {source}")
        return summary.strip(), str(source.resolve()), "l4_global"
    raise ValueError("Provide either --summary-file or --adapter-inputs")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain or markdown-fenced model output."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response does not contain a valid JSON object")


def calculate_sss(claims: list[dict[str, Any]], matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute category-balanced, directional semantic coverage from claim matches."""
    if not claims:
        raise ValueError("Input summary produced no evaluable semantic claims")
    claim_ids = [str(claim.get("claim_id", "")) for claim in claims]
    if any(not claim_id for claim_id in claim_ids) or len(set(claim_ids)) != len(claim_ids):
        raise ValueError("Claims must have unique, non-empty claim_id values")
    claim_id_set = set(claim_ids)

    match_by_id: dict[str, dict[str, Any]] = {}
    for match in matches:
        claim_id = str(match.get("claim_id", ""))
        if claim_id in match_by_id:
            raise ValueError(f"duplicate match for claim_id {claim_id}")
        if claim_id not in claim_id_set:
            raise ValueError(f"Unknown matched claim_id: {claim_id}")
        status = str(match.get("status", ""))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid match status for {claim_id}: {status}")
        match_by_id[claim_id] = match
    if set(match_by_id) != set(claim_ids):
        raise ValueError("Every input claim must have exactly one match")

    scores_by_category: dict[str, list[float]] = {category: [] for category in ALLOWED_CATEGORIES}
    status_counts = {status: 0 for status in ALLOWED_STATUSES}
    for claim in claims:
        category = str(claim.get("category", ""))
        if category not in scores_by_category:
            raise ValueError(f"Unknown claim category: {category}")
        status = str(match_by_id[str(claim["claim_id"])]["status"])
        status_counts[status] += 1
        if status != "not_applicable":
            scores_by_category[category].append(STATUS_SCORES[status])

    category_scores = {
        category: {
            "score": sum(values) / len(values) if values else None,
            "claim_count": len(values),
        }
        for category, values in scores_by_category.items()
    }
    applicable = [entry["score"] for entry in category_scores.values() if entry["score"] is not None]
    if not applicable:
        raise ValueError("No applicable claims remain after excluding not_applicable claims")
    sss = sum(applicable) / len(applicable)
    return {
        "metric": "SSS",
        "definition": "Category-balanced directional coverage of input-story claims in the generated video",
        "higher_is_better": True,
        "sss": sss,
        "sss_100": sss * 100,
        "category_scores": category_scores,
        "status_counts": status_counts,
        "claim_count": len(claims),
        "applicable_category_count": len(applicable),
    }


def _probe_duration(video_path: Path) -> float:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot inspect video duration: {exc}") from exc
    if duration <= 0:
        raise ValueError(f"Video has invalid duration: {duration}")
    return duration


def sample_video_frames(video_path: Path, frame_dir: Path, max_frames: int) -> list[dict[str, Any]]:
    duration = _probe_duration(video_path)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for index in range(max_frames):
        timestamp = duration * (index + 0.5) / max_frames
        frame_path = frame_dir / f"frame_{index + 1:03d}.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}",
            "-i", str(video_path), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2",
            "-q:v", "4", str(frame_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            details = getattr(exc, "stderr", "")
            raise ValueError(f"Could not sample video frame at {timestamp:.3f}s: {details}") from exc
        if not frame_path.is_file() or frame_path.stat().st_size == 0:
            raise ValueError(f"ffmpeg produced an empty frame at {timestamp:.3f}s")
        frames.append({"index": index + 1, "timestamp_sec": round(timestamp, 3), "path": str(frame_path)})
    return frames


def _chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_sec: int,
    request_context: str = "unspecified",
    request_log_path: Path | None = None,
    max_retries: int = 2,
    retry_backoff_sec: float = 5.0,
) -> str:
    request_id = f"{time.time_ns()}"
    started = time.monotonic()
    max_retries = max(0, int(max_retries))
    retry_backoff_sec = max(0.0, float(retry_backoff_sec))

    def log_event(event: str, **fields: Any) -> None:
        if request_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event": event,
            "context": request_context,
            "model": model,
            "url": f"{api_base.rstrip('/')}/chat/completions",
            **fields,
        }
        request_log_path.parent.mkdir(parents=True, exist_ok=True)
        with request_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    log_event("request_started", max_attempts=max_retries + 1)
    for attempt in range(max_retries + 1):
        attempt_started = time.monotonic()
        try:
            response = requests.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            log_event(
                "request_succeeded",
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                attempt_duration_sec=round(time.monotonic() - attempt_started, 3),
                duration_sec=round(time.monotonic() - started, 3),
                http_status=response.status_code,
                usage=payload.get("usage") if isinstance(payload, dict) else None,
            )
            return str(payload["choices"][0]["message"]["content"])
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            response = getattr(exc, "response", None)
            error_fields: dict[str, Any] = {}
            http_status: int | None = None
            if response is not None:
                http_status = response.status_code
                error_fields["http_status"] = http_status
                try:
                    error = response.json().get("error", {})
                    if isinstance(error, dict):
                        error_fields.update({
                            "error_type": error.get("type"),
                            "error_code": error.get("code"),
                            "error_message": str(error.get("message", ""))[:500],
                        })
                except (ValueError, AttributeError):
                    pass
            retryable = isinstance(exc, requests.RequestException) and (
                http_status is None or http_status == 429 or http_status >= 500
            )
            if retryable and attempt < max_retries:
                delay_sec = retry_backoff_sec * (3 ** attempt)
                log_event(
                    "request_retry_scheduled",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    max_attempts=max_retries + 1,
                    retry_delay_sec=delay_sec,
                    attempt_duration_sec=round(time.monotonic() - attempt_started, 3),
                    exception_type=type(exc).__name__,
                    exception=str(exc)[:500],
                    **error_fields,
                )
                print(
                    "sss_smoke_test: transient chat request failure; "
                    f"context={request_context} retry={attempt + 1}/{max_retries} "
                    f"delay_sec={delay_sec:g} error={exc}",
                    file=sys.stderr,
                )
                time.sleep(delay_sec)
                continue
            log_event(
                "request_failed",
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                attempt_duration_sec=round(time.monotonic() - attempt_started, 3),
                duration_sec=round(time.monotonic() - started, 3),
                exception_type=type(exc).__name__,
                exception=str(exc)[:500],
                **error_fields,
            )
            raise RuntimeError(
                f"OpenAI-compatible chat request failed at {request_context} "
                f"after {attempt + 1}/{max_retries + 1} attempts: {exc}"
            ) from exc
    raise AssertionError("unreachable chat retry state")


def _frame_content(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": "Reconstruct only the story supported by these ordered video frames. "
        "Do not infer unseen causes or use outside knowledge. Mark uncertainty. "
        "Each image is labeled with its video timestamp. Return one JSON object with "
        "keys story, events (array of {timestamp_sec, description, characters, relations, confidence}), "
        "and uncertainties (array of strings).",
    }]
    for frame in frames:
        encoded = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Frame {frame['index']} at {frame['timestamp_sec']} seconds:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
        })
    return content


def _ground_reconstruction(
    captions: list[dict[str, Any]],
    evidence_by_shot: dict[str, list[dict[str, Any]]],
    chat_call,
    output_dir: Path,
) -> dict[str, Any]:
    claims = []
    raw_dir = output_dir / "grounding"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, caption in enumerate(captions, start=1):
        shot_id = str(caption.get("shot_id", "video"))
        prompt = build_grounding_prompt([caption])
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in evidence_by_shot.get(shot_id, []):
            encoded = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"Evidence frame at {frame['timestamp_sec']:.3f} seconds."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"}})
        raw = chat_call([{"role": "user", "content": content}], f"grounding.shot_{index:04d}:{shot_id}")
        raw_path = raw_dir / f"shot_{index:04d}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        try:
            grounded_claims = validate_grounding_payload(extract_json_object(raw))
        except (ValueError, json.JSONDecodeError) as first_error:
            # Keep the initial raw response for diagnosis, then make exactly
            # one model retry only when the returned JSON/payload is invalid.
            # API/network failures still propagate from chat_call unchanged.
            retry_content = list(content)
            retry_content[0] = {
                "type": "text",
                "text": (
                    f"{prompt}\n\n"
                    "Your previous response was not valid for the required schema. "
                    "Return only one valid JSON object with a non-empty `claims` array. "
                    "Every array item must contain a non-empty `text` field and a "
                    "boolean or null `grounded` field. Do not include Markdown or commentary."
                ),
            }
            retry_raw = chat_call(
                [{"role": "user", "content": retry_content}],
                f"grounding.shot_{index:04d}:{shot_id}.json_retry_1",
            )
            retry_path = raw_dir / f"shot_{index:04d}.json_retry_1.txt"
            retry_path.write_text(retry_raw, encoding="utf-8")
            try:
                grounded_claims = validate_grounding_payload(extract_json_object(retry_raw))
            except (ValueError, json.JSONDecodeError) as retry_error:
                details_path = output_dir / "grounding_failure.json"
                _write_json(details_path, {
                    "stage": "per_shot_schema_validation",
                    "shot_id": shot_id,
                    "shot_index": index,
                    "first_error": str(first_error),
                    "retry_error": str(retry_error),
                    "first_response_path": str(raw_path),
                    "retry_response_path": str(retry_path),
                })
                print(
                    "sss_smoke_test: grounding JSON retry failed "
                    f"shot={shot_id} first_error={first_error!s} retry_error={retry_error!s} "
                    f"details={details_path} first_response={raw_path} retry_response={retry_path}",
                    file=sys.stderr,
                )
                raise ValueError(
                    f"grounding response remained invalid after one JSON retry for {shot_id}: "
                    f"first={first_error}; retry={retry_error}; details={details_path}"
                ) from retry_error
            print(
                "sss_smoke_test: grounding JSON retry succeeded "
                f"shot={shot_id} first_error={first_error!s} "
                f"first_response={raw_path} retry_response={retry_path}",
                file=sys.stderr,
            )
        # Models commonly restart IDs (for example, c001) for every shot.
        # Grounding is aggregated across the complete video, so retain the
        # local ID but scope it by its source shot before global validation.
        claim_scope = shot_id or f"shot_{index:04d}"
        for claim in grounded_claims:
            local_claim_id = str(claim.get("claim_id") or "")
            claim["claim_id"] = f"{claim_scope}::{local_claim_id}"
        claims.extend(grounded_claims)
    try:
        result = calculate_caption_grounding(claims)
    except ValueError as error:
        occurrences: dict[str, list[dict[str, Any]]] = {}
        for claim in claims:
            claim_id = str(claim.get("claim_id") or "")
            occurrences.setdefault(claim_id, []).append({
                "shot_id": str(claim.get("shot_id") or ""),
                "text": str(claim.get("text") or "")[:300],
                "grounded": claim.get("grounded"),
            })
        duplicates = [
            {"claim_id": claim_id, "occurrences": rows}
            for claim_id, rows in occurrences.items()
            if not claim_id or len(rows) > 1
        ]
        details_path = output_dir / "grounding_failure.json"
        _write_json(details_path, {
            "stage": "cross_shot_claim_aggregation",
            "error": str(error),
            "claim_count": len(claims),
            "duplicate_or_empty_claim_ids": duplicates,
            "raw_response_dir": str(raw_dir),
        })
        duplicate_ids = ", ".join(str(row["claim_id"] or "<empty>") for row in duplicates[:10]) or "none"
        print(
            "sss_smoke_test: grounding aggregation failed "
            f"error={error!s} claim_count={len(claims)} duplicate_claim_ids={duplicate_ids} "
            f"details={details_path} raw_responses={raw_dir}",
            file=sys.stderr,
        )
        raise ValueError(f"{error}; details={details_path}") from error
    result["claims"] = claims
    _write_json(output_dir / "caption_grounding.json", result)
    return result


def _validated_diagnostic_json(
    chat_call,
    prompt: str,
    context: str,
    validator: Callable[[dict[str, Any]], Any],
    raw_path: Path,
    output_dir: Path,
    diagnostic_name: str,
) -> Any:
    """Call one diagnostic judge and retry once only for schema/JSON failures."""

    raw = chat_call([{"role": "user", "content": prompt}], context)
    raw_path.write_text(raw, encoding="utf-8")
    try:
        return validator(extract_json_object(raw))
    except (ValueError, json.JSONDecodeError, TypeError) as first_error:
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous response failed schema validation: "
            f"{first_error}. Return only one valid JSON object that exactly satisfies "
            "the requested top-level keys and nested structure. Do not include Markdown or commentary."
        )
        retry_context = f"{context}.json_retry_1"
        retry_raw = chat_call([{"role": "user", "content": retry_prompt}], retry_context)
        retry_path = raw_path.with_name(f"{raw_path.stem}.json_retry_1{raw_path.suffix}")
        retry_path.write_text(retry_raw, encoding="utf-8")
        try:
            value = validator(extract_json_object(retry_raw))
        except (ValueError, json.JSONDecodeError, TypeError) as retry_error:
            details_path = output_dir / f"{diagnostic_name}_schema_failure.json"
            _write_json(details_path, {
                "stage": "diagnostic_schema_validation",
                "diagnostic": diagnostic_name,
                "context": context,
                "first_error": str(first_error),
                "retry_error": str(retry_error),
                "first_response_path": str(raw_path),
                "retry_response_path": str(retry_path),
            })
            print(
                "sss_smoke_test: diagnostic JSON retry failed "
                f"diagnostic={diagnostic_name} first_error={first_error!s} retry_error={retry_error!s} "
                f"details={details_path} first_response={raw_path} retry_response={retry_path}",
                file=sys.stderr,
            )
            raise ValueError(
                f"{diagnostic_name} response remained invalid after one JSON retry; details={details_path}"
            ) from retry_error
        print(
            "sss_smoke_test: diagnostic JSON retry succeeded "
            f"diagnostic={diagnostic_name} first_error={first_error!s} "
            f"first_response={raw_path} retry_response={retry_path}",
            file=sys.stderr,
        )
        return value


def _score_story_pair(
    reference_story: str,
    l4_story: str,
    l2_events: list[dict[str, Any]],
    chat_call,
    embedding_api_base: str,
    embedding_api_key: str,
    embedding_model: str,
    timeout_sec: int,
    output_dir: Path,
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    reference_embedding = get_embedding(
        reference_story, embedding_api_base, embedding_api_key, embedding_model, timeout_sec,
    )
    reconstruction_embedding = get_embedding(
        l4_story, embedding_api_base, embedding_api_key, embedding_model, timeout_sec,
    )
    primary = calculate_embedding_sss(reference_embedding, reconstruction_embedding, embedding_model)
    primary["reference_embedding_dimension"] = len(reference_embedding)
    primary["reconstruction_embedding_dimension"] = len(reconstruction_embedding)
    result: dict[str, Any] = {"primary": primary, "diagnostics": {}}
    if run_diagnostics:
        structured_prompt = build_structured_alignment_prompt(reference_story, l4_story)
        result["diagnostics"]["structured_alignment"] = _validated_diagnostic_json(
            chat_call, structured_prompt, "diagnostics.structured_alignment",
            calculate_structured_alignment, output_dir / "raw_structured_alignment_response.txt",
            output_dir, "structured_alignment",
        )

        reference_prompt = build_reference_events_prompt(reference_story)
        reference_events = _validated_diagnostic_json(
            chat_call, reference_prompt, "diagnostics.reference_event_extraction",
            validate_reference_events, output_dir / "raw_reference_events_response.txt",
            output_dir, "reference_event_extraction",
        )
        _write_json(output_dir / "reference_events.json", {"events": reference_events})
        coverage_prompt = build_event_coverage_prompt(reference_events, l2_events)
        expected_event_ids = [event["event_id"] for event in reference_events]
        matches = _validated_diagnostic_json(
            chat_call, coverage_prompt, "diagnostics.event_coverage",
            lambda payload: validate_event_matches(payload, expected_event_ids),
            output_dir / "raw_event_coverage_response.txt", output_dir, "event_coverage",
        )
        result["diagnostics"]["event_reconstruction_coverage"] = calculate_event_coverage(
            [event["event_id"] for event in reference_events], matches,
        )
        result["diagnostics"]["event_reconstruction_coverage"]["reference_events"] = reference_events
    else:
        result["diagnostics"]["status"] = "skipped_by_user"
    _write_json(output_dir / "sss_result.json", result)
    return result


def _validate_claims(payload: dict[str, Any]) -> list[dict[str, Any]]:
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Claim extraction response must include a non-empty claims array")
    normalized = []
    seen: set[str] = set()
    for index, raw in enumerate(claims, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Every claim must be a JSON object")
        claim_id = str(raw.get("claim_id") or f"claim_{index:03d}")
        category = str(raw.get("category", ""))
        text = str(raw.get("text", "")).strip()
        if claim_id in seen or category not in ALLOWED_CATEGORIES or not text:
            raise ValueError(f"Invalid or duplicate source claim: {claim_id}")
        seen.add(claim_id)
        normalized.append({"claim_id": claim_id, "category": category, "text": text})
    return normalized


def _validate_matches(payload: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise ValueError("Alignment response must include a matches array")
    expected_ids = {claim["claim_id"] for claim in claims}
    result = []
    seen: set[str] = set()
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Every claim match must be a JSON object")
        claim_id = str(match.get("claim_id", ""))
        status = str(match.get("status", ""))
        if claim_id in seen or claim_id not in expected_ids or status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid or duplicate match for claim {claim_id}")
        evidence = match.get("evidence", "")
        if not isinstance(evidence, str):
            raise ValueError(f"Evidence must be text for claim {claim_id}")
        if status in {"supported", "partial", "contradicted"} and not evidence.strip():
            raise ValueError(f"Evidence is required for {status} claim {claim_id}")
        timestamp = match.get("timestamp_sec")
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timestamp for claim {claim_id}") from exc
            if timestamp < 0:
                raise ValueError(f"Invalid timestamp for claim {claim_id}")
        seen.add(claim_id)
        result.append({
            "claim_id": claim_id,
            "status": status,
            "evidence": evidence,
            "timestamp_sec": timestamp,
            "rationale": str(match.get("rationale", "")),
        })
    if seen != expected_ids:
        raise ValueError("Alignment response must include exactly one match per input claim")
    return result


def run_evaluation(
    video_path: Path,
    summary: str | None,
    frames: list[dict[str, Any]],
    api_base: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    output_dir: Path,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    chat_retries: int = 2,
    chat_retry_backoff_sec: float = 5.0,
    run_diagnostics: bool = True,
    legacy_claim_sss: bool = False,
    mode: str = "v1",
    segments: list[dict[str, Any]] | None = None,
    frames_per_shot: int = 1,
    max_l4_words: int = 180,
    summary_loader: Callable[[], tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Run SSS with reconstruction completed before the reference is loaded or scored."""
    if mode not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unsupported SSS mode: {mode}")
    resolved_embedding_api_base = embedding_api_base or api_base
    resolved_embedding_api_key = embedding_api_key or api_key

    request_log_path = output_dir / "api_requests.jsonl"

    def chat_call(messages: list[dict[str, Any]], context: str = "unspecified") -> str:
        return _chat_completion(
            api_base, api_key, model, messages, timeout_sec,
            request_context=context, request_log_path=request_log_path,
            max_retries=chat_retries, retry_backoff_sec=chat_retry_backoff_sec,
        )

    if not legacy_claim_sss:
        if mode == "v1":
            # Compatibility baseline: global uniform frames, then video-only L4 normalization.
            reconstruction_raw = chat_call(
                [{"role": "user", "content": _frame_content(frames)}],
                "reconstruction.v1.video",
            )
            reconstruction = extract_json_object(reconstruction_raw)
            raw_story = str(reconstruction.get("story", "")).strip()
            if not raw_story:
                raise ValueError("Video reconstruction response has no story text")
            normalize_prompt = build_l4_normalization_prompt(raw_story, max_l4_words)
            (output_dir / "prompts").mkdir(parents=True, exist_ok=True)
            (output_dir / "prompts/l4_normalization.txt").write_text(normalize_prompt, encoding="utf-8")
            normalized_raw = chat_call(
                [{"role": "user", "content": normalize_prompt}],
                "reconstruction.v1.L4_normalization",
            )
            l4 = validate_l4_payload(extract_json_object(normalized_raw), max_l4_words)
            l2_events = reconstruction.get("events", [])
            captions = [
                {"shot_id": "video", "caption": str(event.get("description", event.get("caption", "")))}
                for event in l2_events if isinstance(event, dict)
                and str(event.get("description", event.get("caption", ""))).strip()
            ]
            if not captions:
                captions = [{"shot_id": "video", "caption": raw_story}]
            evidence_by_shot = {"video": frames}
            _write_json(output_dir / "video_reconstruction.json", reconstruction)
            _write_json(output_dir / "L4_normalized.json", l4)
            (output_dir / "raw_reconstruction_response.txt").write_text(reconstruction_raw, encoding="utf-8")
            (output_dir / "raw_l4_normalization_response.txt").write_text(normalized_raw, encoding="utf-8")
        else:
            if not segments:
                raise ValueError(f"{mode} requires resolved shot segments")
            hierarchical_dir = output_dir / "hierarchical"
            reconstruction = run_hierarchical_reconstruction(
                segments, video_path, output_dir / "frames", hierarchical_dir,
                chat_call,
                frames_per_shot=frames_per_shot, max_l4_words=max_l4_words,
            )
            l4 = reconstruction["L4"]
            l2_events = reconstruction["L2"]
            captions = reconstruction["L1"]
            frame_payload = _read_json(hierarchical_dir / "frame_manifest.json")
            evidence_by_shot = {
                str(row["shot_id"]): row["frames"] for row in frame_payload.get("shots", [])
            }

        # Grounding audits are video-only and finish before the reference loader is invoked.
        grounding = _ground_reconstruction(captions, evidence_by_shot, chat_call, output_dir)
        _write_json(output_dir / "reconstructed_story.json", {
            "mode": mode, "L4": l4, "L2": l2_events,
        })
        if summary_loader is not None:
            summary, _, _ = summary_loader()
        if summary is None or not summary.strip():
            raise ValueError("A non-empty reference story is required for SSS scoring")
        result = _score_story_pair(
            summary, l4["story"], l2_events,
            chat_call,
            resolved_embedding_api_base, resolved_embedding_api_key,
            embedding_model, timeout_sec, output_dir, run_diagnostics,
        )
        result["diagnostics"]["caption_grounding"] = grounding
        result.update({"mode": mode, "video_path": str(video_path.resolve()), "model": model})
        _write_json(output_dir / "sss_result.json", result)
        return result

    if summary is None:
        raise ValueError("Legacy claim SSS requires a reference story")
    # The legacy path is retained only for explicit compatibility experiments.
    reconstruction_raw = chat_call(
        [{"role": "user", "content": _frame_content(frames)}],
        "reconstruction.legacy.video",
    )
    reconstruction = extract_json_object(reconstruction_raw)
    if not str(reconstruction.get("story", "")).strip():
        raise ValueError("Video reconstruction response has no story text")
    _write_json(output_dir / "video_reconstruction.json", reconstruction)
    (output_dir / "raw_reconstruction_response.txt").write_text(reconstruction_raw, encoding="utf-8")

    claim_prompt = (
        "Convert this input story summary into atomic, video-checkable claims. "
        "Do not add facts. Use only these categories: " + ", ".join(ALLOWED_CATEGORIES) + ". "
        "Return JSON: {\"claims\":[{\"claim_id\":\"c001\",\"category\":\"...\",\"text\":\"...\"}]}.\n\n"
        "INPUT SUMMARY:\n" + summary
    )
    claims_raw = chat_call(
        [{"role": "user", "content": claim_prompt}],
        "legacy.reference_claim_extraction",
    )
    claims = _validate_claims(extract_json_object(claims_raw))
    _write_json(output_dir / "input_claims.json", {"claims": claims})
    (output_dir / "raw_claim_extraction_response.txt").write_text(claims_raw, encoding="utf-8")

    align_prompt = (
        "Compare each input claim against the independently reconstructed video story. "
        "Do not reward merely shared names or themes. Use supported only when evidence is clear; "
        "partial when meaning is incomplete; absent when not represented; contradicted when video evidence conflicts; "
        "not_applicable only when the claim cannot reasonably be assessed from sampled video/audio. "
        "Do not treat reasonable extra story expansion as an error. Cite a timestamp from the reconstruction when possible. "
        "Return JSON: {\"matches\":[{\"claim_id\":\"...\",\"status\":\"...\",\"evidence\":\"...\",\"timestamp_sec\":0.0,\"rationale\":\"...\"}]}.\n\n"
        "INPUT CLAIMS:\n" + json.dumps(claims, ensure_ascii=False) + "\n\n"
        "VIDEO-ONLY STORY RECONSTRUCTION:\n" + json.dumps(reconstruction, ensure_ascii=False)
    )
    alignment_raw = chat_call(
        [{"role": "user", "content": align_prompt}],
        "legacy.claim_alignment",
    )
    matches = _validate_matches(extract_json_object(alignment_raw), claims)
    (output_dir / "raw_alignment_response.txt").write_text(alignment_raw, encoding="utf-8")
    _write_json(output_dir / "claim_matches.json", {"matches": matches})
    result = calculate_sss(claims, matches)
    result.update({"video_path": str(video_path.resolve()), "model": model, "frame_count": len(frames)})
    _write_json(output_dir / "sss_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    summary_group = parser.add_mutually_exclusive_group(required=True)
    summary_group.add_argument("--summary-file", type=Path, help="Exact generation-time summary or an explicitly labeled proxy")
    summary_group.add_argument("--adapter-inputs", type=Path, help="Read a proxy prompt from the prepared DirectorBench inputs.json")
    summary_group.add_argument("--l4-global", type=Path, help="L4 global directory or movie_bible.json containing the original summary")
    parser.add_argument("--track", choices=("A", "B"), default="B", help="Evaluation track selected from the track adapter")
    parser.add_argument("--track-adapter-inputs", type=Path, help="Track/clip mapping used only for v3 segmentation")
    parser.add_argument("--l4-text", choices=("short_synopsis", "global_storyline"), default="short_synopsis")
    parser.add_argument("--summary-provenance", choices=("original_runtime", "l4_global", "proxy", "unknown"))
    parser.add_argument("--allow-proxy-summary", action="store_true", help="Allow an explicitly exploratory proxy summary for API scoring")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frames-per-shot", type=int, default=1)
    parser.add_argument("--shot-threshold", type=float, default=0.3)
    parser.add_argument("--uniform-fallback-sec", type=float, help="Explicit segmentation fallback if both detectors fail")
    parser.add_argument("--max-l4-words", type=int, default=180)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL_NAME") or "text-embedding-3-small",
        help="Embedding model; defaults to EMBEDDING_MODEL_NAME, then text-embedding-3-small.",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Embedding endpoint base URL; defaults to EMBEDDING_BASE_URL, then OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="Embedding API key; defaults to EMBEDDING_API_KEY, then OPENAI_API_KEY.",
    )
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip structured alignment and event coverage")
    parser.add_argument("--legacy-claim-sss", action="store_true", help="Run the deprecated claim-coverage SSS instead")
    parser.add_argument("--run-llm", action="store_true", help="Call the configured OpenAI-compatible VLM/LLM endpoint")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL") or os.environ.get("SSS_MODEL") or "gpt-4o",
        help="Chat model; defaults to OPENAI_MODEL, then SSS_MODEL, then gpt-4o.",
    )
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per chat/embedding request timeout in seconds (default: 15 minutes).")
    parser.add_argument("--chat-retries", type=int, default=2, help="Retries for transient chat API errors after the first attempt (default: 2).")
    parser.add_argument("--chat-retry-backoff-sec", type=float, default=5.0, help="Initial chat retry delay in seconds; retries use 5s, then 15s by default.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if not 1 <= args.max_frames <= 24:
            raise ValueError("--max-frames must be between 1 and 24")
        if args.frames_per_shot <= 0 or args.max_l4_words <= 0:
            raise ValueError("--frames-per-shot and --max-l4-words must be positive")
        if args.chat_retries < 0 or args.chat_retry_backoff_sec < 0:
            raise ValueError("--chat-retries and --chat-retry-backoff-sec must be non-negative")
        def summary_loader():
            return load_summary(args.summary_file, args.adapter_inputs, args.l4_global, args.track, args.l4_text)

        summary = None
        summary_source = None
        inferred_provenance = "unknown"
        if not args.run_llm:
            summary, summary_source, inferred_provenance = summary_loader()
        elif args.summary_file:
            inferred_provenance = "file"
        elif args.adapter_inputs:
            inferred_provenance = "proxy"
        elif args.l4_global:
            inferred_provenance = "l4_global"
        summary_provenance = args.summary_provenance or inferred_provenance
        if args.summary_provenance == "original_runtime" and inferred_provenance == "proxy":
            raise ValueError("An adapter-derived prompt cannot be labeled original_runtime")
        if args.summary_provenance == "proxy":
            summary_provenance = "proxy"
        if args.run_llm and summary_provenance not in {"original_runtime", "l4_global"} and not args.allow_proxy_summary:
            raise ValueError("Online scoring requires an original runtime summary or explicit --allow-proxy-summary")
        if args.run_llm and not args.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key when using --run-llm")
        if args.run_llm and not args.legacy_claim_sss and not args.embedding_api_key:
            raise ValueError("Set EMBEDDING_API_KEY or pass --embedding-api-key when using SSS_emb")
        if args.run_llm and args.legacy_claim_sss:
            summary, summary_source, inferred_provenance = summary_loader()
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty; pass --overwrite to replace it: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.video.resolve()
        duration = _probe_duration(video_path)
        segments = None
        segmentation_info = None
        if args.mode == "v1":
            frames = sample_video_frames(video_path, output_dir / "frames", args.max_frames)
            frame_sampling = "uniform midpoint sampling over full video"
        else:
            segments, segmentation_info = resolve_segments(
                args.mode, args.track, video_path, duration, args.track_adapter_inputs,
                _probe_duration, args.shot_threshold, args.uniform_fallback_sec,
            )
            frames = []
            frame_rows = []
            if not args.run_llm:
                for segment in segments:
                    shot_frames = extract_segment_frames(
                        video_path, segment, output_dir / "frames", args.frames_per_shot,
                    )
                    frame_rows.append({"shot_id": segment["shot_id"], "frames": shot_frames})
                    frames.extend(shot_frames)
            frame_sampling = "midpoint sampling within resolved shot segments"
        manifest = {
            "schema_version": "cameramaster-sss-v2",
            "sample_id": video_path.stem,
            "mode": args.mode,
            "track": args.track if args.mode == "v3" else None,
            "video_path": str(video_path),
            "video_duration_sec": duration,
            "summary_path": summary_source,
            "summary_provenance": summary_provenance,
            "summary_selector": args.l4_text if args.l4_global else "external_text_file_or_adapter_prompt",
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest() if summary else None,
            "summary_char_count": len(summary) if summary else None,
            "frame_sampling": frame_sampling,
            "frame_count": len(frames),
            "shot_count": len(segments) if segments is not None else None,
            "segmentation": segmentation_info,
            "audio_used": False,
            "evaluation_scope": "hierarchical_video_reconstruction_and_story_similarity" if args.mode != "v1" else "uniform_frame_video_reconstruction_and_story_similarity",
            "primary_metric": "SSS_emb" if not args.legacy_claim_sss else "legacy_claim_sss",
            "warnings": ([
                "Only uniformly sampled video frames are sent to the model; intervening content may be missed.",
                "Audio is not transcribed or evaluated.",
            ] if args.mode == "v1" else [
                "One or more midpoint frames represent each shot; within-shot content may be missed.",
                "Audio is not transcribed or evaluated.",
            ]),
            "blind_reconstruction_inputs": [frame["path"] for frame in frames],
            "summary_exposed_to_reconstruction": False,
            "api_calls_enabled": bool(args.run_llm),
            "model": args.model if args.run_llm else None,
            "embedding_model": args.embedding_model if args.run_llm and not args.legacy_claim_sss else None,
            "embedding_api_base": args.embedding_api_base if args.run_llm and not args.legacy_claim_sss else None,
            "chat_retries": args.chat_retries if args.run_llm else None,
            "chat_retry_backoff_sec": args.chat_retry_backoff_sec if args.run_llm else None,
            "status": "prepared" if not args.run_llm else "running",
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        _write_json(output_dir / "frame_manifest.json", {"frames": frames})
        if segments is not None:
            _write_json(output_dir / "shot_segments.json", {"segments": segments, "frame_groups": frame_rows})
        if not args.run_llm:
            print(json.dumps({**manifest, "next_step": "rerun with --run-llm to call the configured judge endpoint"}, ensure_ascii=False, indent=2))
            return 0
        result = run_evaluation(
            video_path, summary, frames, args.api_base, args.api_key, args.model,
            args.timeout_sec, output_dir,
            embedding_model=args.embedding_model,
            embedding_api_base=args.embedding_api_base,
            embedding_api_key=args.embedding_api_key,
            chat_retries=args.chat_retries,
            chat_retry_backoff_sec=args.chat_retry_backoff_sec,
            run_diagnostics=not args.skip_diagnostics,
            legacy_claim_sss=args.legacy_claim_sss,
            mode=args.mode,
            segments=segments,
            frames_per_shot=args.frames_per_shot,
            max_l4_words=args.max_l4_words,
            summary_loader=summary_loader if not args.legacy_claim_sss else None,
        )
        if segments is not None:
            hierarchical_frames = _read_json(output_dir / "hierarchical/frame_manifest.json")
            frame_rows = hierarchical_frames.get("shots", [])
            frames = [frame for row in frame_rows for frame in row.get("frames", [])]
            _write_json(output_dir / "frame_manifest.json", {"frames": frames, "shots": frame_rows})
            _write_json(output_dir / "shot_segments.json", {"segments": segments, "frame_groups": frame_rows})
            manifest["frame_count"] = len(frames)
        if summary_source is None:
            _, summary_source, _ = summary_loader()
        manifest.update({"summary_path": summary_source, "status": "completed"})
        _write_json(output_dir / "run_manifest.json", manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"sss_smoke_test: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
