"""Primary embedding score and independent SSS reliability diagnostics."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import requests


ALIGNMENT_DIMENSIONS = ("characters", "conflict", "key_events", "relationships_goals", "ending")
EVENT_STATUSES = {"recovered", "missed", "not_assessable"}


def validate_embedding_payload(payload: Mapping[str, Any]) -> list[float]:
    try:
        embedding = payload["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("embedding response must contain data[0].embedding") from exc
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("embedding must be a non-empty numeric array")
    try:
        values = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must contain only numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    return values


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("embedding vectors must be one-dimensional and have the same dimension")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embedding vectors contain non-finite values")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("embedding vectors must have non-zero norm")
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding(text: str, api_base: str, api_key: str, model: str = "text-embedding-3-small", timeout_sec: int = 300) -> list[float]:
    if not text.strip():
        raise ValueError("cannot embed empty text")
    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": text},
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return validate_embedding_payload(response.json())
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise RuntimeError(f"OpenAI-compatible embeddings request failed: {exc}") from exc


def calculate_embedding_sss(reference_embedding: Sequence[float], reconstruction_embedding: Sequence[float], model: str) -> dict[str, Any]:
    score = cosine_similarity(reference_embedding, reconstruction_embedding)
    return {
        "metric": "SSS_emb",
        "definition": "Cosine similarity between embeddings of the reference story and normalized L4 reconstructed story",
        "score": score,
        "higher_is_better": True,
        "embedding_model": model,
    }


def build_structured_alignment_prompt(reference_story: str, reconstructed_story: str) -> str:
    return (
        "Compare the reference story with the independently reconstructed story. Score each dimension from 1 (no meaningful agreement) "
        "to 5 (strong agreement), based only on the texts. Return strict JSON with integer scores for characters, conflict, key_events, "
        "relationships_goals, ending, plus a rationale object with the same keys. Do not combine the dimensions into an overall score.\n\n"
        f"REFERENCE STORY:\n{reference_story}\n\nRECONSTRUCTED L4 STORY:\n{reconstructed_story}"
    )


def calculate_structured_alignment(payload: Mapping[str, Any]) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("structured alignment must contain scores object")
    normalized = {}
    for dimension in ALIGNMENT_DIMENSIONS:
        score = scores.get(dimension)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{dimension} must be an integer in [1, 5]")
        normalized[dimension] = score
    rationale = payload.get("rationale", {})
    if not isinstance(rationale, Mapping):
        raise ValueError("structured alignment rationale must be an object")
    return {"scores": normalized, "score": sum(normalized.values()) / (5 * len(normalized)), "rationale": dict(rationale)}


def build_event_coverage_prompt(reference_events: Sequence[Mapping[str, Any]], reconstructed_events: Sequence[Mapping[str, Any]]) -> str:
    return (
        "For each reference event, decide whether the event is correctly recovered by the reconstructed L2 events. "
        "Use recovered only when the main action/state change is represented, missed when assessable but absent or contradicted, "
        "and not_assessable only when the video evidence cannot establish it. Return strict JSON with one match per reference event: "
        "{event_id, status, matched_reconstructed_event_ids, evidence}. Do not invent matches.\n\n"
        f"REFERENCE EVENTS:\n{json.dumps(reference_events, ensure_ascii=False)}\n\n"
        f"RECONSTRUCTED L2 EVENTS:\n{json.dumps(reconstructed_events, ensure_ascii=False)}"
    )


def calculate_event_coverage(reference_event_ids: Sequence[str], matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(event_id) for event_id in reference_event_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("reference event IDs must be non-empty and unique")
    by_id = {}
    for match in matches:
        event_id = str(match.get("event_id", ""))
        status = match.get("status")
        if event_id not in ids or event_id in by_id:
            raise ValueError(f"invalid or duplicate event match: {event_id}")
        if status not in EVENT_STATUSES:
            raise ValueError(f"invalid event coverage status: {status}")
        by_id[event_id] = status
    if set(by_id) != set(ids):
        raise ValueError("each reference event requires exactly one match")
    assessable = [status for status in by_id.values() if status != "not_assessable"]
    recovered = sum(status == "recovered" for status in assessable)
    return {
        "coverage": recovered / len(assessable) if assessable else None,
        "recovered_count": recovered,
        "assessable_event_count": len(assessable),
        "not_assessable_count": len(ids) - len(assessable),
        "event_matches": [{"event_id": event_id, "status": by_id[event_id]} for event_id in ids],
    }


def build_grounding_prompt(captions: Sequence[Mapping[str, Any]]) -> str:
    return (
        "Audit each caption claim against the provided video frames only. The captions may be inaccurate. "
        "For every atomic claim, return grounded=true only when visibly supported, false when contradicted or unsupported, "
        "and null when not visually assessable. Never use or request a reference story. Return strict JSON with claims: "
        "[{claim_id, shot_id, text, grounded, evidence}].\n\n"
        f"VIDEO-ONLY CAPTIONS TO AUDIT:\n{json.dumps(captions, ensure_ascii=False)}"
    )


def build_reference_events_prompt(reference_story: str) -> str:
    return (
        "Extract atomic, chronologically ordered plot events stated in this reference story. "
        "Include only events that can be checked against video evidence; do not add interpretation. "
        "Return strict JSON with events [{event_id, text}].\n\nREFERENCE STORY:\n" + reference_story
    )


def validate_reference_events(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("reference event response must contain a non-empty events array")
    result = []
    seen = set()
    for index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise ValueError("reference events must be objects")
        event_id = str(event.get("event_id") or f"reference_event_{index:03d}")
        text = event.get("text")
        if event_id in seen or not isinstance(text, str) or not text.strip():
            raise ValueError("reference event IDs must be unique and event text non-empty")
        seen.add(event_id)
        result.append({"event_id": event_id, "text": text.strip()})
    return result


def validate_event_matches(payload: Mapping[str, Any], event_ids: Sequence[str]) -> list[dict[str, Any]]:
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise ValueError("event coverage response must contain a matches array")
    expected = {str(event_id) for event_id in event_ids}
    result = []
    seen = set()
    for match in matches:
        if not isinstance(match, Mapping):
            raise ValueError("event matches must be objects")
        event_id = str(match.get("event_id", ""))
        status = match.get("status")
        if event_id not in expected or event_id in seen or status not in EVENT_STATUSES:
            raise ValueError(f"invalid event match: {event_id}")
        seen.add(event_id)
        result.append({
            "event_id": event_id,
            "status": status,
            "matched_reconstructed_event_ids": match.get("matched_reconstructed_event_ids", []),
            "evidence": str(match.get("evidence", "")),
        })
    if seen != expected:
        raise ValueError("each reference event requires exactly one match")
    return result


def calculate_caption_grounding(claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(claim.get("claim_id", "")) for claim in claims]
    if any(not claim_id for claim_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("grounding claims require unique non-empty claim_id values")
    statuses = [claim.get("grounded") for claim in claims]
    if any(status not in (True, False, None) for status in statuses):
        raise ValueError("grounded must be true, false, or null")
    evaluable = [status for status in statuses if status is not None]
    unsupported = sum(status is False for status in evaluable)
    return {
        "grounded_count": sum(status is True for status in evaluable),
        "unsupported_count": unsupported,
        "not_assessable_count": len(statuses) - len(evaluable),
        "evaluable_claim_count": len(evaluable),
        "hallucination_rate": unsupported / len(evaluable) if evaluable else None,
    }


def validate_grounding_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("grounding response must contain a non-empty claims array")
    normalized = []
    seen = set()
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, Mapping):
            raise ValueError("grounding claims must be objects")
        claim_id = str(claim.get("claim_id") or f"claim_{index:03d}")
        grounded = claim.get("grounded")
        text = claim.get("text")
        if claim_id in seen or grounded not in (True, False, None) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid grounding claim {claim_id}: grounded must be boolean/null and text non-empty")
        seen.add(claim_id)
        normalized.append({
            "claim_id": claim_id,
            "shot_id": str(claim.get("shot_id", "")),
            "text": text.strip(),
            "grounded": grounded,
            "evidence": str(claim.get("evidence", "")),
        })
    return normalized
