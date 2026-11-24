"""Replica of ViMax Appendix Figure 12/13/14 judge protocol.

This module intentionally exposes prompts and JSON validation separately from
the API transport so prompts can be audited and responses can be replayed.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import requests


PLANNING_METRICS = ("CBC", "NC", "PPR", "VSU", "STQ")
NF_METRICS = ("NF",)
PROMPT_VERSION = "vimax-appendix-fig12-14-replica-v1"


def _storyboard_json(storyboards: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for index, row in enumerate(storyboards, start=1):
        rows.append({
            "id": row.get("id") or row.get("shot_id") or row.get("node_id") or f"shot_{index:03d}",
            "storyboard_description": row.get("storyboard_description") or row.get("description") or row.get("story_description") or "",
            "characters": row.get("characters", []),
            "locations": row.get("locations", []),
            "dialogue": row.get("dialogue", ""),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_planning_prompt(storyboards: Sequence[Mapping[str, Any]]) -> str:
    return f"""You are the narrative-planning judge from the ViMax-Bench protocol (replica of Appendix Figures 12-13).
Evaluate only the ordered storyboard batch below. Do not reward verbosity, and do not penalize concise descriptions by themselves.
Score each dimension independently on an integer scale from 1 (severe failure) to 5 (excellent). Distinguish local issues from structural failure and explain missing information conservatively.

Dimensions:
- CBC (Character Behavior Consistency): character identity, goals, behavior, and actions remain coherent across shots.
- NC (Narrative Coherence): the ordered shots form a comprehensible causal and temporal progression.
- PPR (Plot Pacing and Rhythm): event timing, escalation, pauses, and shot granularity support the intended story.
- VSU (Visual Specificity and Shot Usability): descriptions are concrete enough to guide image/video generation, including subjects, composition, setting, and action.
- STQ (Scene Transition and Continuity Quality): transitions preserve spatial, temporal, and state continuity between shots/scenes.

Return exactly one JSON object with these keys:
{{"scores": {{"CBC": 1-5, "NC": 1-5, "PPR": 1-5, "VSU": 1-5, "STQ": 1-5}}, "rationale": {{}}, "key_strengths": [], "key_weaknesses": [], "improvement_suggestions": []}}
All five scores must be integers. The rationale object must contain one short explanation per metric.

Storyboard batch:
{_storyboard_json(storyboards)}"""


def build_nf_prompt(source_text: str, storyboards: Sequence[Mapping[str, Any]]) -> str:
    return f"""You are the Narrative Faithfulness (NF) judge from the ViMax-Bench protocol (replica of Appendix Figure 14).
The source text is authoritative. Evaluate only this storyboard batch, while using the complete source text as ground truth.
Check the order of events, characters, settings, actions, and major plot beats. Penalize invented scenes, wrong identities or locations, changed motivations, and unsupported events. Reasonable visualisation details are not hallucinations. Do not penalize this batch for content that appears in another batch.
Score NF as an integer from 1 to 5: 1 severely unfaithful, 2 mostly unfaithful, 3 partially faithful, 4 largely faithful, 5 fully faithful.

Return exactly one JSON object with keys:
{{"scores": {{"NF": 1-5}}, "rationale": {{}}, "key_strengths": [], "key_weaknesses": [], "improvement_suggestions": []}}
The rationale object must contain one short explanation for NF.

Full source text:
{source_text}

Storyboard batch:
{_storyboard_json(storyboards)}"""


def extract_json_object(text: str) -> dict[str, Any]:
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
    raise ValueError("judge response does not contain a JSON object")


def validate_judge_payload(payload: Mapping[str, Any], metric_names: Sequence[str]) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("judge response must contain a scores object")
    normalized_scores: dict[str, int] = {}
    for metric in metric_names:
        value = scores.get(metric)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{metric} score must be an integer in [1, 5]")
        normalized_scores[metric] = value
    rationale = payload.get("rationale", {})
    if not isinstance(rationale, Mapping):
        raise ValueError("rationale must be a JSON object")
    return {
        "scores": normalized_scores,
        "rationale": dict(rationale),
        "key_strengths": list(payload.get("key_strengths", [])) if isinstance(payload.get("key_strengths", []), list) else [],
        "key_weaknesses": list(payload.get("key_weaknesses", [])) if isinstance(payload.get("key_weaknesses", []), list) else [],
        "improvement_suggestions": list(payload.get("improvement_suggestions", [])) if isinstance(payload.get("improvement_suggestions", []), list) else [],
    }


def aggregate_judge_batches(batches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    batches = list(batches)
    if not batches:
        raise ValueError("at least one judge batch is required")
    metric_names = tuple(batches[0]["scores"])
    values = {metric: [] for metric in metric_names}
    for batch in batches:
        for metric in metric_names:
            if metric not in batch.get("scores", {}):
                raise ValueError(f"missing {metric} score in batch")
            values[metric].append(float(batch["scores"][metric]))
    scores = {metric: sum(items) / len(items) for metric, items in values.items()}
    return {"scores": scores, "average": sum(scores.values()) / len(scores), "batch_count": len(batches), "batch_scores": values}


def chat_completion(api_base: str, api_key: str, model: str, prompt: str, timeout_sec: int = 300) -> tuple[dict[str, Any], str]:
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    raw = str(response.json()["choices"][0]["message"]["content"])
    return extract_json_object(raw), raw

