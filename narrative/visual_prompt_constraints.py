from __future__ import annotations

from typing import Any, Sequence


_MULTI_PERSON_CUES = (
    "conversation",
    "dialogue",
    "exchange",
    "fight",
    "follow",
    "gather",
    "group",
    "help",
    "meet",
    "multiple",
    "argue",
    "ask",
    "reply",
    "says",
    "speak",
    "talk",
    "together",
    "trio",
    "duo",
    "crowd",
    "team",
    "confront",
    "block",
    "blocks",
    "blocking",
    "hand over",
    "give",
    "take",
    "pull",
    "push",
)

_INTERACTION_CUES = (
    "ask",
    "asks",
    "answer",
    "answers",
    "argue",
    "blocks",
    "block",
    "challenge",
    "confront",
    "dialogue",
    "exchange",
    "fight",
    "help",
    "hand over",
    "meet",
    "refuse",
    "reply",
    "respond",
    "speak",
    "talk",
    "trade",
    "warn",
)


def _clean_list(values: Sequence[Any] | None) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]


def build_people_control_section(character_names: Sequence[Any] | None, *, allow_multi_people: bool) -> str:
    names = _clean_list(character_names)
    lines = ["people_policy:"]
    if allow_multi_people:
        focus = ", ".join(names[:4]) if names else "the listed characters"
        lines.extend(
            [
                f"- This event explicitly requires multiple named characters: {focus}.",
                "- Include only the referenced characters.",
                "- Preserve their identity, costume, and staging consistency.",
                "- Do not add pedestrians, crowds, bystanders, or unrelated people.",
            ]
        )
        return "\n".join(lines)
    focus = names[0] if names else "the main subject"
    lines.extend(
        [
            f"- Keep the frame focused on {focus} and the required environment or object interaction.",
            "- Render only the named subject and essential context.",
            "- Do not add pedestrians, crowds, bystanders, or unrelated people.",
        ]
    )
    return "\n".join(lines)


def build_people_control_clause(character_names: Sequence[Any] | None, *, allow_multi_people: bool) -> str:
    return build_people_control_section(character_names, allow_multi_people=allow_multi_people)


def build_node_selection_prompt_policy() -> str:
    return (
        "node_selection_policy:\n"
        "- Default granularity is one role action per step; each selected single_role_step node becomes one video task.\n"
        "- A node should normally contain exactly one character from one step.\n"
        "- Merge two or more role actions into one narrative node only if they occur in the same step, in the same location, and the text shows direct interaction or mutual causal action between those characters.\n"
        "- Do not merge unrelated characters just because they appear in the same step or share the same broader scene.\n"
        "- For example, do not collapse step::1::CharacterA, step::1::CharacterB, and step::1::CharacterC into one node unless all of them participate in the same location-bound interaction event.\n"
        "- A merged node should read like one event with multiple participants, not like separate parallel actions."
    )


def build_path_orchestration_prompt_policy() -> str:
    return (
        "path_orchestration_policy:\n"
        "- Each candidate is already an atomic narrative node.\n"
        "- Preserve node boundaries exactly as provided; do not split or merge candidates.\n"
        "- A single_role_step candidate remains one video task for that one role action.\n"
        "- If a candidate contains multiple characters, treat it as one interaction node only when the node itself already encodes same-step same-location direct interaction.\n"
        "- Sequence nodes in narrative order without inventing new merges across neighboring candidates."
    )


def candidate_location(candidate: dict[str, Any]) -> str:
    return str(candidate.get("location", "") or candidate.get("scene_location", "") or "").strip()


def candidate_role_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("role_name", "") or "").strip()


def candidate_text_blob(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get("role_output_text", ""),
        candidate.get("action_type", ""),
        candidate.get("target_character", ""),
        candidate.get("location", ""),
        candidate.get("reason", ""),
        candidate.get("monologue", ""),
        candidate.get("sam_summary", ""),
        candidate.get("text", ""),
    ]
    return " ".join(str(part) for part in parts if str(part).strip()).lower()


def candidates_have_direct_interaction(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = candidate_role_name(left).lower()
    right_name = candidate_role_name(right).lower()
    if not left_name or not right_name:
        return False
    if candidate_location(left).lower() != candidate_location(right).lower():
        return False
    left_text = candidate_text_blob(left)
    right_text = candidate_text_blob(right)
    has_named_target = (
        candidate_role_name(right).lower() == str(left.get("target_character", "") or "").strip().lower()
        or candidate_role_name(left).lower() == str(right.get("target_character", "") or "").strip().lower()
    )
    mentions_each_other = left_name in right_text or right_name in left_text
    has_interaction_cue = any(cue in f"{left_text} {right_text}" for cue in _INTERACTION_CUES)
    return bool((has_named_target or mentions_each_other) and has_interaction_cue)


def event_needs_multi_person_references(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    assets = node.get("visual_assets", {}) or {}
    if bool(assets.get("multi_person_reference_allowed")):
        return True
    characters = _clean_list(node.get("characters", []) or [])
    if len(characters) < 2:
        return False
    text_bits = [
        node.get("event_summary", ""),
        node.get("story_description", ""),
        node.get("video_description", ""),
        node.get("dialogue", ""),
        node.get("event_type", ""),
        " ".join(characters),
    ]
    text = " ".join(str(bit) for bit in text_bits if str(bit).strip()).lower()
    return any(cue in text for cue in _MULTI_PERSON_CUES)
