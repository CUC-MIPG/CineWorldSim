from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from narrative.visual_prompt_constraints import (
    build_people_control_clause,
    event_needs_multi_person_references,
)

_REFERENCE_NAME_STOPWORDS = {
    "black",
    "blue",
    "broken",
    "cracked",
    "dark",
    "gold",
    "gray",
    "green",
    "hidden",
    "large",
    "left",
    "light",
    "little",
    "new",
    "old",
    "red",
    "right",
    "sealed",
    "silver",
    "small",
    "true",
    "white",
}


def _clean(value: object, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(".,; ") + "."
    return text


def _join(values: list[Any]) -> str:
    return ", ".join(str(v).strip() for v in values or [] if str(v).strip())


def _name_parts(name: str) -> list[str]:
    text = re.sub(r"[_\\-]+", " ", str(name or "")).lower()
    return [
        part
        for part in re.findall(r"[a-z0-9]+", text)
        if len(part) >= 4 and part not in _REFERENCE_NAME_STOPWORDS
    ]


def _mentions_name(text: str, name: str) -> bool:
    normalized = re.sub(r"[_\\-]+", " ", str(name or "")).lower().strip()
    if not normalized:
        return False
    if normalized in text:
        return True
    parts = _name_parts(normalized)
    return bool(parts) and any(part in text for part in parts)


def _reference_selection_text(node: dict[str, Any]) -> str:
    fields = [
        "event_summary",
        "story_description",
        "video_description",
        "dialogue",
        "rendered_dialogue",
        "event_type",
        "story_phase",
    ]
    return " ".join(str(node.get(field) or "") for field in fields).lower()


def referenced_event_items(node: dict[str, Any]) -> list[str]:
    mention_text = _reference_selection_text(node)
    items = [str(x).strip() for x in node.get("items", []) or [] if str(x).strip()]
    return [item for item in items if _mentions_name(mention_text, item)]


def reference_selection_policy(referenced_items: list[str]) -> str:
    item_policy = (
        "Only the explicitly mentioned item references are active: " + _join(referenced_items) + "."
        if referenced_items
        else "No item/object reference is active for this event because the event description does not explicitly mention a story object."
    )
    return (
        "Reference selection policy: use only references required by the current event description. "
        "Prioritize event keyframes, character identity references, and the world-space scene reference. "
        + item_policy
        + " Do not fill unused reference slots, and do not introduce unmentioned props, objects, crowds, pedestrians, or unrelated people."
    )


def _story_description(node: dict[str, Any]) -> str:
    text = _clean(
        node.get("story_description", "") or node.get("video_description", "") or node.get("event_summary", ""),
        limit=260,
    )
    return "" if text.lower() == "none" else text


def _dialogue_from_event(node: dict[str, Any]) -> str:
    dialogue = _clean(node.get("dialogue", "") or node.get("dialogue_text", ""), limit=300)
    if dialogue:
        return dialogue
    monologues = [
        _clean(row.get("next_action", {}).get("monologue", ""), limit=160)
        for row in node.get("role_logs", []) or []
        if isinstance(row, dict) and isinstance(row.get("next_action"), dict)
    ]
    monologues = [line for line in monologues if line]
    return " / ".join(monologues) if monologues else "No spoken dialogue; tell the event through performance, ambient sound, and camera language."


def _h3_dialogue_from_event(node: dict[str, Any], character_subjects: list[tuple[str, str]]) -> str:
    backfill = node.get("_dialogue_backfill")
    if not isinstance(backfill, dict):
        text = _dialogue_from_event(node)
        return "none" if text.strip().lower() == "none" else text

    dialogue = backfill.get("dialogue")
    slot_mapping = backfill.get("slot_mapping")
    if not isinstance(dialogue, dict) or not isinstance(slot_mapping, dict):
        text = _dialogue_from_event(node)
        return "none" if text.strip().lower() == "none" else text

    name_to_label = {name: label for label, name in character_subjects}
    lines: list[str] = []
    speaker_index = 1
    for slot in slot_mapping:
        line = _clean(dialogue.get(slot, ""), limit=260)
        if not line or line.lower() == "none":
            continue
        name = str(slot_mapping.get(slot) or slot)
        subject_label = name_to_label.get(name, slot)
        lines.append(f"{subject_label} ({slot}) (S{speaker_index}) says, <d>[English] {line}</d>")
        speaker_index += 1
    return " ".join(lines) if lines else "none"


def _bgm_from_event(event_type: str, phase: str) -> str:
    context = f"{event_type} {phase}".lower()
    if any(token in context for token in ("climax", "confront", "danger", "reveal", "escalat")):
        return "Restrained suspense score: low strings, distant percussion, and a slow rising pulse that supports tension without masking dialogue."
    if any(token in context for token in ("resolution", "ending", "reconcile")):
        return "A sparse resolving score with warm sustained tones, leaving room for the scene's final emotional beat."
    return "Subtle atmospheric underscore matched to the location, with a restrained pulse that supports forward motion and does not dominate dialogue."


def _build_single_shot_description(
    *,
    event: str,
    event_type: str,
    phase: str,
    char_refs: str,
    env_ref: str,
    item_refs: str,
) -> str:
    return (
        f"[Shot 1] A single coherent cinematic shot opens inside {env_ref}. "
        f"{char_refs} are arranged according to the event logic, with referenced identities and poses preserved rather than treated as separate slideshow images. "
        f"The action centers on: {event} "
        f"If visible, {item_refs} remain integrated as physical objects with stable placement and readable silhouettes. "
        f"The event type is {event_type or 'narrative action'} and the story phase is {phase or 'ongoing development'}."
    )


def _build_multi_shot_description(
    node: dict[str, Any],
    shot_plan: list[dict[str, Any]],
    *,
    storyline: str,
    event: str,
    event_type: str,
    phase: str,
    char_refs: str,
    env_ref: str,
    item_refs: str,
    dialogue_text: str,
) -> str:
    story_description = _story_description(node)
    lines = [
        "The target video is a multi-shot cinematic reference-generation sequence. "
        "All shots preserve the same referenced characters, environment layout, lighting continuity, object placement, and screen-direction logic."
    ]
    for fallback_index, shot in enumerate(shot_plan, start=1):
        if not isinstance(shot, dict):
            continue
        shot_index = int(shot.get("shot_index", fallback_index) or fallback_index)
        prefix = f"[Shot {shot_index}]"
        if shot_index > 1:
            prefix += f" At {shot.get('start_timecode', '00:00.000')},"
        purpose = _clean(shot.get("video_description", ""), limit=360)
        camera = _clean(shot.get("cinematography", ""), limit=220)
        music = _clean(shot.get("bgm", ""), limit=180)
        narration = _clean(shot.get("narration") or shot.get("voiceover"), limit=260)
        audio_text = f"Dialogue or performance audio: {dialogue_text or _dialogue_from_event(node)}"
        if narration:
            audio_text += f" Narration: {narration}"
        lines.append(
            f"{prefix} {purpose or story_description or event} "
            f"The shot uses {char_refs} inside {env_ref}, with {item_refs} kept physically integrated when visible. "
            f"Camera and composition: {camera or 'controlled camera movement with stable spatial continuity'}. "
            f"{audio_text} "
            f"Non-diegetic cue for this shot: {music or _bgm_from_event(event_type, phase)}"
        )
    if len(lines) == 1:
        lines.append(
            _build_single_shot_description(
                event=event,
                event_type=event_type,
                phase=phase,
                char_refs=char_refs,
                env_ref=env_ref,
                item_refs=item_refs,
            )
        )
    story_context = _clean(storyline, limit=260)
    if story_context:
        lines.append(f"The shot sequence remains grounded in the story context: {story_context}")
    return "\n".join(lines)


def build_h3_ref_prompt(
    node: dict[str, Any],
    storyline: str = "",
    shot_plan: list[dict[str, Any]] | None = None,
) -> str:
    """Build a MiniMax-H3 full-reference prompt from a CineWorldSim event node.

    The structure follows MiniMax-H3's full-reference prompt-writing guide: subject_definitions,
    summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
    """
    event = _clean(node.get("event_summary", ""), limit=260)
    story_description = _story_description(node)
    phase = _clean(node.get("story_phase", ""), limit=120)
    event_type = _clean(node.get("event_type", ""), limit=80)
    bgm = _bgm_from_event(event_type, phase)
    characters = [str(x).strip() for x in node.get("characters", []) or [] if str(x).strip()]
    locations = [str(x).strip() for x in node.get("locations", []) or [] if str(x).strip()]
    items = referenced_event_items(node)
    ref_policy = reference_selection_policy(items)

    subject_lines: list[str] = []
    subject_index = 1
    character_subjects: list[tuple[str, str]] = []
    for name in characters:
        label = f"<Subject {subject_index}>"
        character_subjects.append((label, name))
        subject_lines.append(
            f"{label} is the character {name}, using the corresponding character reference image to preserve identity, costume, body language, and action continuity."
        )
        subject_index += 1

    environment_label = ""
    if locations:
        environment_label = f"<Subject {subject_index}>"
        subject_lines.append(
            f"{environment_label} is the main environment, { _join(locations) }, using the world-space reference image to preserve spatial layout, lighting, atmosphere, and material details."
        )
        subject_index += 1

    item_subjects: list[tuple[str, str]] = []
    for item in items[:3]:
        label = f"<Subject {subject_index}>"
        item_subjects.append((label, item))
        subject_lines.append(
            f"{label} is the important story object {item}, preserving its recognizable shape, placement, and narrative function when it appears."
        )
        subject_index += 1

    if not subject_lines:
        subject_lines.append(
            "<Subject 1> is the referenced event scene, preserving the visible environment, characters, and object relationships from the provided reference images."
        )

    allow_multi_people = event_needs_multi_person_references(node)
    people_clause = build_people_control_clause(characters, allow_multi_people=allow_multi_people)
    char_refs = ", ".join(f"{label} ({name})" for label, name in character_subjects) or "the referenced characters"
    env_ref = environment_label or "the referenced environment"
    item_refs = ", ".join(f"{label} ({name})" for label, name in item_subjects) or "the referenced story objects"
    dialogue = _h3_dialogue_from_event(node, character_subjects)

    retention = []
    for label, name in character_subjects:
        retention.append(
            f"{label} (appears in [Shot 1]): fully_preserved - {name}'s identity, costume impression, and action role are retained from the character reference image."
        )
    if environment_label:
        retention.append(
            f"{environment_label} (appears in [Shot 1]): fully_preserved - the location layout, lighting, rain-soaked atmosphere, and spatial continuity are retained from the world reference image."
        )
    for label, name in item_subjects:
        retention.append(
            f"{label} (appears in [Shot 1] when relevant): partially_preserved - {name} remains recognizable and narratively connected to the event, while its exact screen position may adapt to the shot composition."
        )
    if not retention:
        retention.append("<Subject 1> (appears in [Shot 1]): fully_preserved - the visible reference content guides the target shot's scene, action, and atmosphere.")

    plan = [shot for shot in (shot_plan or []) if isinstance(shot, dict)]
    story_description_block = (
        _build_multi_shot_description(
            node,
            plan,
            storyline=storyline,
            event=event,
            event_type=event_type,
            phase=phase,
            char_refs=char_refs,
            env_ref=env_ref,
            item_refs=item_refs,
            dialogue_text=dialogue,
        )
        if plan
        else _build_single_shot_description(
            event=story_description or event,
            event_type=event_type,
            phase=phase,
            char_refs=char_refs,
            env_ref=env_ref,
            item_refs=item_refs,
        )
    )
    cinematography = (
        "Cinematic live-action treatment with coherent spatial continuity, realistic lighting, and controlled shot transitions. "
        "Use readable compositions, then controlled push-ins, lateral glides, or cuts according to the shot plan. "
        "Maintain stable screen direction, clear character-object interaction, and no slideshow cuts or abrupt scene changes."
    )

    sections = [
        "subject_definitions:\n" + "\n".join(subject_lines),
        "summary:\n"
        + f"[reference generation] The target video turns the selected CineWorldSim event into one continuous cinematic shot using {char_refs}, {env_ref}, and {item_refs} as visual references. "
        + people_clause
        + " "
        + ref_policy,
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\n"
        + "story_description:\n"
        + story_description
        + "\n\nstory_description_block:\n"
        + story_description_block
        + "\n\nvisual_constraints:\n"
        + people_clause
        + "\n"
        + ref_policy
        + "\n\ncinematography:\n"
        + cinematography
        + "\n\ndialogue:\n"
        + dialogue
        + "\n\nbgm:\n"
        + bgm,
        "overall_soundscape:\nRain, room tone, footsteps, fabric movement, and subtle object handling sounds match the physical environment without overpowering the action.",
        "non_diegetic_music:\n" + bgm,
    ]
    return "\n\n".join(sections)
