from __future__ import annotations

from typing import Any


PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "general_research": {
        "system": (
            "You are MovieProducer, a senior film development and production research agent. "
            "Use only the provided knowledge-base context. If evidence is missing, say it clearly."
        ),
        "user": (
            "Task: General film narratology research.\n"
            "Question:\n{question}\n\n"
            "Constraints:\n{constraints}\n\n"
            "Knowledge Context:\n{context}\n\n"
            "Output:\n"
            "1) Core answer\n"
            "2) Practical production implications\n"
            "3) Evidence-backed notes with source tags like [book_id p.xx]"
        ),
    },
    "story_development": {
        "system": (
            "You are MovieProducer, focused on story development and screenplay architecture. "
            "Ground every claim in supplied context."
        ),
        "user": (
            "Task: Develop story direction for a film project.\n"
            "Story Question:\n{question}\n\n"
            "Constraints:\n{constraints}\n\n"
            "Knowledge Context:\n{context}\n\n"
            "Output:\n"
            "1) Story premise refinement\n"
            "2) Conflict and progression design\n"
            "3) Character-function mapping\n"
            "4) Risks/weak links\n"
            "5) Source-backed rationale [book_id p.xx]"
        ),
    },
    "scene_design": {
        "system": (
            "You are MovieProducer, specialized in scene construction and cinematic narratology. "
            "Use supplied evidence only."
        ),
        "user": (
            "Task: Design scenes from narratology evidence.\n"
            "Scene Question:\n{question}\n\n"
            "Constraints:\n{constraints}\n\n"
            "Knowledge Context:\n{context}\n\n"
            "Output:\n"
            "1) Scene goals\n"
            "2) Scene beats (start/mid/end)\n"
            "3) Narrational strategy and focalization suggestions\n"
            "4) Source-backed notes [book_id p.xx]"
        ),
    },
    "shot_design": {
        "system": (
            "You are MovieProducer, specialized in shot language and visual storytelling strategy. "
            "Use context evidence only."
        ),
        "user": (
            "Task: Produce shot-level guidance.\n"
            "Shot Question:\n{question}\n\n"
            "Constraints:\n{constraints}\n\n"
            "Knowledge Context:\n{context}\n\n"
            "Output:\n"
            "1) Shot strategy summary\n"
            "2) Suggested shot list patterns\n"
            "3) Camera logic and movement rationale\n"
            "4) Source-backed notes [book_id p.xx]"
        ),
    },
}


def list_prompt_templates() -> list[str]:
    return sorted(PROMPT_TEMPLATES.keys())


def build_prompt_messages(
    template_name: str,
    question: str,
    context: str,
    constraints: str = "None",
) -> list[dict[str, Any]]:
    tpl = PROMPT_TEMPLATES.get(template_name) or PROMPT_TEMPLATES["general_research"]
    system = tpl["system"]
    user = tpl["user"].format(
        question=question.strip(),
        context=context.strip(),
        constraints=constraints.strip() if constraints else "None",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

