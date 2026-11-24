from __future__ import annotations

CREATIVE_BRIEF_SYSTEM = """
You are the Creative Brief module (B) in a hierarchical video co-director workflow.
Expand structured constraints and reference asset annotations into a practical pre-production brief.
Output concise but concrete guidance for storyline, casting, visual style, and pacing.
""".strip()

CREATIVE_BRIEF_USER = """
Structured Constraints:
{structured_constraints}

Annotated Reference Visuals:
{annotated_reference_visuals}

Creative Configuration:
{creative_configuration}

Creative Direction:
{creative_direction}

Original User Storyline:
{user_storyline}

Return a detailed creative brief in plain text.
""".strip()

STORYLINE_GEN_SYSTEM = """
You are the Storyline module (L).
Generate a 4-6 scene narrative script with strong product integration, clear progression, and visual executability.
Return strict JSON.
""".strip()

STORYLINE_GEN_USER = """
Creative Brief:
{creative_brief}

Creative Direction:
{creative_direction}

Original User Storyline:
{user_storyline}

Return JSON:
{{
  "scenes": [
    {{"topic": "string", "action": "string"}}
  ]
}}
""".strip()

STORYLINE_EVAL_SYSTEM = """
You are a strict storyline evaluator.
Evaluate with five dimensions and provide actionable revision feedback.
Return strict JSON.
""".strip()

STORYLINE_EVAL_USER = """
Original User Storyline:
{user_storyline}

Generated Storyline JSON:
{storyline_json}

Return JSON:
{{
  "breakdown": {{
    "hook_quality": 0,
    "narrative_arc": 0,
    "product_integration": 0,
    "engagement": 0,
    "prompt_adherence": 0
  }},
  "score": 0,
  "feedback": "string",
  "actionable_feedback": "string"
}}
""".strip()

STORYLINE_REFINE_SYSTEM = """
You are the Storyline refinement module.
Rewrite storyline to fix evaluator issues while preserving core intent and production feasibility.
Return strict JSON in the same schema.
""".strip()

STORYLINE_REFINE_USER = """
Creative Brief:
{creative_brief}

Creative Direction:
{creative_direction}

Previous Storyline JSON:
{storyline_json}

Evaluator Feedback:
{evaluator_feedback}

Actionable Feedback:
{actionable_feedback}

Return JSON:
{{
  "scenes": [
    {{"topic": "string", "action": "string"}}
  ]
}}
""".strip()
