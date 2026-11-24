#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging
from narrative.simulation_scaffold import OrchestratorAgent, WorldModelAgent


DEFAULT_STORY = (
    "On a stormy coastal night, an investigative journalist returns to Black Tidemark Island after ten years. "
    "The abandoned lighthouse, the flooded underground archive, the cliffside cemetery, the old ferry dock, "
    "and the ruined greenhouse all become active spaces in the search for the truth behind her brother's disappearance. "
    "A cracked brass compass, a salt-stained ledger, a red child's raincoat, a broken voice recorder, a ceremonial key, "
    "and a sealed evidence box are repeatedly mentioned as essential clues. "
    "As the tide rises, every location hides a different fragment of the conspiracy."
)

DEFAULT_CHARACTERS = ["Journalist", "Ferryman", "Caretaker", "Hidden Witness"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the narrative WorldModelAgent only and export comparison assets.")
    parser.add_argument("--storyline", type=str, default=DEFAULT_STORY)
    parser.add_argument("--characters", type=str, default=",".join(DEFAULT_CHARACTERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--image-tools", type=str, default="flux_local,nano_banana")
    parser.add_argument("--flux-width", type=int, default=1280)
    parser.add_argument("--flux-height", type=int, default=720)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    os.environ["NARRATIVE_WORLD_MODEL_IMAGE_TOOLS"] = args.image_tools
    os.environ["FLUX_LOCAL_WIDTH"] = str(int(args.flux_width))
    os.environ["FLUX_LOCAL_HEIGHT"] = str(int(args.flux_height))
    os.environ["FLUX_WIDTH"] = str(int(args.flux_width))
    os.environ["FLUX_HEIGHT"] = str(int(args.flux_height))

    storyline = str(args.storyline).strip()
    character_names = [x.strip() for x in str(args.characters).split(",") if x.strip()]
    if not character_names:
        character_names = list(DEFAULT_CHARACTERS)

    orchestrator = OrchestratorAgent()
    plan = orchestrator.understand_input(storyline, character_names)

    world_model_agent = WorldModelAgent()
    bundle = world_model_agent.build_world_model(
        story_prompt=storyline,
        output_dir=args.output_dir,
    )
    enriched_cfgs = world_model_agent.enrich_character_configs(plan["character_configs"], bundle)
    modeled_prompt = world_model_agent.enrich_story_prompt(storyline, bundle)

    payload = {
        "storyline": storyline,
        "characters": character_names,
        "modeled_prompt": modeled_prompt,
        "image_tools": args.image_tools.split(","),
        "flux_size": {
            "width": int(args.flux_width),
            "height": int(args.flux_height),
        },
        "orchestrator_character_configs": [
            {
                "name": cfg.name,
                "final_goal": cfg.final_goal,
                "role_profile": cfg.role_profile,
                "known_relations": list(cfg.known_relations),
                "current_environment": cfg.current_environment,
                "prompt_arm": cfg.prompt_arm,
                "dynamic_system_prompt": cfg.dynamic_system_prompt,
            }
            for cfg in enriched_cfgs
        ],
        "world_model": bundle,
    }

    output_json = args.output_json or (args.output_dir / "world_model_result.json")
    _write_json(output_json, payload)
    print(output_json)


if __name__ == "__main__":
    main()
