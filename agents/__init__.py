"""Agent pipeline for story -> shots -> video -> post-production generation."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Storyboard",
    "Screenplay",
    "Scene",
    "Shot",
    "DirectorAgent",
    "ShotAgent",
    "VideoAgent",
    "StorySupervisorAgent",
    "PostProductionAgent",
    "InfoPool",
    "MovieProducerAgent",
    "run_pipeline",
]

_LAZY_IMPORTS = {
    "Storyboard": ("agents.schemas", "Storyboard"),
    "Screenplay": ("agents.schemas", "Screenplay"),
    "Scene": ("agents.schemas", "Scene"),
    "Shot": ("agents.schemas", "Shot"),
    "DirectorAgent": ("agents.director_agent", "DirectorAgent"),
    "ShotAgent": ("agents.shot_agent", "ShotAgent"),
    "VideoAgent": ("agents.video_agent", "VideoAgent"),
    "StorySupervisorAgent": ("agents.story_supervisor_agent", "StorySupervisorAgent"),
    "PostProductionAgent": ("agents.post_production_agent", "PostProductionAgent"),
    "InfoPool": ("agents.info_pool", "InfoPool"),
    "MovieProducerAgent": ("agents.movie_producer_agent", "MovieProducerAgent"),
    "run_pipeline": ("agents.orchestrator", "run_pipeline"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
