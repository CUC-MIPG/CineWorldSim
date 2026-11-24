from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.logging_utils import configure_logging
from agents.post_production_agent import PostProductionAgent
from agents.schemas import Storyboard
from agents.story_supervisor_agent import StorySupervisorAgent
from agents.video_agent import VideoAgent

from .director_agent_dynamic import DirectorAgentDynamic
from .shot_agent_dynamic import ShotAgentDynamic

logger = logging.getLogger(__name__)


def _apply_max_scenes(storyboard: Storyboard, max_scenes: int | None) -> Storyboard:
    if not max_scenes or max_scenes <= 0:
        return storyboard
    keep = set(sorted(storyboard.scenes.keys())[: int(max_scenes)])
    storyboard.scenes = {k: v for k, v in storyboard.scenes.items() if k in keep}
    storyboard.shots = {k: v for k, v in storyboard.shots.items() if v.scene_id in keep}
    return storyboard


def _reuse_existing_shot_videos(storyboard: Storyboard, reuse_video_dir: Path) -> tuple[int, int]:
    hit = 0
    miss = 0
    for shot in storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True):
        p = reuse_video_dir / f"video_{shot.shot_id}.mp4"
        if p.exists() and p.stat().st_size > 0:
            shot.video_path = str(p)
            hit += 1
        else:
            miss += 1
    return hit, miss


def run_dynamic_pipeline(
    *,
    storyline: str,
    prompt_library: Path,
    output_root: Path,
    max_scenes: int | None = None,
    max_shots_per_scene: int = 8,
    dry_run_video: bool = True,
    strict_online: bool = True,
    retries: int = 3,
    scene_image_policy: str = "auto",
    character_image_policy: str = "auto",
    video_mode: str = "reference",
    reference_assets_json: Path | None = None,
    reuse_video_dir: Path | None = None,
    enable_story_supervisor: bool = True,
    enable_post_production: bool = True,
    include_narration: bool = True,
    max_storyline_refine_attempts: int = 2,
    storyline_score_threshold: int = 85,
) -> Storyboard:
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("dynamic_pipeline status=start output=%s", str(output_root))

    director = DirectorAgentDynamic(
        prompt_library=prompt_library,
        strict_online=strict_online,
        retries=retries,
    )
    storyboard, pre_bundle = director.run_preproduction(
        storyline=storyline,
        reference_assets_json=reference_assets_json,
        max_storyline_refine_attempts=max_storyline_refine_attempts,
        storyline_score_threshold=storyline_score_threshold,
    )
    storyboard = _apply_max_scenes(storyboard, max_scenes)

    # Keep compatibility with existing generation tools.
    char_dir = output_root / "characters"
    scene_dir = output_root / "images"
    storyboard = director.base_director.ensure_character_images(
        storyboard,
        image_dir=char_dir,
        image_policy=character_image_policy,
    )
    storyboard = director.base_director.ensure_scene_images(
        storyboard,
        image_dir=scene_dir,
        image_policy=scene_image_policy,
    )

    shot_agent = ShotAgentDynamic(
        prompt_library=prompt_library,
        strict_online=strict_online,
        retries=retries,
    )
    storyboard = shot_agent.loop_storyboard_generation(
        storyboard,
        max_shots_per_scene=max_shots_per_scene,
        creative_direction=pre_bundle.get("creative_direction", {}),
    )

    # Reuse existing shot videos if provided.
    if reuse_video_dir and reuse_video_dir.exists():
        hit, miss = _reuse_existing_shot_videos(storyboard, reuse_video_dir)
        logger.info("dynamic_pipeline reuse_video status=done hit=%d miss=%d dir=%s", hit, miss, str(reuse_video_dir))

    video_agent = VideoAgent(strict_online=strict_online, retries=retries)
    video_dir = output_root / "videos"

    ordered = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
    all_reused = bool(ordered) and all(s.video_path and Path(s.video_path).exists() for s in ordered)

    if all_reused:
        logger.info("dynamic_pipeline video stage=skip reason=all_reused")
    else:
        storyboard = video_agent.process_shots(
            storyboard,
            output_dir=video_dir,
            dry_run=dry_run_video,
            save_task_json=True,
            video_mode=video_mode,
        )

    ordered = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
    shot_video_paths = [s.video_path for s in ordered if s.video_path and Path(s.video_path).exists()]
    final_concat = output_root / "final_concat.mp4"
    if shot_video_paths:
        video_agent.concat_videos(shot_video_paths, final_concat)

    story_supervisor_summary: dict[str, Any] | None = None
    if enable_story_supervisor:
        sup = StorySupervisorAgent(strict_online=strict_online, retries=retries)
        story_supervisor_summary = sup.generate_story_plan(
            storyboard=storyboard,
            output_dir=output_root / "story_supervisor",
        )

    post_summary: dict[str, Any] | None = None
    if enable_post_production:
        post_agent = PostProductionAgent()
        post_summary = post_agent.process(
            storyboard=storyboard,
            output_dir=output_root / "post_production",
            final_video_path=str(final_concat) if final_concat.exists() else "",
            supervision_plan=story_supervisor_summary,
            include_narration=include_narration,
        )

    storyboard.save_json(output_root / "storyboard.json")
    (output_root / "dynamic_preproduction_bundle.json").write_text(
        json.dumps(pre_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "title": storyboard.title,
        "scene_count": len(storyboard.scenes),
        "shot_count": len(storyboard.shots),
        "final_concat": str(final_concat) if final_concat.exists() else "",
        "story_supervisor_enabled": bool(enable_story_supervisor),
        "post_production_enabled": bool(enable_post_production),
        "post_production_info": str(output_root / "post_production" / "post_production_info.json") if post_summary else "",
    }
    (output_root / "dynamic_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("dynamic_pipeline status=done scenes=%d shots=%d", len(storyboard.scenes), len(storyboard.shots))
    return storyboard


__all__ = ["run_dynamic_pipeline", "configure_logging"]
