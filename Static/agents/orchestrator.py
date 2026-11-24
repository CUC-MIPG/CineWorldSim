from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .director_agent import DirectorAgent
from .shot_agent import ShotAgent
from .video_agent import VideoAgent
from .story_supervisor_agent import StorySupervisorAgent
from .post_production_agent import PostProductionAgent
from .info_pool import InfoPool
from .schemas import Storyboard

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    storyline: str
    prompt_library: str
    output_root: str
    max_scenes: int
    max_shots_per_scene: int
    dry_run_video: bool
    strict_online: bool
    retries: int
    scene_image_policy: str
    character_image_policy: str
    video_mode: str
    reference_assets_json: str
    character_image_dir: str
    scene_image_dir: str
    run_id: str
    info_pool_dir: str
    enable_story_supervisor: bool
    enable_post_production: bool
    include_narration: bool

    storyboard: dict
    story_supervisor: dict
    post_production: dict
    stages_done: list[str]


def _stage_done(state: PipelineState, name: str) -> bool:
    return name in state.get("stages_done", [])


def _mark_stage(state: PipelineState, name: str) -> PipelineState:
    done = list(state.get("stages_done", []))
    if name not in done:
        done.append(name)
    return {"stages_done": done}


def _save_checkpoint(output_root: Path, state: PipelineState) -> None:
    ckpt = output_root / "pipeline_state.json"
    ckpt.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def _get_info_pool(state: PipelineState) -> InfoPool:
    if state.get("info_pool_dir"):
        pool_dir = Path(str(state["info_pool_dir"]))
    else:
        pool_dir = Path(state["output_root"]) / "info_pool"
    return InfoPool(pool_dir)


def _record_agent_output(
    state: PipelineState,
    *,
    stage_name: str,
    agent_name: str,
    output: dict,
) -> None:
    run_id = str(state.get("run_id", "unknown_run"))
    _get_info_pool(state).record_agent_output(
        run_id=run_id,
        stage_name=stage_name,
        agent_name=agent_name,
        output=output,
    )


def _load_storyboard_from_state(state: PipelineState) -> Storyboard | None:
    data = state.get("storyboard")
    if not data:
        return None
    return Storyboard.from_dict(data)


def _apply_max_scenes(storyboard: Storyboard, max_scenes: int | None) -> Storyboard:
    if not max_scenes or max_scenes <= 0:
        return storyboard
    sorted_ids = sorted(storyboard.scenes.keys())
    keep_ids = set(sorted_ids[:max_scenes])

    storyboard.scenes = {
        sid: scene
        for sid, scene in sorted(storyboard.scenes.items(), key=lambda x: x[0])
        if sid in keep_ids
    }
    storyboard.shots = {
        shot_id: shot
        for shot_id, shot in storyboard.shots.items()
        if shot.scene_id in keep_ids
    }
    return storyboard


def _director_node(state: PipelineState) -> PipelineState:
    if _stage_done(state, "director"):
        return {}
    t0 = time.time()
    logger.info("stage=director status=start")

    prompt_library = Path(state["prompt_library"])
    output_root = Path(state["output_root"])
    strict_online = bool(state.get("strict_online", True))
    retries = int(state.get("retries", 3))

    director = DirectorAgent(
        prompt_library=prompt_library,
        strict_online=strict_online,
        retries=retries,
    )

    storyboard = director.build_storyboard(state["storyline"])
    storyboard = _apply_max_scenes(storyboard, int(state.get("max_scenes", 0) or 0))
    reference_assets_json = state.get("reference_assets_json")
    character_image_dir = state.get("character_image_dir")
    scene_image_dir = state.get("scene_image_dir")
    if reference_assets_json:
        storyboard = director.attach_reference_assets_json(
            storyboard,
            reference_assets_json=Path(reference_assets_json),
        )
    # Backward compatibility with old directory args.
    if character_image_dir:
        storyboard = director.attach_character_reference_images(
            storyboard,
            character_image_dir=Path(character_image_dir),
        )
    if scene_image_dir:
        storyboard = director.attach_scene_reference_images(
            storyboard,
            scene_image_dir=Path(scene_image_dir),
        )

    character_image_out_dir = output_root / "characters"
    storyboard = director.ensure_character_images(
        storyboard,
        image_dir=character_image_out_dir,
        image_policy=str(state.get("character_image_policy", "auto")),
    )

    image_dir = output_root / "images"
    storyboard = director.ensure_scene_images(
        storyboard,
        image_dir=image_dir,
        image_policy=str(state.get("scene_image_policy", "auto")),
    )

    out = {
        "storyboard": storyboard.to_dict(),
        **_mark_stage(state, "director"),
    }
    new_state = {**state, **out}
    _save_checkpoint(output_root, new_state)
    _record_agent_output(
        state,
        stage_name="director",
        agent_name="DirectorAgent",
        output={
            "storyboard": out["storyboard"],
            "output_artifacts": {
                "characters_dir": str(character_image_out_dir),
                "images_dir": str(image_dir),
            },
        },
    )
    logger.info(
        "stage=director status=done scenes=%d shots=%d elapsed_sec=%.2f",
        len(storyboard.scenes),
        len(storyboard.shots),
        time.time() - t0,
    )
    return out


def _shot_node(state: PipelineState) -> PipelineState:
    if _stage_done(state, "shot"):
        return {}
    t0 = time.time()
    logger.info("stage=shot status=start")

    storyboard = _load_storyboard_from_state(state)
    if storyboard is None:
        raise RuntimeError("Shot stage requires storyboard from director stage")
    storyboard = _apply_max_scenes(storyboard, int(state.get("max_scenes", 0) or 0))

    prompt_library = Path(state["prompt_library"])
    output_root = Path(state["output_root"])
    strict_online = bool(state.get("strict_online", True))
    retries = int(state.get("retries", 3))

    shot_agent = ShotAgent(
        prompt_library=prompt_library,
        strict_online=strict_online,
        retries=retries,
    )
    storyboard = shot_agent.loop_storyboard_generation(
        storyboard,
        max_shots_per_scene=int(state.get("max_shots_per_scene", 8)),
    )

    out = {
        "storyboard": storyboard.to_dict(),
        **_mark_stage(state, "shot"),
    }
    new_state = {**state, **out}
    _save_checkpoint(output_root, new_state)
    _record_agent_output(
        state,
        stage_name="shot",
        agent_name="ShotAgent",
        output={"storyboard": out["storyboard"]},
    )
    logger.info(
        "stage=shot status=done scenes=%d shots=%d elapsed_sec=%.2f",
        len(storyboard.scenes),
        len(storyboard.shots),
        time.time() - t0,
    )
    return out


def _story_supervisor_node(state: PipelineState) -> PipelineState:
    if _stage_done(state, "story_supervisor"):
        return {}
    t0 = time.time()
    logger.info("stage=story_supervisor status=start")

    output_root = Path(state["output_root"])
    if not bool(state.get("enable_story_supervisor", True)):
        out = {
            "story_supervisor": {
                "enabled": False,
                "reason": "enable_story_supervisor is false",
            },
            **_mark_stage(state, "story_supervisor"),
        }
        new_state = {**state, **out}
        _save_checkpoint(output_root, new_state)
        _record_agent_output(
            state,
            stage_name="story_supervisor",
            agent_name="StorySupervisorAgent",
            output=out["story_supervisor"],
        )
        logger.info("stage=story_supervisor status=skipped reason=disabled elapsed_sec=%.2f", time.time() - t0)
        return out

    storyboard = _load_storyboard_from_state(state)
    if storyboard is None:
        raise RuntimeError("Story supervisor stage requires storyboard from previous stages")
    storyboard = _apply_max_scenes(storyboard, int(state.get("max_scenes", 0) or 0))

    strict_online = bool(state.get("strict_online", True))
    retries = int(state.get("retries", 3))
    supervisor = StorySupervisorAgent(strict_online=strict_online, retries=retries)
    summary = supervisor.generate_story_plan(
        storyboard=storyboard,
        output_dir=output_root / "story_supervisor",
    )
    out = {
        "story_supervisor": summary,
        **_mark_stage(state, "story_supervisor"),
    }
    new_state = {**state, **out}
    _save_checkpoint(output_root, new_state)
    _record_agent_output(
        state,
        stage_name="story_supervisor",
        agent_name="StorySupervisorAgent",
        output=summary,
    )
    logger.info(
        "stage=story_supervisor status=done scene_count=%d elapsed_sec=%.2f",
        len(summary.get("scene_plans", [])) if isinstance(summary, dict) else -1,
        time.time() - t0,
    )
    return out


def _video_node(state: PipelineState) -> PipelineState:
    if _stage_done(state, "video"):
        return {}
    t0 = time.time()
    logger.info("stage=video status=start")

    storyboard = _load_storyboard_from_state(state)
    if storyboard is None:
        raise RuntimeError("Video stage requires storyboard from previous stages")
    storyboard = _apply_max_scenes(storyboard, int(state.get("max_scenes", 0) or 0))

    output_root = Path(state["output_root"])
    strict_online = bool(state.get("strict_online", True))
    retries = int(state.get("retries", 3))

    video_agent = VideoAgent(strict_online=strict_online, retries=retries)

    video_dir = output_root / "videos"
    storyboard = video_agent.process_shots(
        storyboard,
        output_dir=video_dir,
        dry_run=bool(state.get("dry_run_video", True)),
        save_task_json=True,
        video_mode=str(state.get("video_mode", "reference")),
    )

    ordered_shots = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
    video_paths = [s.video_path for s in ordered_shots if s.video_path]
    video_agent.concat_videos(video_paths, output_root / "final_concat.mp4")

    storyboard.save_json(output_root / "storyboard.json")

    out = {
        "storyboard": storyboard.to_dict(),
        **_mark_stage(state, "video"),
    }
    new_state = {**state, **out}
    _save_checkpoint(output_root, new_state)
    _record_agent_output(
        state,
        stage_name="video",
        agent_name="VideoAgent",
        output={
            "storyboard": out["storyboard"],
            "output_artifacts": {
                "video_task_json": str(video_dir / "video_info.json"),
                "final_concat": str(output_root / "final_concat.mp4"),
            },
        },
    )
    logger.info(
        "stage=video status=done assigned_video_paths=%d first_shot=%s last_shot=%s elapsed_sec=%.2f",
        len(video_paths),
        ordered_shots[0].shot_id if ordered_shots else "",
        ordered_shots[-1].shot_id if ordered_shots else "",
        time.time() - t0,
    )
    return out


def _post_production_node(state: PipelineState) -> PipelineState:
    if _stage_done(state, "post_production"):
        return {}
    t0 = time.time()
    logger.info("stage=post_production status=start")

    output_root = Path(state["output_root"])
    if not bool(state.get("enable_post_production", False)):
        out = {
            "post_production": {
                "enabled": False,
                "reason": "enable_post_production is false",
            },
            **_mark_stage(state, "post_production"),
        }
        new_state = {**state, **out}
        _save_checkpoint(output_root, new_state)
        _record_agent_output(
            state,
            stage_name="post_production",
            agent_name="PostProductionAgent",
            output=out["post_production"],
        )
        logger.info("stage=post_production status=skipped reason=disabled elapsed_sec=%.2f", time.time() - t0)
        return out

    storyboard = _load_storyboard_from_state(state)
    if storyboard is None:
        raise RuntimeError("Post-production stage requires storyboard from previous stages")
    storyboard = _apply_max_scenes(storyboard, int(state.get("max_scenes", 0) or 0))

    post_agent = PostProductionAgent()
    post_dir = output_root / "post_production"
    final_video = output_root / "final_concat.mp4"
    summary = post_agent.process(
        storyboard=storyboard,
        output_dir=post_dir,
        final_video_path=str(final_video) if final_video.exists() else "",
        supervision_plan=state.get("story_supervisor"),
        include_narration=bool(state.get("include_narration", True)),
    )

    out = {
        "post_production": summary,
        **_mark_stage(state, "post_production"),
    }
    new_state = {**state, **out}
    _save_checkpoint(output_root, new_state)
    _record_agent_output(
        state,
        stage_name="post_production",
        agent_name="PostProductionAgent",
        output=summary,
    )
    logger.info(
        "stage=post_production status=done scene_count=%d elapsed_sec=%.2f",
        int(summary.get("scene_count", -1)) if isinstance(summary, dict) else -1,
        time.time() - t0,
    )
    return out


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("director", _director_node)
    graph.add_node("shot", _shot_node)
    graph.add_node("story_supervisor", _story_supervisor_node)
    graph.add_node("video", _video_node)
    graph.add_node("post_production", _post_production_node)
    graph.add_edge(START, "director")
    graph.add_edge("director", "shot")
    graph.add_edge("shot", "story_supervisor")
    graph.add_edge("story_supervisor", "video")
    graph.add_edge("video", "post_production")
    graph.add_edge("post_production", END)
    return graph.compile()


def run_pipeline(
    storyline: str,
    prompt_library: Path,
    output_root: Path,
    max_scenes: int | None = None,
    max_shots_per_scene: int = 8,
    dry_run_video: bool = True,
    strict_online: bool = True,
    retries: int = 3,
    resume: bool = True,
    scene_image_policy: str = "auto",
    character_image_policy: str = "auto",
    video_mode: str = "reference",
    reference_assets_json: Path | None = None,
    character_image_dir: Path | None = None,
    scene_image_dir: Path | None = None,
    enable_story_supervisor: bool = True,
    enable_post_production: bool = False,
    include_narration: bool = True,
) -> Storyboard:
    t0 = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt = output_root / "pipeline_state.json"

    if resume and ckpt.exists():
        state = json.loads(ckpt.read_text(encoding="utf-8"))
        if not state.get("prompt_library") and state.get("cameraatist_dir"):
            state["prompt_library"] = state["cameraatist_dir"]
        # Runtime flags can be overridden on resume.
        state["dry_run_video"] = dry_run_video
        state["strict_online"] = strict_online
        state["retries"] = retries
        state["max_scenes"] = int(max_scenes or 0)
        state["scene_image_policy"] = scene_image_policy
        state["character_image_policy"] = character_image_policy
        state["video_mode"] = video_mode
        if reference_assets_json is not None:
            state["reference_assets_json"] = str(reference_assets_json)
        if character_image_dir is not None:
            state["character_image_dir"] = str(character_image_dir)
        if scene_image_dir is not None:
            state["scene_image_dir"] = str(scene_image_dir)
        state["enable_story_supervisor"] = enable_story_supervisor
        state["enable_post_production"] = enable_post_production
        state["include_narration"] = include_narration
        if not state.get("run_id"):
            state["run_id"] = _new_run_id()
        if not state.get("info_pool_dir"):
            state["info_pool_dir"] = str(output_root / "info_pool")
        done = list(state.get("stages_done", []))
        # Migration safety:
        # if post-production was done before story-supervisor stage existed,
        # re-run post-production so it can consume the new supervision plan.
        if (
            enable_post_production
            and enable_story_supervisor
            and ("story_supervisor" not in done)
            and ("post_production" in done)
        ):
            done = [x for x in done if x != "post_production"]
            state["stages_done"] = done
    else:
        state = {
            "storyline": storyline,
            "prompt_library": str(prompt_library),
            "output_root": str(output_root),
            "max_scenes": int(max_scenes or 0),
            "max_shots_per_scene": max_shots_per_scene,
            "dry_run_video": dry_run_video,
            "strict_online": strict_online,
            "retries": retries,
            "scene_image_policy": scene_image_policy,
            "character_image_policy": character_image_policy,
            "video_mode": video_mode,
            "reference_assets_json": str(reference_assets_json) if reference_assets_json else "",
            "character_image_dir": str(character_image_dir) if character_image_dir else "",
            "scene_image_dir": str(scene_image_dir) if scene_image_dir else "",
            "run_id": _new_run_id(),
            "info_pool_dir": str(output_root / "info_pool"),
            "enable_story_supervisor": enable_story_supervisor,
            "enable_post_production": enable_post_production,
            "include_narration": include_narration,
            "stages_done": [],
        }

    _save_checkpoint(output_root, state)
    logger.info(
        "pipeline status=start output_root=%s resume=%s max_scenes=%s max_shots_per_scene=%s dry_run_video=%s video_mode=%s",
        str(output_root),
        resume,
        max_scenes,
        max_shots_per_scene,
        dry_run_video,
        video_mode,
    )
    run_id = str(state.get("run_id", "unknown_run"))
    _get_info_pool(state).record_pipeline_event(
        run_id=run_id,
        event_type="pipeline_start",
        payload={
            "storyline": storyline,
            "output_root": str(output_root),
            "prompt_library": str(prompt_library),
            "resume": resume,
            "enable_story_supervisor": enable_story_supervisor,
            "enable_post_production": enable_post_production,
            "include_narration": include_narration,
            "reference_assets_json": str(state.get("reference_assets_json", "")),
        },
    )

    app = build_graph()
    final_state = app.invoke(state)

    storyboard = _load_storyboard_from_state(final_state)
    if storyboard is None:
        raise RuntimeError("Pipeline completed without storyboard")

    _get_info_pool(final_state).record_pipeline_event(
        run_id=run_id,
        event_type="pipeline_complete",
        payload={
            "title": storyboard.title,
            "scene_count": len(storyboard.scenes),
            "shot_count": len(storyboard.shots),
            "stages_done": final_state.get("stages_done", []),
            "storyboard_json": str(output_root / "storyboard.json"),
        },
    )
    logger.info(
        "pipeline status=done title=%s scenes=%d shots=%d elapsed_sec=%.2f",
        storyboard.title,
        len(storyboard.scenes),
        len(storyboard.shots),
        time.time() - t0,
    )
    return storyboard
