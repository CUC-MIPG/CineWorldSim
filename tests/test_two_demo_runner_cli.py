from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_two_adventure_demo_pipelines import _build_cmd
from scripts.resume_two_adventure_demo_pipelines import _demo2_full_cmd


def _args(**overrides):
    base = dict(
        backend_mode="all_banana",
        path_mode="both",
        max_steps=15,
        node_selection_mode="top_k",
        node_min_nodes=15,
        node_max_nodes=15,
        event_shot_count=4,
        event_duration_sec=None,
        max_reference_images=4,
        log_level="INFO",
        use_previous_tail_frame=False,
        dialogue_backfill=True,
        insert_auxiliary_videos=True,
        video_reference_mode="all_allowed",
        output_root=Path("/tmp/cameramaster_out"),
        cuda_visible_devices="2,3",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_two_demo_runner_defaults_allow_video_agent_duration_planning():
    cmd = _build_cmd(_args(), Path("story.txt"), Path("out"), "Mara,Ilya,Nora")

    assert "--event-duration-sec" not in cmd
    assert "--insert-auxiliary-videos" in cmd
    assert cmd[cmd.index("--video-reference-mode") + 1] == "all_allowed"


def test_two_demo_runner_can_still_force_duration_when_requested():
    cmd = _build_cmd(_args(event_duration_sec=10.0), Path("story.txt"), Path("out"), "Mara,Ilya,Nora")

    assert cmd[cmd.index("--event-duration-sec") + 1] == "10.0"


def test_resume_demo2_uses_new_video_reference_flags():
    cmd = _demo2_full_cmd(_args())

    assert "--insert-auxiliary-videos" in cmd
    assert cmd[cmd.index("--video-reference-mode") + 1] == "all_allowed"
    assert "--event-duration-sec" not in cmd
