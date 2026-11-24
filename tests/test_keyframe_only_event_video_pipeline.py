from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_keyframe_only_event_video_pipeline import _build_cmd, _summarize_outputs


def _args(**overrides):
    base = dict(
        story_file=Path("story.txt"),
        storyline="",
        characters="Mara,Ilya,Nora",
        output_dir=Path("out"),
        path_mode="both",
        max_steps=15,
        stop_on_complete=False,
        node_selection_mode="top_k",
        node_min_nodes=1,
        node_max_nodes=45,
        node_llm_scoring=True,
        strict_online=True,
        retries=3,
        video_dry_run=False,
        event_shot_count=4,
        event_duration_sec=None,
        max_reference_images=4,
        use_previous_tail_frame=False,
        use_previous_audio=False,
        upload_previous_audio=False,
        dialogue_backfill=True,
        insert_auxiliary_videos=True,
        log_level="INFO",
        mcts_model=None,
        mcts_budget_iters=1,
        mcts_topk_actions=8,
        mcts_cpuct=1.2,
        mcts_rollout_horizon=32,
        mcts_rollout_topk=5,
        mcts_rollout_epsilon=0.2,
        mcts_pw_base=1.5,
        mcts_pw_alpha=0.55,
        pov_switch_penalty=0.05,
        over_focus_threshold=4,
        over_focus_penalty=0.08,
        w_narrative_causality=0.4,
        w_visual_transition=0.3,
        w_rhythmic_pacing=0.3,
        random_seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_cmd_uses_banana_45_cap_keyframe_only_and_keeps_auxiliary_videos():
    cmd = _build_cmd(_args())

    assert cmd[0] == sys.executable
    assert "scripts/run_full_narrative_world_video_pipeline.py" in cmd[1]
    assert cmd[cmd.index("--backend-mode") + 1] == "all_banana"
    assert cmd[cmd.index("--max-steps") + 1] == "15"
    assert "--no-stop-on-complete" in cmd
    assert cmd[cmd.index("--node-min-nodes") + 1] == "1"
    assert cmd[cmd.index("--node-max-nodes") + 1] == "45"
    assert "--node-llm-scoring" in cmd
    assert "--insert-auxiliary-videos" in cmd
    assert cmd[cmd.index("--video-reference-mode") + 1] == "event_keyframe_only"
    assert "--no-use-previous-tail-frame" in cmd
    assert "--event-duration-sec" not in cmd


def test_build_cmd_can_disable_auxiliary_and_force_duration():
    cmd = _build_cmd(_args(insert_auxiliary_videos=False, event_duration_sec=10.0, stop_on_complete=True))

    assert "--stop-on-complete" in cmd
    assert "--no-insert-auxiliary-videos" in cmd
    assert cmd[cmd.index("--event-duration-sec") + 1] == "10.0"
    assert cmd[cmd.index("--video-reference-mode") + 1] == "event_keyframe_only"


def test_summarize_outputs_accepts_auxiliary_semantic_refs_but_requires_event_keyframe_only(tmp_path: Path):
    run_dir = tmp_path / "all_banana"
    path_dir = run_dir / "path_orchestration"
    video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
    path_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    (path_dir / "narrative_nodes.json").write_text(
        json.dumps({"selected_nodes": [{"node_id": "NN_001"}, {"node_id": "NN_002"}]}),
        encoding="utf-8",
    )
    (video_dir / "event_video_tasks.json").write_text(
        json.dumps(
            [
                {"task_type": "establishing", "reference_mode": "semantic_assets"},
                {"task_type": "event", "reference_mode": "event_keyframe_only"},
                {"task_type": "world_transition", "reference_mode": "semantic_assets"},
                {"task_type": "event", "reference_mode": "event_keyframe_only"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (video_dir / "event_video_summary.json").write_text(
        json.dumps({"concat_ok": True, "success": 4, "failed": 0}),
        encoding="utf-8",
    )

    summary = _summarize_outputs(tmp_path)

    assert summary["selected_node_count"] == 2
    assert summary["event_task_count"] == 2
    assert summary["auxiliary_task_count"] == 2
    assert summary["event_reference_mode_ok"] is True
    assert summary["unexpected_event_reference_modes"] == []


def test_summarize_outputs_flags_non_keyframe_event_reference_mode(tmp_path: Path):
    run_dir = tmp_path / "all_banana"
    video_dir = run_dir / "event_video_pipeline_minimax_h3_vllm_multishot"
    video_dir.mkdir(parents=True)
    (video_dir / "event_video_tasks.json").write_text(
        json.dumps([{"task_type": "event", "reference_mode": "all_allowed", "node_id": "NN_001"}]),
        encoding="utf-8",
    )

    summary = _summarize_outputs(tmp_path)

    assert summary["event_reference_mode_ok"] is False
    assert summary["unexpected_event_reference_modes"] == [{"node_id": "NN_001", "reference_mode": "all_allowed"}]


def test_summarize_outputs_reports_path_plan_from_path_manifest(tmp_path: Path):
    path_dir = tmp_path / "all_banana" / "path_orchestration"
    video_dir = tmp_path / "all_banana" / "event_video_pipeline_minimax_h3_vllm_multishot"
    path_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    mcts_plan = path_dir / "path_mcts" / "narrative_path_plan_mcts.json"
    mcts_plan.parent.mkdir(parents=True)
    mcts_plan.write_text(json.dumps({"ordered_node_ids": []}), encoding="utf-8")
    (path_dir / "pipeline_manifest.json").write_text(
        json.dumps({"artifacts": {"mcts_path_json": str(mcts_plan)}}),
        encoding="utf-8",
    )
    (video_dir / "event_video_tasks.json").write_text(json.dumps([]), encoding="utf-8")

    summary = _summarize_outputs(tmp_path)

    assert summary["path_plan_json"] == str(mcts_plan)
