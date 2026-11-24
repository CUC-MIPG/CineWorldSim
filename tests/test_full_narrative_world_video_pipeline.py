from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_narrative_world_video_pipeline import run_pipeline


def test_run_pipeline_orchestrates_world_path_and_video(tmp_path, monkeypatch):
    world_run_dir = tmp_path / "world" / "all_banana"
    world_run_dir.mkdir(parents=True)
    (world_run_dir / "pipeline_result.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    calls: list[str] = []

    def fake_run_world_stage(args, story, characters):
        calls.append("world")
        del args, story, characters
        return [world_run_dir]

    def fake_run_path_for_result(*, pipeline_result_json, storyline, output_dir, args):
        calls.append("path")
        del pipeline_result_json, storyline, args
        output_dir.mkdir(parents=True, exist_ok=True)
        narrative_nodes = output_dir / "narrative_nodes.json"
        narrative_nodes.write_text(
            json.dumps({"selected_nodes": [{"node_id": "NN_001", "characters": ["Hero"]}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        path_plan = output_dir / "path_llm_direct" / "narrative_path_plan.json"
        path_plan.parent.mkdir(parents=True, exist_ok=True)
        path_plan.write_text(json.dumps({"ordered_node_ids": ["NN_001"]}, ensure_ascii=False), encoding="utf-8")
        manifest = {"artifacts": {"narrative_nodes_json": str(narrative_nodes), "llm_path_json": str(path_plan)}}
        (output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return manifest

    class FakeVideoAgent:
        concat_calls: list[tuple[list[str], Path]] = []

        def __init__(self, **kwargs):
            del kwargs

        def process_orchestrated_events(self, **kwargs):
            calls.append("video")
            assert Path(kwargs["narrative_nodes_json"]).exists()
            assert Path(kwargs["path_plan_json"]).exists()
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            event_video = out / "event_video_001_NN_001.mp4"
            event_video.write_bytes(b"video")
            (out / "event_video_summary.json").write_text(
                json.dumps({"total": 1, "success": 1, "failed": 0, "skipped": 0}, ensure_ascii=False),
                encoding="utf-8",
            )
            (out / "event_video_tasks.json").write_text(json.dumps([{"node_id": "NN_001"}], ensure_ascii=False), encoding="utf-8")
            return {
                "total": 1,
                "success": 1,
                "failed": 0,
                "skipped": 0,
                "generated": [
                    {
                        "node_id": "NN_001",
                        "status": "done",
                        "output_video_path": str(event_video),
                    }
                ],
            }

        def concat_videos(self, video_paths, output_path):
            self.concat_calls.append((list(video_paths), Path(output_path)))
            Path(output_path).write_bytes(b"final video")
            return True

    monkeypatch.setattr("scripts.run_full_narrative_world_video_pipeline._run_world_stage", fake_run_world_stage)
    monkeypatch.setattr("scripts.run_full_narrative_world_video_pipeline._run_path_for_result", fake_run_path_for_result)
    monkeypatch.setattr("scripts.run_full_narrative_world_video_pipeline.VideoAgent", FakeVideoAgent)

    args = argparse.Namespace(
        story_file=None,
        storyline="A test story.",
        characters="Hero",
        backend_mode="all_banana",
        run_both=False,
        max_steps=3,
        stop_on_complete=True,
        output_dir=tmp_path / "out",
        dry_run=False,
        path_mode="llm",
        enable_narrative_node_selector=True,
        node_selection_mode="hybrid",
        node_score_threshold=0.55,
        node_min_nodes=1,
        node_max_nodes=5,
        node_llm_scoring=False,
        strict_online=True,
        retries=1,
        video_dry_run=False,
        max_reference_images=4,
        multi_shot_events=True,
        event_shot_count=4,
        event_duration_sec=8.0,
        use_previous_tail_frame=False,
        use_previous_audio=False,
        upload_previous_audio=False,
        dialogue_backfill=True,
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

    manifest = run_pipeline(args)

    assert calls == ["world", "path", "video"]
    assert manifest["runs"][0]["video_success"] == 1
    assert FakeVideoAgent.concat_calls == [
        (
            [str(world_run_dir / "event_video_pipeline_minimax_h3_vllm_multishot" / "event_video_001_NN_001.mp4")],
            world_run_dir / "event_video_pipeline_minimax_h3_vllm_multishot" / "final_orchestrated_events.mp4",
        )
    ]
    assert manifest["runs"][0]["concat_ok"] is True
    assert (args.output_dir / "full_pipeline_manifest.json").exists()
