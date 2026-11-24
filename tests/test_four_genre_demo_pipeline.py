from __future__ import annotations

import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_four_genre_demo_pipeline import (
    DEMOS,
    build_pipeline_command,
    build_resume_path_command,
    build_video_command,
    collect_missing_pipeline_assets,
    collect_missing_video_assets,
    prepare_child_env,
    run_command,
)


def test_four_genre_demo_definitions_have_unique_role_counts():
    genres = [demo["genre"] for demo in DEMOS]
    role_counts = [len(demo["characters"]) for demo in DEMOS]

    assert genres == ["suspense", "adventure", "action", "urban"]
    assert role_counts == [2, 3, 4, 5]
    assert len(set(role_counts)) == 4
    for demo in DEMOS:
        assert demo["storyline"].strip()
        assert all(str(name).strip() for name in demo["characters"])


def test_batch_runner_builds_full_pipeline_and_multishot_video_commands(tmp_path):
    demo = DEMOS[0]
    pipeline_cmd = build_pipeline_command(
        demo=demo,
        output_root=tmp_path,
        backend_mode="all_banana",
        max_steps=10,
        dry_run=True,
        path_mode="llm",
        node_llm_scoring=True,
        strict_online=False,
        retries=2,
        log_level="INFO",
    )
    joined_pipeline = " ".join(str(x) for x in pipeline_cmd)

    assert "scripts/run_narrative_world_path_pipeline.py" in joined_pipeline
    assert "--storyline" in pipeline_cmd
    assert demo["storyline"] in pipeline_cmd
    assert "--characters" in pipeline_cmd
    assert ",".join(demo["characters"]) in pipeline_cmd
    assert "--dry-run" in pipeline_cmd
    assert "--node-llm-scoring" in pipeline_cmd
    assert "--no-strict-online" in pipeline_cmd

    run_dir = tmp_path / f"01_{demo['slug']}" / "pipeline" / "all_banana"
    video_cmd = build_video_command(
        run_dir=run_dir,
        output_dir=run_dir / "event_video_pipeline_minimax_h3_vllm_multishot",
        dry_run=True,
        max_reference_images=4,
        multi_shot_events=True,
        event_shot_count=4,
        event_duration_sec=8.0,
        use_previous_tail_frame=True,
        use_previous_audio=True,
        upload_previous_audio=False,
        cuda_visible_devices="2,3",
        log_level="INFO",
    )
    joined_video = " ".join(str(x) for x in video_cmd)

    assert "scripts/run_orchestrated_event_video_generation.py" in joined_video
    assert "--pipeline-run-dir" in video_cmd
    assert "--multi-shot-events" in video_cmd
    assert "--event-shot-count" in video_cmd
    assert "4" in video_cmd
    assert "--event-duration-sec" in video_cmd
    assert "8.0" in video_cmd
    assert "--use-previous-tail-frame" in video_cmd
    assert "--use-previous-audio" in video_cmd
    assert "--no-upload-previous-audio" in video_cmd
    assert "--cuda-visible-devices" in video_cmd
    assert "2,3" in video_cmd


def test_resume_path_command_uses_existing_pipeline_result_json(tmp_path):
    demo = DEMOS[0]
    pipeline_result_json = tmp_path / "pipeline" / "all_banana" / "pipeline_result.json"
    output_dir = tmp_path / "pipeline" / "all_banana" / "path_orchestration"

    cmd = build_resume_path_command(
        demo=demo,
        pipeline_result_json=pipeline_result_json,
        output_dir=output_dir,
        path_mode="llm",
        node_llm_scoring=True,
        strict_online=False,
        retries=2,
        log_level="INFO",
    )

    joined = " ".join(str(x) for x in cmd)
    assert "scripts/run_narrative_world_path_pipeline.py" in joined
    assert "--existing-pipeline-result-json" in cmd
    assert str(pipeline_result_json) in cmd
    assert "--output-dir" in cmd
    assert str(output_dir) in cmd
    assert demo["storyline"] in cmd
    assert "--node-llm-scoring" in cmd
    assert "--no-strict-online" in cmd


def test_batch_runner_can_request_fifteen_narrative_nodes(tmp_path):
    demo = DEMOS[1]
    cmd = build_pipeline_command(
        demo=demo,
        output_root=tmp_path,
        backend_mode="all_banana",
        max_steps=15,
        dry_run=False,
        path_mode="llm",
        stop_on_complete=False,
        node_selection_mode="top_k",
        node_score_threshold=0.0,
        node_min_nodes=15,
        node_max_nodes=15,
        node_llm_scoring=True,
        strict_online=True,
        retries=2,
        log_level="INFO",
    )

    assert "--max-steps" in cmd
    assert "15" == cmd[cmd.index("--max-steps") + 1]
    assert "--no-stop-on-complete" in cmd
    assert "--node-selection-mode" in cmd
    assert "top_k" == cmd[cmd.index("--node-selection-mode") + 1]
    assert "--node-score-threshold" in cmd
    assert "0.0" == cmd[cmd.index("--node-score-threshold") + 1]
    assert "--node-min-nodes" in cmd
    assert "15" == cmd[cmd.index("--node-min-nodes") + 1]
    assert "--node-max-nodes" in cmd
    assert "15" == cmd[cmd.index("--node-max-nodes") + 1]


def test_prepare_child_env_sets_unbuffered_and_call_log(tmp_path):
    call_log = tmp_path / "logs" / "model_tool_calls.jsonl"
    base_env = {
        "CAMERAMASTER_SKILL_ROOTS": "/tmp/skills",
        "CAMERAMASTER_SKILL_POLICY_PATH": "/tmp/policy.yaml",
        "DIRECTOR_SKILL_NAME": "director-skill",
        "SHOT_SKILL_NAME": "shot-skill",
        "SUPERVISOR_SKILL_NAME": "supervisor-skill",
        "VIDEO_SKILL_NAME": "video-skill",
    }

    env = prepare_child_env(base_env=base_env, call_log_jsonl=call_log)

    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CAMERAMASTER_CALL_LOG_JSONL"] == str(call_log)
    assert env["CAMERAMASTER_RUN_LOG_DIR"] == str(call_log.parent)
    assert env["MINIMAX_H3_PROMPT_STYLE"] == "official_ref"
    for key, value in base_env.items():
        assert env[key] == value


def test_run_command_streams_stdout_and_stderr_to_log_file(tmp_path):
    log_file = tmp_path / "logs" / "stage.log"
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('stdout-line', flush=True); print('stderr-line', file=sys.stderr, flush=True)",
    ]

    rc = run_command(cmd, env=os.environ.copy(), log_file=log_file, stage_name="unit")

    assert rc == 0
    text = log_file.read_text(encoding="utf-8")
    assert "[STAGE START] unit" in text
    assert "stdout-line" in text
    assert "[stderr] stderr-line" in text
    assert "[STAGE DONE] unit rc=0" in text


def test_collect_missing_pipeline_assets_detects_failed_and_missing_outputs(tmp_path):
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"png")
    result_json = tmp_path / "pipeline_result.json"
    result_json.write_text(
        json.dumps(
            {
                "world_model": {
                    "locations": [
                        {"images": [{"ok": True, "output_path": str(existing)}]},
                        {"images": [{"ok": False, "output_path": str(tmp_path / "failed.png")}]},
                    ],
                    "items": [
                        {"images": [{"ok": True, "output_path": str(tmp_path / "missing.png")}]}
                    ],
                },
                "event_keyframes": [
                    {"ok": True, "output_path": str(existing)},
                    {"ok": False, "output_path": str(tmp_path / "failed_kf.png")},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    missing = collect_missing_pipeline_assets(result_json)

    assert len(missing) == 3
    assert any("failed.png" in row for row in missing)
    assert any("missing.png" in row for row in missing)
    assert any("failed_kf.png" in row for row in missing)


def test_collect_missing_video_assets_detects_missing_outputs(tmp_path):
    summary = tmp_path / "event_video_summary.json"
    ok_mp4 = tmp_path / "ok.mp4"
    ok_mp4.write_bytes(b"mp4")
    summary.write_text(
        json.dumps(
            {
                "generated": [
                    {"output_video_path": str(ok_mp4)},
                    {"output_video_path": str(tmp_path / "missing.mp4")},
                ],
                "concat_output": str(tmp_path / "final.mp4"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    missing = collect_missing_video_assets(summary)

    assert len(missing) == 2
    assert any("missing.mp4" in row for row in missing)
    assert any("final.mp4" in row for row in missing)
