from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_pre_video_pipeline import (
    build_recovery_plan,
    check_pipeline,
    ensure_repair_output_dir,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_healthy_pipeline(root: Path) -> Path:
    image = root / "narrative_artifacts" / "world_model" / "locations" / "Vault" / "wide.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    portrait = root / "narrative_artifacts" / "portraits" / "Mara.jpg"
    portrait.parent.mkdir(parents=True, exist_ok=True)
    portrait.write_bytes(b"portrait")
    keyframe = root / "narrative_artifacts" / "event_keyframes" / "step_001" / "Mara.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"keyframe")

    _write_json(
        root / "story_bundle.json",
        {"story": "A short vault adventure.", "characters": ["Mara"]},
    )
    _write_json(
        root / "pipeline_result.json",
        {
            "backend_mode": "all_banana",
            "backend_tool": "nano_banana",
            "story": "A short vault adventure.",
            "characters": ["Mara"],
            "character_portraits": {"Mara": str(portrait)},
            "world_model": {
                "locations": [
                    {
                        "name": "Vault",
                        "images": [{"ok": True, "output_path": str(image)}],
                    }
                ],
                "items": [],
            },
            "world_states": [{"step": 0}, {"step": 1}],
            "world_deltas": [{"step": 1, "decision_source": "llm"}],
            "step_logs": [
                {
                    "step": 1,
                    "role_logs": [{"role_name": "Mara", "next_action": {"location": "Vault"}}],
                }
            ],
            "event_keyframes": [
                {
                    "step": 1,
                    "role_name": "Mara",
                    "ok": True,
                    "output_path": str(keyframe),
                }
            ],
        },
    )
    _write_json(
        root / "narrative_checkpoint.json",
        {"completed_steps": 1, "max_steps": 1},
    )
    _write_json(
        root / "step_logs.json",
        [
            {
                "step": 1,
                "role_logs": [{"role_name": "Mara", "next_action": {"location": "Vault"}}],
            }
        ],
    )
    _write_json(root / "world_state_step0.json", {"step": 0})
    _write_json(root / "world_state_step1.json", {"step": 1})
    _write_json(root / "world_deltas" / "world_delta_step1.json", {"step": 1})
    _write_json(root / "path_orchestration" / "path_candidates.json", [{"node_id": "step::1::Mara"}])
    _write_json(
        root / "path_orchestration" / "narrative_nodes.json",
        {"selected_nodes": [{"node_id": "step::1::Mara", "characters": ["Mara"]}]},
    )
    _write_json(
        root / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json",
        {"mode": "llm", "ordered_node_ids": ["step::1::Mara"]},
    )
    _write_json(
        root / "path_orchestration" / "path_mcts" / "narrative_path_plan_mcts.json",
        {"mode": "mcts", "ordered_node_ids": ["step::1::Mara"]},
    )
    return root


def test_check_pipeline_reports_healthy_complete_pre_video_snapshot(tmp_path: Path) -> None:
    report = check_pipeline(_make_healthy_pipeline(tmp_path / "pipeline"))

    assert report["status"] == "healthy"
    assert report["stages"]["world_initialization"]["status"] == "healthy"
    assert report["stages"]["narrative_simulation"]["status"] == "healthy"
    assert report["stages"]["path_llm"]["status"] == "healthy"
    assert report["stages"]["path_mcts"]["status"] == "healthy"
    assert report["fallback_calls"] == []


def test_check_pipeline_detects_missing_asset_as_world_initialization_failure(tmp_path: Path) -> None:
    root = _make_healthy_pipeline(tmp_path / "pipeline")
    missing = root / "does-not-exist.png"
    payload = json.loads((root / "pipeline_result.json").read_text(encoding="utf-8"))
    payload["world_model"]["locations"][0]["images"][0]["output_path"] = str(missing)
    _write_json(root / "pipeline_result.json", payload)

    report = check_pipeline(root)

    assert report["status"] == "failed"
    assert report["stages"]["world_initialization"]["status"] == "failed"
    assert any("does-not-exist.png" in item for item in report["missing_artifacts"])


def test_check_pipeline_maps_fallback_log_to_simulation_and_path_plan(tmp_path: Path) -> None:
    root = _make_healthy_pipeline(tmp_path / "pipeline")
    (root / "run.log").write_text(
        "WARNING narrative simulation model call failed, using fallback: 503\n"
        "INFO path orchestrator mode=fallback\n",
        encoding="utf-8",
    )
    report = check_pipeline(root)
    plan = build_recovery_plan(report)

    assert report["status"] == "failed"
    assert {item["stage"] for item in report["fallback_calls"]} == {
        "narrative_simulation",
        "path_llm",
    }
    assert plan["earliest_failed_stage"] == "narrative_simulation"
    assert plan["affected_stages"] == [
        "narrative_simulation",
        "world_delta",
        "event_keyframe",
        "node_selection",
        "path_llm",
        "path_mcts",
    ]


def test_check_pipeline_detects_structured_role_fallback_and_step(tmp_path: Path) -> None:
    root = _make_healthy_pipeline(tmp_path / "pipeline")
    step_logs = json.loads((root / "step_logs.json").read_text(encoding="utf-8"))
    step_logs[0]["role_logs"][0].update(
        {
            "model_calls": [
                {
                    "call_id": "llm-13",
                    "call_label": "character_action",
                    "status": "fallback",
                    "fallback_used": True,
                    "error": "Error code: 503 Service temporarily unavailable",
                }
            ],
            "fallback_used": True,
        }
    )
    _write_json(root / "step_logs.json", step_logs)

    report = check_pipeline(root)
    plan = build_recovery_plan(report)

    assert report["status"] == "failed"
    assert report["stages"]["narrative_simulation"]["status"] == "failed"
    assert any("step=1 actor=Mara" in item["reason"] for item in report["fallback_calls"])
    assert 1 in plan["affected_steps"]


def test_check_pipeline_detects_step_gap_and_invalid_path_plan(tmp_path: Path) -> None:
    root = _make_healthy_pipeline(tmp_path / "pipeline")
    (root / "world_state_step1.json").unlink()
    _write_json(
        root / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json",
        {"mode": "llm", "ordered_node_ids": ["not-a-candidate"]},
    )

    report = check_pipeline(root)

    assert report["stages"]["world_delta"]["status"] == "failed"
    assert report["stages"]["path_llm"]["status"] == "failed"
    assert report["step_gaps"]
    assert report["path_gaps"]


def test_repair_plan_for_path_only_does_not_rerun_world_stages(tmp_path: Path) -> None:
    root = _make_healthy_pipeline(tmp_path / "pipeline")
    _write_json(
        root / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json",
        {"mode": "fallback", "ordered_node_ids": ["step::1::Mara"]},
    )

    plan = build_recovery_plan(check_pipeline(root))

    assert plan["earliest_failed_stage"] == "path_llm"
    assert plan["affected_stages"] == ["path_llm"]


def test_repair_output_must_be_new_empty_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="source"):
        ensure_repair_output_dir(source, source)

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "file.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        ensure_repair_output_dir(source, existing)


def test_path_repair_uses_new_directory_and_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_healthy_pipeline(tmp_path / "source")
    original = (source / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").read_text(
        encoding="utf-8"
    )
    _write_json(
        source / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json",
        {"mode": "fallback", "ordered_node_ids": ["step::1::Mara"]},
    )

    from scripts import run_narrative_world_path_pipeline as path_pipeline

    def fake_llm_path(**kwargs: object) -> Path:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "narrative_path_plan.json"
        _write_json(
            path,
            {
                "mode": "llm",
                "ordered_node_ids": ["step::1::Mara"],
                "candidate_count": 1,
            },
        )
        return path

    monkeypatch.setattr(path_pipeline, "_run_llm_path", fake_llm_path)
    report = check_pipeline(source)
    output = tmp_path / "repaired" / "custom_name"
    manifest = __import__("scripts.recover_pre_video_pipeline", fromlist=["repair_pipeline"]).repair_pipeline(
        pipeline_dir=source,
        output_dir=output,
        report=report,
        strict_online=True,
    )

    assert manifest["status"] == "done"
    assert output != source
    assert (output / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").exists()
    assert json.loads(
        (output / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").read_text(
            encoding="utf-8"
        )
    )["mode"] == "llm"
    assert json.loads(
        (source / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").read_text(
            encoding="utf-8"
        )
    )["mode"] == "fallback"
    assert original != (source / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json").read_text(
        encoding="utf-8"
    )


def test_world_repair_rewrites_staging_paths_and_rebuilds_downstream_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_healthy_pipeline(tmp_path / "source")
    source_payload = json.loads((source / "pipeline_result.json").read_text(encoding="utf-8"))
    source_payload["world_model"]["locations"][0]["images"][0]["output_path"] = str(
        source / "missing-after-world-failure.png"
    )
    _write_json(source / "pipeline_result.json", source_payload)
    report = check_pipeline(source)

    from scripts import recover_pre_video_pipeline as recovery
    from scripts import run_dual_backend_narrative_world_pipeline as world_pipeline
    from scripts import run_narrative_world_path_pipeline as path_pipeline
    from agents.narrative_node_selector import NarrativeNodeSelector

    world_calls: list[dict[str, object]] = []
    path_calls: list[str] = []

    def fake_world_run(**kwargs: object) -> Path:
        world_calls.append(kwargs)
        generated_dir = Path(str(kwargs["output_dir"])) / str(kwargs["backend_mode"])
        return _make_healthy_pipeline(generated_dir)

    def fake_node_select(
        self: object,
        payload: dict[str, object],
        *,
        source_pipeline_result_json: str = "",
    ) -> dict[str, object]:
        del self, payload
        path_calls.append("node_selection")
        return {
            "source_pipeline_result_json": source_pipeline_result_json,
            "selected_nodes": [
                {
                    "event_id": "EVT_001",
                    "node_id": "NN_001",
                    "source_steps": [1],
                    "characters": ["Mara"],
                    "event_summary": "Mara studies the vault compass.",
                }
            ],
        }

    def fake_llm_path(**kwargs: object) -> Path:
        path_calls.append("path_llm")
        output_dir = Path(str(kwargs["output_dir"]))
        path = output_dir / "narrative_path_plan.json"
        _write_json(path, {"mode": "llm", "ordered_node_ids": ["NN_001"]})
        return path

    def fake_mcts_path(**kwargs: object) -> Path:
        path_calls.append("path_mcts")
        output_dir = Path(str(kwargs["output_dir"]))
        path = output_dir / "narrative_path_plan_mcts.json"
        _write_json(path, {"mode": "mcts", "ordered_node_ids": ["NN_001"]})
        return path

    monkeypatch.setattr(world_pipeline, "_run_one_backend", fake_world_run)
    monkeypatch.setattr(NarrativeNodeSelector, "select", fake_node_select)
    monkeypatch.setattr(path_pipeline, "_run_llm_path", fake_llm_path)
    monkeypatch.setattr(path_pipeline, "_run_mcts_path", fake_mcts_path)

    output = tmp_path / "repaired" / "custom_name"
    manifest = recovery.repair_pipeline(
        pipeline_dir=source,
        output_dir=output,
        report=report,
        strict_online=True,
    )

    assert manifest["status"] == "done"
    assert world_calls and world_calls[0]["backend_mode"] == "all_banana"
    assert path_calls == ["node_selection", "path_llm", "path_mcts"]
    assert manifest["post_repair_report"]["status"] == "healthy"
    repaired_payload = json.loads((output / "pipeline_result.json").read_text(encoding="utf-8"))
    repaired_asset = repaired_payload["world_model"]["locations"][0]["images"][0]["output_path"]
    assert str(output) in repaired_asset
    assert Path(repaired_asset).exists()
    assert not any(".custom_name.pre_video_stage" in str(path) for path in output.rglob("*.json"))
