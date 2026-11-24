from pathlib import Path
import json
import subprocess
import sys


def test_dual_backend_pipeline_dry_run(tmp_path: Path) -> None:
    story_file = tmp_path / "story.txt"
    story_file.write_text(
        "A journalist enters a lighthouse, opens a box, and discovers a compass.",
        encoding="utf-8",
    )
    output_dir = tmp_path / "pipeline"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_dual_backend_narrative_world_pipeline.py",
            "--story-file",
            str(story_file),
            "--run-both",
            "--max-steps",
            "1",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "all_flux" / "pipeline_result.json").exists()
    assert (output_dir / "all_banana" / "pipeline_result.json").exists()
    assert (output_dir / "comparison_overview.html").exists()
    payload = json.loads((output_dir / "all_banana" / "pipeline_result.json").read_text(encoding="utf-8"))
    assert payload["character_portraits"]
    for path in payload["character_portraits"].values():
        assert Path(path).exists()


def test_real_step_logs_reuses_complete_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_dual_backend_narrative_world_pipeline as pipeline

    checkpoint = tmp_path / "narrative_checkpoint.json"
    checkpoint.write_text(
        '{"completed_steps": 15, "step_logs": [{"step": 1, "role_logs": []}], "agents": [{"state": {"name": "Leona", "final_goal": "goal", "role_profile": "profile", "known_relations": [], "current_environment": "env"}, "model_config": {}, "portrait_path": ""}]}',
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("SAM should not run when a complete checkpoint exists")

    monkeypatch.setattr(pipeline.StateAnalysisMachine, "run_narrative_deduction", fail_run)

    step_logs, character_configs = pipeline._real_step_logs(
        "story",
        ["Leona"],
        {"summary": "world"},
        15,
        checkpoint_path=checkpoint,
    )

    assert step_logs == [{"step": 1, "role_logs": []}]
    assert character_configs[0]["name"] == "Leona"


def test_world_stage_passes_strict_online_and_retries_to_backend(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_narrative_world_path_pipeline as pipeline
    import argparse

    calls = []

    def fake_run_one_backend(**kwargs):
        calls.append(kwargs)
        return tmp_path / kwargs["backend_mode"]

    monkeypatch.setattr(pipeline, "_run_one_backend", fake_run_one_backend)
    args = argparse.Namespace(
        run_both=False,
        backend_mode="all_banana",
        max_steps=3,
        output_dir=tmp_path,
        dry_run=False,
        stop_on_complete=True,
        strict_online=True,
        retries=7,
    )

    pipeline._run_world_stage(args, "story", ["Hero"])

    assert calls[0]["strict_online"] is True
    assert calls[0]["retries"] == 7


def test_real_step_logs_renders_portraits_from_complete_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_dual_backend_narrative_world_pipeline as pipeline

    checkpoint = tmp_path / "narrative_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "completed_steps": 15,
                "step_logs": [{"step": 1, "role_logs": []}],
                "agents": [
                    {
                        "state": {
                            "name": "Hero",
                            "final_goal": "Find the route",
                            "role_profile": "Explorer",
                            "known_relations": [],
                            "current_environment": "Vault",
                        },
                        "model_config": {},
                        "portrait_path": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rendered = []

    def fake_render_character_initial_assets(**kwargs):
        rendered.append(kwargs)
        return {"Hero": str(tmp_path / "Hero_portrait.png")}

    monkeypatch.setattr(pipeline, "render_character_initial_assets", fake_render_character_initial_assets)
    world = {}
    step_logs, configs = pipeline._real_step_logs(
        "story",
        ["Hero"],
        world,
        15,
        checkpoint_path=checkpoint,
        renderer=object(),
        run_dir=tmp_path,
    )

    assert step_logs == [{"step": 1, "role_logs": []}]
    assert configs[0]["portrait_path"] == ""
    assert rendered and rendered[0]["run_dir"] == tmp_path
    assert world["character_portraits"] == {"Hero": str(tmp_path / "Hero_portrait.png")}


def test_character_config_serialization_preserves_rendered_views_for_world_state():
    from types import SimpleNamespace
    from scripts.run_dual_backend_narrative_world_pipeline import _serialize_character_config
    from narrative.world_state_tracker import build_initial_world_state

    config = SimpleNamespace(
        name="Hero", final_goal="Find the key", role_profile="Explorer",
        known_relations=[], current_environment="Vault", prompt_arm="",
        model_config=None, dynamic_system_prompt="", portrait_path="/tmp/hero.png",
        views={"front_full": {"output_path": "/tmp/hero_front.png"}},
    )

    character = _serialize_character_config(config)
    state = build_initial_world_state(
        world_model={"characters": [character]}, backend_tool="nano_banana"
    )

    assert state["characters"][0]["name"] == "Hero"
    assert state["characters"][0]["views"]["front_full"]["last_image_path"] == "/tmp/hero_front.png"
