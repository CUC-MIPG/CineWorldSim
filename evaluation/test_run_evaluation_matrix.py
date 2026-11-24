import json
from pathlib import Path

from run_evaluation_matrix import build_report


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_keeps_metric_families_separate(tmp_path):
    root = tmp_path / "moviebench"
    arc = root / "movie" / "all_arcs" / "movie_arc_01"
    write_json(arc / "moviebench_arc_video_eval_summary.json", {"status": "done", "event_count": 2, "groups": {"a": {"total": 2, "success": 2}, "b": {"total": 2, "success": 1, "failed": 1}}})
    write_json(arc / "track_a_metrics.json", {"KER": 0.8, "FDC": 0.7})
    write_json(arc / "track_b_metrics.json", {"EVR": 0.5})
    write_json(root / "external" / "constory.json", {"CED": 0.3})

    report = build_report(root, external_paths={"constory": root / "external" / "constory.json"})
    assert report["scope"]["arc_count"] == 1
    assert report["generation"]["total_jobs"] == 4
    assert report["generation"]["failed_jobs"] == 1
    assert report["track_a"]["sample_count"] == 1
    assert report["track_a"]["metrics"]["KER"]["mean"] == 0.8
    assert report["track_b"]["metrics"]["EVR"]["mean"] == 0.5
    assert report["external_benchmarks"]["constory"]["CED"] == 0.3
    assert "moviebench_blind_review" not in report
    assert "overall_score" not in report


def test_missing_outputs_are_explicit(tmp_path):
    report = build_report(tmp_path)
    assert report["scope"]["arc_count"] == 0
    assert report["track_a"]["status"] == "unavailable"
    assert report["external_benchmarks"] == {}


def test_dry_run_planned_jobs_are_counted(tmp_path):
    arc = tmp_path / "arc"
    write_json(arc / "moviebench_arc_video_eval_summary.json", {"status": "dry_run", "planned_video_jobs": 8})
    report = build_report(tmp_path)
    assert report["generation"]["total_jobs"] == 8
