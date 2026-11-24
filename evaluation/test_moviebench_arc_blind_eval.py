import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import moviebench_arc_blind_eval as evaluator


CONDITIONS = ("fl2va_event_keyframe_only", "ref2va_semantic_assets")
DIMENSIONS = evaluator.DIMENSIONS


def setup_run(tmp_path):
    root = tmp_path / "arc_01"
    plan = [
        {"node_id": "N1", "prompt": "same prompt", "duration_sec": 5, "num_frames": 124},
        {"node_id": "N2", "prompt": "second prompt", "duration_sec": 5, "num_frames": 124},
    ]
    (root / "video_comparison").mkdir(parents=True)
    (root / "video_comparison_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for condition in CONDITIONS:
        folder = root / "video_comparison" / condition
        folder.mkdir()
        entries = []
        for node in ("N1", "N2"):
            video = folder / f"{node}.mp4"
            video.write_bytes(b"video")
            entries.append({"node_id": node, "status": "done", "output_video_path": str(video)})
        (folder / "event_video_tasks.json").write_text(json.dumps(entries), encoding="utf-8")
    return root


def test_export_blind_tasks_and_key(tmp_path):
    root = setup_run(tmp_path)
    result = evaluator.export_blind_tasks(root)
    tasks = json.loads(Path(result["tasks_path"]).read_text(encoding="utf-8"))
    key = json.loads(Path(result["key_path"]).read_text(encoding="utf-8"))
    assert len(tasks["items"]) == 4
    assert all("condition" not in item for item in tasks["items"])
    assert all("blind_code" in item and (root / item["video_path"]).is_file() for item in tasks["items"])
    assert len(key["assignments"]) == 4
    assert {row["condition"] for row in key["assignments"]} == set(CONDITIONS)


def test_missing_videos_are_reported_and_omitted(tmp_path):
    root = setup_run(tmp_path)
    condition_dir = root / "video_comparison" / CONDITIONS[0]
    manifest = json.loads((condition_dir / "event_video_tasks.json").read_text())
    manifest.pop()
    (condition_dir / "event_video_tasks.json").write_text(json.dumps(manifest))
    result = evaluator.export_blind_tasks(root)
    tasks = json.loads(Path(result["tasks_path"]).read_text())
    assert len(tasks["items"]) == 3
    assert tasks["missing"] == [{"node_id": "N2"}]
    key = json.loads(Path(result["key_path"]).read_text())
    assert any(row["condition"] == CONDITIONS[0] and row["node_id"] == "N2" for row in key["missing"])


def test_score_validation_and_na_pairwise_summary(tmp_path):
    root = setup_run(tmp_path)
    result = evaluator.export_blind_tasks(root, seed=3)
    key = json.loads(Path(result["key_path"]).read_text())
    codes = {(row["condition"], row["node_id"]): row["blind_code"] for row in key["assignments"]}
    scores = []
    for node in ("N1", "N2"):
        for condition, score in ((CONDITIONS[0], 3), (CONDITIONS[1], 5)):
            row = {"blind_code": codes[(condition, node)]}
            row.update({dimension: score for dimension in DIMENSIONS})
            scores.append(row)
    scores[0][DIMENSIONS[0]] = "N/A"
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps(scores), encoding="utf-8")
    summary = evaluator.score_and_summarize(score_path, Path(result["key_path"]))
    assert summary["dimensions"][DIMENSIONS[0]]["n_by_condition"][CONDITIONS[0]] == 1
    assert summary["paired_differences"][DIMENSIONS[1]]["mean_difference_b_minus_a"] == 2

    scores[0][DIMENSIONS[1]] = 8
    score_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="1-5"):
        evaluator.score_and_summarize(score_path, Path(result["key_path"]))
