from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_moviebench_clip_metadata import collect_groups, enrich_groups


def _write_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"Plot": "x"}, ensure_ascii=False), encoding="utf-8")


def test_clip_continuity_allows_less_than_one_second_gap(tmp_path):
    movie = tmp_path / "MOVIE_A"
    _write_clip(movie / "MOVIE_A_00.00.00.000-00.00.10.000.json")
    _write_clip(movie / "MOVIE_A_00.00.10.999-00.00.20.000.json")
    _write_clip(movie / "MOVIE_A_00.00.21.000-00.00.30.000.json")

    stats = enrich_groups(collect_groups(tmp_path), write=True)

    assert stats["updated"] == 3
    second = json.loads((movie / "MOVIE_A_00.00.10.999-00.00.20.000.json").read_text(encoding="utf-8"))
    third = json.loads((movie / "MOVIE_A_00.00.21.000-00.00.30.000.json").read_text(encoding="utf-8"))
    assert second["clip_meta"]["time_diff_from_prev_ms"] == 999
    assert second["clip_meta"]["is_continuous_with_prev"] is True
    assert second["clip_meta"]["prev_file_name"] == "MOVIE_A_00.00.00.000-00.00.10.000.json"
    assert third["clip_meta"]["time_diff_from_prev_ms"] == 1000
    assert third["clip_meta"]["is_continuous_with_prev"] is False
    assert third["clip_meta"]["prev_file_name"] is None
    assert third["clip_meta"]["prev_chronological_file_name"] == "MOVIE_A_00.00.10.999-00.00.20.000.json"


def test_collect_groups_skips_non_clip_json_outputs(tmp_path):
    movie = tmp_path / "MOVIE_A"
    _write_clip(movie / "MOVIE_A_00.00.00.000-00.00.10.000.json")
    generated = movie / "llm_scene_segments" / "scene30.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps({"scene_id": "scene30"}, ensure_ascii=False), encoding="utf-8")

    groups = collect_groups(tmp_path)

    assert list(groups) == ["MOVIE_A"]
    assert len(groups["MOVIE_A"]) == 1
