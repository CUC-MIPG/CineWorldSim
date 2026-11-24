import importlib.util
import json
import sys
from pathlib import Path


MODULE = Path(__file__).with_name("run_cameramaster_vimax_comparison.py")
SPEC = importlib.util.spec_from_file_location("comparison", MODULE)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepare_comparison_outputs_normalized_media_and_html(tmp_path):
    case = tmp_path / "case.json"
    _write(case, {"story_overview": "A painter finishes a landscape.", "scenes": [{"scene_num": 1, "shots": [{"shot_id": 1, "first_frame": "a painter", "video_prompt": "the painter paints"}]}]})
    cm, vm = tmp_path / "cm", tmp_path / "vm"
    for root in (cm,):
        _write(root / "track_A" / "narrative_nodes.json", {"storyline": "A painter finishes a landscape.", "selected_nodes": [{"node_id": "n1", "event_summary": "the painter paints"}]})
        _write(root / "track_A" / "path_plan.json", {"ordered_node_ids": ["n1"]})
        video = root / "track_A" / "video" / "event_video_001_n1.mp4"; video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(b"x")
    video = vm / "track_A" / "vimax_pipeline" / "shots" / "0" / "video.mp4"; video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(b"x")
    out = tmp_path / "out"
    assert comparison.main(["--benchmark-case", str(case), "--cameramaster-output-root", str(cm), "--vimax-output-root", str(vm), "--output-dir", str(out), "--tracks", "A"]) == 0
    assert (out / "normalized" / "cameramaster_track_A.json").is_file()
    assert (out / "normalized" / "vimax_track_A.json").is_file()
    assert (out / "qualitative" / "index.html").is_file()
    assert (out / "quantitative" / "comparison.csv").is_file()
