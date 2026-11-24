import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.run_vimax_moviebench_pipeline import (
    build_track_a_script,
    build_track_a_characters,
    build_track_b_idea,
    load_moviebench_dataset,
    run_pipeline,
)
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline


def _write_dataset(root: Path) -> None:
    (root / "l4_global").mkdir(parents=True)
    (root / "l3_arcs").mkdir(parents=True)
    (root / "l2_events" / "units").mkdir(parents=True)
    (root / "l1_shots" / "units").mkdir(parents=True)
    (root / "l4_global" / "movie_bible.json").write_text(json.dumps({
        "result": {
            "movie_id": "movie_01",
            "global_storyline": {"story": "A family changes over one summer."},
            "movie_synopsis": {"short_synopsis": "Fallback synopsis."},
            "global_character_profiles": [{"character": "Alex", "observed_profile": "A guarded parent."}],
        }
    }))
    (root / "l3_arcs" / "story_arcs.json").write_text(json.dumps({"story_arcs": [{
        "l3_story_arc_id": "movie_01_L3ARC_001",
        "arc_title": "The first morning",
        "full_story_S": "A parent and child prepare to leave home.",
        "l2_event_ids": ["event_01"],
        "l1_shot_ids": ["shot_01"],
    }]}))
    (root / "l2_events" / "units" / "event.json").write_text(json.dumps({"l2_events": [{
        "l2_event_id": "event_01", "event_description": "Alex opens the curtains.",
        "main_characters": ["Alex"], "source_shot_ids": ["shot_01"],
        "time_range": {"start_ms": 10},
    }]}))
    (root / "l1_shots" / "units" / "shot.json").write_text(json.dumps({"result": {
        "l1_shot_id": "shot_01", "order_index": 1,
        "time_range": {"start_ms": 10, "end_ms": 3000},
        "characters": [{"name": "Alex", "visible_action": "opens curtains", "visible_state": "standing"}],
        "location_mentions": [{"name": "Bedroom"}],
        "observable_facts": ["Alex opens the curtains."],
    }}))
    (root / "track_b.json").write_text(json.dumps({"answer": "WITHHELD ANSWER MUST NOT BE USED"}))


class MovieBenchViMaxPipelineTests(unittest.TestCase):
    def test_loads_restructured_hierarchy_and_builds_native_track_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root)
            data = load_moviebench_dataset(root)

            script = build_track_a_script(data, "movie_01_L3ARC_001")
            characters = build_track_a_characters(data, "movie_01_L3ARC_001")
            idea = build_track_b_idea(data)

        self.assertIn("The first morning", script)
        self.assertIn("Alex opens the curtains", script)
        self.assertIn("Linked L2 events", script)
        self.assertIn("Bedroom", script)
        self.assertEqual(characters[0]["identifier_in_scene"], "Alex")
        self.assertIn("A family changes over one summer", idea)
        self.assertNotIn("WITHHELD ANSWER", idea)
        self.assertIn("A guarded parent", idea)

    def test_dry_run_writes_independent_track_inputs_without_pipeline_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "movie"
            _write_dataset(data_root)
            output = root / "out"
            args = Namespace(
                dataset_root=data_root, output_dir=output, track="both",
                sample_id="movie_01_L3ARC_001", max_l1_nodes=0, style="cinematic",
                script_config=Path("configs/script2video.yaml"),
                idea_config=Path("configs/idea2video.yaml"), run=False,
            )
            result = run_pipeline(args)

            self.assertEqual(result["status"], "dry_run")
            self.assertTrue((output / "track_A" / "input_script.txt").is_file())
            self.assertTrue((output / "track_B" / "input_idea.txt").is_file())
            self.assertFalse((output / "track_B" / "answer.txt").exists())
            self.assertEqual([row["track"] for row in result["tracks"]], ["A", "B"])

    def test_dry_run_refuses_to_overwrite_nonempty_output_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "movie"
            _write_dataset(data_root)
            output = root / "out"
            output.mkdir()
            (output / "keep.txt").write_text("preserve")
            args = Namespace(
                dataset_root=data_root, output_dir=output, track="A",
                sample_id="movie_01_L3ARC_001", max_l1_nodes=0, style="cinematic",
                script_config=Path("configs/script2video.yaml"),
                idea_config=Path("configs/idea2video.yaml"), run=False, resume=False,
            )
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                run_pipeline(args)
            self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_online_run_dispatches_tracks_to_distinct_native_vimax_pipelines(self):
        calls = []

        class FakeScriptPipeline:
            async def __call__(self, **kwargs):
                calls.append(("A", kwargs["script"], len(kwargs["characters"])))
                return "track_A/final_video.mp4"

        class FakeIdeaPipeline:
            async def __call__(self, **kwargs):
                calls.append(("B", kwargs["idea"]))
                return "track_B/final_video.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "movie"
            _write_dataset(data_root)
            args = Namespace(
                dataset_root=data_root, output_dir=root / "out", track="both",
                sample_id="movie_01_L3ARC_001", max_l1_nodes=0, style="cinematic",
                script_config=Path("configs/script2video.yaml"),
                idea_config=Path("configs/idea2video.yaml"), run=True, resume=False,
            )
            with patch.object(Script2VideoPipeline, "init_from_config", return_value=FakeScriptPipeline()), \
                 patch.object(Idea2VideoPipeline, "init_from_config", return_value=FakeIdeaPipeline()):
                result = run_pipeline(args)

        self.assertEqual([row[0] for row in calls], ["A", "B"])
        self.assertIn("The first morning", calls[0][1])
        self.assertIn("A family changes over one summer", calls[1][1])
        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
