import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interfaces.video_output import VideoOutput
from scripts.run_vimax_moviebench_arc01_baseline import (
    CONDITIONS,
    build_jobs,
    run,
)


def _plan(reference_paths):
    return [
        {
            "rank": 1,
            "node_id": "arc_scene_001",
            "prompt": "Exact baseline prompt",
            "duration_sec": 5,
            "num_frames": 124,
            "groups": {
                condition: {
                    "minimax_task": task,
                    "reference_images": refs,
                    "reference_mode": condition,
                }
                for condition, task, refs in zip(
                    CONDITIONS,
                    ("fl2va", "ref2va"),
                    reference_paths,
                )
            },
            "same_prompt_for_both_groups": True,
        }
    ]


class MovieBenchBaselineTests(unittest.TestCase):
    def test_build_jobs_preserves_plan_prompt_task_and_references(self):
        jobs = build_jobs(_plan([["keyframe.png"], ["character.png", "room.png"]]))

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["condition"], CONDITIONS[0])
        self.assertEqual(jobs[0]["prompt"], "Exact baseline prompt")
        self.assertEqual(jobs[0]["task"], "fl2va")
        self.assertEqual(jobs[1]["task"], "ref2va")
        self.assertEqual(jobs[1]["reference_images"], ["character.png", "room.png"])

    def test_build_jobs_rejects_malformed_or_duplicate_node_ids(self):
        with self.assertRaisesRegex(ValueError, "same prompt"):
            row = _plan([["one.png"], ["two.png"]])[0]
            row["same_prompt_for_both_groups"] = False
            build_jobs([row])

        with self.assertRaisesRegex(ValueError, "duplicate node_id"):
            plan = _plan([["one.png"], ["two.png"]])
            build_jobs(plan + plan)

    def test_default_run_writes_dry_run_manifests_and_missing_reference_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "comparison.json"
            output_dir = root / "result"
            plan_path.write_text(json.dumps(_plan([["missing-keyframe.png"], ["missing-room.png"]])))

            summary = run(plan_path, output_dir, run_generation=False, concat=False)

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["planned_jobs"], 2)
            self.assertEqual(summary["missing_reference_count"], 2)
            for condition in CONDITIONS:
                manifest = json.loads((output_dir / condition / "event_video_tasks.json").read_text())
                self.assertEqual(manifest["jobs"][0]["status"], "blocked_missing_references")
                self.assertEqual(manifest["jobs"][0]["prompt"], "Exact baseline prompt")

    def test_run_mode_fails_preflight_before_any_generation_if_reference_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "comparison.json"
            plan_path.write_text(json.dumps(_plan([["missing-keyframe.png"], ["missing-room.png"]])))

            with self.assertRaisesRegex(FileNotFoundError, "reference image"):
                run(plan_path, root / "result", run_generation=True, concat=False)

    def test_run_mode_generates_each_condition_and_writes_success_manifests(self):
        calls = []

        class FakeGenerator:
            async def generate_single_video(self, prompt, reference_image_paths, output_video_path):
                calls.append((prompt, list(reference_image_paths), output_video_path))
                return VideoOutput(fmt="bytes", ext="mp4", data=b"fake-mp4")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references = [[root / "keyframe.png"], [root / "room.png"]]
            for path in references[0] + references[1]:
                path.write_bytes(b"image")
            plan_path = root / "comparison.json"
            plan_path.write_text(json.dumps(_plan([[str(path) for path in group] for group in references])))

            with patch("scripts.run_vimax_moviebench_arc01_baseline.VideoGeneratorMinimaxH3VLLM", FakeGenerator):
                summary = run(plan_path, root / "result", run_generation=True, concat=False)

            self.assertEqual(summary["status"], "done")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call[0] == "Exact baseline prompt" for call in calls))
            for condition in CONDITIONS:
                manifest = json.loads((root / "result" / condition / "event_video_tasks.json").read_text())
                self.assertEqual(manifest["jobs"][0]["status"], "success")
                self.assertEqual(Path(manifest["jobs"][0]["output_path"]).read_bytes(), b"fake-mp4")


if __name__ == "__main__":
    unittest.main()
