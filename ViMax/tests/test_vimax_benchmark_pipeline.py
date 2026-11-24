import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_vimax_benchmark_pipeline import (
    build_track_a_script,
    build_track_b_idea,
    load_cases,
    select_cases,
)


class ViMaxBenchmarkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "vimax_benchmark"
        cls.cases = load_cases(cls.root)

    def test_index_contains_all_35_cases(self):
        self.assertEqual(len(self.cases), 35)

    def test_select_type_a_cases(self):
        selected = select_cases(self.cases, case_type="A")
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(case[2]["consistency_type"] == "Type A" for case in selected))

    def test_track_a_contains_shot_fields(self):
        theme, source, case = self.cases[0]
        text = build_track_a_script(case, max_shots=1)
        self.assertIn("First-frame specification:", text)
        self.assertIn("Motion and video specification:", text)
        self.assertIn(case["scenes"][0]["shots"][0]["first_frame"], text)
        self.assertIn(case["scenes"][0]["shots"][0]["video_prompt"], text)

    def test_track_b_excludes_shot_level_fields(self):
        _, _, case = self.cases[0]
        text = build_track_b_idea(case)
        self.assertEqual(text.strip(), case["story_overview"])
        self.assertNotIn(case["scenes"][0]["shots"][0]["first_frame"], text)
        self.assertNotIn(case["scenes"][0]["shots"][0]["video_prompt"], text)

    def test_dry_run_writes_separate_track_inputs(self):
        from scripts.run_vimax_benchmark_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            self.assertEqual(main([
                "--benchmark-root", str(self.root),
                "--case", "chef_international_kitchens",
                "--track", "both",
                "--output-dir", str(output),
            ]), 0)
            a = output / "chef_international_kitchens" / "track_A" / "input_script.txt"
            b = output / "chef_international_kitchens" / "track_B" / "input_idea.txt"
            self.assertTrue(a.exists())
            self.assertTrue(b.exists())
            manifest = json.loads((output / "chef_international_kitchens" / "case_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run")
            self.assertEqual({row["track"] for row in manifest["tracks"]}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
