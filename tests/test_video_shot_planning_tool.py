from agents.video_shot_planning_tool import VideoShotPlanningTool


def test_shot_planner_uses_storyboard_chosen_count_but_caps_at_four():
    raw_shots = [
        {
            "shot_index": index,
            "duration_sec": index,
            "video_description": f"Beat {index}",
        }
        for index in range(1, 6)
    ]

    normalized = VideoShotPlanningTool._normalize_shots(raw_shots, 10)

    assert len(normalized) == 4
    assert [row["shot_index"] for row in normalized] == [1, 2, 3, 4]
    assert sum(row["duration_sec"] for row in normalized) == 10
    assert normalized[-1]["start_sec"] + normalized[-1]["duration_sec"] == 10


def test_fallback_plan_is_single_shot_with_unified_h3_duration():
    plan = VideoShotPlanningTool.fallback_plan({"story_description": "A concise action."})

    assert plan["duration_sec"] == 10
    assert plan["num_frames"] == 243
    assert len(plan["shots"]) == 1
    assert plan["shots"][0]["duration_sec"] == 10


def test_generate_preserves_planner_selected_one_to_four_shots_and_ten_second_total():
    class FakeLLM:
        system_prompt = ""

        def run_json(self, *, system_prompt, **kwargs):
            del kwargs
            self.system_prompt = system_prompt
            return {
                "duration_sec": 5,
                "num_frames": 124,
                "shots": [
                    {"shot_index": index, "duration_sec": index, "video_description": f"Beat {index}"}
                    for index in range(1, 4)
                ],
            }

    llm = FakeLLM()
    planner = VideoShotPlanningTool(llm_tool=llm)
    result = planner.generate(node={"event_summary": "Three distinct story beats."}, storyline="Story")

    assert "exactly 10 seconds and 243 frames" in llm.system_prompt
    assert result["duration_sec"] == 10
    assert result["num_frames"] == 243
    assert len(result["shots"]) == 3
    assert sum(row["duration_sec"] for row in result["shots"]) == 10
