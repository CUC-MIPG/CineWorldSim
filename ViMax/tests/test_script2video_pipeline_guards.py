import json
import tempfile
import unittest
import asyncio
from pathlib import Path

from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from interfaces.video_output import VideoOutput
from pipelines.script2video_pipeline import Script2VideoPipeline, _camera_frame_priority, _group_shots_into_cameras


class FlakyCameraImageGenerator:
    def __init__(self):
        self.calls = 0

    async def construct_camera_tree(self, cameras, shot_descs):
        self.calls += 1
        if self.calls == 1:
            return ["not-a-camera"]
        return cameras


class CaptureVideoGenerator:
    def __init__(self):
        self.kwargs = None

    async def generate_single_video(self, **kwargs):
        self.kwargs = kwargs
        return VideoOutput(fmt="bytes", ext="mp4", data=b"video")


class Script2VideoPipelineGuardTests(unittest.IsolatedAsyncioTestCase):
    def test_root_camera_with_earliest_shot_is_prioritized(self):
        cameras = [
            Camera(idx=2, active_shot_idxs=[2], parent_cam_idx=None, parent_shot_idx=None),
            Camera(idx=0, active_shot_idxs=[0, 1], parent_cam_idx=None, parent_shot_idx=None),
            Camera(idx=1, active_shot_idxs=[3], parent_cam_idx=2, parent_shot_idx=2),
        ]

        ordered = sorted(cameras, key=_camera_frame_priority)

        self.assertEqual([camera.idx for camera in ordered], [0, 2, 1])

    async def test_partial_portrait_registry_is_resumed_for_missing_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp,
            )
            registry_path = Path(tmp) / "character_portraits_registry.json"
            registry_path.write_text(json.dumps({"Existing": {"front": {"path": "existing.png"}}}))
            characters = [
                CharacterInScene(idx=0, identifier_in_scene="Existing", is_visible=True, static_features="person"),
                CharacterInScene(idx=1, identifier_in_scene="Missing", is_visible=True, static_features="person"),
            ]

            async def generate_missing(*, characters, character_portraits_registry, style, progress=None):
                self.assertEqual(set(character_portraits_registry), {"Existing"})
                return {**character_portraits_registry, "Missing": {"front": {"path": "missing.png"}}}

            pipeline.generate_character_portraits = generate_missing
            result = await pipeline._load_or_generate_character_portraits(characters, "cinematic")

            self.assertEqual(set(result), {"Existing", "Missing"})
            self.assertEqual(json.loads(registry_path.read_text()), result)

    async def test_video_generator_receives_output_path_for_provider_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = CaptureVideoGenerator()
            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=object(),
                video_generator=generator,
                working_dir=tmp,
            )
            pipeline.frame_events[0] = {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            shot_dir = Path(tmp) / "shots" / "0"
            shot_dir.mkdir(parents=True)
            (shot_dir / "first_frame.png").write_bytes(b"frame")
            shot = ShotDescription(
                idx=0,
                is_last=True,
                cam_idx=0,
                visual_desc="a person walks",
                variation_type="small",
                variation_reason="simple movement",
                ff_desc="walking",
                ff_vis_char_idxs=[],
                lf_desc="walking",
                lf_vis_char_idxs=[],
                motion_desc="walks forward",
                audio_desc="quiet street",
            )

            await pipeline.generate_video_for_single_shot(shot)

            self.assertEqual(generator.kwargs["output_video_path"], str(shot_dir / "video.mp4"))

    def test_group_shots_into_cameras_does_not_use_camera_idx_as_list_index(self):
        shots = [
            ShotDescription(idx=0, is_last=False, cam_idx=2, visual_desc="a", variation_type="small", variation_reason="same", ff_desc="a", ff_vis_char_idxs=[], lf_desc="a", lf_vis_char_idxs=[], motion_desc="a", audio_desc="none"),
            ShotDescription(idx=1, is_last=True, cam_idx=5, visual_desc="b", variation_type="small", variation_reason="same", ff_desc="b", ff_vis_char_idxs=[], lf_desc="b", lf_vis_char_idxs=[], motion_desc="b", audio_desc="none"),
            ShotDescription(idx=2, is_last=True, cam_idx=2, visual_desc="c", variation_type="small", variation_reason="same", ff_desc="c", ff_vis_char_idxs=[], lf_desc="c", lf_vis_char_idxs=[], motion_desc="c", audio_desc="none"),
        ]
        cameras = _group_shots_into_cameras(shots)
        self.assertEqual([camera.idx for camera in cameras], [2, 5])
        self.assertEqual(cameras[0].active_shot_idxs, [0, 2])
        self.assertEqual(cameras[1].active_shot_idxs, [1])

    async def test_plan_text_artifacts_retries_bad_camera_tree_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.camera_image_generator = FlakyCameraImageGenerator()

            async def design_storyboard(script, characters, user_requirement, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "audio_desc": "waves"}]

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "variation_type": "small", "variation_reason": "simple", "ff_desc": "start", "ff_vis_char_idxs": [], "lf_desc": "end", "lf_vis_char_idxs": [], "motion_desc": "walk", "audio_desc": "waves"}]

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            events = []
            result = await pipeline.plan_text_artifacts(
                "script",
                "req",
                "style",
                characters=[{"idx": 0, "identifier_in_scene": "Man", "is_visible": True, "static_features": "adult", "dynamic_features": "coat"}],
                progress=lambda stage, message, metadata=None: events.append(stage),
                quiet=True,
            )

            self.assertEqual(pipeline.camera_image_generator.calls, 2)
            self.assertIn("construct_camera_tree_retry", events)
            self.assertEqual(result["camera_tree"][0].idx, 3)
            self.assertTrue((Path(tmp) / "camera_tree.json").exists())


if __name__ == "__main__":
    unittest.main()
