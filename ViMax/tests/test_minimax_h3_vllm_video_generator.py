import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interfaces.video_output import VideoOutput
from tools.video_generator_minimax_h3_vllm import (
    VideoGeneratorMinimaxH3VLLM,
    _post_multipart,
    _request_fields,
    _resolve_task,
)


class MinimaxH3VLLMVideoGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_requests_are_serialized_per_generator(self):
        active = 0
        max_active = 0

        async def fake_post(url, *, fields, file_fields, timeout):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return 200, {"content-type": "video/mp4"}, b"video"

        generator = VideoGeneratorMinimaxH3VLLM()
        with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
             patch.dict("os.environ", {}, clear=True):
            await asyncio.gather(
                generator.generate_single_video("first"),
                generator.generate_single_video("second"),
                generator.generate_single_video("third"),
            )

        self.assertEqual(max_active, 1)

    def test_auto_task_depends_on_reference_count(self):
        self.assertEqual(_resolve_task("auto", 0), "t2va")
        self.assertEqual(_resolve_task("auto", 1), "fl2va")
        self.assertEqual(_resolve_task("auto", 3), "ref2va")

    def test_unseeded_sentinel_omits_seed_without_changing_default(self):
        with patch.dict("os.environ", {"MINIMAX_H3_SEED": "none"}, clear=True):
            self.assertNotIn("seed", _request_fields("prompt", "fl2va", "16:9"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_request_fields("prompt", "t2va", "16:9")["seed"], "42")

    async def test_post_helper_uses_multipart_and_repeated_file_field_names(self):
        captured = {}

        class FakeResponse:
            status = 200
            headers = {"content-type": "video/mp4"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def read(self):
                return b"video"

        class FakeSession:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, data):
                captured["url"] = url
                captured["multipart"] = data._is_multipart
                captured["file_names"] = [
                    (field[0].get("name"), field[0].get("filename"))
                    for field in data._fields
                    if field[0].get("filename")
                ]
                return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_paths = [Path(tmp) / "one.png", Path(tmp) / "two.png"]
            for path in image_paths:
                path.write_bytes(b"image")
            with patch("tools.video_generator_minimax_h3_vllm.aiohttp.ClientSession", FakeSession):
                status, headers, body = await _post_multipart(
                    "http://vllm.test/v1/videos/sync",
                    fields={"prompt": "test"},
                    file_fields=[("input_references", str(path)) for path in image_paths],
                    timeout=30,
                )

        self.assertEqual(status, 200)
        self.assertEqual(body, b"video")
        self.assertEqual(headers["content-type"], "video/mp4")
        self.assertTrue(captured["multipart"])
        self.assertEqual(
            captured["file_names"],
            [("input_references", "one.png"), ("input_references", "two.png")],
        )

    async def test_sends_camera_master_form_fields_and_returns_mp4_bytes(self):
        captured = {}

        async def fake_post(url, *, fields, file_fields, timeout):
            captured.update(url=url, fields=fields, file_fields=file_fields, timeout=timeout)
            return 200, {"content-type": "video/mp4", "x-request-id": "req-1"}, b"mp4-data"

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            first.write_bytes(b"first-frame")
            output_path = Path(tmp) / "shot.mp4"
            generator = VideoGeneratorMinimaxH3VLLM(api_url="http://vllm.test/v1/videos/sync")
            with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
                 patch.dict("os.environ", {}, clear=True):
                output = await generator.generate_single_video(
                    prompt="A cyclist crosses the bridge",
                    reference_image_paths=[str(first)],
                    output_video_path=str(output_path),
                )
            report = json.loads(output_path.with_suffix(".minimax_h3_vllm.json").read_text(encoding="utf-8"))

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(output.fmt, "bytes")
        self.assertEqual(output.ext, "mp4")
        self.assertEqual(output.data, b"mp4-data")
        self.assertEqual(captured["url"], "http://vllm.test/v1/videos/sync")
        self.assertEqual(captured["fields"]["prompt"], "A cyclist crosses the bridge")
        self.assertEqual(captured["fields"]["width"], "1024")
        self.assertEqual(captured["fields"]["height"], "576")
        self.assertEqual(captured["fields"]["fps"], "24")
        self.assertEqual(captured["fields"]["num_inference_steps"], "30")
        self.assertEqual(captured["fields"]["flow_shift"], "12")
        self.assertEqual(captured["fields"]["num_frames"], "124")
        self.assertEqual(json.loads(captured["fields"]["extra_params"])["task"], "fl2va")
        self.assertEqual(json.loads(captured["fields"]["extra_params"])["audio_flow_shift"], 3.0)
        self.assertEqual(captured["file_fields"], [("input_reference", str(first))])
        self.assertEqual(report["status_code"], 200)
        self.assertEqual(report["reference_image_count"], 1)
        self.assertNotIn("prompt", report)

    async def test_text_generation_uses_aspect_ratio_and_environment_settings(self):
        captured = {}

        async def fake_post(url, *, fields, file_fields, timeout):
            captured.update(url=url, fields=fields, file_fields=file_fields, timeout=timeout)
            return 200, {"content-type": "video/mp4"}, b"video"

        generator = VideoGeneratorMinimaxH3VLLM()
        with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
             patch.dict("os.environ", {
                 "MINIMAX_H3_TASK": "auto",
                 "MINIMAX_H3_WIDTH": "1280",
                 "MINIMAX_H3_HEIGHT": "720",
                 "MINIMAX_H3_FPS": "30",
                 "MINIMAX_H3_NUM_FRAMES": "90",
                 "MINIMAX_H3_STEPS": "36",
                 "MINIMAX_H3_VLLM_FLOW_SHIFT": "9.5",
                 "MINIMAX_H3_SEED": "7",
                 "MINIMAX_H3_VLLM_TIMEOUT_SEC": "77",
                 "MINIMAX_H3_VLLM_ASPECT_RATIO": "4:3",
             }, clear=True):
            await generator.generate_single_video("text prompt", aspect_ratio="2:1")

        self.assertEqual(captured["url"], "http://127.0.0.1:9001/v1/videos/sync")
        self.assertEqual(captured["fields"]["width"], "1280")
        self.assertEqual(captured["fields"]["height"], "720")
        self.assertEqual(captured["fields"]["fps"], "30")
        self.assertEqual(captured["fields"]["num_inference_steps"], "36")
        self.assertEqual(captured["fields"]["flow_shift"], "9.5")
        self.assertEqual(captured["fields"]["seed"], "7")
        self.assertEqual(captured["fields"]["aspect_ratio"], "4:3")
        self.assertNotIn("num_frames", captured["fields"])
        self.assertEqual(json.loads(captured["fields"]["extra_params"])["task"], "t2va")
        self.assertEqual(captured["file_fields"], [])
        self.assertEqual(captured["timeout"], 77.0)

    async def test_ref2va_uploads_repeated_input_references(self):
        captured = {}

        async def fake_post(url, *, fields, file_fields, timeout):
            captured["task"] = json.loads(fields["extra_params"])["task"]
            captured["file_fields"] = file_fields
            return 200, {"content-type": "video/mp4"}, b"video"

        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "one.png", Path(tmp) / "two.png"]
            for path in paths:
                path.write_bytes(b"frame")
            generator = VideoGeneratorMinimaxH3VLLM()
            with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
                 patch.dict("os.environ", {}, clear=True):
                await generator.generate_single_video("prompt", [str(p) for p in paths])

        self.assertEqual(captured["task"], "ref2va")
        self.assertEqual(captured["file_fields"], [("input_references", str(p)) for p in paths])

    async def test_non_success_response_raises_with_status_and_body(self):
        async def fake_post(url, *, fields, file_fields, timeout):
            return 503, {"content-type": "application/json"}, b"vllm unavailable"

        generator = VideoGeneratorMinimaxH3VLLM()
        with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
             patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503.*vllm unavailable"):
                await generator.generate_single_video("prompt")

    async def test_empty_success_response_is_rejected(self):
        async def fake_post(url, *, fields, file_fields, timeout):
            return 200, {"content-type": "video/mp4"}, b""

        generator = VideoGeneratorMinimaxH3VLLM()
        with patch("tools.video_generator_minimax_h3_vllm._post_multipart", fake_post), \
             patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "empty video response"):
                await generator.generate_single_video("prompt")


if __name__ == "__main__":
    unittest.main()
