import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from tools.image_generator_nanobanana_compatible_api import (
    ImageGeneratorNanobananaCompatibleAPI,
    NanoBananaResponseError,
    _retryable,
    _decode_image_response,
)


def _png_b64(size: tuple[int, int] = (16, 9)) -> str:
    output = BytesIO()
    Image.new("RGB", size, "green").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _png_bytes() -> bytes:
    return base64.b64decode(_png_b64())


class NanoBananaCompatibleImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def test_empty_chat_response_is_retryable(self):
        self.assertTrue(_retryable(NanoBananaResponseError("choices_count=0")))
        self.assertTrue(_retryable(ValueError("Generated image is not landscape-oriented (1024x1024); retrying for a landscape frame")))

    async def test_empty_chat_response_reports_schema(self):
        post = AsyncMock(return_value=(200, {"choices": [], "usage": {}}))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")

        with self.assertRaises(NanoBananaResponseError) as raised:
            with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
                await generator.generate_single_image("a picture")

        message = str(raised.exception)
        self.assertIn("top_level_keys=choices,usage", message)
        self.assertIn("choices_count=0", message)
        self.assertEqual(post.await_count, 3)

    async def test_unrecognized_success_payload_reports_schema_without_image_data(self):
        payload = "Image generation unavailable: quota exceeded. API key=super-secret https://cdn.example/error?token=secret"

        with self.assertRaisesRegex(ValueError, "no image data URI") as raised:
            await _decode_image_response(payload, None)

        message = str(raised.exception)
        self.assertIn("chat completion", message)
        self.assertIn("quota exceeded", message)
        self.assertIn("API key=[REDACTED]", message)
        self.assertIn("[URL REDACTED]", message)
        self.assertNotIn("super-secret", message)
        self.assertNotIn("token=secret", message)

    async def test_generation_error_includes_finish_reason_and_sanitized_content(self):
        post = AsyncMock(return_value=(200, {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Image generation was refused by policy. Bearer private-token",
                },
            }],
            "usage": {"completion_tokens": 12},
        }))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")

        with self.assertRaises(NanoBananaResponseError) as raised:
            with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
                await generator.generate_single_image("a picture")

        message = str(raised.exception)
        self.assertIn("finish_reason=stop", message)
        self.assertIn("Image generation was refused by policy", message)
        self.assertIn("Bearer [REDACTED]", message)
        self.assertNotIn("private-token", message)
        self.assertEqual(post.await_count, 3)

    async def test_generation_posts_chat_completion_and_decodes_base64_data_uri(self):
        post = AsyncMock(return_value=(200, {
            "choices": [{"message": {"content": f"Generated image: data:image/png;base64,{_png_b64()}"}}]
        }))
        generator = ImageGeneratorNanobananaCompatibleAPI(
            api_key="secret", base_url="https://api.example/v1", model="nano-banana-2.1", timeout_sec=480
        )
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            result = await generator.generate_single_image("cinematic coast", aspect_ratio="16:9")

        self.assertEqual(post.await_args.args[0], "https://api.example/v1/chat/completions")
        self.assertEqual(post.await_args.kwargs["headers"]["Authorization"], "Bearer secret")
        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["model"], "nano-banana-2.1")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertIn("16:9 aspect ratio", payload["messages"][0]["content"][0]["text"])
        self.assertEqual(post.await_args.kwargs["timeout"].total, 480)
        self.assertEqual(result.data.size, (16, 9))

    async def test_chat_request_explicitly_requests_text_and_image_modalities(self):
        post = AsyncMock(return_value=(200, {
            "choices": [{"message": {"content": f"data:image/png;base64,{_png_b64()}"}}]
        }))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            await generator.generate_single_image("a picture")

        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["modalities"], ["TEXT", "IMAGE"])
        request_text = payload["messages"][0]["content"][0]["text"]
        self.assertIn("wholly original fictional image", request_text)
        self.assertIn("Do not reproduce or imitate", request_text)

    async def test_originality_guard_can_be_disabled_for_provider_compatibility(self):
        post = AsyncMock(return_value=(200, {
            "choices": [{"message": {"content": f"data:image/png;base64,{_png_b64()}"}}]
        }))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        with patch.dict("os.environ", {"NANO_BANANA_ORIGINALITY_GUARD": "false"}), \
             patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            await generator.generate_single_image("a picture")

        request_text = post.await_args.kwargs["payload"]["messages"][0]["content"][0]["text"]
        self.assertNotIn("wholly original fictional image", request_text)

    async def test_blank_white_frame_is_rendered_locally(self):
        post = AsyncMock()
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        prompt = (
            "Create a full-frame static image of a completely uniform, untextured white field. "
            "No characters, canvas, easel, horizon, shadows, gradients, texture, or any other "
            "visual elements should be visible."
        )
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            result = await generator.generate_single_image(prompt, size="1600x900")

        self.assertEqual(result.data.size, (1600, 900))
        self.assertEqual(result.data.getpixel((800, 450)), (255, 255, 255))
        post.assert_not_awaited()

    async def test_structured_message_images_are_decoded(self):
        post = AsyncMock(return_value=(200, {
            "choices": [{
                "message": {
                    "content": "> unknown finish reason: [IMAGE_OTHER]",
                    "images": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_png_b64()}"},
                    }],
                }
            }]
        }))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            result = await generator.generate_single_image("a picture")

        self.assertEqual(result.data.size, (16, 9))

    async def test_square_response_is_retried_until_strict_landscape(self):
        post = AsyncMock(side_effect=[
            (200, {"choices": [{"message": {"content": f"data:image/png;base64,{_png_b64((1024, 1024))}"}}]}),
            (200, {"choices": [{"message": {"content": f"data:image/png;base64,{_png_b64()}"}}]}),
        ])
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", post):
            result = await generator.generate_single_image("a picture")

        self.assertEqual(result.data.size, (16, 9))
        self.assertEqual(post.await_count, 2)

    async def test_reference_image_uses_chat_completion_multimodal_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "ref.png"
            Image.new("RGB", (8, 8), "red").save(reference)
            response = AsyncMock(return_value=(200, {
                "choices": [{"message": {"content": f"data:image/png;base64,{_png_b64()}"}}]
            }))
            generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
            with patch("tools.image_generator_nanobanana_compatible_api._post_json", response):
                await generator.generate_single_image("edit", [str(reference)])

        self.assertEqual(response.await_args.args[0], "https://api.xi-ai.cn/v1/chat/completions")
        content = response.await_args.kwargs["payload"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("edit", content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    async def test_chat_completion_image_url_is_downloaded(self):
        response = AsyncMock(return_value=(200, {
            "choices": [{"message": {"content": "Image: https://cdn.example/generated.png"}}]
        }))
        download = AsyncMock(return_value=(_png_bytes(), "image/png"))
        generator = ImageGeneratorNanobananaCompatibleAPI(api_key="secret")
        with patch("tools.image_generator_nanobanana_compatible_api._post_json", response), \
             patch("tools.image_generator_nanobanana_compatible_api._download_image", download):
            result = await generator.generate_single_image("a picture")
        self.assertEqual(download.await_args.args[0], "https://cdn.example/generated.png")
        self.assertEqual(result.data.size, (16, 9))

if __name__ == "__main__":
    unittest.main()
