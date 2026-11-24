from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.generation_backends import generate_video_minimax_h3


def _write_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (20, 40, 60)).save(path)
    return str(path)


def test_minimax_h3_video_modes_build_expected_tasks(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    seen_envs = []

    def fake_run(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        seen_envs.append(kwargs.get("env", {}))
        task = cmd[cmd.index("--task") + 1]
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"minimax_h3_{task}_muxed.mp4").write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("agents.generation_backends.subprocess.run", fake_run)

    img1 = _write_image(tmp_path / "first.png")
    img2 = _write_image(tmp_path / "last.png")
    common = {
        "prompt": "cinematic test",
        "model_dir": "/models/MiniMax-H3",
        "conda_env": "minimaxh3hf",
        "script_path": "/data1/AgentG/hhb/minimax_h3/generate_minimax_h3_all.py",
        "width": 128,
        "height": 72,
        "num_frames": 124,
        "steps": 1,
        "timeout_sec": None,
    }

    assert generate_video_minimax_h3(
        **common,
        output_video_path=str(tmp_path / "text.mp4"),
        image_paths=[],
        video_mode="text",
    )
    assert "--task" in calls[-1] and calls[-1][calls[-1].index("--task") + 1] == "t2va"
    assert "--image" not in calls[-1]

    assert generate_video_minimax_h3(
        **common,
        output_video_path=str(tmp_path / "image.mp4"),
        image_paths=[img1],
        video_mode="image",
    )
    assert calls[-1][calls[-1].index("--task") + 1] == "fl2va"
    assert calls[-1][calls[-1].index("--image") + 1] == str(Path(img1).resolve())

    assert generate_video_minimax_h3(
        **common,
        output_video_path=str(tmp_path / "first_last.mp4"),
        image_paths=[img1, img2],
        video_mode="first_last",
    )
    assert calls[-1][calls[-1].index("--task") + 1] == "fl2va_first_last"
    assert calls[-1][calls[-1].index("--image") + 1] == str(Path(img1).resolve())
    assert calls[-1][calls[-1].index("--last-image") + 1] == str(Path(img2).resolve())

    assert generate_video_minimax_h3(
        **common,
        output_video_path=str(tmp_path / "reference.mp4"),
        image_paths=[img1, img2],
        video_mode="reference",
    )
    assert calls[-1][calls[-1].index("--task") + 1] == "ref2va"
    assert calls[-1].count("--ref-image") == 2
    assert seen_envs[-1]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    for name in ["text.mp4", "image.mp4", "first_last.mp4", "reference.mp4"]:
        assert (tmp_path / name).read_bytes() == b"mp4"


def test_minimax_h3_vllm_builds_multipart_for_text_image_and_ref(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agents.generation_backends import generate_video_minimax_h3_vllm

    img1 = _write_image(tmp_path / "first.png")
    img2 = _write_image(tmp_path / "ref2.png")
    calls = []

    class FakeResp:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"mp4bytes"
        text = ""

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResp()

    monkeypatch.setattr("agents.generation_backends.requests_post", fake_post)

    assert generate_video_minimax_h3_vllm(
        prompt="text prompt",
        output_video_path=str(tmp_path / "t2va.mp4"),
        image_paths=[],
        video_mode="text",
        task="t2va",
        api_url="http://127.0.0.1:9001/v1/videos/sync",
    )
    assert calls[-1][1]["data"]["extra_params"].find('"task": "t2va"') >= 0
    assert calls[-1][1]["files"] is None
    assert calls[-1][1]["data"]["aspect_ratio"] == "16:9"

    assert generate_video_minimax_h3_vllm(
        prompt="image prompt",
        output_video_path=str(tmp_path / "fl2va.mp4"),
        image_paths=[img1],
        video_mode="image",
        task="fl2va",
        api_url="http://127.0.0.1:9001/v1/videos/sync",
    )
    assert calls[-1][1]["data"]["extra_params"].find('"task": "fl2va"') >= 0
    assert calls[-1][1]["files"][0][0] == "input_reference"

    assert generate_video_minimax_h3_vllm(
        prompt="ref prompt",
        output_video_path=str(tmp_path / "ref2va.mp4"),
        image_paths=[],
        reference_image_paths=[img1, img2],
        video_mode="reference",
        task="ref2va",
        api_url="http://127.0.0.1:9001/v1/videos/sync",
    )
    assert calls[-1][1]["data"]["extra_params"].find('"task": "ref2va"') >= 0
    assert calls[-1][1]["data"]["num_frames"] == "243"
    assert [row[0] for row in calls[-1][1]["files"]] == ["input_references", "input_references"]

    for name in ["t2va.mp4", "fl2va.mp4", "ref2va.mp4"]:
        assert (tmp_path / name).read_bytes() == b"mp4bytes"


def test_minimax_h3_vllm_optionally_uploads_reference_audio(tmp_path, monkeypatch):
    from agents.generation_backends import generate_video_minimax_h3_vllm

    img = _write_image(tmp_path / "ref.png")
    audio = tmp_path / "continuity.wav"
    audio.write_bytes(b"wav")
    calls = []

    class FakeResp:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"mp4bytes"
        text = ""

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResp()

    monkeypatch.setattr("agents.generation_backends.requests_post", fake_post)

    assert generate_video_minimax_h3_vllm(
        prompt="continuity prompt",
        output_video_path=str(tmp_path / "ref_audio.mp4"),
        reference_image_paths=[img],
        reference_audio_paths=[str(audio)],
        upload_reference_audio=True,
        video_mode="reference",
        task="ref2va",
    )
    uploaded_names = [row[1][0] for row in calls[-1][1]["files"]]
    assert uploaded_names == [Path(img).name, audio.name]


def test_minimax_h3_vllm_uses_num_frames_without_duration_param(tmp_path, monkeypatch):
    from agents.generation_backends import generate_video_minimax_h3_vllm

    img = _write_image(tmp_path / "ref.png")
    calls = []

    class FakeResp:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"mp4bytes"
        text = ""

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResp()

    monkeypatch.setattr("agents.generation_backends.requests_post", fake_post)

    assert generate_video_minimax_h3_vllm(
        prompt="243 frame prompt",
        output_video_path=str(tmp_path / "ref243.mp4"),
        reference_image_paths=[img],
        video_mode="reference",
        task="ref2va",
        num_frames=243,
        duration=None,
    )
    data = calls[-1][1]["data"]
    assert data["num_frames"] == "243"
    assert '"duration"' not in data["extra_params"]


def test_minimax_h3_vllm_writes_call_log_jsonl(tmp_path, monkeypatch):
    from agents.generation_backends import generate_video_minimax_h3_vllm

    img = _write_image(tmp_path / "ref.png")
    call_log = tmp_path / "logs" / "model_tool_calls.jsonl"
    monkeypatch.setenv("CAMERAMASTER_CALL_LOG_JSONL", str(call_log))

    class FakeResp:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"mp4bytes"
        text = ""

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        return FakeResp()

    monkeypatch.setattr("agents.generation_backends.requests_post", fake_post)

    assert generate_video_minimax_h3_vllm(
        prompt="logged prompt",
        output_video_path=str(tmp_path / "logged.mp4"),
        reference_image_paths=[img],
        video_mode="reference",
        task="ref2va",
        num_frames=243,
        duration=None,
    )

    rows = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows if row.get("tool") == "minimax_h3_vllm"] == ["start", "done"]
    start = next(row for row in rows if row.get("tool") == "minimax_h3_vllm" and row["status"] == "start")
    assert start["num_frames"] == 243
    assert start["reference_image_count"] == 1
