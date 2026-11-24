from pathlib import Path


def _env_value(path: Path, key: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(f"{key}="):
            return raw.split("=", 1)[1].strip()
    raise AssertionError(f"{key} not found in {path}")


def test_flux_local_env_defaults_are_16_9() -> None:
    root = Path(__file__).resolve().parents[1]
    for env_file in [root / ".env", root / ".env.example"]:
        width = int(_env_value(env_file, "FLUX_LOCAL_WIDTH"))
        height = int(_env_value(env_file, "FLUX_LOCAL_HEIGHT"))
        assert width == 1024
        assert height == 576
        assert abs((width / height) - (16 / 9)) < 0.01


def test_flux_local_infer_uses_new_pytorch_alloc_conf() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "tool_hub/code/image_generation/providers/flux_local_infer.py").read_text(encoding="utf-8")
    assert "PYTORCH_ALLOC_CONF" in text
    assert "PYTORCH_CUDA_ALLOC_CONF" not in text
