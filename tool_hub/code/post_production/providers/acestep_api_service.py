from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


DEFAULT_ACESTEP_ROOT = "/data1/AgentG/hhb/ACE-Step-1.5"
DEFAULT_CHECKPOINTS_DIR = "/data1/AgentG/hhb/CameraMaster/tool_hub/models/bgm/Ace-Step1.5"
DEFAULT_OUTPUT_DIR = "/data1/AgentG/hhb/CameraMaster/outputs/tool_hub/acestep_api"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Music prompt/caption")
    lyrics: str = Field(default="[Instrumental]")
    duration: float = Field(default=20.0, ge=10.0, le=600.0)
    bpm: int | None = Field(default=100, ge=30, le=300)
    keyscale: str = Field(default="C Major")
    timesignature: str = Field(default="4")
    vocal_language: str = Field(default="en")
    inference_steps: int = Field(default=8, ge=1, le=200)
    guidance_scale: float = Field(default=7.0)
    seed: int = Field(default=-1)
    batch_size: int = Field(default=1, ge=1, le=8)
    audio_format: str = Field(default="wav")
    with_lm: bool = Field(default=False)
    thinking: bool = Field(default=False)


def _sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            if obj.numel() <= 64:
                return obj.detach().cpu().tolist()
            return {
                "_type": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
    except Exception:
        pass
    return str(obj)


class AceStepRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.initialized = False
        self.dit_handler: Any | None = None
        self.llm_handler: Any | None = None
        self.GenerationParams: Any | None = None
        self.GenerationConfig: Any | None = None
        self.generate_music: Any | None = None
        self.output_dir: Path = Path(os.getenv("ACESTEP_API_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.acestep_root = Path(os.getenv("ACESTEP_PROJECT_ROOT", DEFAULT_ACESTEP_ROOT)).resolve()
        self.checkpoints_dir = Path(os.getenv("ACESTEP_CHECKPOINTS_DIR", DEFAULT_CHECKPOINTS_DIR)).resolve()
        self.config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
        self.lm_model_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
        self.device = os.getenv("ACESTEP_DEVICE", "auto")
        self.lm_backend = os.getenv("ACESTEP_LM_BACKEND", "pt")

    def initialize(self) -> None:
        if self.initialized:
            return

        if not self.acestep_root.exists():
            raise FileNotFoundError(f"acestep root not found: {self.acestep_root}")
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"checkpoints dir not found: {self.checkpoints_dir}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(self.checkpoints_dir)

        if str(self.acestep_root) not in sys.path:
            sys.path.insert(0, str(self.acestep_root))

        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler

        self.dit_handler = AceStepHandler()
        status, ok = self.dit_handler.initialize_service(
            project_root=str(self.acestep_root),
            config_path=self.config_path,
            device=self.device,
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {status}")

        self.llm_handler = LLMHandler()
        self.GenerationParams = GenerationParams
        self.GenerationConfig = GenerationConfig
        self.generate_music = generate_music
        self.initialized = True

    def _ensure_lm(self) -> None:
        assert self.llm_handler is not None
        if getattr(self.llm_handler, "llm_initialized", False):
            return
        llm_status, llm_ok = self.llm_handler.initialize(
            checkpoint_dir=str(self.checkpoints_dir),
            lm_model_path=self.lm_model_path,
            backend=self.lm_backend,
            device=self.device,
        )
        if not llm_ok:
            raise RuntimeError(f"LLM init failed: {llm_status}")

    def generate(self, req: GenerateRequest) -> dict[str, Any]:
        if not self.initialized:
            self.initialize()

        assert self.GenerationParams is not None
        assert self.GenerationConfig is not None
        assert self.generate_music is not None

        with self.lock:
            if req.with_lm:
                self._ensure_lm()

            params = self.GenerationParams(
                task_type="text2music",
                caption=req.prompt,
                lyrics=req.lyrics,
                duration=req.duration,
                bpm=req.bpm,
                keyscale=req.keyscale,
                timesignature=req.timesignature,
                vocal_language=req.vocal_language,
                inference_steps=req.inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                thinking=bool(req.with_lm and req.thinking),
                use_cot_caption=bool(req.with_lm and req.thinking),
                use_cot_language=bool(req.with_lm and req.thinking),
                use_cot_metas=bool(req.with_lm and req.thinking),
            )
            config = self.GenerationConfig(
                batch_size=req.batch_size,
                audio_format=req.audio_format,
                allow_lm_batch=bool(req.with_lm),
            )

            t0 = time.time()
            result = self.generate_music(
                dit_handler=self.dit_handler,
                llm_handler=self.llm_handler,
                params=params,
                config=config,
                save_dir=str(self.output_dir),
            )
            elapsed = time.time() - t0

            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            payload["elapsed_sec"] = round(elapsed, 3)
            payload["output_dir"] = str(self.output_dir)
            payload["audio_urls"] = []
            for row in payload.get("audios", []) or []:
                p = str(row.get("path", ""))
                if p:
                    name = Path(p).name
                    payload["audio_urls"].append(f"/files/{name}")
            return _sanitize_for_json(payload)


runtime = AceStepRuntime()
app = FastAPI(title="ACE-Step Minimal API Service", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(runtime.output_dir)), name="files")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "initialized": runtime.initialized,
        "acestep_root": str(runtime.acestep_root),
        "checkpoints_dir": str(runtime.checkpoints_dir),
        "output_dir": str(runtime.output_dir),
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    try:
        return runtime.generate(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
