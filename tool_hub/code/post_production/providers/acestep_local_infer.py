from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ACESTEP_ROOT = "/data1/AgentG/hhb/ACE-Step-1.5"
DEFAULT_CHECKPOINTS_DIR = "/data1/AgentG/hhb/CameraMaster/tool_hub/models/bgm/Ace-Step1.5"
DEFAULT_OUTPUT_DIR = "/data1/AgentG/hhb/CameraMaster/outputs/tool_hub/acestep_local"


def _ensure_import_path(acestep_root: Path) -> None:
    if str(acestep_root) not in sys.path:
        sys.path.insert(0, str(acestep_root))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ACE-Step local inference (single-shot)")
    p.add_argument("--acestep-root", type=Path, default=Path(os.getenv("ACESTEP_PROJECT_ROOT", DEFAULT_ACESTEP_ROOT)))
    p.add_argument("--checkpoints-dir", type=Path, default=Path(os.getenv("ACESTEP_CHECKPOINTS_DIR", DEFAULT_CHECKPOINTS_DIR)))
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--config-path", type=str, default="acestep-v15-turbo")
    p.add_argument("--lm-model-path", type=str, default="acestep-5Hz-lm-1.7B")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps", "xpu"])
    p.add_argument("--lm-backend", type=str, default="pt", choices=["pt", "vllm", "mlx"])
    p.add_argument("--with-lm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--lyrics", type=str, default="[Instrumental]")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--bpm", type=int, default=100)
    p.add_argument("--keyscale", type=str, default="C Major")
    p.add_argument("--timesignature", type=str, default="4")
    p.add_argument("--vocal-language", type=str, default="en")
    p.add_argument("--inference-steps", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--audio-format", type=str, default="wav", choices=["wav", "flac", "mp3", "wav32", "opus", "aac"])
    return p


def _result_to_jsonable(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict()
        except Exception:
            pass
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


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


def main() -> None:
    args = _build_parser().parse_args()

    args.acestep_root = args.acestep_root.resolve()
    args.checkpoints_dir = args.checkpoints_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.acestep_root.exists():
        raise FileNotFoundError(f"acestep root not found: {args.acestep_root}")
    if not args.checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {args.checkpoints_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(args.checkpoints_dir)

    _ensure_import_path(args.acestep_root)

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(args.acestep_root),
        config_path=args.config_path,
        device=args.device,
    )
    if not ok:
        raise RuntimeError(f"DiT init failed: {status}")

    llm_handler = LLMHandler()
    if args.with_lm:
        llm_status, llm_ok = llm_handler.initialize(
            checkpoint_dir=str(args.checkpoints_dir),
            lm_model_path=args.lm_model_path,
            backend=args.lm_backend,
            device=args.device,
        )
        if not llm_ok:
            raise RuntimeError(f"LLM init failed: {llm_status}")

    params = GenerationParams(
        task_type="text2music",
        caption=args.prompt,
        lyrics=args.lyrics,
        duration=args.duration,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        vocal_language=args.vocal_language,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        thinking=bool(args.with_lm and args.thinking),
        use_cot_caption=bool(args.with_lm and args.thinking),
        use_cot_language=bool(args.with_lm and args.thinking),
        use_cot_metas=bool(args.with_lm and args.thinking),
    )
    config = GenerationConfig(
        batch_size=args.batch_size,
        audio_format=args.audio_format,
        allow_lm_batch=bool(args.with_lm),
    )

    t0 = time.time()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(args.output_dir),
    )
    dt = time.time() - t0

    payload = _result_to_jsonable(result)
    payload["elapsed_sec"] = round(dt, 3)
    payload["run_config"] = {
        "acestep_root": str(args.acestep_root),
        "checkpoints_dir": str(args.checkpoints_dir),
        "config_path": args.config_path,
        "lm_model_path": args.lm_model_path,
        "device": args.device,
        "with_lm": args.with_lm,
        "thinking": args.thinking,
        "audio_format": args.audio_format,
    }

    payload = _sanitize_for_json(payload)
    report_path = args.output_dir / "acestep_local_infer_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok_flag = bool(payload.get("success", True))
    print(f"[acestep_local_infer] success={ok_flag} elapsed_sec={payload['elapsed_sec']}")
    print(f"[acestep_local_infer] report={report_path}")
    for row in payload.get("audios", []) or []:
        print(f"[acestep_local_infer] audio={row.get('path', '')}")

    if not ok_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
