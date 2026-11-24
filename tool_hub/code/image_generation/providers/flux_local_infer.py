from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FLUX.2-klein-9B text-to-image inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch
    from diffusers import Flux2KleinPipeline

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(str(args.dtype).strip().lower(), torch.bfloat16)
    device = str(args.device).strip().lower() or "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = Flux2KleinPipeline.from_pretrained(
        str(args.model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    if device == "cpu":
        pipe.to("cpu")
    elif device.startswith("cuda"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to(device)
    else:
        pipe.to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(int(args.seed))

    result = pipe(
        prompt=args.prompt,
        width=int(args.width),
        height=int(args.height),
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        generator=generator,
    )
    image = result.images[0]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_path)

    payload = {
        "ok": True,
        "output_path": str(args.output_path),
        "model_dir": str(args.model_dir),
        "width": int(args.width),
        "height": int(args.height),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "device": device,
        "dtype": str(args.dtype),
        "seed": args.seed,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
