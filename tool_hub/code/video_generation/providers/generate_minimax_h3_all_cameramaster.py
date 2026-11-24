#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
import diffusers.hooks.group_offloading as group_offloading
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from torchao.quantization import Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig


TRANSFORMER_INT8_SKIP = [
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder",
    "time_proj",
    "token_refiner",
    "norm_out",
    "proj_out",
    "audio_proj_out",
]

TEXT_ENCODER_INT8_SKIP = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]

MiniMaxH3Reference = None
MiniMaxH3ImageReference = None
MiniMaxH3VideoReference = None
MiniMaxH3AudioReference = None


def workflow_for_task(task: str) -> str:
    if task == "t2va":
        return "t2va"
    if task in {"fl2va", "fl2va_last_frame", "fl2va_first_last"}:
        return "fl2va"
    if task == "ref2va":
        return "ref2va"
    raise ValueError(f"unsupported task: {task}")


def load_ref2va_reference_symbols():
    global MiniMaxH3Reference, MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference
    if (
        MiniMaxH3Reference is not None
        and MiniMaxH3ImageReference is not None
        and MiniMaxH3VideoReference is not None
        and MiniMaxH3AudioReference is not None
    ):
        return MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference
    try:
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference as _MiniMaxH3AudioReference,
            MiniMaxH3ImageReference as _MiniMaxH3ImageReference,
            MiniMaxH3Reference as _MiniMaxH3Reference,
            MiniMaxH3VideoReference as _MiniMaxH3VideoReference,
        )
    except ImportError as exc:
        raise ImportError(
            "The installed diffusers MiniMax-H3 branch does not expose MiniMax-H3 reference classes. "
            "Install a diffusers revision that provides the MiniMax-H3 ref2va workflow."
        ) from exc
    MiniMaxH3Reference = _MiniMaxH3Reference
    MiniMaxH3ImageReference = _MiniMaxH3ImageReference
    MiniMaxH3VideoReference = _MiniMaxH3VideoReference
    MiniMaxH3AudioReference = _MiniMaxH3AudioReference
    return MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference


def patch_torchao_group_offload_transfer() -> None:
    def move_torchao_tensor_no_alias(tensor: torch.Tensor, device: torch.device):
        cls = type(tensor)
        tensor_names = list(getattr(cls, "tensor_data_names", []))
        optional_tensor_names = [
            name for name in getattr(cls, "optional_tensor_data_names", []) if getattr(tensor, name, None) is not None
        ]
        attr_names = list(getattr(cls, "tensor_attribute_names", []))
        optional_attr_names = list(getattr(cls, "optional_tensor_attribute_names", []))

        tensors = [getattr(tensor, name).to(device) for name in tensor_names]
        attrs = [device if name == "device" else getattr(tensor, name) for name in attr_names]
        optional_tensors = [getattr(tensor, name).to(device) for name in optional_tensor_names]
        optional_attrs = [device if name == "device" else getattr(tensor, name) for name in optional_attr_names]
        return cls(*tensors, *attrs, *optional_tensors, *optional_attrs)

    def patched_transfer(self, tensor, source_tensor, default_stream):
        if group_offloading._is_torchao_tensor(source_tensor):
            moved = move_torchao_tensor_no_alias(source_tensor, self.onload_device)
        else:
            moved = source_tensor.to(self.onload_device, non_blocking=self.non_blocking)
        if group_offloading._is_torchao_tensor(tensor):
            group_offloading._swap_torchao_tensor(tensor, moved)
        else:
            tensor.data = moved
        if self.record_stream:
            if group_offloading._is_torchao_tensor(tensor):
                group_offloading._record_stream_torchao_tensor(tensor, default_stream)
            else:
                tensor.data.record_stream(default_stream)

    def offload_tensor_inplace(tensor, device: torch.device) -> None:
        if group_offloading._is_torchao_tensor(tensor):
            moved = move_torchao_tensor_no_alias(tensor, device)
            group_offloading._swap_torchao_tensor(tensor, moved)
        else:
            tensor.data = tensor.data.to(device, non_blocking=False)

    def patched_offload_to_memory(self):
        if self.stream is not None:
            if not self.record_stream:
                self._torch_accelerator_module.current_stream().synchronize()

            for group_module in self.modules:
                for param in group_module.parameters():
                    if group_offloading._is_torchao_tensor(param):
                        group_offloading._restore_torchao_tensor(param, self.cpu_param_dict[param])
                    else:
                        param.data = self.cpu_param_dict[param]
                for buffer in group_module.buffers():
                    if group_offloading._is_torchao_tensor(buffer):
                        group_offloading._restore_torchao_tensor(buffer, self.cpu_param_dict[buffer])
                    else:
                        buffer.data = self.cpu_param_dict[buffer]
            for param in self.parameters:
                if group_offloading._is_torchao_tensor(param):
                    group_offloading._restore_torchao_tensor(param, self.cpu_param_dict[param])
                else:
                    param.data = self.cpu_param_dict[param]
            for buffer in self.buffers:
                if group_offloading._is_torchao_tensor(buffer):
                    group_offloading._restore_torchao_tensor(buffer, self.cpu_param_dict[buffer])
                else:
                    buffer.data = self.cpu_param_dict[buffer]
        else:
            for group_module in self.modules:
                for param in group_module.parameters():
                    offload_tensor_inplace(param, self.offload_device)
                for buffer in group_module.buffers():
                    offload_tensor_inplace(buffer, self.offload_device)
            for param in self.parameters:
                offload_tensor_inplace(param, self.offload_device)
            for buffer in self.buffers:
                offload_tensor_inplace(buffer, self.offload_device)

    group_offloading.ModuleGroup._transfer_tensor_to_device = patched_transfer
    group_offloading.ModuleGroup._offload_to_memory = patched_offload_to_memory


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def cuda_device(index: int | None = None) -> torch.device:
    if index is None:
        index = int(os.environ.get("MINIMAX_H3_RUNTIME_DEVICE_INDEX", "0"))
    return torch.device(f"cuda:{index}")


def gpu_mem(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(cuda_device())
    parts = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            allocated = torch.cuda.memory_allocated(index) / 1024**3
            reserved = torch.cuda.memory_reserved(index) / 1024**3
            peak = torch.cuda.max_memory_allocated(index) / 1024**3
            try:
                free, total = torch.cuda.mem_get_info(index)
                parts.append(
                    f"cuda:{index} alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB "
                    f"peak={peak:.2f}GiB free={free / 1024**3:.2f}GiB total={total / 1024**3:.2f}GiB"
                )
            except RuntimeError:
                parts.append(f"cuda:{index} alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")
    log(f"{label}: " + " | ".join(parts))


def output_offload_folder(output_dir: str) -> Path:
    return Path(output_dir).resolve() / "accelerate_offload"


def parse_max_memory(entries: list[str] | None) -> dict[int | str, str] | None:
    if not entries:
        return None
    max_memory: dict[int | str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"--max-memory must look like 0:30GiB, got {entry!r}")
        key, value = entry.split(":", 1)
        max_memory[int(key) if key.isdigit() else key] = value
    return max_memory


def transformer_quant_config() -> TorchAoConfig:
    return TorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=TRANSFORMER_INT8_SKIP,
    )


def text_encoder_quant_config() -> TransformersTorchAoConfig:
    return TransformersTorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=TEXT_ENCODER_INT8_SKIP,
    )


def init_pipeline(model_dir: Path, task: str) -> ModularPipeline:
    workflow = workflow_for_task(task)
    log(f"initializing {workflow} modular pipeline")
    return ModularPipeline.from_pretrained(str(model_dir), workflow=workflow, local_files_only=True)


def load_transformer(
    model_dir: Path,
    subfolder: str,
    parallel_mode: str,
    device_map: str,
    max_memory: dict[int | str, str] | None,
    offload_folder: Path | None = None,
) -> MiniMaxH3Transformer3DModel:
    kwargs = {}
    if parallel_mode == "balanced":
        kwargs.update(device_map=device_map, max_memory=max_memory)
        if offload_folder is not None:
            kwargs["offload_folder"] = str(offload_folder)
    return MiniMaxH3Transformer3DModel.from_pretrained(
        str(model_dir),
        subfolder=subfolder,
        dtype=torch.bfloat16,
        quantization_config=transformer_quant_config(),
        local_files_only=True,
        low_cpu_mem_usage=True,
        **kwargs,
    )


def load_text_encoder(
    model_dir: Path,
    parallel_mode: str,
    device_map: str,
    max_memory: dict[int | str, str] | None,
    text_encoder_placement: str = "offload",
    offload_folder: Path | None = None,
) -> Qwen3VLForConditionalGeneration:
    kwargs = {}
    if parallel_mode == "balanced" and text_encoder_placement == "cuda":
        kwargs.update(device_map=device_map, max_memory=max_memory, low_cpu_mem_usage=True)
        if offload_folder is not None:
            kwargs["offload_folder"] = str(offload_folder)
    elif parallel_mode == "balanced":
        # In the 2x5090 non-vLLM path, keep Qwen3-VL off GPU residency and use group offload below.
        kwargs.update(low_cpu_mem_usage=True)
    return Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=text_encoder_quant_config(),
        local_files_only=True,
        **kwargs,
    )


def load_pipeline(args: argparse.Namespace) -> ModularPipeline:
    model_dir = Path(args.model_dir).resolve()
    max_memory = parse_max_memory(args.max_memory)

    pipe = init_pipeline(model_dir, args.task)

    transformer_component = "transformer_ref" if args.task == "ref2va" else "transformer"
    transformer_subfolder = "transformer_ref" if args.task == "ref2va" else "transformer"

    log(f"loading int8 {transformer_component} with parallel_mode={args.parallel_mode}")
    offload_folder = Path(args.offload_folder).resolve() if args.offload_folder else output_offload_folder(args.output_dir)
    if args.parallel_mode == "balanced":
        offload_folder.mkdir(parents=True, exist_ok=True)
        log(f"accelerate offload_folder: {offload_folder}")
    transformer = load_transformer(
        model_dir, transformer_subfolder, args.parallel_mode, args.device_map, max_memory, offload_folder
    )
    gpu_mem(f"after {transformer_component} load")

    log(
        f"loading int8 Qwen3-VL text_encoder with parallel_mode={args.parallel_mode} "
        f"placement={args.text_encoder_placement}"
    )
    text_encoder = load_text_encoder(
        model_dir, args.parallel_mode, args.device_map, max_memory, args.text_encoder_placement, offload_folder
    )
    gpu_mem("after text_encoder load")

    pipe.update_components(**{transformer_component: transformer, "text_encoder": text_encoder})

    log("loading shared components")
    pipe.load_components(
        ["tokenizer", "processor", "scheduler", "audio_scheduler", "vae", "audio_vae"],
        pretrained_model_name_or_path=str(model_dir),
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if args.vae_use_tiling or args.vae_parallel_mode == "tile":
        for name in ("vae", "audio_vae"):
            component = getattr(pipe, name, None)
            if hasattr(component, "enable_tiling"):
                log(f"enabling {name} tiling")
                component.enable_tiling()
    if args.vae_use_slicing:
        for name in ("vae", "audio_vae"):
            component = getattr(pipe, name, None)
            if hasattr(component, "enable_slicing"):
                log(f"enabling {name} slicing")
                component.enable_slicing()

    getattr(pipe, transformer_component).requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    runtime_device = cuda_device(args.runtime_device_index)
    if args.parallel_mode == "single_offload":
        offload = dict(onload_device=runtime_device, offload_device=torch.device("cpu"), use_stream=True)
        log(f"enabling {transformer_component} block-level offload")
        getattr(pipe, transformer_component).enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload
        )
        if args.text_encoder_placement == "offload":
            log(f"enabling text_encoder leaf-level offload, use_stream={args.text_encoder_use_stream}")
            text_offload = dict(
                onload_device=runtime_device,
                offload_device=torch.device("cpu"),
                use_stream=args.text_encoder_use_stream,
            )
            apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **text_offload)
        elif args.text_encoder_placement == "cuda":
            log(f"moving text_encoder to {runtime_device}")
            pipe.text_encoder.to(runtime_device)
        elif args.text_encoder_placement == "cpu":
            log("keeping text_encoder on cpu")
    else:
        log(f"using accelerate device_map={args.device_map} for transformer components")
        if args.text_encoder_placement == "offload":
            log(f"enabling balanced-mode text_encoder leaf-level offload, use_stream={args.text_encoder_use_stream}")
            text_offload = dict(
                onload_device=runtime_device,
                offload_device=torch.device("cpu"),
                use_stream=args.text_encoder_use_stream,
            )
            apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **text_offload)
        elif args.text_encoder_placement == "cpu":
            log("keeping balanced-mode text_encoder on cpu")
        else:
            log("keeping balanced-mode text_encoder on accelerate device_map")

    if args.vae_placement == "cuda":
        log(f"moving VAE and audio VAE to {runtime_device}")
        pipe.vae.to(runtime_device)
        pipe.audio_vae.to(runtime_device)
    elif args.vae_placement == "offload":
        log(f"enabling VAE and audio VAE leaf-level offload to {runtime_device}")
        vae_offload = dict(onload_device=runtime_device, offload_device=torch.device("cpu"), use_stream=False)
        apply_group_offloading(pipe.vae, offload_type="leaf_level", **vae_offload)
        apply_group_offloading(pipe.audio_vae, offload_type="leaf_level", **vae_offload)
    else:
        log("keeping VAE and audio VAE on cpu")
    gpu_mem("after pipeline setup")
    return pipe


def load_audio_tensor(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32).T
    return torch.from_numpy(audio), int(sample_rate)


def resize_image_16x9(src: str, output_dir: Path, width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return src
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    out = output_dir / f"{src_path.stem}_{width}x{height}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        target_ratio = width / height
        current_ratio = im.width / im.height
        if current_ratio > target_ratio:
            new_w = int(im.height * target_ratio)
            left = max((im.width - new_w) // 2, 0)
            im = im.crop((left, 0, left + new_w, im.height))
        elif current_ratio < target_ratio:
            new_h = int(im.width / target_ratio)
            top = max((im.height - new_h) // 2, 0)
            im = im.crop((0, top, im.width, top + new_h))
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(out, quality=92)
    return str(out)


def prepare_ref_image(path: str, args: argparse.Namespace) -> str:
    if args.reference_resize_width <= 0 or args.reference_resize_height <= 0:
        return path
    return resize_image_16x9(
        path,
        Path(args.output_dir).resolve() / "prepared_references",
        args.reference_resize_width,
        args.reference_resize_height,
    )


def parse_reference(entry: str, args: argparse.Namespace | None = None) -> MiniMaxH3Reference:
    image_cls, video_cls, audio_cls = load_ref2va_reference_symbols()
    if "=" in entry:
        kind, value = entry.split("=", 1)
    elif ":" in entry:
        kind, value = entry.split(":", 1)
    else:
        raise ValueError(f"--reference must look like image:/path, video:/path or audio:/path, got {entry!r}")
    kind = kind.strip().lower()
    value = value.strip()
    if kind == "image":
        if args is not None:
            value = prepare_ref_image(value, args)
        return image_cls(load_image(value))
    if kind == "video":
        return video_cls(load_video(value))
    if kind == "audio":
        audio, sample_rate = load_audio_tensor(value)
        return audio_cls(audio, sample_rate=sample_rate)
    raise ValueError(f"unsupported reference kind {kind!r}; expected image, video or audio")


def build_call_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "generator": torch.Generator(device=cuda_device(args.runtime_device_index)).manual_seed(args.seed),
    }
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames

    if args.task == "t2va":
        return kwargs
    if args.task == "fl2va":
        if not args.image:
            raise ValueError("--task fl2va requires --image")
        kwargs["image"] = load_image(args.image)
        return kwargs
    if args.task == "fl2va_last_frame":
        if not args.last_image:
            raise ValueError("--task fl2va_last_frame requires --last-image")
        kwargs["last_image"] = load_image(args.last_image)
        return kwargs
    if args.task == "fl2va_first_last":
        if not args.image or not args.last_image:
            raise ValueError("--task fl2va_first_last requires both --image and --last-image")
        kwargs["image"] = load_image(args.image)
        kwargs["last_image"] = load_image(args.last_image)
        return kwargs
    if args.task == "ref2va":
        image_cls, video_cls, audio_cls = load_ref2va_reference_symbols()
        references = [parse_reference(entry, args) for entry in args.reference]
        references.extend(image_cls(load_image(prepare_ref_image(path, args))) for path in args.ref_image)
        references.extend(video_cls(load_video(path)) for path in args.ref_video)
        for path in args.ref_audio:
            audio, sample_rate = load_audio_tensor(path)
            references.append(audio_cls(audio, sample_rate=sample_rate))
        if not references:
            raise ValueError("--task ref2va requires at least one --reference/--ref-image/--ref-video/--ref-audio")
        kwargs["references"] = references
        return kwargs
    raise ValueError(f"unsupported task: {args.task}")


def normalize_audio(audio) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
        if audio.ndim == 3:
            audio = audio[0]
    else:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 3:
            audio = audio[0]
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2:
        raise ValueError(f"unsupported audio shape: {audio.shape}")
    if audio.shape[-1] in (1, 2) and audio.shape[0] > audio.shape[-1]:
        samples_by_channel = audio
    else:
        samples_by_channel = audio.T
    pcm = np.clip(samples_by_channel, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16)


def save_outputs(result, output_dir: Path, prefix: str, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = result.videos
    frames = videos[0] if isinstance(videos, list) and videos and isinstance(videos[0], list) else videos
    video_path = output_dir / f"{prefix}_video.mp4"
    log(f"saving video: {video_path}")
    export_to_video(frames, str(video_path), fps=fps)

    audio_path = output_dir / f"{prefix}_audio.wav"
    sampling_rate = int(result.sampling_rate)
    pcm = normalize_audio(result.audio)
    log(f"saving audio: {audio_path}")
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(pcm.tobytes())

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log("ffmpeg not found; saved video and audio separately")
        return

    muxed_path = output_dir / f"{prefix}_muxed.mp4"
    log(f"muxing video+audio: {muxed_path}")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(muxed_path),
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 t2va/fl2va/ref2va locally with visible progress.")
    parser.add_argument("--model-dir", default="./MiniMax-H3")
    parser.add_argument("--output-dir", default="./outputs/minimax_h3_all")
    parser.add_argument(
        "--task",
        choices=["t2va", "fl2va", "fl2va_last_frame", "fl2va_first_last", "ref2va"],
        default="t2va",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="First-frame/keyframe image path for fl2va.")
    parser.add_argument("--last-image", help="Last-frame image path for fl2va_last_frame or fl2va_first_last.")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Ordered ref2va reference, e.g. image:/path/a.png, video:/path/a.mp4, audio:/path/a.wav.",
    )
    parser.add_argument("--ref-image", action="append", default=[], help="Append an image reference for ref2va.")
    parser.add_argument("--ref-video", action="append", default=[], help="Append a video reference for ref2va.")
    parser.add_argument("--ref-audio", action="append", default=[], help="Append an audio reference for ref2va.")
    parser.add_argument("--reference-resize-width", type=int, default=0)
    parser.add_argument("--reference-resize-height", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--parallel-mode",
        choices=["single_offload", "balanced"],
        default="single_offload",
        help="single_offload is the verified 32GB single-card mode; balanced uses accelerate device_map across visible GPUs.",
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default="balanced",
        help="Only used with --parallel-mode balanced.",
    )
    parser.add_argument(
        "--max-memory",
        action="append",
        help="Per visible device memory limit for balanced mode, e.g. --max-memory 0:30GiB --max-memory 1:30GiB.",
    )
    parser.add_argument(
        "--text-encoder-placement",
        choices=["cuda", "offload", "cpu"],
        default="offload",
        help="Controls Qwen3-VL placement. In balanced mode, offload/cpu avoids keeping it resident on GPU.",
    )
    parser.add_argument("--text-encoder-use-stream", action="store_true")
    parser.add_argument("--runtime-device-index", type=int, default=0, help="Visible CUDA index for generator and offload onload device.")
    parser.add_argument("--vae-placement", choices=["cuda", "offload", "cpu"], default="offload")
    parser.add_argument("--offload-folder", default="", help="Accelerate disk offload folder for balanced mode.")
    parser.add_argument(
        "--vae-parallel-mode",
        choices=["none", "tile"],
        default="none",
        help="Non-vLLM approximation of official --vae-parallel-mode tile: enables VAE tiling.",
    )
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run inside a GPU-enabled shell.")

    patch_torchao_group_offload_transfer()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log(f"visible CUDA devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        log(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    os.environ["MINIMAX_H3_RUNTIME_DEVICE_INDEX"] = str(args.runtime_device_index)
    log(f"runtime_device_index: {args.runtime_device_index}")
    log(f"task: {args.task}")
    log(f"model_dir: {model_dir}")
    log(f"output_dir: {output_dir}")
    log(f"prompt: {args.prompt}")
    gpu_mem("start")

    pipe = load_pipeline(args)
    call_kwargs = build_call_kwargs(args)
    log(
        f"generating {args.task}: {args.width}x{args.height}, frames={args.num_frames}, "
        f"steps={args.steps}, seed={args.seed}"
    )

    with torch.inference_mode():
        result = pipe(**call_kwargs)
    gpu_mem("after generation")

    save_outputs(result, output_dir, prefix=f"minimax_h3_{args.task}", fps=args.fps)
    log("generation finished")

    del result, pipe
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem("after cleanup")


if __name__ == "__main__":
    main()
