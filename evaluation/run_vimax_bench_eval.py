"""Run ViMax-Bench-style visual consistency evaluation on CameraMaster outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vimax_bench_adapter import build_vimax_samples, load_scene_map
from vimax_metrics import (
    DeterministicFakeEncoder,
    ViCLIPEncoder,
    ViCLIPHTTPEncoder,
    aggregate_viclip_metrics,
)


def _feature_cache_path(cache_dir: Path, video_path: str, encoder_config: dict[str, Any]) -> Path:
    path = Path(video_path).resolve()
    stat = path.stat()
    key = json.dumps({
        "video": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        **encoder_config,
    }, sort_keys=True)
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npy"


def _encode_with_cache(encoder: Any, paths: list[str], cache_dir: Path | None, encoder_config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    features = []
    cache_hits = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        cache_path = _feature_cache_path(cache_dir, path, encoder_config) if cache_dir else None
        if cache_path and cache_path.is_file():
            value = np.load(cache_path)
            cache_hits.append(path)
        else:
            value = np.asarray(encoder.encode_video(path), dtype=np.float32)
            if cache_path:
                np.save(cache_path, value)
        if value.ndim != 1:
            raise ValueError(f"encoder must return one feature vector per shot; got {value.shape} for {path}")
        features.append(value)
    return np.stack(features), cache_hits


def aggregate_dataset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report story-macro means and pair-weighted means without mixing levels."""
    summary: dict[str, Any] = {}
    for metric in ("IC", "CC", "GC"):
        available = [
            record["metrics"][metric]
            for record in records
            if record.get("metrics", {}).get(metric, {}).get("score") is not None
        ]
        weighted_pair_count = sum(int(row.get("pair_count", 0)) for row in available)
        summary[metric] = {
            "story_macro_mean": float(np.mean([row["score"] for row in available])) if available else None,
            "pair_weighted_mean": (
                float(sum(row["score"] * row.get("pair_count", 0) for row in available) / weighted_pair_count)
                if weighted_pair_count else None
            ),
            "story_count": len(available),
            "pair_count": weighted_pair_count,
        }
    return summary


def evaluate(
    inputs_json: str | Path,
    output: str | Path,
    track: str | None = None,
    shots_per_sample: int | None = None,
    scene_map: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    fake_encoder: bool = False,
    backend: str = "http",
    embed_url: str = "http://172.17.34.43:8890/embed",
    num_frames: int = 8,
    resolution: int = 224,
    timeout: float = 600,
) -> dict[str, Any]:
    scene_labels = load_scene_map(scene_map)
    samples = build_vimax_samples(inputs_json, track=track, shots_per_sample=shots_per_sample, scene_map=scene_labels)
    if fake_encoder:
        encoder: Any = DeterministicFakeEncoder()
        backend = "fake"
    elif backend == "http":
        encoder = ViCLIPHTTPEncoder(embed_url, num_frames, resolution, timeout)
    elif backend == "local":
        encoder = ViCLIPEncoder(checkpoint=checkpoint, device=device, frame_count=num_frames, image_size=resolution)
    else:
        raise ValueError("backend must be one of: http, local")
    cache_path = Path(cache_dir).resolve() if cache_dir else None
    if backend == "http":
        encoder_config = {"backend": "http", "endpoint": embed_url, "num_frames": num_frames, "resolution": resolution}
    elif backend == "local":
        encoder_config = {"backend": "local", "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
                          "device": device, "num_frames": num_frames, "resolution": resolution}
    else:
        encoder_config = {"backend": "fake", "num_frames": num_frames, "resolution": resolution}
    records = []
    for sample in samples:
        shots = sample["shots"]
        paths = [shot["video_path"] for shot in shots]
        features, cache_hits = _encode_with_cache(encoder, paths, cache_path, encoder_config)
        metrics = aggregate_viclip_metrics(
            [shot["shot_id"] for shot in shots],
            [shot.get("scene_id") for shot in shots],
            features,
        )
        records.append({
            "sample_id": sample["sample_id"],
            "track": sample.get("track"),
            "protocol": sample["protocol"],
            "metrics": metrics,
            "feature_extraction": {
                "encoder": "deterministic_fake" if fake_encoder else "ViCLIP",
                "backend": backend,
                "endpoint": embed_url if backend == "http" else None,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "device": device,
                "frame_count": num_frames,
                "image_size": resolution,
                "cache_hit_count": len(cache_hits),
                "cache_key_config": encoder_config,
                "shot_count": len(paths),
            },
            "shots": [{"shot_id": shot["shot_id"], "scene_id": shot.get("scene_id"), "video_path": shot["video_path"]} for shot in shots],
        })
    report = {
        "schema_version": "vimax-bench-eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_inputs_json": str(Path(inputs_json).resolve()),
        "source_modified": False,
        "results_are_official_vimax_bench": False,
        "result_note": "CameraMaster adaptations are custom samples; use official ViMax-Bench files for official benchmark claims.",
        "encoder": "deterministic_fake" if fake_encoder else "ViCLIP",
        "backend": backend,
        "dataset_aggregate": aggregate_dataset_metrics(records),
        "samples": records,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track")
    parser.add_argument("--shots-per-sample", type=int)
    parser.add_argument("--scene-map", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("http", "local"), default="http")
    parser.add_argument("--embed-url", default="http://172.17.34.43:8890/embed")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--fake-encoder", action="store_true", help="offline smoke test only; never use for benchmark reporting")
    args = parser.parse_args()
    report = evaluate(
        args.inputs_json, args.output, args.track, args.shots_per_sample,
        args.scene_map, args.checkpoint, args.device, args.cache_dir, args.fake_encoder,
        args.backend, args.embed_url, args.num_frames, args.resolution, args.timeout,
    )
    print(json.dumps({"sample_count": len(report["samples"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
