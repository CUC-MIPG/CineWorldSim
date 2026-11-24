"""ViMax-Bench visual consistency metrics and a lazy ViCLIP wrapper."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import requests


def extract_uniform_indices(num_frames: int, count: int = 8) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if num_frames == 1:
        return np.zeros(count, dtype=np.int64)
    return np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int64)


def normalize_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features must be [N,D], got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("features contain a zero vector")
    return array / norms


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        raise ValueError("cosine vectors must be non-empty and have the same dimension")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or norm_a == 0 or norm_b == 0:
        raise ValueError("cosine vectors must be finite and have non-zero norm")
    score = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(score, -1.0, 1.0))


def _summary(values: list[float], pairs: list[list[str]]) -> dict[str, Any]:
    if not values:
        return {"score": None, "pair_count": 0, "mean": None, "std": None, "pairs": []}
    return {
        "score": float(np.mean(values)),
        "pair_count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "pairs": [{"shot_a": p[0], "shot_b": p[1], "cosine": v} for p, v in zip(pairs, values)],
    }


def aggregate_viclip_metrics(shot_ids: Sequence[str], scene_ids: Sequence[Any], features: np.ndarray) -> dict[str, Any]:
    if len(shot_ids) != len(scene_ids) or len(shot_ids) != len(features):
        raise ValueError("shot_ids, scene_ids and features must have the same length")
    if len(shot_ids) < 2:
        empty = _summary([], [])
        return {"IC": empty, "CC": empty, "GC": empty, "warnings": ["At least two shots are required for pairwise metrics."]}
    normalized = normalize_features(features)
    same_values: list[float] = []
    same_pairs: list[list[str]] = []
    cross_values: list[float] = []
    cross_pairs: list[list[str]] = []
    all_values: list[float] = []
    all_pairs: list[list[str]] = []
    missing_scene = any(scene is None or str(scene).strip() == "" for scene in scene_ids)
    for i, j in itertools.combinations(range(len(shot_ids)), 2):
        cosine = cosine_similarity(normalized[i], normalized[j])
        pair = [str(shot_ids[i]), str(shot_ids[j])]
        all_values.append(cosine)
        all_pairs.append(pair)
        if not missing_scene:
            if str(scene_ids[i]) == str(scene_ids[j]):
                same_values.append(cosine)
                same_pairs.append(pair)
            else:
                cross_values.append(cosine)
                cross_pairs.append(pair)
    warnings = []
    if missing_scene:
        warnings.append("Scene labels are incomplete; IC and CC are not computed. GC remains valid.")
    return {
        "IC": _summary(same_values, same_pairs) if not missing_scene else _summary([], []),
        "CC": _summary(cross_values, cross_pairs) if not missing_scene else _summary([], []),
        "GC": _summary(all_values, all_pairs),
        "warnings": warnings,
    }


def sample_frames_from_array(frames: np.ndarray, count: int = 8, size: int = 224) -> np.ndarray:
    """Prepare RGB uint8 frames as normalized float [1,T,C,H,W]."""
    from PIL import Image
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("frames must have shape [T,H,W,3]")
    indices = extract_uniform_indices(len(array), count)
    selected = []
    for index in indices:
        image = Image.fromarray(array[int(index)].astype(np.uint8), mode="RGB").resize((size, size), Image.Resampling.BILINEAR)
        selected.append(np.asarray(image, dtype=np.float32) / 255.0)
    tensor = np.stack(selected, axis=0)
    tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose(tensor, (0, 3, 1, 2))[None, ...]


def read_video_frames(video_path: str | Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio
        with imageio.get_reader(str(video_path), "ffmpeg") as reader:
            frames = [frame for frame in reader]
    except Exception as exc:
        raise RuntimeError(f"cannot decode video {video_path}; install imageio-ffmpeg/ffmpeg: {exc}") from exc
    if not frames:
        raise ValueError(f"video has no frames: {video_path}")
    return np.asarray(frames)


class ViCLIPEncoder:
    """Thin wrapper around the repository ViCLIP implementation.

    Imports torch and ViCLIP only on construction so protocol/aggregation tests
    work in environments where the model environment is not installed.
    """

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cuda", frame_count: int = 8, image_size: int = 224):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("ViCLIPEncoder requires the ViCLIP conda environment with torch") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.frame_count = frame_count
        self.image_size = image_size
        viclip_dir = Path(__file__).resolve().parent / "ViCLIP"
        sys.path.insert(0, str(viclip_dir))
        from simple_tokenizer import SimpleTokenizer
        from viclip import ViCLIP
        checkpoint = Path(checkpoint) if checkpoint else viclip_dir / "ViClip-InternVid-10M-FLT.pth"
        self.model = ViCLIP(SimpleTokenizer(), pretrain=str(checkpoint)).to(self.device).eval()

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        batch = sample_frames_from_array(frames, self.frame_count, self.image_size)
        tensor = self.torch.from_numpy(batch).to(self.device).float()
        with self.torch.no_grad():
            value = self.model.get_vid_features(tensor)
        return value.detach().float().cpu().numpy()[0]

    def encode_video(self, video_path: str | Path) -> np.ndarray:
        return self.encode_frames(read_video_frames(video_path))

    def encode_batch(self, video_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([self.encode_video(path) for path in video_paths], axis=0)


class ViCLIPHTTPEncoder:
    """Feature encoder for the verified ViCLIP multipart ``/embed`` API."""

    def __init__(
        self,
        endpoint: str = "http://172.17.34.43:8890/embed",
        num_frames: int = 8,
        resolution: int = 224,
        timeout: float = 600,
        expected_dimension: int = 768,
    ):
        if not endpoint.rstrip("/").endswith("/embed"):
            raise ValueError("ViCLIP embedding endpoint must end with /embed")
        if min(num_frames, resolution, expected_dimension) <= 0 or timeout <= 0:
            raise ValueError("frame count, resolution, dimension, and timeout must be positive")
        self.endpoint = endpoint
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)
        self.timeout = float(timeout)
        self.expected_dimension = int(expected_dimension)

    def encode_video(self, video_path: str | Path) -> np.ndarray:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"video does not exist: {path}")
        try:
            with path.open("rb") as video_file:
                response = requests.post(
                    self.endpoint,
                    data={"num_frames": str(self.num_frames), "resolution": str(self.resolution)},
                    files={"file": (path.name, video_file, "application/octet-stream")},
                    timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"ViCLIP /embed request failed for {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("ViCLIP /embed response must be a JSON object")
        try:
            embedding = np.asarray(payload["embedding"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ViCLIP /embed response must contain a numeric embedding array") from exc
        if embedding.ndim != 1 or embedding.size != self.expected_dimension:
            raise ValueError(
                f"ViCLIP embedding dimension must be {self.expected_dimension}, got {embedding.shape}"
            )
        if int(payload.get("dimension", -1)) != self.expected_dimension:
            raise ValueError("ViCLIP response dimension metadata does not match the embedding")
        if int(payload.get("sampled_frames", -1)) != self.num_frames:
            raise ValueError("ViCLIP response sampled_frames does not match the request")
        if int(payload.get("resolution", -1)) != self.resolution:
            raise ValueError("ViCLIP response resolution does not match the request")
        if payload.get("normalized") is not True:
            raise ValueError("ViCLIP response must declare normalized=true")
        if not np.isfinite(embedding).all():
            raise ValueError("ViCLIP embedding contains non-finite values")
        norm = float(np.linalg.norm(embedding))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(f"ViCLIP embedding norm must be approximately 1.0, got {norm}")
        return (embedding / norm).astype(np.float32)

    def encode_batch(self, video_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([self.encode_video(path) for path in video_paths], axis=0)


class DeterministicFakeEncoder:
    """Offline smoke-test encoder; never use for reported benchmark results."""

    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def encode_video(self, video_path: str | Path) -> np.ndarray:
        digest = hashlib.sha256(str(video_path).encode("utf-8")).digest()
        raw = np.frombuffer((digest * ((self.dimension * 4 // len(digest)) + 1))[: self.dimension * 4], dtype=np.uint32)
        return raw.astype(np.float32)
