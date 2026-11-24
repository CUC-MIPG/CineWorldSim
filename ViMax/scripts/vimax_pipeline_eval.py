"""Offline, protocol-aware ViCLIP evaluation for ViMax Pipeline outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
_SHOT_DIR = re.compile(r"^\d+$")
_SCENE_DIR = re.compile(r"^scene_(\d+)$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _empty_metric() -> dict[str, Any]:
    return {"score": None, "pair_count": 0, "mean": None, "std": None, "pairs": []}


def _pair_metrics(shot_ids: Sequence[str], scene_ids: Sequence[Any], features: np.ndarray) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(shot_ids) or len(scene_ids) != len(shot_ids):
        raise ValueError("shot ids, scene labels, and feature rows must have matching lengths")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("features contain a zero vector")
    features = features / norms
    buckets: dict[str, list[tuple[float, dict[str, Any]]]] = {"IC": [], "CC": [], "GC": []}
    labeled = all(value is not None and str(value).strip() for value in scene_ids)
    for i, j in itertools.combinations(range(len(shot_ids)), 2):
        cosine = float(np.clip(np.dot(features[i], features[j]), -1.0, 1.0))
        pair = {"shot_a": str(shot_ids[i]), "shot_b": str(shot_ids[j]), "cosine": cosine}
        buckets["GC"].append((cosine, pair))
        if labeled:
            buckets["IC" if str(scene_ids[i]) == str(scene_ids[j]) else "CC"].append((cosine, pair))
    result = {}
    for key, values in buckets.items():
        scores = [item[0] for item in values]
        result[key] = {
            "score": float(np.mean(scores)) if scores else None,
            "pair_count": len(scores),
            "mean": float(np.mean(scores)) if scores else None,
            "std": float(np.std(scores)) if scores else None,
            "pairs": [item[1] for item in values],
        }
    if not labeled:
        result["IC"] = _empty_metric()
        result["CC"] = _empty_metric()
    result["warnings"] = [] if labeled else ["Source scene labels unavailable; IC/CC are not computed."]
    return result


def aggregate_viclip_metrics(shot_ids: Sequence[str], scene_ids: Sequence[Any], features: np.ndarray) -> dict[str, Any]:
    return _pair_metrics(shot_ids, scene_ids, features)


def _metric_shell(shot_count: int, labeled: bool) -> dict[str, Any]:
    total_pairs = shot_count * (shot_count - 1) // 2
    return {
        "IC": _empty_metric(),
        "CC": _empty_metric(),
        "GC": {**_empty_metric(), "pair_count": total_pairs},
        "warnings": [] if labeled else ["Source scene labels unavailable; IC/CC are not computed."],
    }


def _numeric_directories(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and _SHOT_DIR.fullmatch(p.name))


def discover_track_a(source_case: dict[str, Any], pipeline_root: str | Path) -> dict[str, Any]:
    pipeline = Path(pipeline_root)
    source_shots = []
    for scene in sorted(source_case.get("scenes", []), key=lambda x: int(x.get("scene_num", 0))):
        for shot in sorted(scene.get("shots", []), key=lambda x: int(x.get("shot_id", 0))):
            source_shots.append((str(scene.get("scene_num", "")), str(shot.get("shot_id", ""))))
    found_indices = _numeric_directories(pipeline / "shots")
    expected_indices = list(range(len(source_shots)))
    clips: list[dict[str, Any]] = []
    missing, zero_byte = [], []
    for idx in found_indices:
        path = pipeline / "shots" / str(idx) / "video.mp4"
        if not path.is_file():
            missing.append(str(idx))
            continue
        if path.stat().st_size == 0:
            zero_byte.append(str(idx))
        clips.append({"shot_id": str(idx), "video_path": str(path.resolve()), "scene_id": None, "generated_index": idx})
    missing_indices = sorted(set(expected_indices) - set(found_indices))
    extra_indices = sorted(set(found_indices) - set(expected_indices))
    contiguous = found_indices == list(range(len(found_indices)))
    complete_mapping = bool(source_shots) and len(found_indices) == len(source_shots) and contiguous
    errors = []
    if not source_shots:
        errors.append("source case contains no shots")
    if len(found_indices) != len(source_shots):
        errors.append(f"generated clip directory count {len(found_indices)} != expected source shot count {len(source_shots)}")
    if not contiguous:
        errors.append("generated shot indices are not contiguous from zero")
    if missing:
        errors.append(f"missing video.mp4 for shot indices: {', '.join(missing)}")
    if zero_byte:
        errors.append(f"zero-byte videos for shot indices: {', '.join(zero_byte)}")
    if complete_mapping:
        for row, (scene_id, source_shot_id) in zip(clips, source_shots):
            row["scene_id"] = scene_id
            row["source_shot_id"] = source_shot_id
    else:
        for row in clips:
            row["source_shot_id"] = None
    complete = complete_mapping and not missing and not zero_byte and len(clips) == len(source_shots)
    return {
        "track": "A", "status": "complete" if complete else "incomplete", "pipeline_root": str(pipeline.resolve()),
        "expected_shot_count": len(source_shots), "found_shot_count": len(found_indices), "shots": clips,
        "missing_shots": [str(x) for x in missing_indices] + missing,
        "extra_shots": [str(x) for x in extra_indices], "non_contiguous_indices": not contiguous,
        "scene_label_source": "source_positional" if complete_mapping else None,
        "protocol_structure": {"shot_count_8_16": 8 <= len(source_shots) <= 16,
                                "scene_count_2_4": 2 <= len({s for s, _ in source_shots}) <= 4},
        "metrics": _metric_shell(len(clips), complete_mapping) if complete else None,
        "diagnostics": errors,
    }


def _storyboard_indices(path: Path) -> list[int]:
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("shots", data.get("storyboard", data))
    if not isinstance(data, list):
        raise ValueError(f"{path}: storyboard must be a list")
    indices = []
    for pos, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: storyboard entry {pos} must be an object")
        indices.append(int(item.get("idx", pos)))
    return sorted(indices)


def discover_track_b(pipeline_root: str | Path, track_b_scene_diagnostics: bool = False) -> dict[str, Any]:
    pipeline = Path(pipeline_root)
    errors: list[str] = []
    script_path = pipeline / "script.json"
    if not script_path.is_file():
        expected_scene_indices: list[int] = []
        errors.append("missing script.json")
    else:
        script = _read_json(script_path)
        if not isinstance(script, list):
            expected_scene_indices = []
            errors.append("script.json must contain a scene list")
        else:
            expected_scene_indices = list(range(len(script)))
            if not script:
                errors.append("script.json contains no scenes")
    actual_scene_indices = sorted(int(m.group(1)) for p in pipeline.iterdir() if p.is_dir() and (m := _SCENE_DIR.fullmatch(p.name))) if pipeline.is_dir() else []
    missing_scene_indices = sorted(set(expected_scene_indices) - set(actual_scene_indices))
    extra_scene_indices = sorted(set(actual_scene_indices) - set(expected_scene_indices))
    if missing_scene_indices:
        errors.append(f"missing scene directories: {missing_scene_indices}")
    if extra_scene_indices:
        errors.append(f"unexpected scene directories: {extra_scene_indices}")
    shots: list[dict[str, Any]] = []
    missing_shots: list[str] = []
    extra_shots: list[str] = []
    zero_byte: list[str] = []
    diagnostics_scene_labels: list[Any] = []
    expected_count = 0
    for scene_idx in expected_scene_indices:
        scene_dir = pipeline / f"scene_{scene_idx}"
        storyboard_path = scene_dir / "storyboard.json"
        if not storyboard_path.is_file():
            errors.append(f"missing scene_{scene_idx}/storyboard.json")
            expected_indices = []
        else:
            try:
                expected_indices = _storyboard_indices(storyboard_path)
                if len(expected_indices) != len(set(expected_indices)):
                    errors.append(f"scene_{scene_idx}/storyboard.json contains duplicate shot indices")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid scene_{scene_idx}/storyboard.json: {exc}")
                expected_indices = []
        expected_count += len(expected_indices)
        actual_indices = _numeric_directories(scene_dir / "shots")
        for idx in sorted(set(expected_indices) - set(actual_indices)):
            missing_shots.append(f"{scene_idx}:{idx}")
        for idx in sorted(set(actual_indices) - set(expected_indices)):
            extra_shots.append(f"{scene_idx}:{idx}")
        for idx in sorted(set(expected_indices) | set(actual_indices)):
            path = scene_dir / "shots" / str(idx) / "video.mp4"
            if idx not in expected_indices:
                continue
            if not path.is_file():
                missing_shots.append(f"{scene_idx}:{idx}")
                continue
            if path.stat().st_size == 0:
                zero_byte.append(f"{scene_idx}:{idx}")
            shots.append({"shot_id": f"{scene_idx}:{idx}", "video_path": str(path.resolve()),
                          "scene_id": None,
                          "scene_idx": scene_idx, "shot_idx": idx})
    for scene_idx in extra_scene_indices:
        actual_indices = _numeric_directories(pipeline / f"scene_{scene_idx}" / "shots")
        extra_shots.extend(f"{scene_idx}:{idx}" for idx in actual_indices)
    if missing_shots:
        errors.append(f"missing shots/videos: {missing_shots}")
    if extra_shots:
        errors.append(f"unexpected shots: {extra_shots}")
    if zero_byte:
        errors.append(f"zero-byte videos: {zero_byte}")
    complete = not errors and len(shots) == expected_count and expected_count > 0
    result: dict[str, Any] = {
        "track": "B", "status": "complete" if complete else "incomplete", "pipeline_root": str(pipeline.resolve()),
        "expected_shot_count": expected_count, "found_shot_count": len(shots), "expected_scene_count": len(expected_scene_indices),
        "found_scene_count": len(actual_scene_indices), "shots": shots, "missing_shots": missing_shots,
        "extra_shots": extra_shots, "scene_label_source": None,
        "protocol_structure": {"shot_count_8_16": 8 <= expected_count <= 16, "scene_count_2_4": None},
        "metrics": _metric_shell(len(shots), False) if complete else None, "diagnostics": errors,
    }
    if track_b_scene_diagnostics and complete:
        result["scene_diagnostics"] = {
            "scene_label_source": "pipeline_generated",
            "metrics": _metric_shell(len(shots), True),
            "warning": "Generated scene groups are diagnostic only and are not source-scene annotations.",
        }
    return result


def aggregate_dataset_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for track in ("A", "B"):
        output[track] = {}
        for metric in ("IC", "CC", "GC"):
            samples = [row["metrics"][metric] for row in rows
                       if row.get("track") == track and row.get("status") == "complete"
                       and row.get("metrics") and row["metrics"].get(metric, {}).get("score") is not None]
            pair_count = sum(int(item.get("pair_count", 0)) for item in samples)
            output[track][metric] = {
                "story_macro_mean": float(np.mean([item["score"] for item in samples])) if samples else None,
                "pair_weighted_mean": (sum(item["score"] * item["pair_count"] for item in samples) / pair_count) if pair_count else None,
                "sample_count": len(samples), "pair_count": pair_count,
            }
    return output


def _validate_video(path: Path, sample_name: str, shot_id: str) -> None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            ok, _ = cap.read()
        finally:
            cap.release()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"sample={sample_name} shot={shot_id}: video decoding requires OpenCV or ffmpeg")
        try:
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "null", "-"],
                                    capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"sample={sample_name} shot={shot_id}: cannot decode {path}: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip()[-1000:]
            raise RuntimeError(f"sample={sample_name} shot={shot_id}: cannot decode {path}: {detail}")
        return
    except Exception as exc:
        raise RuntimeError(f"sample={sample_name} shot={shot_id}: cannot decode {path}: {exc}") from exc
    if not ok:
        raise RuntimeError(f"sample={sample_name} shot={shot_id}: video decode produced no frame: {path}")


def _cache_key(path: Path, encoder: Any) -> str:
    stat = path.stat()
    data = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "endpoint": getattr(encoder, "endpoint", "fake"), "frames": getattr(encoder, "num_frames", None),
            "resolution": getattr(encoder, "resolution", None),
            "dimension": getattr(encoder, "expected_dimension", getattr(encoder, "dimension", None))}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _encode_cached(path: Path, encoder: Any, cache_dir: Path) -> tuple[np.ndarray, bool]:
    key = _cache_key(path, encoder)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        try:
            value = np.asarray(_read_json(cache_path), dtype=np.float32)
            expected_dimension = getattr(encoder, "expected_dimension", getattr(encoder, "dimension", None))
            valid_dimension = expected_dimension is None or value.size == expected_dimension
            if (value.ndim == 1 and value.size and valid_dimension and np.isfinite(value).all()
                    and np.linalg.norm(value) > 0):
                return value, True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    value = np.asarray(encoder.encode_video(path), dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError("encoder returned an invalid embedding")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(value.tolist()), encoding="utf-8")
    return value, False


def evaluate_sample(sample: dict[str, Any], encoder: Any, cache_dir: str | Path,
                    sample_name: str, track_b_scene_diagnostics: bool = False) -> dict[str, Any]:
    if sample["status"] != "complete":
        return sample
    vectors, hits = [], 0
    cache_root = Path(cache_dir)
    for shot in sample["shots"]:
        path = Path(shot["video_path"])
        _validate_video(path, sample_name, shot["shot_id"])
        try:
            vector, hit = _encode_cached(path, encoder, cache_root)
        except Exception as exc:
            raise RuntimeError(f"sample={sample_name} shot={shot['shot_id']}: embedding failed: {exc}") from exc
        vectors.append(vector)
        hits += int(hit)
    features = np.stack(vectors)
    ids = [shot["shot_id"] for shot in sample["shots"]]
    labels = [shot.get("scene_id") for shot in sample["shots"]]
    sample["metrics"] = aggregate_viclip_metrics(ids, labels, features)
    sample["cache"] = {"hits": hits, "misses": len(vectors) - hits}
    if sample["track"] == "B" and track_b_scene_diagnostics:
        diag_labels = [str(shot["scene_idx"]) for shot in sample["shots"]]
        sample["scene_diagnostics"] = {
            "scene_label_source": "pipeline_generated",
            "metrics": aggregate_viclip_metrics(ids, diag_labels, features),
            "warning": "Generated scene groups are diagnostic only and are not source-scene annotations.",
        }
    return sample


def load_benchmark_cases(benchmark_root: str | Path) -> list[tuple[str, Path, dict[str, Any]]]:
    root = Path(benchmark_root)
    index_path = root / "benchmark_index.json"
    index = _read_json(index_path)
    rows = index.get("stories", []) if isinstance(index, dict) else []
    if not isinstance(rows, list):
        raise ValueError(f"{index_path}: stories must be a list")
    result = []
    for row in rows:
        source = root / str(row["file"])
        case = _read_json(source)
        theme = str(row.get("theme") or case.get("metadata", {}).get("theme_key") or source.stem)
        result.append((theme, source, case))
    return result


def make_report(rows: list[dict[str, Any]], encoder: Any, args: dict[str, Any]) -> dict[str, Any]:
    scores = []
    for row in rows:
        metrics = row.get("metrics") or {}
        score_row = {
            "case": row.get("case"),
            "track": row.get("track"),
            "status": row.get("status"),
            "IC": metrics.get("IC", {}).get("score"),
            "CC": metrics.get("CC", {}).get("score"),
            "GC": metrics.get("GC", {}).get("score"),
        }
        diagnostic = row.get("scene_diagnostics", {}).get("metrics")
        if diagnostic:
            score_row["track_b_scene_diagnostic_scores"] = {
                name: diagnostic.get(name, {}).get("score") for name in ("IC", "CC", "GC")
            }
        if row.get("status") != "complete":
            score_row["diagnostic"] = "; ".join(row.get("diagnostics", [])) or "Sample is incomplete"
        scores.append(score_row)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_are_official_vimax_bench": False,
        "encoder": {"name": type(encoder).__name__, "reportable": bool(getattr(encoder, "reportable", False)), **args},
        "sample_count": len(scores), "complete_sample_count": sum(r["status"] == "complete" for r in scores),
        "case_scores": scores,
    }


class DeterministicFakeEncoder:
    reportable = False

    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def encode_video(self, video_path: str | Path) -> np.ndarray:
        digest = hashlib.sha256(Path(video_path).resolve().as_posix().encode()).digest()
        raw = np.frombuffer((digest * ((self.dimension * 4 // len(digest)) + 1))[:self.dimension * 4], dtype=np.uint32)
        value = raw.astype(np.float64) / np.iinfo(np.uint32).max - 0.5
        return (value / np.linalg.norm(value)).astype(np.float32)


class ViCLIPHTTPEncoder:
    reportable = True

    def __init__(self, endpoint: str = "http://172.17.34.43:8890/embed", num_frames: int = 8,
                 resolution: int = 224, timeout: float = 600, expected_dimension: int = 768):
        if not endpoint.rstrip("/").endswith("/embed"):
            raise ValueError("ViCLIP endpoint must end with /embed")
        if min(num_frames, resolution, timeout, expected_dimension) <= 0:
            raise ValueError("encoder settings must be positive")
        self.endpoint, self.num_frames, self.resolution = endpoint, int(num_frames), int(resolution)
        self.timeout, self.expected_dimension = float(timeout), int(expected_dimension)

    def encode_video(self, video_path: str | Path) -> np.ndarray:
        import requests
        path = Path(video_path)
        try:
            with path.open("rb") as stream:
                response = requests.post(self.endpoint,
                    data={"num_frames": str(self.num_frames), "resolution": str(self.resolution)},
                    files={"file": (path.name, stream, "application/octet-stream")}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"ViCLIP /embed request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("ViCLIP response must be a JSON object")
        try:
            vector = np.asarray(payload["embedding"], dtype=np.float64)
            metadata_ok = (int(payload["dimension"]) == self.expected_dimension
                           and int(payload["sampled_frames"]) == self.num_frames
                           and int(payload["resolution"]) == self.resolution
                           and payload["normalized"] is True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ViCLIP response is missing valid embedding metadata") from exc
        if vector.ndim != 1 or vector.size != self.expected_dimension:
            raise ValueError(f"expected {self.expected_dimension}-D embedding, got {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if not metadata_ok or not np.isfinite(vector).all() or norm == 0 or not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError("ViCLIP embedding or response metadata failed validation")
        return (vector / norm).astype(np.float32)
