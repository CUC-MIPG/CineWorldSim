#!/usr/bin/env python3
"""Check, plan, and repair CineWorldSim artifacts before video generation.

The checker is deliberately conservative.  A recorded fallback is treated as
untrusted even when a placeholder artifact exists, because downstream video
generation should not silently consume fallback narrative or path decisions.
Repair never writes into the source pipeline directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)

STAGE_ORDER = [
    "world_initialization",
    "narrative_simulation",
    "world_delta",
    "event_keyframe",
    "node_selection",
    "path_llm",
    "path_mcts",
]

STAGE_LABELS = {
    "world_initialization": "world initialization",
    "narrative_simulation": "narrative simulation",
    "world_delta": "world delta/world state",
    "event_keyframe": "event keyframe",
    "node_selection": "narrative node selection",
    "path_llm": "LLM path orchestration",
    "path_mcts": "MCTS path orchestration",
}

FALLBACK_WORDS = (
    "fallback",
    "failed",
    "error",
    "disabled_or_fallback",
    "missing_dependency",
    "skipped",
    "partial",
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dict(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _read_list(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _path_exists(path_value: Any, source_dir: Path) -> bool:
    raw = _safe_text(path_value)
    if not raw:
        return False
    candidates = [Path(raw)]
    path = Path(raw)
    if not path.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / path,
                source_dir / path,
            ]
        )
    return any(candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0 for candidate in candidates)


def _record_failure(record: dict[str, Any], source_dir: Path, label: str) -> str:
    if not bool(record.get("ok", True)):
        return f"{label}: ok=false"
    output_path = _safe_text(record.get("output_path"))
    if not output_path:
        return f"{label}: output_path is empty"
    if not _path_exists(output_path, source_dir):
        return f"{label}: missing or empty output {output_path}"
    return ""


def _stage_result(
    *,
    status: str = "healthy",
    failures: Iterable[str] = (),
    fallbacks: Iterable[dict[str, Any]] = (),
    artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    failure_list = list(failures)
    fallback_list = list(fallbacks)
    if failure_list or fallback_list:
        status = "failed"
    return {
        "status": status,
        "failures": failure_list,
        "fallbacks": fallback_list,
        "artifacts_checked": list(artifacts),
    }


def _iter_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _status_marker(value: str) -> bool:
    lower = value.strip().lower()
    return any(word in lower for word in FALLBACK_WORDS)


def _stage_from_context(text: str, *, default: str = "narrative_simulation") -> str:
    lower = text.lower()
    if any(token in lower for token in ["mcts", "monte carlo"]):
        return "path_mcts"
    if any(token in lower for token in ["path_llm", "llm path", "path orchestrator", "path plan"]):
        return "path_llm"
    if any(token in lower for token in ["node selector", "node_selection", "narrative node"]):
        return "node_selection"
    if any(token in lower for token in ["event_keyframe", "event keyframe", "keyframe renderer"]):
        return "event_keyframe"
    if any(token in lower for token in ["world_delta", "world delta", "world state", "world update"]):
        return "world_delta"
    if any(token in lower for token in ["world_model", "world model", "initial asset", "portrait", "nano_banana"]):
        return "world_initialization"
    if "video" in lower or "vllm" in lower:
        return ""
    return default


def _scan_log_files(
    pipeline_dir: Path,
    explicit_files: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    files: list[Path] = []
    seen: set[Path] = set()
    search_roots = [pipeline_dir]
    if pipeline_dir.parent.name == "pipeline":
        search_roots.append(pipeline_dir.parent.parent / "logs")
    search_roots.append(pipeline_dir.parent / "logs")
    for path in explicit_files:
        if path.exists():
            search_roots.append(path.parent)
            files.append(path)
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [
                item
                for item in root.rglob("*")
                if item.is_file() and item.suffix.lower() in {".log", ".jsonl", ".txt"}
            ]
        for item in candidates:
            resolved = item.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(item)

    markers: list[dict[str, Any]] = []
    for path in files:
        if path.name in {
            "pre_video_health_report.json",
            "pre_video_recovery_plan.json",
            "pre_video_recovery_manifest.json",
        }:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record: dict[str, Any] | None = None
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    record = parsed
            except Exception:
                pass
            if record is not None:
                status = _safe_text(record.get("status") or record.get("mode") or record.get("source"))
                http_status = _safe_int(record.get("status_code"))
                is_bad = _status_marker(status) or http_status >= 400
                if not is_bad:
                    continue
                context = " ".join(
                    _safe_text(record.get(key))
                    for key in [
                        "scope",
                        "tool",
                        "kind",
                        "url",
                        "output_path",
                        "reason",
                        "error",
                    ]
                )
                stage = _stage_from_context(context)
                if not stage:
                    continue
                markers.append(
                    {
                        "stage": stage,
                        "source": str(path),
                        "line": line_number,
                        "status": status or f"http_{http_status}",
                        "reason": _safe_text(record.get("reason") or record.get("error")),
                        "task_id": _safe_text(record.get("task_id") or record.get("call_id")),
                    }
                )
                continue

            lower = stripped.lower()
            if not any(pattern in lower for pattern in [
                "using fallback",
                "status=fallback",
                "mode=fallback",
                "fallback_after_error",
                "model call failed",
                "status=failed",
                "status=error",
                "ok=false",
            ]):
                continue
            stage = _stage_from_context(stripped)
            if not stage:
                continue
            markers.append(
                {
                    "stage": stage,
                    "source": str(path),
                    "line": line_number,
                    "status": "text_marker",
                    "reason": stripped[:1000],
                    "task_id": "",
                }
            )
    return markers


def _check_world_initialization(
    source_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    artifacts: list[str] = []
    world_model = payload.get("world_model")
    if not isinstance(world_model, dict):
        return _stage_result(status="failed", failures=["world_model is missing or not an object"])
    artifacts.append("pipeline_result.json:world_model")
    for group_name in ("locations", "items"):
        rows = world_model.get(group_name)
        if not isinstance(rows, list):
            failures.append(f"world_model.{group_name}: missing or invalid")
            continue
        for index, asset in enumerate(rows):
            if not isinstance(asset, dict):
                failures.append(f"world_model.{group_name}[{index}]: invalid record")
                continue
            asset_name = _safe_text(asset.get("asset_id") or asset.get("name") or index)
            views = asset.get("views") if isinstance(asset.get("views"), dict) else {}
            images = asset.get("images") if isinstance(asset.get("images"), list) else []
            records: list[tuple[str, dict[str, Any]]] = []
            if group_name == "locations" and views:
                for view_name in ("wide", "medium", "close", "extreme_close"):
                    view = views.get(view_name)
                    if isinstance(view, dict):
                        records.append((f"{asset_name}.{view_name}", view))
                    else:
                        failures.append(f"world_model.locations.{asset_name}: missing view {view_name}")
            else:
                records.extend(
                    (f"{asset_name}.images[{image_index}]", image)
                    for image_index, image in enumerate(images)
                    if isinstance(image, dict)
                )
            if not records:
                failures.append(f"world_model.{group_name}.{asset_name}: no image records")
            for label, record in records:
                artifacts.append(f"world_model.{group_name}.{label}")
                reason = _record_failure(record, source_dir, label)
                if reason:
                    failures.append(reason)

    portraits = payload.get("character_portraits")
    if not isinstance(portraits, dict) or not portraits:
        failures.append("character_portraits: empty")
    else:
        for character, path in portraits.items():
            artifacts.append(f"character_portraits.{character}")
            if not _path_exists(path, source_dir):
                failures.append(f"character_portraits.{character}: missing or empty output {path}")
    return _stage_result(failures=failures, artifacts=artifacts)


def _load_step_logs(source_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = source_dir / "step_logs.json"
    if path.exists():
        return _read_list(path)
    return [item for item in payload.get("step_logs", []) or [] if isinstance(item, dict)]


def _check_narrative_simulation(
    source_dir: Path,
    payload: dict[str, Any],
    *,
    step_gaps: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    failures: list[str] = []
    artifacts: list[str] = []
    logs: list[dict[str, Any]] = []
    path = source_dir / "step_logs.json"
    if path.exists():
        artifacts.append(str(path))
    else:
        failures.append("step_logs.json: missing")
    try:
        logs = _load_step_logs(source_dir, payload)
    except Exception as exc:
        failures.append(f"step_logs.json: unreadable: {exc}")
        return _stage_result(failures=failures, artifacts=artifacts), logs, 0
    if not logs:
        failures.append("step_logs: empty")
        return _stage_result(failures=failures, artifacts=artifacts), logs, 0
    steps = sorted({_safe_int(row.get("step")) for row in logs if _safe_int(row.get("step")) > 0})
    max_step = max(steps, default=0)
    expected = list(range(1, max_step + 1))
    missing = [step for step in expected if step not in steps]
    if missing:
        step_gaps.extend(f"missing narrative step {step}" for step in missing)
        failures.extend(step_gaps[-len(missing):])
    for step in steps:
        row = next(item for item in logs if _safe_int(item.get("step")) == step)
        roles = row.get("role_logs")
        if not isinstance(roles, list) or not roles:
            failures.append(f"step {step}: role_logs is empty")
            continue
        for role in roles:
            if not isinstance(role, dict) or not _safe_text(role.get("role_name")):
                failures.append(f"step {step}: invalid role log")
    checkpoint = source_dir / "narrative_checkpoint.json"
    if not checkpoint.exists():
        failures.append("narrative_checkpoint.json: missing")
    else:
        artifacts.append(str(checkpoint))
        try:
            checkpoint_payload = _read_dict(checkpoint)
            if _safe_int(checkpoint_payload.get("completed_steps")) < max_step:
                failures.append("narrative_checkpoint.json: completed_steps is behind step_logs")
        except Exception as exc:
            failures.append(f"narrative_checkpoint.json: unreadable: {exc}")
    return _stage_result(failures=failures, artifacts=artifacts), logs, max_step


def _check_world_delta(
    source_dir: Path,
    payload: dict[str, Any],
    max_step: int,
    *,
    step_gaps: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    artifacts: list[str] = []
    states = {
        _safe_int(row.get("step")): row
        for row in payload.get("world_states", []) or []
        if isinstance(row, dict)
    }
    deltas = {
        _safe_int(row.get("step")): row
        for row in payload.get("world_deltas", []) or []
        if isinstance(row, dict)
    }
    for step in range(0, max_step + 1):
        state_path = source_dir / f"world_state_step{step}.json"
        artifacts.append(str(state_path))
        if not state_path.exists():
            step_gaps.append(f"missing world state step {step}")
            failures.append(step_gaps[-1])
    for step in range(1, max_step + 1):
        delta_path = source_dir / "world_deltas" / f"world_delta_step{step}.json"
        artifacts.append(str(delta_path))
        if not delta_path.exists():
            step_gaps.append(f"missing world delta step {step}")
            failures.append(step_gaps[-1])
    for step, delta in deltas.items():
        decision_source = _safe_text(delta.get("decision_source")).lower()
        if "fallback" in decision_source or decision_source in {"disabled", "disabled_or_fallback"}:
            failures.append(f"world delta step {step}: decision_source={decision_source}")
    return _stage_result(failures=failures, artifacts=artifacts)


def _check_event_keyframes(
    source_dir: Path,
    payload: dict[str, Any],
    logs: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    artifacts: list[str] = []
    keyframes = [
        item for item in payload.get("event_keyframes", []) or []
        if isinstance(item, dict)
    ]
    by_key = {
        (_safe_int(item.get("step")), _safe_text(item.get("role_name"))): item
        for item in keyframes
    }
    for step_log in logs:
        step = _safe_int(step_log.get("step"))
        for role in step_log.get("role_logs", []) or []:
            if not isinstance(role, dict):
                continue
            role_name = _safe_text(role.get("role_name"))
            if not role_name:
                continue
            record = by_key.get((step, role_name))
            if record is None:
                failures.append(f"event keyframe step {step} role {role_name}: missing record")
                continue
            label = f"event_keyframes.step_{step}.{role_name}"
            artifacts.append(label)
            reason = _record_failure(record, source_dir, label)
            if reason:
                failures.append(reason)
    return _stage_result(failures=failures, artifacts=artifacts)


def _check_node_selection(
    source_dir: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    artifacts: list[str] = []
    if not required:
        return _stage_result(status="not_applicable"), []
    path = source_dir / "path_orchestration" / "narrative_nodes.json"
    artifacts.append(str(path))
    if not path.exists():
        return _stage_result(failures=["narrative_nodes.json: missing"], artifacts=artifacts), []
    try:
        payload = _read_dict(path)
    except Exception as exc:
        return _stage_result(failures=[f"narrative_nodes.json: unreadable: {exc}"], artifacts=artifacts), []
    nodes = [item for item in payload.get("selected_nodes", []) or [] if isinstance(item, dict)]
    if not nodes:
        failures.append("narrative_nodes.json: selected_nodes is empty")
    candidates_path = source_dir / "path_orchestration" / "path_candidates.json"
    candidates: list[dict[str, Any]] = []
    artifacts.append(str(candidates_path))
    if not candidates_path.exists():
        failures.append("path_candidates.json: missing")
    else:
        try:
            candidates = _read_list(candidates_path)
        except Exception as exc:
            failures.append(f"path_candidates.json: unreadable: {exc}")
    candidate_ids = {
        _safe_text(item.get("node_id") or item.get("event_id"))
        for item in candidates
    }
    for node in nodes:
        node_id = _safe_text(node.get("node_id") or node.get("event_id"))
        if not node_id:
            failures.append("narrative_nodes.json: selected node has empty id")
        elif candidate_ids and node_id not in candidate_ids:
            failures.append(f"narrative_nodes.json: selected id is not a candidate: {node_id}")
    return _stage_result(failures=failures, artifacts=artifacts), candidates


def _check_path_plan(
    source_dir: Path,
    *,
    kind: str,
    required: bool,
    candidates: list[dict[str, Any]],
    path_gaps: list[str] | None = None,
) -> dict[str, Any]:
    if not required:
        return _stage_result(status="not_applicable")
    if kind == "llm":
        path = source_dir / "path_orchestration" / "path_llm_direct" / "narrative_path_plan.json"
        stage = "path_llm"
    else:
        path = source_dir / "path_orchestration" / "path_mcts" / "narrative_path_plan_mcts.json"
        stage = "path_mcts"
    artifacts = [str(path)]
    if not path.exists():
        failure = f"{path.name}: missing"
        if path_gaps is not None:
            path_gaps.append(failure)
        return _stage_result(failures=[failure], artifacts=artifacts)
    try:
        payload = _read_dict(path)
    except Exception as exc:
        failure = f"{path.name}: unreadable: {exc}"
        if path_gaps is not None:
            path_gaps.append(failure)
        return _stage_result(failures=[failure], artifacts=artifacts)
    failures: list[str] = []
    mode = _safe_text(payload.get("mode")).lower()
    if "fallback" in mode:
        failures.append(f"{path.name}: mode={mode}")
    ordered = payload.get("ordered_node_ids")
    if not isinstance(ordered, list) or not ordered:
        failures.append(f"{path.name}: ordered_node_ids is empty")
        ordered = []
    candidate_ids = {
        _safe_text(item.get("node_id") or item.get("event_id"))
        for item in candidates
    }
    ordered_ids = [_safe_text(item) for item in ordered]
    if len(ordered_ids) != len(set(ordered_ids)):
        failures.append(f"{path.name}: duplicate ordered node ids")
    if candidate_ids:
        unknown = [item for item in ordered_ids if item not in candidate_ids]
        if unknown:
            failures.append(f"{path.name}: unknown node ids: {unknown[:5]}")
    if path_gaps is not None:
        path_gaps.extend(failures)
    return _stage_result(failures=failures, artifacts=artifacts)


def _iter_marker_fields(value: Any, path: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            field_path = f"{path}.{key}"
            if key in {"status", "mode", "decision_source", "source", "fallback_used"}:
                yield field_path, key, item
            yield from _iter_marker_fields(item, field_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_marker_fields(item, f"{path}[{index}]")


def _detect_json_fallbacks(source_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    json_files = list(source_dir.rglob("*.json"))
    for path in json_files:
        try:
            data = _read_json(path)
        except Exception:
            continue
        for location, field_name, raw_value in _iter_marker_fields(data):
            value = _safe_text(raw_value)
            if field_name == "fallback_used":
                is_marker = bool(raw_value)
            else:
                is_marker = _status_marker(value)
            if not is_marker:
                continue
            context = f"{path} {location} {value}"
            stage = _stage_from_context(context)
            if not stage:
                continue
            markers.append(
                {
                    "stage": stage,
                    "source": str(path),
                    "line": 0,
                    "status": "json_marker",
                    "reason": value[:1000],
                    "task_id": "",
                }
            )
    return markers


def _detect_structured_narrative_fallbacks(
    source_dir: Path,
    logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read per-role call status persisted by the narrative simulator.

    A complete artifact is not trustworthy when a role or step used an LLM
    fallback.  These markers are intentionally limited to explicit structured
    fields so ordinary story text containing words such as ``fallback`` does
    not create false failures.
    """
    markers: list[dict[str, Any]] = []
    for step_row in logs:
        step = _safe_int(step_row.get("step"))
        if step <= 0:
            continue
        role_logs = step_row.get("role_logs", []) or []
        for role in role_logs:
            if not isinstance(role, dict):
                continue
            actor = _safe_text(role.get("role_name")) or "unknown"
            calls = role.get("model_calls", []) or []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                if not bool(call.get("fallback_used")) and _safe_text(call.get("status")).lower() not in {
                    "fallback",
                    "skipped",
                    "failed",
                    "error",
                }:
                    continue
                reason = _safe_text(call.get("error")) or _safe_text(call.get("status"))
                label = _safe_text(call.get("call_label")) or "unknown_call"
                markers.append(
                    {
                        "stage": "narrative_simulation",
                        "source": str(source_dir / "step_logs.json"),
                        "line": 0,
                        "status": "structured_fallback",
                        "reason": f"step={step} actor={actor} call={label}: {reason}",
                        "task_id": _safe_text(call.get("call_id")),
                    }
                )
        for call in step_row.get("model_calls", []) or []:
            if not isinstance(call, dict):
                continue
            if not bool(call.get("fallback_used")) and _safe_text(call.get("status")).lower() not in {
                "fallback",
                "skipped",
                "failed",
                "error",
            }:
                continue
            reason = _safe_text(call.get("error")) or _safe_text(call.get("status"))
            label = _safe_text(call.get("call_label")) or "unknown_call"
            markers.append(
                {
                    "stage": "narrative_simulation",
                    "source": str(source_dir / "step_logs.json"),
                    "line": 0,
                    "status": "structured_fallback",
                    "reason": f"step={step} call={label}: {reason}",
                    "task_id": _safe_text(call.get("call_id")),
                }
            )
    return markers


def check_pipeline(
    pipeline_dir: Path,
    *,
    call_log_files: Iterable[Path] = (),
    scan_logs: bool = True,
) -> dict[str, Any]:
    source_dir = Path(pipeline_dir).resolve()
    report: dict[str, Any] = {
        "status": "failed",
        "pipeline_dir": str(source_dir),
        "stages": {},
        "fallback_calls": [],
        "missing_artifacts": [],
        "invalid_artifacts": [],
        "step_gaps": [],
        "path_gaps": [],
    }
    pipeline_result_path = source_dir / "pipeline_result.json"
    if not pipeline_result_path.exists():
        report["invalid_artifacts"].append(f"missing {pipeline_result_path}")
        for stage in STAGE_ORDER:
            report["stages"][stage] = _stage_result(
                status="failed" if stage == "world_initialization" else "blocked",
                failures=["pipeline_result.json is missing"],
            )
        return report
    try:
        payload = _read_dict(pipeline_result_path)
    except Exception as exc:
        report["invalid_artifacts"].append(f"unreadable pipeline_result.json: {exc}")
        for stage in STAGE_ORDER:
            report["stages"][stage] = _stage_result(
                status="failed" if stage == "world_initialization" else "blocked",
                failures=["pipeline_result.json is unreadable"],
            )
        return report

    report["stages"]["world_initialization"] = _check_world_initialization(source_dir, payload)
    narrative_stage, logs, max_step = _check_narrative_simulation(
        source_dir,
        payload,
        step_gaps=report["step_gaps"],
    )
    report["stages"]["narrative_simulation"] = narrative_stage
    report["stages"]["world_delta"] = _check_world_delta(
        source_dir,
        payload,
        max_step,
        step_gaps=report["step_gaps"],
    )
    report["stages"]["event_keyframe"] = _check_event_keyframes(source_dir, payload, logs)

    path_root = source_dir / "path_orchestration"
    node_required = path_root.exists() or (source_dir / "full_pipeline_manifest.json").exists()
    node_stage, candidates = _check_node_selection(source_dir, required=node_required)
    report["stages"]["node_selection"] = node_stage
    llm_required = (
        (path_root / "path_llm_direct").exists()
        or (path_root / "path_llm_direct" / "narrative_path_plan.json").exists()
    )
    mcts_required = (
        (path_root / "path_mcts").exists()
        or (path_root / "path_mcts" / "narrative_path_plan_mcts.json").exists()
    )
    report["stages"]["path_llm"] = _check_path_plan(
        source_dir,
        kind="llm",
        required=llm_required,
        candidates=candidates,
        path_gaps=report["path_gaps"],
    )
    report["stages"]["path_mcts"] = _check_path_plan(
        source_dir,
        kind="mcts",
        required=mcts_required,
        candidates=candidates,
        path_gaps=report["path_gaps"],
    )

    markers = _scan_log_files(source_dir, call_log_files) if scan_logs else []
    markers.extend(_detect_structured_narrative_fallbacks(source_dir, logs))
    markers.extend(_detect_json_fallbacks(source_dir, payload))
    unique_markers: list[dict[str, Any]] = []
    seen_markers: set[tuple[str, str, int, str]] = set()
    for marker in markers:
        key = (
            _safe_text(marker.get("stage")),
            _safe_text(marker.get("source")),
            _safe_int(marker.get("line")),
            _safe_text(marker.get("reason")),
        )
        if key in seen_markers:
            continue
        seen_markers.add(key)
        unique_markers.append(marker)
        stage = _safe_text(marker.get("stage"))
        if stage in report["stages"]:
            report["stages"][stage]["fallbacks"].append(marker)
            report["stages"][stage]["status"] = "failed"
    report["fallback_calls"] = unique_markers

    for stage in STAGE_ORDER:
        stage_report = report["stages"].get(stage, {})
        report["missing_artifacts"].extend(
            failure
            for failure in stage_report.get("failures", [])
            if "missing" in str(failure).lower() or "empty" in str(failure).lower()
        )
    report["status"] = (
        "failed"
        if any(item.get("status") == "failed" for item in report["stages"].values())
        else "healthy"
    )
    return report


def _affected_stages(failed: set[str]) -> list[str]:
    affected: set[str] = set()
    if any(stage in failed for stage in {
        "world_initialization",
        "narrative_simulation",
        "world_delta",
        "event_keyframe",
    }):
        first = min(STAGE_ORDER.index(stage) for stage in failed if stage in STAGE_ORDER[:4])
        affected.update(STAGE_ORDER[first:])
    if "node_selection" in failed:
        affected.update({"node_selection", "path_llm", "path_mcts"})
    if "path_llm" in failed:
        affected.add("path_llm")
    if "path_mcts" in failed:
        affected.add("path_mcts")
    return [stage for stage in STAGE_ORDER if stage in affected]


def build_recovery_plan(report: dict[str, Any]) -> dict[str, Any]:
    failed = {
        stage
        for stage, stage_report in (report.get("stages") or {}).items()
        if isinstance(stage_report, dict) and stage_report.get("status") == "failed"
    }
    affected = _affected_stages(failed)
    if not failed:
        return {
            "status": "healthy",
            "pipeline_dir": report.get("pipeline_dir", ""),
            "earliest_failed_stage": "",
            "affected_stages": [],
            "affected_steps": [],
            "tasks": [],
            "reused_artifacts": STAGE_ORDER,
        }
    earliest = min(failed, key=STAGE_ORDER.index)
    affected_steps: set[int] = set()
    for marker in report.get("fallback_calls", []) or []:
        reason = _safe_text(marker.get("reason"))
        for match in re.finditer(r"\bstep(?:_|=|\s+)(\d+)\b", reason.lower()):
            affected_steps.add(int(match.group(1)))
    for gap in report.get("step_gaps", []) or []:
        match = re.search(r"\b(?:step|state|delta)\s+(\d+)\b", str(gap).lower())
        if match:
            affected_steps.add(int(match.group(1)))
    tasks = []
    for stage in affected:
        action = "rebuild_world_stage" if stage in STAGE_ORDER[:4] else "rerun_path_stage"
        tasks.append(
            {
                "task_id": f"{stage}.repair",
                "stage": stage,
                "action": action,
                "reason": "; ".join(
                    list((report["stages"].get(stage, {}) or {}).get("failures", []))
                    + [
                        _safe_text(item.get("reason"))
                        for item in report.get("fallback_calls", [])
                        if item.get("stage") == stage and _safe_text(item.get("reason"))
                    ]
                )[:2000],
            }
        )
    return {
        "status": "repairable",
        "pipeline_dir": report.get("pipeline_dir", ""),
        "earliest_failed_stage": earliest,
        "affected_stages": affected,
        "affected_steps": sorted(affected_steps),
        "tasks": tasks,
        "reused_artifacts": [stage for stage in STAGE_ORDER if stage not in affected],
    }


def ensure_repair_output_dir(source_dir: Path, output_dir: Path) -> Path:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise RuntimeError(
            f"Refusing to write repair output inside source pipeline directory: {output}"
        )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Repair output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _copy_tree_and_rewrite_paths(source_dir: Path, output_dir: Path) -> None:
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
    source = source_dir.resolve()
    output = output_dir.resolve()
    try:
        relative_source = source.relative_to(PROJECT_ROOT.resolve())
        project_prefix = str(PROJECT_ROOT.resolve() / relative_source)
        relative_prefix = str(relative_source)
    except ValueError:
        project_prefix = str(source)
        relative_prefix = ""
    for path in output.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        text = text.replace(str(source), str(output))
        if project_prefix:
            replacement_prefix = str(output)
            text = text.replace(project_prefix, replacement_prefix)
        if relative_prefix:
            text = text.replace(
                f'"{relative_prefix}/',
                f'"{output}/',
            )
        path.write_text(text, encoding="utf-8")


def _pipeline_story_and_characters(source_dir: Path, payload: dict[str, Any]) -> tuple[str, list[str]]:
    story = _safe_text(payload.get("story"))
    characters = [
        _safe_text(item)
        for item in payload.get("characters", []) or []
        if _safe_text(item)
    ]
    if not story:
        bundle = source_dir / "story_bundle.json"
        if bundle.exists():
            bundle_payload = _read_dict(bundle)
            story = _safe_text(bundle_payload.get("story"))
            if not characters:
                characters = [
                    _safe_text(item)
                    for item in bundle_payload.get("characters", []) or []
                    if _safe_text(item)
                ]
    if not story or not characters:
        raise RuntimeError("Cannot repair world stage: story or characters are missing")
    return story, characters


def _source_max_step(source_dir: Path, payload: dict[str, Any]) -> int:
    values = [
        _safe_int(row.get("step"))
        for row in payload.get("step_logs", []) or []
        if isinstance(row, dict)
    ]
    values.extend(
        _safe_int(row.get("step"))
        for row in payload.get("world_states", []) or []
        if isinstance(row, dict)
    )
    checkpoint = source_dir / "narrative_checkpoint.json"
    if checkpoint.exists():
        try:
            values.append(_safe_int(_read_dict(checkpoint).get("completed_steps")))
        except Exception:
            pass
    return max(values or [0])


def _run_world_repair(
    *,
    source_dir: Path,
    output_dir: Path,
    report: dict[str, Any],
    backend_mode: str,
    strict_online: bool,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    payload = _read_dict(source_dir / "pipeline_result.json")
    story, characters = _pipeline_story_and_characters(source_dir, payload)
    from scripts.run_dual_backend_narrative_world_pipeline import _run_one_backend

    # `_run_one_backend` owns the backend-directory layout.  Accept an
    # arbitrary user-provided output directory by using a temporary layout
    # when the target directory is not already named after the backend.
    target_dir = output_dir
    if output_dir.name == backend_mode:
        generated = _run_one_backend(
            story=story,
            characters=characters,
            backend_mode=backend_mode,
            max_steps=max(1, _source_max_step(source_dir, payload)),
            output_dir=output_dir.parent,
            dry_run=dry_run,
            stop_on_complete=False,
            strict_online=strict_online,
            retries=retries,
        )
    else:
        staging_root = output_dir.parent / f".{output_dir.name}.pre_video_stage"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        generated = _run_one_backend(
            story=story,
            characters=characters,
            backend_mode=backend_mode,
            max_steps=max(1, _source_max_step(source_dir, payload)),
            output_dir=staging_root,
            dry_run=dry_run,
            stop_on_complete=False,
            strict_online=strict_online,
            retries=retries,
        )
        _copy_tree_and_rewrite_paths(generated, target_dir)
        shutil.rmtree(staging_root, ignore_errors=True)
        generated = target_dir

    return {
        "mode": "world_stage_rebuild",
        "generated_dir": str(generated),
        "affected_stages": report["plan"]["affected_stages"],
    }


def _path_args(
    *,
    strict_online: bool,
    retries: int,
    path_mode: str,
    node_config: dict[str, Any] | None,
) -> SimpleNamespace:
    config = node_config or {}
    selector = config.get("selector_config", {}) if isinstance(config, dict) else {}
    return SimpleNamespace(
        enable_narrative_node_selector=True,
        node_selection_mode=selector.get("selection_mode", "hybrid"),
        node_score_threshold=float(selector.get("score_threshold", 0.55) or 0.55),
        node_min_nodes=int(selector.get("min_nodes", 3) or 3),
        node_max_nodes=int(selector.get("max_nodes", 8) or 8),
        node_llm_scoring=bool(selector.get("llm_enabled", True)),
        strict_online=strict_online,
        retries=retries,
        path_mode=path_mode,
        mcts_model=None,
        mcts_budget_iters=1,
        mcts_topk_actions=8,
        mcts_cpuct=1.2,
        mcts_rollout_horizon=32,
        mcts_rollout_topk=5,
        mcts_rollout_epsilon=0.2,
        mcts_pw_base=1.5,
        mcts_pw_alpha=0.55,
        pov_switch_penalty=0.05,
        over_focus_threshold=4,
        over_focus_penalty=0.08,
        w_narrative_causality=0.4,
        w_visual_transition=0.3,
        w_rhythmic_pacing=0.3,
        random_seed=42,
    )


def _run_path_repair(
    *,
    source_dir: Path,
    output_dir: Path,
    report: dict[str, Any],
    strict_online: bool,
    retries: int,
) -> dict[str, Any]:
    if source_dir.resolve() != output_dir.resolve():
        _copy_tree_and_rewrite_paths(source_dir, output_dir)
    path_root = output_dir / "path_orchestration"
    source_path_root = source_dir / "path_orchestration"
    node_payload: dict[str, Any] = {}
    node_path = source_path_root / "narrative_nodes.json"
    if node_path.exists():
        node_payload = _read_dict(node_path)
    payload = _read_dict(output_dir / "pipeline_result.json")
    storyline = _safe_text(payload.get("story"))
    affected = set(report["plan"]["affected_stages"])
    from scripts.run_narrative_world_path_pipeline import (
        _run_llm_path,
        _run_mcts_path,
        build_path_candidates_from_narrative_nodes,
        build_path_candidates_from_pipeline_result,
    )

    if "node_selection" in affected:
        shutil.rmtree(path_root, ignore_errors=True)
        path_root.mkdir(parents=True, exist_ok=True)
        args = _path_args(
            strict_online=strict_online,
            retries=retries,
            path_mode="both",
            node_config=None,
        )
        from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig

        selector = NarrativeNodeSelector(
            NarrativeNodeSelectorConfig(
                selection_mode=args.node_selection_mode,
                score_threshold=args.node_score_threshold,
                min_nodes=args.node_min_nodes,
                max_nodes=args.node_max_nodes,
                llm_enabled=args.node_llm_scoring,
            )
        )
        node_payload = selector.select(
            payload,
            source_pipeline_result_json=str(output_dir / "pipeline_result.json"),
        )
        _write_json(path_root / "narrative_nodes.json", node_payload)
        candidates = build_path_candidates_from_narrative_nodes(node_payload)
        _write_json(path_root / "path_candidates.json", candidates)
        affected.update({"path_llm", "path_mcts"})
    else:
        candidates_path = path_root / "path_candidates.json"
        if candidates_path.exists():
            candidates = _read_list(candidates_path)
        elif node_payload:
            candidates = build_path_candidates_from_narrative_nodes(node_payload)
        else:
            candidates = build_path_candidates_from_pipeline_result(payload)
            _write_json(path_root / "path_candidates.json", candidates)

    runs: list[dict[str, Any]] = []
    for stage in ("path_llm", "path_mcts"):
        if stage not in affected:
            continue
        if stage == "path_llm":
            out_dir = path_root / "path_llm_direct"
            shutil.rmtree(out_dir, ignore_errors=True)
            args = _path_args(
                strict_online=strict_online,
                retries=retries,
                path_mode="llm",
                node_config=node_payload,
            )
            path = _run_llm_path(
                storyline=storyline,
                candidates=candidates,
                output_dir=out_dir,
                strict_online=args.strict_online,
                retries=args.retries,
            )
            runs.append({"stage": stage, "output_json": str(path)})
        else:
            out_dir = path_root / "path_mcts"
            shutil.rmtree(out_dir, ignore_errors=True)
            args = _path_args(
                strict_online=strict_online,
                retries=retries,
                path_mode="mcts",
                node_config=node_payload,
            )
            path = _run_mcts_path(
                storyline=storyline,
                candidates=candidates,
                output_dir=out_dir,
                args=args,
            )
            runs.append({"stage": stage, "output_json": str(path)})
    _write_json(
        path_root / "pipeline_manifest.json",
        {
            "status": "done",
            "source": "pre_video_recovery",
            "path_mode": "both" if len(runs) == 2 else (runs[0]["stage"] if runs else "none"),
            "runs": runs,
        },
    )
    return {
        "mode": "path_repair",
        "output_dir": str(output_dir),
        "affected_stages": sorted(affected, key=STAGE_ORDER.index),
        "runs": runs,
    }


def repair_pipeline(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    report: dict[str, Any],
    backend_mode: str = "all_banana",
    strict_online: bool = True,
    retries: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = Path(pipeline_dir).resolve()
    output = ensure_repair_output_dir(source, Path(output_dir))
    plan = build_recovery_plan(report)
    report = copy.deepcopy(report)
    report["plan"] = plan
    if plan["status"] == "healthy":
        _copy_tree_and_rewrite_paths(source, output)
        execution = {"mode": "no_op_copy", "output_dir": str(output)}
    elif any(stage in plan["affected_stages"] for stage in STAGE_ORDER[:4]):
        execution = _run_world_repair(
            source_dir=source,
            output_dir=output,
            report=report,
            backend_mode=backend_mode,
            strict_online=strict_online,
            retries=retries,
            dry_run=dry_run,
        )
        path_stages = {
            stage
            for stage in ("node_selection", "path_llm", "path_mcts")
            if stage in plan["affected_stages"]
        }
        if path_stages:
            path_report = copy.deepcopy(report)
            path_plan = copy.deepcopy(plan)
            path_plan["affected_stages"] = [
                stage for stage in plan["affected_stages"] if stage in path_stages
            ]
            path_report["plan"] = path_plan
            execution["path_repair"] = _run_path_repair(
                source_dir=output,
                output_dir=output,
                report=path_report,
                strict_online=strict_online,
                retries=retries,
            )
    else:
        execution = _run_path_repair(
            source_dir=source,
            output_dir=output,
            report=report,
            strict_online=strict_online,
            retries=retries,
        )
    # Copied source logs may contain the original failed call. Keep them for
    # audit, but evaluate the repaired snapshot from regenerated artifacts and
    # JSON status fields rather than replaying historical log markers.
    post_report = check_pipeline(output, scan_logs=False)
    post_plan = build_recovery_plan(post_report)
    manifest = {
        "status": "done" if post_report["status"] == "healthy" else "partial",
        "source_pipeline_dir": str(source),
        "output_dir": str(output),
        "pre_repair_report": report,
        "post_repair_report": post_report,
        "post_repair_plan": post_plan,
        "execution": execution,
    }
    _write_json(output / "pre_video_recovery_manifest.json", manifest)
    _write_json(output / "pre_video_health_report.json", post_report)
    _write_json(output / "pre_video_recovery_plan.json", post_plan)
    return manifest


def _default_report_dir(pipeline_dir: Path) -> Path:
    return pipeline_dir.parent / f"{pipeline_dir.name}_pre_video_recovery_reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check, plan, or repair all CineWorldSim stages before video generation."
    )
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["check", "plan", "repair"], default="check")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--call-log", action="append", type=Path, default=[])
    parser.add_argument("--backend-mode", choices=["all_banana", "all_flux"], default="all_banana")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    pipeline_dir = args.pipeline_dir.resolve()
    report = check_pipeline(pipeline_dir, call_log_files=args.call_log)
    plan = build_recovery_plan(report)
    if args.mode in {"check", "plan"}:
        report_dir = (args.report_dir or _default_report_dir(pipeline_dir)).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        health_path = _write_json(report_dir / "pre_video_health_report.json", report)
        plan_path = _write_json(report_dir / "pre_video_recovery_plan.json", plan)
        print(json.dumps({
            "mode": args.mode,
            "status": report["status"],
            "health_report": str(health_path),
            "recovery_plan": str(plan_path),
        }, ensure_ascii=False, indent=2))
        return

    if args.output_dir is None:
        raise SystemExit("--output-dir is required for --mode repair")
    manifest = repair_pipeline(
        pipeline_dir=pipeline_dir,
        output_dir=args.output_dir,
        report=report,
        backend_mode=args.backend_mode,
        strict_online=args.strict_online,
        retries=max(1, int(args.retries)),
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
