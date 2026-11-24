#!/usr/bin/env python3
"""Validate CineWorldSim output against ViStoryBench shot anchors."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


SHOT_RE = re.compile(r"(?:^|_)(\d+)(?:\.[^.]+)$")


def load_story(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_indices(story: dict[str, Any]) -> list[int]:
    return [int(item["index"]) for item in story.get("Shots", []) if isinstance(item, dict) and "index" in item]


def validate_story_output(story: dict[str, Any], story_dir: Path) -> dict[str, Any]:
    expected = expected_indices(story)
    shot_dir = story_dir / "shots"
    actual: dict[int, str] = {}
    errors: list[str] = []
    if not shot_dir.is_dir():
        errors.append(f"missing shots directory: {shot_dir}")
    else:
        for path in sorted(shot_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            match = SHOT_RE.search(path.name)
            if not match:
                errors.append(f"unparseable shot filename: {path.name}")
                continue
            index = int(match.group(1))
            if index in actual:
                errors.append(f"duplicate shot index {index}: {actual[index]} and {path.name}")
            actual[index] = str(path)
            if path.stat().st_size <= 0:
                errors.append(f"empty image: {path}")
                continue
            try:
                with Image.open(path) as image:
                    if image.format != "PNG":
                        errors.append(f"not PNG encoded: {path} (format={image.format})")
                    image.verify()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid image: {path} ({type(exc).__name__}: {exc})")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    errors.extend(f"missing shot {index}" for index in missing)
    errors.extend(f"unexpected shot {index}" for index in extra)
    return {
        "expected_indices": expected,
        "actual_indices": sorted(actual),
        "missing": missing,
        "extra": extra,
        "errors": errors,
        "ok": not errors,
    }


def validate_root(input_root: Path, dataset_root: Path, story_ids: list[str] | None = None) -> dict[str, Any]:
    stories_root = dataset_root
    selected = story_ids or sorted(path.name for path in input_root.iterdir() if path.is_dir())
    results: dict[str, Any] = {}
    for story_id in selected:
        story_path = stories_root / story_id / "story.json"
        if not story_path.exists():
            results[story_id] = {"ok": False, "errors": [f"missing story.json: {story_path}"]}
            continue
        results[story_id] = validate_story_output(load_story(story_path), input_root / story_id)
    return {"ok": all(item.get("ok") for item in results.values()), "stories": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--story-id", action="append", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = validate_root(args.input_root, args.dataset_root, args.story_id)
    path = args.manifest or args.input_root / "vistorybench_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
