from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.client import make_client, pick_model


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<start>\d{2}\.\d{2}\.\d{2}\.\d{3})-(?P<end>\d{2}\.\d{2}\.\d{2}\.\d{3})\.json$"
)


@dataclass
class Clip:
    path: Path
    file_name: str
    movie_id: str
    order_index: int
    start_tc: str
    end_tc: str
    start_ms: int
    end_ms: int
    plot: str
    background_description: str


def tc_to_ms(tc: str) -> int:
    hh, mm, ss, mmm = [int(x) for x in tc.split(".")]
    return ((hh * 60 + mm) * 60 + ss) * 1000 + mmm


def ms_to_sec(ms: int) -> float:
    return round(ms / 1000.0, 3)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"LLM response is not valid JSON: {text[:240]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root is not an object.")
    return obj


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _parse_time_from_name(name: str) -> tuple[str, str, int, int]:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Bad file name format: {name}")
    start_tc = m.group("start")
    end_tc = m.group("end")
    start_ms = tc_to_ms(start_tc)
    end_ms = tc_to_ms(end_tc)
    return start_tc, end_tc, start_ms, end_ms


def load_clips(movie_dir: Path) -> list[Clip]:
    files = sorted(movie_dir.glob("*.json"))
    clips: list[Clip] = []

    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON: {p}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"JSON must be object: {p}")

        plot = _safe_text(payload.get("Plot", ""))
        bg = _safe_text(payload.get("Background Description", ""))
        start_tc, end_tc, start_ms, end_ms = _parse_time_from_name(p.name)

        meta = payload.get("clip_meta") if isinstance(payload.get("clip_meta"), dict) else {}
        movie_id = _safe_text(meta.get("movie_id")) or movie_dir.name
        order_index = meta.get("order_index_in_movie")
        if not isinstance(order_index, int):
            order_index = -1

        clips.append(
            Clip(
                path=p,
                file_name=p.name,
                movie_id=movie_id,
                order_index=order_index,
                start_tc=start_tc,
                end_tc=end_tc,
                start_ms=start_ms,
                end_ms=end_ms,
                plot=plot,
                background_description=bg,
            )
        )

    # Use time-based order as primary requirement. Keep deterministic tie-break.
    clips.sort(key=lambda x: (x.start_ms, x.end_ms, x.file_name))
    for i, c in enumerate(clips, start=1):
        if c.order_index < 0:
            c.order_index = i
    return clips


class SceneSegmenterLLM:
    def __init__(self, model: str | None = None, retries: int = 3):
        self.client = make_client(scope="SCENE_SEG")
        self.model = model or pick_model("gpt-5.5", scope="SCENE_SEG")
        self.retries = retries

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for _ in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content if resp.choices else "{}"
                if not isinstance(content, str):
                    content = str(content)
                return _extract_json_object(content)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def bootstrap_scene(self, clips: list[Clip]) -> dict[str, Any]:
        if not clips:
            raise ValueError("bootstrap_scene clips is empty")
        n = len(clips)
        clip_rows = []
        for i, c in enumerate(clips, start=1):
            clip_rows.append(
                {
                    "clip_pos": i,
                    "file_name": c.file_name,
                    "plot": c.plot,
                    "background_description": c.background_description,
                }
            )

        system = (
            "You are a film scene segmentation assistant. "
            "Use only Plot and Background Description. "
            "Return strict JSON."
        )
        user = (
            "Task: Determine how many clips from the beginning belong to the same scene/storyline.\n"
            "Rules:\n"
            f"- You are given {n} consecutive clips.\n"
            "- Compute contiguous prefix count from clip_pos=1 that still belongs to same scene/storyline.\n"
            f"- prefix_count must be integer in [1, {n}].\n"
            "- Provide a concise storyline summary for the accepted prefix.\n\n"
            "Input clips JSON:\n"
            f"{json.dumps(clip_rows, ensure_ascii=False, indent=2)}\n\n"
            "Output JSON schema:\n"
            "{\n"
            '  "prefix_count": <int>,\n'
            '  "scene_storyline_summary": "<string>",\n'
            '  "reason": "<short string>"\n'
            "}"
        )
        obj = self._chat_json(system, user)
        prefix_count = obj.get("prefix_count", 1)
        if not isinstance(prefix_count, int):
            prefix_count = 1
        prefix_count = max(1, min(n, prefix_count))
        summary = _safe_text(obj.get("scene_storyline_summary"))
        if not summary:
            summary = self._fallback_summary(clips[:prefix_count])
        return {
            "prefix_count": prefix_count,
            "scene_storyline_summary": summary,
            "reason": _safe_text(obj.get("reason")),
            "raw": obj,
        }

    def check_merge_with_current(self, current_summary: str, candidate: Clip) -> dict[str, Any]:
        system = (
            "You are a film scene segmentation assistant. "
            "Use only Plot and Background Description to decide if candidate clip still belongs to current storyline. "
            "Return strict JSON."
        )
        user = (
            "Current scene storyline summary:\n"
            f"{current_summary}\n\n"
            "Candidate clip:\n"
            f"{json.dumps({'file_name': candidate.file_name, 'plot': candidate.plot, 'background_description': candidate.background_description}, ensure_ascii=False, indent=2)}\n\n"
            "Output JSON schema:\n"
            "{\n"
            '  "belongs_to_current_scene": <true|false>,\n'
            '  "updated_scene_storyline_summary": "<string>",\n'
            '  "reason": "<short string>"\n'
            "}"
        )
        obj = self._chat_json(system, user)
        belongs = obj.get("belongs_to_current_scene")
        belongs = bool(belongs) if isinstance(belongs, bool) else False
        updated_summary = _safe_text(obj.get("updated_scene_storyline_summary"))
        if belongs and not updated_summary:
            updated_summary = current_summary
        return {
            "belongs_to_current_scene": belongs,
            "updated_scene_storyline_summary": updated_summary,
            "reason": _safe_text(obj.get("reason")),
            "raw": obj,
        }

    @staticmethod
    def _fallback_summary(clips: list[Clip]) -> str:
        parts = []
        for c in clips[:3]:
            txt = (c.plot or c.background_description or "").strip()
            if txt:
                parts.append(txt)
        if not parts:
            return "No clear storyline from provided clips."
        joined = " ".join(parts)
        return joined[:500]


def scene_payload(
    scene_id: int,
    movie_id: str,
    clips: list[Clip],
    storyline: str,
    bootstrap_trace: dict[str, Any],
    incremental_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    start = clips[0]
    end = clips[-1]
    return {
        "scene_id": f"scene{scene_id}",
        "movie_id": movie_id,
        "scene_storyline": storyline,
        "clip_count": len(clips),
        "time_range": {
            "start_tc": start.start_tc,
            "end_tc": end.end_tc,
            "start_ms": start.start_ms,
            "end_ms": end.end_ms,
            "duration_ms": end.end_ms - start.start_ms,
            "duration_sec": ms_to_sec(end.end_ms - start.start_ms),
        },
        "clip_files": [c.file_name for c in clips],
        "clip_order_indices": [c.order_index for c in clips],
        "segments": [
            {
                "file_name": c.file_name,
                "order_index_in_movie": c.order_index,
                "clip_start_tc": c.start_tc,
                "clip_end_tc": c.end_tc,
                "clip_start_ms": c.start_ms,
                "clip_end_ms": c.end_ms,
                "plot": c.plot,
                "background_description": c.background_description,
            }
            for c in clips
        ],
        "llm_trace": {
            "bootstrap": bootstrap_trace,
            "incremental_checks": incremental_trace,
        },
    }


def run_segmentation(movie_dir: Path, output_dir: Path, model: str | None, retries: int) -> dict[str, Any]:
    clips = load_clips(movie_dir)
    if not clips:
        raise RuntimeError(f"No JSON files in movie dir: {movie_dir}")

    llm = SceneSegmenterLLM(model=model, retries=retries)
    movie_id = clips[0].movie_id or movie_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)
    scenes_summary: list[dict[str, Any]] = []

    i = 0
    scene_id = 1
    n = len(clips)

    while i < n:
        seed = clips[i : min(i + 3, n)]
        bootstrap = llm.bootstrap_scene(seed)
        prefix_count = bootstrap["prefix_count"]
        current = seed[:prefix_count]
        storyline = bootstrap["scene_storyline_summary"]
        i += prefix_count

        incremental_trace: list[dict[str, Any]] = []
        while i < n:
            cand = clips[i]
            inc = llm.check_merge_with_current(storyline, cand)
            incremental_trace.append(
                {
                    "candidate_file": cand.file_name,
                    "candidate_order_index": cand.order_index,
                    "belongs_to_current_scene": inc["belongs_to_current_scene"],
                    "reason": inc["reason"],
                }
            )
            if inc["belongs_to_current_scene"]:
                current.append(cand)
                storyline = inc["updated_scene_storyline_summary"] or storyline
                i += 1
            else:
                break

        payload = scene_payload(
            scene_id=scene_id,
            movie_id=movie_id,
            clips=current,
            storyline=storyline,
            bootstrap_trace=bootstrap,
            incremental_trace=incremental_trace,
        )
        out_file = output_dir / f"scene{scene_id}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        scenes_summary.append(
            {
                "scene_id": f"scene{scene_id}",
                "clip_count": len(current),
                "start_tc": current[0].start_tc,
                "end_tc": current[-1].end_tc,
                "scene_storyline": storyline,
                "file": str(out_file),
            }
        )
        scene_id += 1

    summary = {
        "movie_id": movie_id,
        "movie_dir": str(movie_dir),
        "total_clips": len(clips),
        "total_scenes": len(scenes_summary),
        "output_dir": str(output_dir),
        "scenes": scenes_summary,
    }
    (output_dir / "scene_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based MovieBench scene segmentation.\n"
            "Uses only Plot + Background Description.\n"
            "Algorithm: bootstrap with first 3 consecutive clips, then merge clips one by one."
        )
    )
    parser.add_argument(
        "--movie-dir",
        type=Path,
        required=True,
        help="Path to a single movie folder containing clip JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for scene*.json. Default: <movie-dir>/llm_scene_segments",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--retries", type=int, default=3, help="LLM retry times per request.")
    args = parser.parse_args()

    movie_dir = args.movie_dir
    if not movie_dir.exists() or not movie_dir.is_dir():
        raise SystemExit(f"movie-dir is invalid: {movie_dir}")

    output_dir = args.output_dir or (movie_dir / "llm_scene_segments")
    summary = run_segmentation(movie_dir, output_dir, model=args.model, retries=args.retries)
    print("=== Scene Segmentation Completed ===")
    print(f"movie_id: {summary['movie_id']}")
    print(f"total_clips: {summary['total_clips']}")
    print(f"total_scenes: {summary['total_scenes']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"scene_index: {output_dir / 'scene_index.json'}")


if __name__ == "__main__":
    main()
