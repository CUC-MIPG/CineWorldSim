from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.client import make_client, pick_model
from agents.retry import call_with_retries


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<start>\d{2}\.\d{2}\.\d{2}\.\d{3})-(?P<end>\d{2}\.\d{2}\.\d{2}\.\d{3})\.json$"
)

CATEGORY_MAP = {
    "dialogue": "对话类",
    "action": "动作类",
    "empty": "无人空镜头类",
}


@dataclass
class ClipRecord:
    file_name: str
    path: Path
    duration_sec: float
    plot: str
    start_ms: int
    end_ms: int


def tc_to_ms(tc: str) -> int:
    hh, mm, ss, mmm = [int(x) for x in tc.split(".")]
    return ((hh * 60 + mm) * 60 + ss) * 1000 + mmm


def _extract_time_from_name(file_name: str) -> tuple[int, int]:
    m = FILENAME_RE.match(file_name)
    if not m:
        return 0, 0
    start_ms = tc_to_ms(m.group("start"))
    end_ms = tc_to_ms(m.group("end"))
    return start_ms, end_ms


def _duration_from_json_or_name(obj: dict[str, Any], file_name: str) -> float:
    cm = obj.get("clip_meta")
    if isinstance(cm, dict):
        d = cm.get("duration_sec")
        if isinstance(d, (int, float)):
            return float(d)
    start_ms, end_ms = _extract_time_from_name(file_name)
    if end_ms > start_ms:
        return round((end_ms - start_ms) / 1000.0, 3)
    return 0.0


def load_records(movie_dir: Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for p in sorted(movie_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        duration_sec = _duration_from_json_or_name(obj, p.name)
        plot = str(obj.get("Plot", "") or "").strip()
        start_ms, end_ms = _extract_time_from_name(p.name)
        records.append(
            ClipRecord(
                file_name=p.name,
                path=p,
                duration_sec=duration_sec,
                plot=plot,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    records.sort(key=lambda x: (x.start_ms, x.end_ms, x.file_name))
    return records


def _nearest_idx(x: float, centers: list[float]) -> int:
    best_i = 0
    best_d = abs(x - centers[0])
    for i in range(1, len(centers)):
        d = abs(x - centers[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def kmeans_1d(values: list[float], k: int, max_iter: int = 100) -> tuple[list[int], list[float], float]:
    if not values:
        return [], [], 0.0
    if k <= 1:
        center = sum(values) / len(values)
        labels = [0 for _ in values]
        inertia = sum((x - center) ** 2 for x in values)
        return labels, [center], inertia

    centers = [_percentile(values, (i + 1) / (k + 1)) for i in range(k)]
    labels = [0 for _ in values]

    for _ in range(max_iter):
        new_labels = [_nearest_idx(x, centers) for x in values]
        if new_labels == labels:
            break
        labels = new_labels

        buckets: list[list[float]] = [[] for _ in range(k)]
        for x, lab in zip(values, labels):
            buckets[lab].append(x)

        for i in range(k):
            if buckets[i]:
                centers[i] = sum(buckets[i]) / len(buckets[i])
            else:
                # Re-init empty center to a farthest point.
                far_x = max(values, key=lambda v: min(abs(v - c) for c in centers))
                centers[i] = far_x

    # Sort centers for deterministic cluster ids.
    order = sorted(range(k), key=lambda i: centers[i])
    remap = {old: new for new, old in enumerate(order)}
    centers = [centers[i] for i in order]
    labels = [remap[x] for x in labels]

    inertia = sum((x - centers[labels[i]]) ** 2 for i, x in enumerate(values))
    return labels, centers, inertia


def silhouette_score_1d(values: list[float], labels: list[int], k: int) -> float:
    n = len(values)
    if n <= 2 or k <= 1:
        return 0.0

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)

    if len(clusters) <= 1:
        return 0.0

    scores = []
    for i in range(n):
        ci = labels[i]
        same = clusters.get(ci, [])
        if len(same) <= 1:
            a = 0.0
        else:
            a = sum(abs(values[i] - values[j]) for j in same if j != i) / (len(same) - 1)

        b = float("inf")
        for c, idxs in clusters.items():
            if c == ci or not idxs:
                continue
            dist = sum(abs(values[i] - values[j]) for j in idxs) / len(idxs)
            if dist < b:
                b = dist
        if b == float("inf"):
            b = 0.0
        denom = max(a, b)
        s = 0.0 if denom == 0 else (b - a) / denom
        scores.append(s)

    return sum(scores) / len(scores) if scores else 0.0


def choose_k_and_cluster(values: list[float], k_min: int = 2, k_max: int = 6) -> dict[str, Any]:
    uniq = len(set(values))
    if not values:
        return {
            "selected_k": 0,
            "labels": [],
            "centers": [],
            "inertia": 0.0,
            "candidates": [],
        }
    if uniq == 1 or len(values) < 4:
        labels, centers, inertia = kmeans_1d(values, 1)
        return {
            "selected_k": 1,
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "candidates": [
                {
                    "k": 1,
                    "silhouette": 0.0,
                    "inertia": inertia,
                    "score": 0.0,
                }
            ],
        }

    k_low = max(2, k_min)
    k_high = min(k_max, uniq, max(2, len(values) // 4 + 1))
    if k_high < k_low:
        k_high = k_low

    candidates = []
    best = None
    for k in range(k_low, k_high + 1):
        labels, centers, inertia = kmeans_1d(values, k)
        sil = silhouette_score_1d(values, labels, k)
        # Slight penalty for too many segments.
        score = sil - (0.015 * k)
        row = {
            "k": k,
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "silhouette": sil,
            "score": score,
        }
        candidates.append(
            {
                "k": k,
                "inertia": round(inertia, 6),
                "silhouette": round(sil, 6),
                "score": round(score, 6),
            }
        )
        if best is None or score > best["score"]:
            best = row

    assert best is not None
    return {
        "selected_k": best["k"],
        "labels": best["labels"],
        "centers": best["centers"],
        "inertia": best["inertia"],
        "candidates": candidates,
    }


class PlotClassifierLLM:
    def __init__(
        self,
        model: str | None = None,
        retries: int = 3,
        strict_online: bool = True,
    ):
        self.client = make_client(scope="ANALYSIS")
        self.model = model or pick_model("gpt-5.5", scope="ANALYSIS")
        self.retries = retries
        self.strict_online = strict_online

    @staticmethod
    def _extract_json_obj(text: str) -> dict[str, Any]:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM response")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON root must be object")
        return obj

    @staticmethod
    def _fallback_label(plot: str) -> tuple[str, str]:
        t = (plot or "").lower()
        if not t.strip():
            return "empty", "Plot为空，按无人空镜头兜底。"

        empty_kw = ["无人", "空镜", "街景", "风景", "建筑", "empty", "landscape", "establishing"]
        dialogue_kw = ["对话", "交谈", "说", "问", "回答", "谈", "conversation", "talk", "says", "asks", "replies"]
        action_kw = ["跑", "追", "打", "冲", "开车", "战斗", "fight", "run", "chase", "shoot", "grab", "move", "enter"]

        if any(k in t for k in empty_kw):
            return "empty", "命中空镜关键词。"
        if any(k in t for k in dialogue_kw):
            return "dialogue", "命中对话关键词。"
        if any(k in t for k in action_kw):
            return "action", "命中动作关键词。"
        return "action", "无明显对话/空镜信号，按动作类兜底。"

    def classify_batch(self, batch: list[ClipRecord]) -> dict[str, dict[str, str]]:
        items = [{"id": r.file_name, "plot": r.plot} for r in batch]
        system = (
            "你是电影镜头内容分类器。"
            "根据Plot文本将片段归类到三类之一：dialogue/action/empty。"
            "只输出JSON。"
        )
        user = (
            "分类规则：\n"
            "1) dialogue: 以人物交流、对白推进为主。\n"
            "2) action: 以动作、行为、冲突、运动推进为主。\n"
            "3) empty: 无人物主体的环境/空镜头。\n\n"
            "输入：\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
            "输出JSON格式：\n"
            "{\n"
            '  "results": [\n'
            '    {"id":"...", "category":"dialogue|action|empty", "reason":"一句话理由"}\n'
            "  ]\n"
            "}\n"
        )

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content if resp.choices else "{}"
            if not isinstance(content, str):
                content = str(content)
            return self._extract_json_obj(content)

        try:
            obj = call_with_retries(_call, attempts=self.retries, delay_sec=1.0)
            rows = obj.get("results", [])
            out: dict[str, dict[str, str]] = {}
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    cid = str(row.get("id", ""))
                    cat = str(row.get("category", "")).strip().lower()
                    reason = str(row.get("reason", "")).strip()
                    if not cid:
                        continue
                    if cat not in {"dialogue", "action", "empty"}:
                        cat, fb = self._fallback_label(next((r.plot for r in batch if r.file_name == cid), ""))
                        reason = reason or fb
                    out[cid] = {"category": cat, "reason": reason}
            # Fill missing ids with fallback.
            for rec in batch:
                if rec.file_name not in out:
                    cat, reason = self._fallback_label(rec.plot)
                    out[rec.file_name] = {"category": cat, "reason": reason}
            return out
        except Exception:
            if self.strict_online:
                raise
            out = {}
            for rec in batch:
                cat, reason = self._fallback_label(rec.plot)
                out[rec.file_name] = {"category": cat, "reason": reason}
            return out

    def classify(self, records: list[ClipRecord], batch_size: int = 20) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            result.update(self.classify_batch(batch))
        return result


def build_clusters_report(values: list[float], labels: list[int], centers: list[float]) -> list[dict[str, Any]]:
    if not values:
        return []
    by_cluster: dict[int, list[float]] = {}
    for x, lab in zip(values, labels):
        by_cluster.setdefault(lab, []).append(x)

    rows = []
    for lab in sorted(by_cluster.keys()):
        vals = by_cluster[lab]
        rows.append(
            {
                "cluster_id": int(lab),
                "cluster_label": f"T{lab + 1}",
                "count": len(vals),
                "centroid_sec": round(float(centers[lab]), 4),
                "min_sec": round(min(vals), 4),
                "max_sec": round(max(vals), 4),
                "mean_sec": round(sum(vals) / len(vals), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MovieBench clip JSONs by duration clustering (KMeans) and Plot semantic classification."
        )
    )
    parser.add_argument(
        "--movie-dir",
        type=Path,
        default=Path(
            "/data1/AgentG/hhb/CameraMaster/MovieBench/Annotation_Shot_Desc_11.15_V2_160movies/3075_THE_SITTER"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <movie-dir>/analysis_duration_plot",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", type=str, default=None, help="Override OpenAI model for Plot classification.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--strict-online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail when LLM API errors occur. If false, fallback to keyword heuristics.",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="Write analysis_meta back into each source JSON.",
    )
    args = parser.parse_args()

    movie_dir = args.movie_dir
    if not movie_dir.exists() or not movie_dir.is_dir():
        raise SystemExit(f"Invalid movie-dir: {movie_dir}")

    output_dir = args.output_dir or (movie_dir / "analysis_duration_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(movie_dir)
    if not records:
        raise SystemExit(f"No valid JSON records found in: {movie_dir}")

    durations = [r.duration_sec for r in records]
    clustering = choose_k_and_cluster(
        values=durations,
        k_min=max(1, args.k_min),
        k_max=max(1, args.k_max),
    )
    labels = clustering["labels"]
    centers = clustering["centers"]
    cluster_rows = build_clusters_report(durations, labels, centers)

    classifier = PlotClassifierLLM(
        model=args.model,
        retries=max(1, args.retries),
        strict_online=bool(args.strict_online),
    )
    classify_map = classifier.classify(records, batch_size=max(1, args.batch_size))

    clip_rows: list[dict[str, Any]] = []
    cat_count = {"dialogue": 0, "action": 0, "empty": 0}
    for rec, lab in zip(records, labels):
        c = classify_map.get(rec.file_name, {})
        cat = str(c.get("category", "action"))
        if cat not in cat_count:
            cat = "action"
        cat_count[cat] += 1
        clip_row = {
            "file_name": rec.file_name,
            "path": str(rec.path),
            "duration_sec": rec.duration_sec,
            "duration_cluster_id": int(lab),
            "duration_cluster_label": f"T{lab + 1}",
            "plot": rec.plot,
            "plot_category_code": cat,
            "plot_category": CATEGORY_MAP[cat],
            "plot_category_reason": str(c.get("reason", "")),
        }
        clip_rows.append(clip_row)

        if args.write_back:
            obj = json.loads(rec.path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj["analysis_meta"] = {
                    "duration_sec": rec.duration_sec,
                    "duration_cluster_id": int(lab),
                    "duration_cluster_label": f"T{lab + 1}",
                    "plot_category_code": cat,
                    "plot_category": CATEGORY_MAP[cat],
                    "plot_category_reason": str(c.get("reason", "")),
                }
                rec.path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "movie_dir": str(movie_dir),
        "total_clips": len(records),
        "duration_statistics": {
            "min_sec": round(min(durations), 4),
            "max_sec": round(max(durations), 4),
            "mean_sec": round(sum(durations) / len(durations), 4),
            "median_sec": round(statistics.median(durations), 4),
        },
        "duration_kmeans": {
            "selected_k": clustering["selected_k"],
            "candidates": clustering["candidates"],
            "clusters": cluster_rows,
        },
        "plot_classification_statistics": {
            "dialogue_count": cat_count["dialogue"],
            "action_count": cat_count["action"],
            "empty_count": cat_count["empty"],
        },
        "clips": clip_rows,
    }

    report_json = output_dir / "analysis_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "clips_analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "duration_sec",
                "duration_cluster_id",
                "duration_cluster_label",
                "plot_category_code",
                "plot_category",
                "plot_category_reason",
            ],
        )
        writer.writeheader()
        for row in clip_rows:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "duration_sec": row["duration_sec"],
                    "duration_cluster_id": row["duration_cluster_id"],
                    "duration_cluster_label": row["duration_cluster_label"],
                    "plot_category_code": row["plot_category_code"],
                    "plot_category": row["plot_category"],
                    "plot_category_reason": row["plot_category_reason"],
                }
            )

    print("=== MovieBench Analysis Completed ===")
    print(f"Movie dir: {movie_dir}")
    print(f"Total clips: {len(records)}")
    print(f"Selected K (duration clusters): {clustering['selected_k']}")
    print(
        "Plot classes: "
        f"dialogue={cat_count['dialogue']}, "
        f"action={cat_count['action']}, "
        f"empty={cat_count['empty']}"
    )
    print(f"Report JSON: {report_json}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
