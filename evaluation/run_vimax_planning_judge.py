"""Run the ViMax Appendix Figure 12/13/14 LLM judge on adapted samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vimax_llm_judge import (
    NF_METRICS,
    PLANNING_METRICS,
    aggregate_judge_batches,
    build_nf_prompt,
    build_planning_prompt,
    chat_completion,
    validate_judge_payload,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _batches(shots: list[dict[str, Any]], batch_size: int, max_items: int | None) -> list[list[dict[str, Any]]]:
    selected = shots[:max_items] if max_items else shots
    return [selected[index:index + batch_size] for index in range(0, len(selected), batch_size)]


def run(
    samples_json: str | Path,
    output: str | Path,
    metric: str,
    source_text: str | Path | None = None,
    sample_id: str | None = None,
    batch_size: int = 20,
    max_items: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-5.4",
    timeout_sec: int = 300,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = _read_json(Path(samples_json))
    samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    if sample_id:
        samples = [sample for sample in samples if sample.get("sample_id") == sample_id]
    if not samples:
        raise ValueError("no matching adapted sample")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if metric not in {"planning", "nf", "both"}:
        raise ValueError("metric must be planning, nf, or both")
    source = None
    if metric in {"nf", "both"}:
        if source_text is None:
            raise ValueError("--source-text is required for NF")
        source_path = Path(source_text)
        source = source_path.read_text(encoding="utf-8") if source_path.is_file() else str(source_text)
    api_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    records = []
    for sample in samples:
        shot_batches = _batches(sample.get("shots", []), batch_size, max_items)
        sample_record: dict[str, Any] = {"sample_id": sample.get("sample_id"), "batch_count": len(shot_batches)}
        for kind in (("planning",) if metric == "planning" else ("nf",) if metric == "nf" else ("planning", "nf")):
            metric_names = PLANNING_METRICS if kind == "planning" else NF_METRICS
            results = []
            prompts = []
            for batch_index, batch in enumerate(shot_batches, start=1):
                prompt = build_planning_prompt(batch) if kind == "planning" else build_nf_prompt(source or "", batch)
                prompts.append(prompt)
                if dry_run:
                    continue
                if not api_key:
                    raise ValueError("OPENAI_API_KEY or --api-key is required unless --dry-run is used")
                raw_payload, raw_text = chat_completion(api_base, api_key, model, prompt, timeout_sec)
                validated = validate_judge_payload(raw_payload, metric_names)
                validated["batch_index"] = batch_index
                validated["raw_response"] = raw_text
                results.append(validated)
            sample_record[kind] = {
                "metric_names": list(metric_names),
                "batch_size": batch_size,
                "max_items": max_items,
                "batches": results,
                "aggregate": aggregate_judge_batches(results) if results else None,
                "prompts": prompts if dry_run else None,
            }
        records.append(sample_record)
    report = {
        "schema_version": "vimax-planning-judge-v1",
        "prompt_version": "vimax-appendix-fig12-14-replica-v1",
        "source_samples_json": str(Path(samples_json).resolve()),
        "source_text_provided": source is not None,
        "model": model,
        "dry_run": dry_run,
        "samples": records,
        "protocol_note": "For the paper protocol, use 20-storyboard batches; baselines report first 20 and ViMax reports first 100 as five batches.",
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-json", type=Path, required=True, help="JSON produced by vimax_bench_adapter.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("planning", "nf", "both"), default="planning")
    parser.add_argument("--source-text", help="source novel text or a text file, required for NF")
    parser.add_argument("--sample-id")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({"sample_count": len(report["samples"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
