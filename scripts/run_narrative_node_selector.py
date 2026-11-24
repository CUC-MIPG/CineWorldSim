#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.narrative_node_selector import NarrativeNodeSelector, NarrativeNodeSelectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Select key narrative nodes from a pipeline_result.json file.")
    parser.add_argument("--pipeline-result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-mode", choices=["hybrid", "threshold", "top_k"], default="hybrid")
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--min-nodes", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--llm-scoring", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    selector = NarrativeNodeSelector(
        NarrativeNodeSelectorConfig(
            selection_mode=args.selection_mode,
            score_threshold=args.score_threshold,
            min_nodes=max(1, int(args.min_nodes)),
            max_nodes=max(1, int(args.max_nodes)),
            llm_enabled=bool(args.llm_scoring),
        )
    )
    out = selector.select_from_file(args.pipeline_result_json, args.output_dir)
    print(out)


if __name__ == "__main__":
    main()
