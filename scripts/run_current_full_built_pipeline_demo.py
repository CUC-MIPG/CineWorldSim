#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STORYLINE = (
    "On a rain-lashed floating market at midnight, an exiled cartographer follows a glowing paper crane "
    "to a locked observatory. A masked archivist tries to stop her from opening a star map that reveals "
    "the city has been drifting toward a hidden waterfall. A young mechanic secretly moves the broken "
    "tide engine while the market lights begin to fail."
)

DEFAULT_CHARACTERS = "Cartographer,MaskedArchivist,YoungMechanic"


def _bool_flag(enabled: bool, name: str) -> str:
    return name if enabled else f"--no-{name[2:]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the currently built CineWorldSim pipeline with a built-in demo story: "
            "spatiotemporal simulation -> world updates -> event keyframes -> "
            "NarrativeNodeSelector -> PathOrchestrator."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/current_full_built_pipeline_demo"))
    parser.add_argument("--storyline", type=str, default=DEFAULT_STORYLINE)
    parser.add_argument("--characters", type=str, default=DEFAULT_CHARACTERS)
    parser.add_argument(
        "--backend-mode",
        choices=["all_flux", "all_banana", "all_flux2_klein"],
        default="all_banana",
    )
    parser.add_argument("--run-both", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--path-mode", choices=["llm", "mcts", "both"], default="llm")
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--node-llm-scoring", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--node-score-threshold", type=float, default=0.55)
    parser.add_argument("--node-min-nodes", type=int, default=3)
    parser.add_argument("--node-max-nodes", type=int, default=8)
    parser.add_argument("--mcts-budget-iters", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--real-run",
        action="store_true",
        help="Call real LLM/image backends. Default is dry-run so every built module can be smoke-tested quickly.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_narrative_world_path_pipeline.py"),
        "--storyline",
        args.storyline,
        "--characters",
        args.characters,
        "--max-steps",
        str(max(1, int(args.max_steps))),
        "--output-dir",
        str(args.output_dir),
        "--path-mode",
        args.path_mode,
        "--node-score-threshold",
        str(float(args.node_score_threshold)),
        "--node-min-nodes",
        str(max(1, int(args.node_min_nodes))),
        "--node-max-nodes",
        str(max(1, int(args.node_max_nodes))),
        "--mcts-budget-iters",
        str(max(1, int(args.mcts_budget_iters))),
        "--log-level",
        args.log_level,
        _bool_flag(args.strict_online, "--strict-online"),
        _bool_flag(args.node_llm_scoring, "--node-llm-scoring"),
    ]
    if args.run_both:
        cmd.append("--run-both")
    else:
        cmd.extend(["--backend-mode", args.backend_mode])
    if not args.real_run:
        cmd.append("--dry-run")

    print("=== CineWorldSim Current Full Built Pipeline Demo ===")
    print(f"Output dir : {args.output_dir}")
    print(f"Real run   : {bool(args.real_run)}")
    print(f"Backends   : {'all_flux,all_banana' if args.run_both else args.backend_mode}")
    print(f"Path mode  : {args.path_mode}")
    print(f"Storyline  : {args.storyline}")
    print("[RUN] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(project_root), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print("=== Demo Completed ===")
    print(args.output_dir / "pipeline_manifest.json")


if __name__ == "__main__":
    main()
