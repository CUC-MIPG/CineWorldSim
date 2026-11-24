from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_STORYLINE = (
    "A quiet museum heist begins at midnight. "
    "The curator notices a missing key artifact, "
    "while a security guard and an unknown visitor each react differently."
)


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, float]:
    t0 = time.time()
    print(f"[RUN] {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    dt = time.time() - t0
    print(f"[DONE] rc={proc.returncode} elapsed_sec={dt:.2f}")
    return proc.returncode, dt


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full auto pipeline: narrative simulation -> state analysis output -> path orchestration (LLM + MCTS)."
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Root output directory for this full run.")
    parser.add_argument("--storyline", type=str, default=DEFAULT_STORYLINE)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-online", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--mcts-model", type=str, default="gpt-5.2")
    parser.add_argument("--mcts-budget-iters", type=int, default=1)
    parser.add_argument("--pov-switch-penalty", type=float, default=0.05)
    parser.add_argument("--over-focus-threshold", type=int, default=4)
    parser.add_argument("--over-focus-penalty", type=float, default=0.08)
    parser.add_argument("--w-narrative-causality", type=float, default=0.4)
    parser.add_argument("--w-visual-transition", type=float, default=0.3)
    parser.add_argument("--w-rhythmic-pacing", type=float, default=0.3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    narrative_dir = out_root / "narrative"
    narrative_json = narrative_dir / "result.json"
    llm_path_dir = out_root / "path_llm_direct"
    mcts_path_dir = out_root / "path_mcts"
    manifest = out_root / "pipeline_manifest.json"

    runs: list[dict[str, Any]] = []

    # Step 1: narrative + state analysis
    cmd1 = [
        "python",
        str(project_root / "scripts" / "run_narrative_simulation_example.py"),
        "--output-dir",
        str(narrative_dir),
        "--output-json",
        str(narrative_json),
        "--max-steps",
        str(int(args.max_steps)),
        "--log-level",
        args.log_level,
    ]
    cmd1.append("--resume" if args.resume else "--no-resume")
    rc1, dt1 = _run(cmd1, cwd=project_root)
    runs.append(
        {
            "step": "narrative_state_analysis",
            "cmd": cmd1,
            "rc": rc1,
            "elapsed_sec": dt1,
            "output_json": str(narrative_json),
        }
    )
    _write_json(
        manifest,
        {
            "status": "running" if rc1 == 0 else "failed",
            "storyline": args.storyline,
            "output_root": str(out_root),
            "runs": runs,
        },
    )
    if rc1 != 0:
        raise SystemExit(rc1)

    # Step 2: LLM direct path orchestrator
    cmd2 = [
        "python",
        str(project_root / "scripts" / "run_path_orchestrator_narrative.py"),
        "--storyline",
        args.storyline,
        "--narrative-json",
        str(narrative_json),
        "--output-dir",
        str(llm_path_dir),
        "--retries",
        str(int(args.retries)),
        "--log-level",
        args.log_level,
    ]
    cmd2.append("--strict-online" if args.strict_online else "--no-strict-online")
    rc2, dt2 = _run(cmd2, cwd=project_root)
    runs.append(
        {
            "step": "path_orchestrator_llm_direct",
            "cmd": cmd2,
            "rc": rc2,
            "elapsed_sec": dt2,
            "output_json": str(llm_path_dir / "narrative_path_plan.json"),
        }
    )
    _write_json(
        manifest,
        {
            "status": "running" if rc2 == 0 else "failed",
            "storyline": args.storyline,
            "output_root": str(out_root),
            "runs": runs,
        },
    )
    if rc2 != 0:
        raise SystemExit(rc2)

    # Step 3: MCTS path orchestrator
    cmd3 = [
        "python",
        str(project_root / "scripts" / "run_path_orchestrator_mcts.py"),
        "--storyline",
        args.storyline,
        "--narrative-json",
        str(narrative_json),
        "--output-dir",
        str(mcts_path_dir),
        "--model",
        args.mcts_model,
        "--budget-iters",
        str(int(args.mcts_budget_iters)),
        "--retries",
        str(int(args.retries)),
        "--log-level",
        args.log_level,
        "--pov-switch-penalty",
        str(float(args.pov_switch_penalty)),
        "--over-focus-threshold",
        str(int(args.over_focus_threshold)),
        "--over-focus-penalty",
        str(float(args.over_focus_penalty)),
        "--w-narrative-causality",
        str(float(args.w_narrative_causality)),
        "--w-visual-transition",
        str(float(args.w_visual_transition)),
        "--w-rhythmic-pacing",
        str(float(args.w_rhythmic_pacing)),
    ]
    cmd3.append("--strict-online" if args.strict_online else "--no-strict-online")
    rc3, dt3 = _run(cmd3, cwd=project_root)
    runs.append(
        {
            "step": "path_orchestrator_mcts",
            "cmd": cmd3,
            "rc": rc3,
            "elapsed_sec": dt3,
            "output_json": str(mcts_path_dir / "narrative_path_plan_mcts.json"),
        }
    )

    final_status = "done" if rc3 == 0 else "failed"
    payload = {
        "status": final_status,
        "storyline": args.storyline,
        "output_root": str(out_root),
        "artifacts": {
            "narrative_result_json": str(narrative_json),
            "llm_path_json": str(llm_path_dir / "narrative_path_plan.json"),
            "mcts_path_json": str(mcts_path_dir / "narrative_path_plan_mcts.json"),
            "mcts_trace_jsonl": str(mcts_path_dir / "mcts_trace.jsonl"),
        },
        "runs": runs,
    }
    _write_json(manifest, payload)
    print(manifest)
    if rc3 != 0:
        raise SystemExit(rc3)


if __name__ == "__main__":
    main()

