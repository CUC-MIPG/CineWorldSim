# ViMax MovieBench Arc Baseline

## Goal
Build a reproducible ViMax baseline runner for the CameraMaster MovieBench arc_01 protocol, consuming its exact comparison plan, prompts, and shared reference images.

## Phases
- [complete] Inspect protocol, plan schema, ViMax video-generator contract, and asset availability
- [complete] Add tested dry-run/preflight and paired-condition generation runner
- [complete] Add usage documentation and verify dry-run plus focused tests

## Constraints
- Default execution must not make API calls.
- Never regenerate baseline reference assets; require exact planned files for real runs.
- Preserve the prompt, node ordering, references, task mapping, and separate condition outputs from the protocol.
- Do not alter unrelated worktree changes.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Bare system Python cannot import project runtime because `cv2` is absent | `python scripts/run_vimax_moviebench_arc01_baseline.py --help` | Run the script with the documented `conda run -n Vimax` environment, where project dependencies and tests pass |
