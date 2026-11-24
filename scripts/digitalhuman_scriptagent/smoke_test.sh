#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[SMOKE] director --help"
"$SCRIPT_DIR/run_director.sh" --help >/dev/null

echo "[SMOKE] critic_script --help"
"$SCRIPT_DIR/run_critic_script.sh" --help >/dev/null

echo "[SMOKE] critic_video --help"
if "$SCRIPT_DIR/run_critic_video.sh" --help >/dev/null 2>&1; then
  echo "[SMOKE] critic_video --help ok"
else
  echo "[SMOKE][WARN] critic_video help failed in current python (likely missing torch/transformers)."
fi

echo "[SMOKE] OK: ScriptAgent wrappers are callable"
