#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

MODEL_PATH="${1:-/data1/AgentG/hhb/digitalhuman/XD-MU-ScriptAgent}"

echo "[INFO] verify omni compat with model: $MODEL_PATH"
print_effective_env

"$SCRIPTAGENT_PYTHON" - << 'PY'
import os
from qwen_omni_utils import vision_process as vp
print('PYTHONPATH=', os.environ.get('PYTHONPATH',''))
print('IMAGE_FACTOR exists:', hasattr(vp, 'IMAGE_FACTOR'))
print('MIN_PIXELS exists:', hasattr(vp, 'MIN_PIXELS'))
print('VIDEO_TOTAL_PIXELS exists:', hasattr(vp, 'VIDEO_TOTAL_PIXELS'))
PY

"$SCRIPTAGENT_PYTHON" - << PY
from swift.llm import PtEngine
model_path = r"$MODEL_PATH"
engine = PtEngine(model_path, max_batch_size=1)
print('PTENGINE_OK', type(engine).__name__)
PY
