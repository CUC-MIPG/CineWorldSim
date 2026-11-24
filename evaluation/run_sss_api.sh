#!/usr/bin/env bash
# Run the SSS smoke evaluator against an OpenAI-compatible API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SSS_ENV_FILE:-/data1/AgentG/hhb/CameraMaster/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 2
fi

# Load project configuration without printing its values. Explicit SSS_*
# variables supplied by the caller below still take precedence.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# Required after loading the project environment.
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from ${ENV_FILE} or the shell environment}"

VIDEO_PATH="${SSS_VIDEO:-${SCRIPT_DIR}/adapted_cases/0001_American_Beauty_arc_001_real/directorbench/track_B.mp4}"
L4_GLOBAL_PATH="${SSS_L4_GLOBAL:-/data1/AgentG/hhb/CameraMaster/outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4/l4_global}"
OUTPUT_DIR="${SSS_OUTPUT_DIR:-${SCRIPT_DIR}/sss_results/0001_American_Beauty_track_B}"
API_BASE="${SSS_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://api.openai.com/v1}}}"
MODEL="${SSS_MODEL:-${OPENAI_MODEL:-gpt-4o}}"
MAX_FRAMES="${SSS_MAX_FRAMES:-12}"
TIMEOUT_SEC="${SSS_TIMEOUT_SEC:-300}"
L4_TEXT="${SSS_L4_TEXT:-short_synopsis}"
MODE="${SSS_MODE:-v1}"
TRACK="${SSS_TRACK:-B}"
TRACK_ADAPTER_INPUTS="${SSS_TRACK_ADAPTER_INPUTS:-}"
FRAMES_PER_SHOT="${SSS_FRAMES_PER_SHOT:-1}"
SHOT_THRESHOLD="${SSS_SHOT_THRESHOLD:-0.3}"
MAX_L4_WORDS="${SSS_MAX_L4_WORDS:-180}"
EMBEDDING_MODEL="${SSS_EMBEDDING_MODEL:-text-embedding-3-small}"
UNIFORM_FALLBACK_SEC="${SSS_UNIFORM_FALLBACK_SEC:-}"
SKIP_DIAGNOSTICS="${SSS_SKIP_DIAGNOSTICS:-0}"

if [[ "${SSS_OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
else
  OVERWRITE_FLAG=()
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Missing video: ${VIDEO_PATH}" >&2
  exit 2
fi
if [[ ! -d "${L4_GLOBAL_PATH}" && ! -f "${L4_GLOBAL_PATH}" ]]; then
  echo "Missing L4 global source: ${L4_GLOBAL_PATH}" >&2
  exit 2
fi

echo "Running SSS API evaluation"
echo "  video: ${VIDEO_PATH}"
echo "  l4 source: ${L4_GLOBAL_PATH}"
echo "  l4 field: ${L4_TEXT}"
echo "  endpoint: ${API_BASE}"
echo "  model: ${MODEL}"
echo "  mode: ${MODE}"
echo "  track: ${TRACK}"
echo "  embedding model: ${EMBEDDING_MODEL}"
echo "  output: ${OUTPUT_DIR}"

ARGS=(
  --video "${VIDEO_PATH}"
  --l4-global "${L4_GLOBAL_PATH}"
  --l4-text "${L4_TEXT}"
  --summary-provenance l4_global
  --output-dir "${OUTPUT_DIR}"
  --mode "${MODE}"
  --track "${TRACK}"
  --max-frames "${MAX_FRAMES}"
  --frames-per-shot "${FRAMES_PER_SHOT}"
  --shot-threshold "${SHOT_THRESHOLD}"
  --max-l4-words "${MAX_L4_WORDS}"
  --embedding-model "${EMBEDDING_MODEL}"
  --run-llm
  --api-base "${API_BASE}"
  --model "${MODEL}"
  --timeout-sec "${TIMEOUT_SEC}"
  "${OVERWRITE_FLAG[@]}"
)
if [[ -n "${TRACK_ADAPTER_INPUTS}" ]]; then
  ARGS+=(--track-adapter-inputs "${TRACK_ADAPTER_INPUTS}")
fi
if [[ -n "${UNIFORM_FALLBACK_SEC}" ]]; then
  ARGS+=(--uniform-fallback-sec "${UNIFORM_FALLBACK_SEC}")
fi
if [[ "${SKIP_DIAGNOSTICS}" == "1" ]]; then
  ARGS+=(--skip-diagnostics)
fi

exec python "${SCRIPT_DIR}/sss_smoke_test.py" \
  "${ARGS[@]}"
