#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERAMASTER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${SCRIPTAGENT_ENV_FILE:-}" ]]; then
  SCRIPTAGENT_ENV_FILE="$CAMERAMASTER_ROOT/.env"
fi

if [[ -f "$SCRIPTAGENT_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SCRIPTAGENT_ENV_FILE"
  set +a
fi

DIGITALHUMAN_ROOT="${DIGITALHUMAN_ROOT:-/data1/AgentG/hhb/digitalhuman}"
SCRIPTAGENT_ROOT="${SCRIPTAGENT_ROOT:-$DIGITALHUMAN_ROOT/ScriptAgent}"
SCRIPTAGENT_CODE_DIR="${SCRIPTAGENT_CODE_DIR:-$SCRIPTAGENT_ROOT/code}"
if [[ -z "${SCRIPTAGENT_PYTHON:-}" ]]; then
  if [[ -x "/data1/AgentG/anaconda3/envs/mifage/bin/python" ]]; then
    SCRIPTAGENT_PYTHON="/data1/AgentG/anaconda3/envs/mifage/bin/python"
  else
    SCRIPTAGENT_PYTHON="python"
  fi
fi
SCRIPTAGENT_MODELSCOPE_CACHE="${SCRIPTAGENT_MODELSCOPE_CACHE:-/tmp/modelscope_cache}"
SCRIPTAGENT_HF_HOME="${SCRIPTAGENT_HF_HOME:-/tmp/hf_cache}"
SCRIPTAGENT_HF_HUB_CACHE="${SCRIPTAGENT_HF_HUB_CACHE:-$SCRIPTAGENT_HF_HOME/hub}"
SCRIPTAGENT_TRANSFORMERS_CACHE="${SCRIPTAGENT_TRANSFORMERS_CACHE:-$SCRIPTAGENT_HF_HOME/transformers}"

mkdir -p "$SCRIPTAGENT_MODELSCOPE_CACHE" "$SCRIPTAGENT_HF_HOME" "$SCRIPTAGENT_HF_HUB_CACHE" "$SCRIPTAGENT_TRANSFORMERS_CACHE"
export MODELSCOPE_CACHE="$SCRIPTAGENT_MODELSCOPE_CACHE"
export HF_HOME="$SCRIPTAGENT_HF_HOME"
export HF_HUB_CACHE="$SCRIPTAGENT_HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$SCRIPTAGENT_TRANSFORMERS_CACHE"

DIRECTOR_ENTRY="$SCRIPTAGENT_CODE_DIR/director_agent.py"
CRITIC_SCRIPT_ENTRY="$SCRIPTAGENT_CODE_DIR/critic_agent_script.py"
CRITIC_VIDEO_ENTRY="$SCRIPTAGENT_CODE_DIR/critic_agent_video.py"
SCRIPTAGENT_COMPAT_DIR="$SCRIPT_DIR/pycompat"
SCRIPTAGENT_ENABLE_OMNI_COMPAT="${SCRIPTAGENT_ENABLE_OMNI_COMPAT:-1}"

if [[ "$SCRIPTAGENT_ENABLE_OMNI_COMPAT" == "1" ]] && [[ -d "$SCRIPTAGENT_COMPAT_DIR" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$SCRIPTAGENT_COMPAT_DIR:$PYTHONPATH"
  else
    export PYTHONPATH="$SCRIPTAGENT_COMPAT_DIR"
  fi
fi

contains_arg() {
  local key="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$key" ]] || [[ "$arg" == "$key="* ]]; then
      return 0
    fi
  done
  return 1
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERR] file not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERR] directory not found: $path" >&2
    exit 1
  fi
}

print_effective_env() {
  echo "[INFO] SCRIPTAGENT_ENV_FILE=$SCRIPTAGENT_ENV_FILE"
  echo "[INFO] DIGITALHUMAN_ROOT=$DIGITALHUMAN_ROOT"
  echo "[INFO] SCRIPTAGENT_ROOT=$SCRIPTAGENT_ROOT"
  echo "[INFO] SCRIPTAGENT_CODE_DIR=$SCRIPTAGENT_CODE_DIR"
  echo "[INFO] SCRIPTAGENT_PYTHON=$SCRIPTAGENT_PYTHON"
  echo "[INFO] SCRIPTAGENT_ENABLE_OMNI_COMPAT=$SCRIPTAGENT_ENABLE_OMNI_COMPAT"
  echo "[INFO] SCRIPTAGENT_COMPAT_DIR=$SCRIPTAGENT_COMPAT_DIR"
  echo "[INFO] MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
  echo "[INFO] HF_HOME=$HF_HOME"
  echo "[INFO] HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "[INFO] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
}
