#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_file "$CRITIC_VIDEO_ENTRY"

BACKEND="${SCRIPTAGENT_VIDEO_BACKEND:-gemini}"
USER_ID="${SCRIPTAGENT_CRITIC_USER_ID:-${DH_CRITIC_USER_ID:-}}"
API_KEY="${SCRIPTAGENT_CRITIC_API_KEY:-${DH_CRITIC_API_KEY:-}}"
MODEL_NAME="${SCRIPTAGENT_CRITIC_MODEL:-${DH_CRITIC_MODEL:-gemini-2.5-pro}}"

if contains_arg "--help" "$@" || contains_arg "-h" "$@"; then
  exec "$SCRIPTAGENT_PYTHON" "$CRITIC_VIDEO_ENTRY" "$@"
fi

CMD=("$SCRIPTAGENT_PYTHON" "$CRITIC_VIDEO_ENTRY")

if ! contains_arg "--backend" "$@"; then
  CMD+=("--backend" "$BACKEND")
fi

if [[ "$BACKEND" == "gemini" ]]; then
  if [[ -z "$USER_ID" || -z "$API_KEY" ]]; then
    echo "[ERR] backend=gemini requires critic user_id/api_key in env" >&2
    exit 1
  fi
  if ! contains_arg "--user_id" "$@"; then
    CMD+=("--user_id" "$USER_ID")
  fi
  if ! contains_arg "--api_key" "$@"; then
    CMD+=("--api_key" "$API_KEY")
  fi
  if ! contains_arg "--model" "$@"; then
    CMD+=("--model" "$MODEL_NAME")
  fi
fi

CMD+=("$@")

echo "[INFO] running critic_agent_video.py"
print_effective_env
printf '[INFO] cmd: %q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
