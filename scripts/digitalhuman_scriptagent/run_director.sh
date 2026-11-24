#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_file "$DIRECTOR_ENTRY"

APP_ID="${SCRIPTAGENT_APP_ID:-${DH_APP_ID:-}}"
APP_KEY="${SCRIPTAGENT_APP_KEY:-${DH_APP_KEY:-}}"

if contains_arg "--help" "$@" || contains_arg "-h" "$@"; then
  exec "$SCRIPTAGENT_PYTHON" "$DIRECTOR_ENTRY" "$@"
fi

if [[ -z "$APP_ID" || -z "$APP_KEY" ]]; then
  echo "[ERR] missing SCRIPTAGENT_APP_ID/SCRIPTAGENT_APP_KEY (or DH_APP_ID/DH_APP_KEY) in env" >&2
  exit 1
fi

CMD=("$SCRIPTAGENT_PYTHON" "$DIRECTOR_ENTRY")

if ! contains_arg "--app_id" "$@"; then
  CMD+=("--app_id" "$APP_ID")
fi
if ! contains_arg "--app_key" "$@"; then
  CMD+=("--app_key" "$APP_KEY")
fi

CMD+=("$@")

echo "[INFO] running director_agent.py"
print_effective_env
printf '[INFO] cmd: %q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
