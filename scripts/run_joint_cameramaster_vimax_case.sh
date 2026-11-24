#!/usr/bin/env bash
# Run one ViMax benchmark case through CameraMaster and ViMax with
# independently resumable stages.  Run this script on 172.17.34.44.
set -Eeuo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMERA_MASTER_ROOT="${CAMERA_MASTER_ROOT:-${SCRIPT_ROOT}}"
VIMAX_ROOT="${VIMAX_ROOT:-${CAMERA_MASTER_ROOT}/ViMax}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${VIMAX_ROOT}/vimax_benchmark}"
MIFAGE_PY="${MIFAGE_PY:-/data1/AgentG/anaconda3/envs/mifage/bin/python}"
VIMAX_PY="${VIMAX_PY:-/data1/AgentG/anaconda3/envs/Vimax/bin/python}"

CASE=""
CAMERA_TRACK="B"
VIMAX_TRACK="both"
CAMERA_OUTPUT_ROOT="${CAMERA_OUTPUT_ROOT:-${CAMERA_MASTER_ROOT}/outputs/vimax_track_ab_4030_storycontrol_v4}"
VIMAX_OUTPUT_DIR="${VIMAX_OUTPUT_DIR:-${VIMAX_ROOT}/outputs/vimax_benchmark/all_cases}"
BACKEND="all_banana"
MAX_STEPS=40
MAX_NODES=30
MAX_REFERENCE_IMAGES=4
RETRIES=3
LOG_LEVEL="INFO"
FORCE=false

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_joint_cameramaster_vimax_case.sh --case CASE [options]

Required:
  --case CASE                  Benchmark JSON filename, stem, or theme.

Options:
  --camera-track {B|both}      CameraMaster tracks (default: B).
  --vimax-track {A|B|both}     ViMax tracks (default: both).
  --camera-output-root PATH    CameraMaster output root.
  --vimax-output-dir PATH      ViMax output root.
  --backend NAME               CameraMaster backend (default: all_banana).
  --max-steps N                CameraMaster narrative candidate horizon (default: 40).
  --max-nodes N                CameraMaster selected node cap (default: 30).
  --max-reference-images N     CameraMaster video reference cap (default: 4).
  --retries N                  Online retry count (default: 3).
  --log-level LEVEL            Logging level (default: INFO).
  --force                      Run all selected stages again; does not delete outputs.
  -h, --help                   Show this help.

Environment overrides:
  VIMAX_ROOT                  ViMax source root (default: CameraMaster/ViMax).
  VIMAX_OUTPUT_DIR            ViMax output root; set both variables to resume
                              a run made in the former standalone repository.

Recovery policy:
  - A .joint_run_state/<case>/STAGE.done marker is written only after a stage
    exits successfully.
  - Re-running the same command skips marked stages and continues at the
    first unfinished one.
  - CameraMaster keeps its own narrative checkpoint. ViMax A and B are
    separate stages because ViMax --resume otherwise does not skip an already
    successful track.
  - StoryControl v4 cannot resume a v3 narrative checkpoint. Use a new
    --camera-output-root for v4 when changing an existing run.
EOF
}

die() { echo "joint-run: ERROR: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case) CASE="${2:-}"; shift 2 ;;
        --camera-track) CAMERA_TRACK="${2:-}"; shift 2 ;;
        --vimax-track) VIMAX_TRACK="${2:-}"; shift 2 ;;
        --camera-output-root) CAMERA_OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --vimax-output-dir) VIMAX_OUTPUT_DIR="${2:-}"; shift 2 ;;
        --backend) BACKEND="${2:-}"; shift 2 ;;
        --max-steps) MAX_STEPS="${2:-}"; shift 2 ;;
        --max-nodes) MAX_NODES="${2:-}"; shift 2 ;;
        --max-reference-images) MAX_REFERENCE_IMAGES="${2:-}"; shift 2 ;;
        --retries) RETRIES="${2:-}"; shift 2 ;;
        --log-level) LOG_LEVEL="${2:-}"; shift 2 ;;
        --force) FORCE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$CASE" ]] || die "--case is required"
[[ "$CAMERA_TRACK" == "B" || "$CAMERA_TRACK" == "both" ]] || die "--camera-track must be B or both"
[[ "$VIMAX_TRACK" == "A" || "$VIMAX_TRACK" == "B" || "$VIMAX_TRACK" == "both" ]] || die "--vimax-track must be A, B, or both"
[[ -x "$MIFAGE_PY" ]] || die "mifage Python not executable: $MIFAGE_PY"
[[ -x "$VIMAX_PY" ]] || die "Vimax Python not executable: $VIMAX_PY"

# Keep state independent from either method's own output layout.  The case
# token is only a directory key; both underlying pipelines still resolve CASE
# through the shared benchmark index.
case_key="$(basename "$CASE")"
case_key="${case_key%.json}"
case_key="$(printf '%s' "$case_key" | tr -cs '[:alnum:]._- ' '_')"
run_key="$(printf '%s\n' "$CASE" "$BENCHMARK_ROOT" "$CAMERA_TRACK" "$VIMAX_TRACK" "$BACKEND" "$MAX_STEPS" "$MAX_NODES" "$MAX_REFERENCE_IMAGES" "$VIMAX_OUTPUT_DIR" | sha256sum | cut -c1-12)"
STATE_DIR="${CAMERA_OUTPUT_ROOT}/.joint_run_state/v4/${case_key}/${run_key}"
LOG_DIR="${STATE_DIR}/logs"
mkdir -p "$LOG_DIR"

# Fail before generation if this output root contains an obsolete checkpoint.
camera_case_dir="${CAMERA_OUTPUT_ROOT}/${case_key}"
camera_checkpoint="${camera_case_dir}/track_B/narrative_simulation/${BACKEND}/narrative_checkpoint.json"
if [[ -f "$camera_checkpoint" ]]; then
    "$MIFAGE_PY" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding="utf-8")); v=(d.get("story_control") or {}).get("version"); sys.exit(0 if v==4 else 1)' "$camera_checkpoint" || \
        die "existing Track B checkpoint is not StoryControl v4: ${camera_checkpoint}; choose a fresh --camera-output-root"
fi

write_done() {
    local stage="$1"
    local marker="${STATE_DIR}/${stage}.done"
    local temp
    temp="$(mktemp "${STATE_DIR}/.${stage}.XXXXXX")"
    {
        printf 'stage=%s\n' "$stage"
        printf 'completed_at=%s\n' "$(date -Is)"
        printf 'case=%s\n' "$CASE"
    } > "$temp"
    mv -f "$temp" "$marker"
}

run_stage() {
    local stage="$1"
    shift
    local marker="${STATE_DIR}/${stage}.done"
    local log_file="${LOG_DIR}/${stage}.log"
    if [[ "$FORCE" != true && -f "$marker" ]]; then
        echo "joint-run: SKIP ${stage} (already successful: ${marker})"
        return 0
    fi
    echo "joint-run: START ${stage}; log=${log_file}"
    printf '\n===== %s %s =====\n' "$stage" "$(date -Is)" | tee -a "$log_file"
    "$@" 2>&1 | tee -a "$log_file"
    write_done "$stage"
    echo "joint-run: DONE ${stage}"
}

camera_command=(
    "$MIFAGE_PY" -u "${CAMERA_MASTER_ROOT}/scripts/run_vimax_track_ab_pipeline.py"
    --benchmark-root "$BENCHMARK_ROOT"
    --case "$CASE"
    --track "$CAMERA_TRACK"
    --output-root "$CAMERA_OUTPUT_ROOT"
    --backend "$BACKEND"
    --max-steps "$MAX_STEPS"
    --max-nodes "$MAX_NODES"
    --max-reference-images "$MAX_REFERENCE_IMAGES"
    --strict-online
    --track-b-story-control
    --track-b-require-full-coverage
    --retries "$RETRIES"
    --log-level "$LOG_LEVEL"
)
run_stage "cameramaster_${CAMERA_TRACK}" "${camera_command[@]}"

# Use one ViMax invocation per track.  Its native --resume only permits an
# existing directory; our stage markers make the composition actually
# resumable without regenerating a completed A track after B is interrupted.
if [[ "$VIMAX_TRACK" == "A" || "$VIMAX_TRACK" == "both" ]]; then
    run_stage "vimax_A" \
        "$VIMAX_PY" -u "${VIMAX_ROOT}/scripts/run_vimax_benchmark_pipeline.py" \
        --benchmark-root "$BENCHMARK_ROOT" --case "$CASE" --track A \
        --output-dir "$VIMAX_OUTPUT_DIR" --run --resume --log-level "$LOG_LEVEL"
fi
if [[ "$VIMAX_TRACK" == "B" || "$VIMAX_TRACK" == "both" ]]; then
    run_stage "vimax_B" \
        "$VIMAX_PY" -u "${VIMAX_ROOT}/scripts/run_vimax_benchmark_pipeline.py" \
        --benchmark-root "$BENCHMARK_ROOT" --case "$CASE" --track B \
        --output-dir "$VIMAX_OUTPUT_DIR" --run --resume --log-level "$LOG_LEVEL"
fi

echo "joint-run: all selected stages completed"
echo "joint-run: state=${STATE_DIR}"
