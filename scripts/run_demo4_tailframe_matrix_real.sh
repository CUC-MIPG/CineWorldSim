#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="outputs/four_genre_demos_15nodes_20260816/04_urban_midnight_rooftop_diner/pipeline/all_banana"
COMMON_ENV=(
  PYTHONUNBUFFERED=1
  MINIMAX_H3_NUM_FRAMES=243
  MINIMAX_H3_VLLM_ENABLED=true
)

run_case() {
  local name="$1"
  shift
  echo "===== ${name} ====="
  "$@" 2>&1 | tee "/tmp/${name}.log"
}

run_case demo4_skill_no_tail env \
  CAMERAMASTER_SKILL_ROOTS=/data1/AgentG/hhb/CameraMaster/docs/skills \
  CAMERAMASTER_SKILL_POLICY_PATH=/data1/AgentG/hhb/CameraMaster/configs/agent_skill_policies.yaml \
  VIDEO_SKILL_NAME=h3-prompt-writing \
  MINIMAX_H3_PROMPT_STYLE=official_ref \
  "${COMMON_ENV[@]}" \
  conda run -n mifage python scripts/run_orchestrated_event_video_generation.py \
    --pipeline-run-dir "$PIPELINE_DIR" \
    --output-dir /tmp/demo4_skill_no_tail_real \
    --multi-shot-events \
    --event-shot-count 4 \
    --event-duration-sec 10 \
    --no-dry-run \
    --no-use-previous-tail-frame \
    --no-use-previous-audio \
    --cuda-visible-devices 2,3 \
    --log-level INFO

run_case demo4_no_skill_no_tail env \
  MINIMAX_H3_PROMPT_STYLE=compact \
  "${COMMON_ENV[@]}" \
  conda run -n mifage python scripts/run_orchestrated_event_video_generation.py \
    --pipeline-run-dir "$PIPELINE_DIR" \
    --output-dir /tmp/demo4_no_skill_no_tail_real \
    --multi-shot-events \
    --event-shot-count 4 \
    --event-duration-sec 10 \
    --no-dry-run \
    --no-use-previous-tail-frame \
    --no-use-previous-audio \
    --cuda-visible-devices 2,3 \
    --log-level INFO

run_case demo4_skill_auto_tail env \
  CAMERAMASTER_SKILL_ROOTS=/data1/AgentG/hhb/CameraMaster/docs/skills \
  CAMERAMASTER_SKILL_POLICY_PATH=/data1/AgentG/hhb/CameraMaster/configs/agent_skill_policies.yaml \
  VIDEO_SKILL_NAME=h3-prompt-writing \
  MINIMAX_H3_PROMPT_STYLE=official_ref \
  "${COMMON_ENV[@]}" \
  conda run -n mifage python scripts/run_orchestrated_event_video_generation.py \
    --pipeline-run-dir "$PIPELINE_DIR" \
    --output-dir /tmp/demo4_skill_auto_tail_real \
    --multi-shot-events \
    --event-shot-count 4 \
    --event-duration-sec 10 \
    --no-dry-run \
    --use-previous-tail-frame \
    --no-use-previous-audio \
    --cuda-visible-devices 2,3 \
    --log-level INFO
