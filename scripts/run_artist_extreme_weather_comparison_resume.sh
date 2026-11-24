#!/usr/bin/env bash
# Resume the complete CameraMaster-versus-ViMax evaluation for the Artist case.
# Existing valid visual/planning/SSS JSON results are skipped by default.

set -euo pipefail

CAMERAMASTER_ROOT="/data1/AgentG/hhb/CameraMaster"
CONDA_BIN="/data1/AgentG/anaconda3/bin/conda"
BENCHMARK_CASE="/data1/AgentG/hhb/ViMax/vimax_benchmark/artist_extreme_weather_typeA.json"
CAMERAMASTER_OUTPUT="${CAMERAMASTER_ROOT}/outputs/vimax_track_ab_4030_storycontrol_v4/artist_extreme_weather_typeA"
VIMAX_OUTPUT="/data1/AgentG/hhb/ViMax/outputs/vimax_benchmark/all_cases/artist_extreme_weather"
COMPARISON_OUTPUT="${CAMERAMASTER_ROOT}/outputs/comparisons/artist_extreme_weather_typeA_4030_storycontrol_v4_vs_vimax"

cd "${CAMERAMASTER_ROOT}"

exec "${CONDA_BIN}" run --no-capture-output -n mifage \
  python -u evaluation/run_cameramaster_vimax_comparison.py \
  --benchmark-case "${BENCHMARK_CASE}" \
  --cameramaster-output-root "${CAMERAMASTER_OUTPUT}" \
  --vimax-output-root "${VIMAX_OUTPUT}" \
  --output-dir "${COMPARISON_OUTPUT}" \
  --tracks B \
  --run-all \
  --resume
