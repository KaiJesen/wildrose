#!/usr/bin/env bash
# 训练 + 验证报告（改进版）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARGS=( "examples/plot_pattern_model_report.py"
       "--output-dir" "${OUTPUT_DIR:-reports/pattern_model_v2}"
       "--checkpoint-dir" "${CHECKPOINT_DIR:-checkpoints/pattern_v2}"
       "--epochs1" "${EPOCHS1:-30}" "--epochs2" "${EPOCHS2:-40}" "--epochs3" "${EPOCHS3:-40}" )
[[ "${SYNTHETIC:-1}" -eq 1 ]] && ARGS+=( "--synthetic" )
[[ "${SKIP_TRAIN:-0}" -eq 1 ]] && ARGS+=( "--skip-train" )
"${PYTHON_BIN}" "${ARGS[@]}"
