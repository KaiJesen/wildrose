#!/usr/bin/env bash
# 自动切分 MHA 实验 + 绘图
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="${OUTPUT_DIR:-reports/auto_segment}"
CKPT="${CHECKPOINT_DIR:-checkpoints/auto_seg}"
E1="${EPOCHS1:-15}"
E2="${EPOCHS2:-12}"
E3="${EPOCHS3:-15}"
ARGS=( "examples/plot_auto_segment_report.py" "--output-dir" "$OUT" "--checkpoint-dir" "$CKPT"
       "--epochs1" "$E1" "--epochs2" "$E2" "--epochs3" "$E3" )
[[ "${SYNTHETIC:-1}" -eq 1 ]] && ARGS+=( "--synthetic" )
[[ "${SKIP_TRAIN:-0}" -eq 1 ]] && ARGS+=( "--skip-train" )
echo "[run] ${PYTHON_BIN} ${ARGS[*]}"
"${PYTHON_BIN}" "${ARGS[@]}"
