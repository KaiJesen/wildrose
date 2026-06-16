#!/usr/bin/env bash
# 真实 K 线数据实验 + 可视化报告（默认 Binance Vision BTCUSDT 1h）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE="${SOURCE:-binance_vision}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"
DAYS="${DAYS:-365}"
OUT="${OUTPUT_DIR:-reports/real_btc_1h}"
CKPT="${CHECKPOINT_DIR:-checkpoints/real_btc_1h}"
E1="${EPOCHS1:-12}"
E2="${EPOCHS2:-10}"
E3="${EPOCHS3:-12}"
ARGS=( "examples/plot_auto_segment_report.py"
       "--source" "$SOURCE" "--symbol" "$SYMBOL" "--interval" "$INTERVAL" "--days" "$DAYS"
       "--output-dir" "$OUT" "--checkpoint-dir" "$CKPT"
       "--epochs1" "$E1" "--epochs2" "$E2" "--epochs3" "$E3"
       "--samples-per-epoch" "${SAMPLES_PER_EPOCH:-1200}" )
[[ "${SKIP_TRAIN:-0}" -eq 1 ]] && ARGS+=( "--skip-train" )
echo "[run] ${PYTHON_BIN} ${ARGS[*]}"
"${PYTHON_BIN}" "${ARGS[@]}"
