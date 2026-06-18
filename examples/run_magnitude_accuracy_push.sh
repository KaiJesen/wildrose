#!/usr/bin/env bash
# 幅度准确性冲刺：原始尺度 MSE + 相对幅度损失 + 相对误差校准
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="${OUTPUT_DIR:-reports/magnitude_accuracy_push}"
CKPT="${CHECKPOINT_DIR:-checkpoints/magnitude_accuracy_push}"

"${PYTHON_BIN}" examples/plot_auto_segment_report.py \
  --source binance_vision --symbol BTCUSDT --interval 1h --days 365 \
  --trend-features \
  --output-dir "${OUT}" --checkpoint-dir "${CKPT}" \
  --epochs1 "${EPOCHS1:-12}" --epochs2 "${EPOCHS2:-8}" --epochs3 "${EPOCHS3:-35}" \
  --pred-horizon "${PRED_HORIZON:-1}" \
  --samples-per-epoch "${SAMPLES_PER_EPOCH:-1500}" \
  --encoder-lr-scale 0.0 \
  --mse-weight 0.15 --raw-mse-weight 1.5 \
  --step-corr-weight 0.03 --cum-corr-weight 0.03 --sign-weight 0.05 \
  --cum-magnitude-weight 1.0 --relative-magnitude-weight 0.9 \
  --magnitude-tolerance 0.2 --magnitude-min-move 0.0005
