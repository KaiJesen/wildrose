#!/usr/bin/env bash
# 方向准确率冲刺：raw-context 旁路 + 方向辅助头 + 历史阈值校准。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
EMBED_CKPT="${EMBED_CKPT:-checkpoints/real_btc_ic_push}"
OUT="${OUTPUT_DIR:-reports/real_btc_raw_context_direction_calibrated}"
CKPT="${CHECKPOINT_DIR:-checkpoints/real_btc_raw_context_direction_calibrated}"

"${PYTHON_BIN}" examples/plot_auto_segment_report.py \
  --source "${SOURCE:-binance_vision}" \
  --symbol "${SYMBOL:-BTCUSDT}" \
  --interval "${INTERVAL:-1h}" \
  --days "${DAYS:-365}" \
  --skip-stage12 \
  --init-checkpoint-dir "${EMBED_CKPT}" \
  --output-dir "${OUT}" \
  --checkpoint-dir "${CKPT}" \
  --epochs3 "${EPOCHS3:-50}" \
  --encoder-lr-scale 0.0 \
  --mse-weight "${MSE_WEIGHT:-0.65}" \
  --step-corr-weight "${STEP_CORR_WEIGHT:-0.2}" \
  --cum-corr-weight "${CUM_CORR_WEIGHT:-0.3}" \
  --rank-weight "${RANK_WEIGHT:-0.1}" \
  --direction-weight "${DIRECTION_WEIGHT:-0.4}" \
  --vol-focus-weight "${VOL_FOCUS_WEIGHT:-2.0}" \
  --vol-focus-top-frac "${VOL_FOCUS_TOP_FRAC:-0.2}" \
  --sign-weight "${SIGN_WEIGHT:-0.15}" \
  --samples-per-epoch "${SAMPLES_PER_EPOCH:-1500}" \
  --device "${DEVICE:-cpu}"
