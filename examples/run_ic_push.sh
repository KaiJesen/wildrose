#!/usr/bin/env bash
# IC 冲刺（推荐两阶段）：
#   1) Stage1/2：波动率伪标签切分 + VQ 多样性
#   2) Stage3：平衡 IC 联合损失（复用 S1/2 checkpoint）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
EMB_CKPT="${EMBED_CKPT:-checkpoints/real_btc_ic_push}"
S3_CKPT="${S3_CKPT:-checkpoints/real_btc_ic_push_balanced}"
S3_OUT="${S3_OUT:-reports/real_btc_ic_push_balanced}"

if [[ "${STAGE3_ONLY:-0}" -ne 1 ]]; then
  echo "=== Stage 1/2: vol-break embedding ==="
  "${PYTHON_BIN}" examples/plot_auto_segment_report.py \
    --source binance_vision --symbol BTCUSDT --interval 1h --days 365 \
    --output-dir "${EMB_CKPT}" --checkpoint-dir "${EMB_CKPT}" \
    --epochs1 "${EPOCHS1:-18}" --epochs2 "${EPOCHS2:-12}" --epochs3 1 \
    --samples-per-epoch "${SAMPLES_PER_EPOCH:-1500}" \
    --break-vol-weight "${BREAK_VOL_WEIGHT:-0.15}" \
    --diversity-weight "${DIVERSITY_WEIGHT:-0.25}" \
    --usage-balance-weight "${USAGE_BALANCE_WEIGHT:-0.35}" \
    --z-spread-weight "${Z_SPREAD_WEIGHT:-0.15}" \
    --vq-dead-threshold "${VQ_DEAD_THRESHOLD:-0.1}" \
    --vq-max-code-frac "${VQ_MAX_CODE_FRAC:-0.18}" \
    --mse-weight 0.7 --step-corr-weight 0.25 --cum-corr-weight 0.35
fi

echo "=== Stage 3: balanced IC loss ==="
"${PYTHON_BIN}" examples/plot_auto_segment_report.py \
  --source binance_vision --symbol BTCUSDT --interval 1h --days 365 \
  --skip-stage12 --init-checkpoint-dir "${EMB_CKPT}" \
  --output-dir "${S3_OUT}" --checkpoint-dir "${S3_CKPT}" \
  --epochs3 "${EPOCHS3:-40}" --encoder-lr-scale 0.0 \
  --mse-weight 0.7 --step-corr-weight 0.25 --cum-corr-weight 0.35 --rank-weight 0.1 \
  --sign-weight 0.15 --anti-lag-weight "${ANTI_LAG_WEIGHT:-0.15}" \
  --anti-lag-margin "${ANTI_LAG_MARGIN:-0.05}" \
  --samples-per-epoch "${SAMPLES_PER_EPOCH:-1500}"
