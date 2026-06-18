#!/usr/bin/env bash
# 三阶段形态编码训练（默认真实 K 线：Binance Vision BTCUSDT 1h）
set -euo pipefail
STAGE="${STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pattern}"
SOURCE="${SOURCE:-binance_vision}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"
DAYS="${DAYS:-365}"
EPOCHS1="${EPOCHS1:-30}"
EPOCHS2="${EPOCHS2:-40}"
EPOCHS3="${EPOCHS3:-40}"
D_MODEL="${D_MODEL:-256}"
NUM_CODES="${NUM_CODES:-16}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
COMMON=(
  "--source" "${SOURCE}" "--symbol" "${SYMBOL}" "--interval" "${INTERVAL}" "--days" "${DAYS}"
  "--checkpoint-dir" "${CHECKPOINT_DIR}" "--d-model" "${D_MODEL}" "--num-codes" "${NUM_CODES}"
)
[[ "${SYNTHETIC:-0}" -eq 1 ]] && COMMON+=( "--synthetic" )

run_stage1() {
  "${PYTHON_BIN}" examples/train_stage1_segment_encoder.py "${COMMON[@]}" --epochs "${EPOCHS1}"
}
run_stage2() {
  "${PYTHON_BIN}" examples/train_stage2_vqvae.py "${COMMON[@]}" --epochs "${EPOCHS2}" \
    --init-checkpoint "${CHECKPOINT_DIR}/stage1_segment_encoder.pt"
}
run_stage3() {
  "${PYTHON_BIN}" examples/train_stage3_predictor.py "${COMMON[@]}" --epochs "${EPOCHS3}" \
    --init-checkpoint "${CHECKPOINT_DIR}/stage2_vqvae.pt"
}

case "${STAGE}" in
  all) run_stage1; run_stage2; run_stage3 ;;
  1) run_stage1 ;;
  2) run_stage2 ;;
  3) run_stage3 ;;
  *) echo "unknown STAGE=${STAGE}" >&2; exit 1 ;;
esac
