#!/usr/bin/env bash
# ============================================================================
# 一键运行 BERT 风格 K 线 Embedding 演示。
#
# 用法：
#   bash examples/run_bert_embedding_demo.sh
#
# 修改下方"参数区"即可换标的 / 周期 / 模型大小，不接收任何命令行参数。
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 参数区（直接改这里）
# ---------------------------------------------------------------------------

# 数据
SOURCE="akshare_em"     # 数据源 id：akshare_em | binance_futures | binance_vision
SYMBOL="600519"         # 标的代码（A 股代码 / BTCUSDT 等）
INTERVAL="60m"          # K 线周期：1m / 5m / 15m / 30m / 60m / 1d
DAYS=90                 # 向前取多少自然日

# 模型 / 滑窗
WINDOW=64               # 序列窗口 T
D_MODEL=128             # embedding 隐层维度
VALUE_PROJ="linear"     # value 投影：linear | mlp
POSITION_TYPE="learned" # 位置编码：learned | sincos
ZSCORE_WINDOW=60        # 因果 z-score 窗口

# 运行环境
PYTHON_BIN="python3"    # 默认走 pyenv shim / PATH 上的 python3
INSTALL_DEPS=0          # 0 = 不装依赖；1 = 跑 pip install（首次运行改成 1）

# ---------------------------------------------------------------------------
# 切到项目根目录（脚本所在目录的上一级）
# ---------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "[info] cwd = $(pwd)"
echo "[info] python = $(${PYTHON_BIN} -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ---------------------------------------------------------------------------
# 可选：安装依赖
# ---------------------------------------------------------------------------

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    echo "[info] installing torch (CPU) + project[all]"
    "${PYTHON_BIN}" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"
    "${PYTHON_BIN}" -m pip install -e ".[all]"
fi

# ---------------------------------------------------------------------------
# 运行 demo
# ---------------------------------------------------------------------------

"${PYTHON_BIN}" examples/bert_embedding_demo.py \
    --source        "${SOURCE}" \
    --symbol        "${SYMBOL}" \
    --interval      "${INTERVAL}" \
    --days          "${DAYS}" \
    --window        "${WINDOW}" \
    --d-model       "${D_MODEL}" \
    --value-proj    "${VALUE_PROJ}" \
    --position-type "${POSITION_TYPE}" \
    --zscore-window "${ZSCORE_WINDOW}"
