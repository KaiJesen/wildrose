# wildrose 交易监控网站

一期实现：**实时行情**（永续 `binance_vision` 归档 + `fapi` / `data-api.binance.vision` 尾部补齐）+ 运行态监控 + Dashboard + K 线交易标注。

## 快速启动

```bash
cd /path/to/wildrose
chmod +x website/runtime/start.sh
./website/runtime/start.sh
```

浏览器打开：**http://127.0.0.1:8765/**

启用纸交易引擎（需 torch + prod checkpoint）：

```bash
MONITOR_WITH_ENGINE=1 ./website/runtime/start.sh
```

## 目录

| 路径 | 说明 |
|------|------|
| `src/monitor/` | Python 后端（FastAPI、行情轮询、SQLite、WebSocket） |
| `src/frontend/dist/` | 前端静态页（lightweight-charts） |
| `runtime/` | 可运行环境：venv、配置、数据、启动脚本 |
| `runtime/configs/backends.yaml` | 后台版本注册 |
| `runtime/bar_runner.py` | 心跳 + 新 bar 事件推送 |
| `scripts/import_backtest.py` | 归档回测 CSV 导入 |

## 手动步骤

```bash
# 1. 创建环境
python3 -m venv website/runtime/venv
website/runtime/venv/bin/pip install -r website/runtime/requirements.txt

# 2. 导入样例回测（开平仓标注 / 决策 / 权益）
export MONITOR_RUNTIME=website/runtime
export PYTHONPATH=website/src:.
website/runtime/venv/bin/python website/scripts/import_backtest.py --all

# 3. 启动 API
website/runtime/venv/bin/python -m uvicorn monitor.api.app:app --host 0.0.0.0 --port 8765

# 4. 另开终端：bar_runner
website/runtime/venv/bin/python website/runtime/bar_runner.py
```

## API

- `GET /api/backends` — 版本列表
- `GET /api/backends/{id}/dashboard` — 门类聚合
- `GET /api/backends/{id}/ohlcv` — 真实实时 K 线
- `GET /api/backends/{id}/trades` — 成交（标注用）
- `WS /ws/backends/{id}` — 事件推送

## 文档

- [001 立项说明书 v0.3](doc/001_交易监控网站立项说明书.md)
