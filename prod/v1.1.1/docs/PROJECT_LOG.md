# wildrose prod v1.1.1 — 项目日志

## 2026-06-26 — v1.1.1 发布（024 B0 研究基线）

### 背景

- 025 Phase 1 A3a、Phase 2 A3b 均未能通过探索双门（return≥8.84% 且 coverage≥28%）
- 架构师裁定：025 结案；驳回 coverage-only 探索线；跳过 A3c Floor
- prod 交易维持 v1.1.0（0062e）

### 固化内容

| 资产 | 说明 |
|------|------|
| `checkpoint/market_state_best.pt` | c1_pw20 B0 模型 |
| `data/kline/..._end20260625.csv` | 钉死 K 线切片 |
| `calibration/teq_edge_calibration.json` | TEQ valid 校准 |
| `config/trading_rule.json` | B0 规则（TEQ wp=0.35） |
| `code/` | 含 `channel_edge.py`、A3 门控工程 |

### 参考指标（test）

- return **9.01%** · coverage **26.7%** · teq **3**

### 相对 v1.1.0

| 维度 | v1.1.0 | v1.1.1 |
|------|--------|--------|
| 用途 | 生产候选 | 研究基线 |
| 模型 | 0062e | c1_pw20 |
| test return | 9.31% | 9.01% |
| coverage | 16.7% | 26.7% |
