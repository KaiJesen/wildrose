---
name: transformer-invest-embedding
description: Build structured Transformer study notes for investment analysis from public K-line and technical indicator data, with emphasis on embedding layer design choices. Include synchronized Python implementation and backtesting practice. Use when the user mentions Transformer learning notes, quant research notes, K-line/indicator features, time-series tokenization, embedding architecture decisions, notebook topic notes, or Chinese markdown study records.
---

# Transformer Embedding Notes for Investment Analysis

## Goal

Produce reusable study notes for Transformer-based investment analysis with a strict focus on embedding-layer decisions for public K-line and indicator inputs.

## Scope and Constraints

- Data source scope: public OHLCV/K-line data and derived technical indicators.
- Main target: embedding design, not full strategy backtest.
- Priority: preserve temporal structure, avoid feature leakage, and keep notes implementation-ready.
- Learning process includes synchronized Python implementation and backtesting.
- Study notes must be stored under `notebook/` as topic-based markdown files, not in the skill directory.
- Notes should be written in Chinese.

## Note File Policy

- Create or update topic files under `notebook/` using clear topic names.
- One topic, one markdown file (for example: `notebook/embedding-design.md`).
- Write all explanatory content in Chinese.
- During study, classify new questions and log them in `notebook/learning-questions-log.md`.
- For each topic note, include three linked parts:
  - Concept understanding
  - Python implementation plan or snippets
  - Backtest setup and verification checklist

## Output Format

Use this Chinese structure for every topic note entry:

```markdown
# [专题名称]

## 问题定义
- 当前要解决的Embedding问题是什么？

## 方案候选
- 方案A：
- 方案B：
- 方案C：

## 选型结论
- 选择：
- 选择原因：
- 代价与权衡：

## Python实现
- 输入与特征工程：
- Embedding实现要点：
- 关键代码片段：

## 回测设计
- 数据切分：
- 回测区间与频率：
- 指标与基线：
- 消融实验：

## 泄漏与稳健性检查
- 潜在泄漏路径：
- 规避措施：
- 稳健性验证：
```

## Embedding Design Workflow

Follow this checklist and keep it in the notes:

```text
Embedding Study Progress
- [ ] Step 1: Define prediction horizon and label
- [ ] Step 2: Define tokenization unit (bar/patch/window)
- [ ] Step 3: Define feature groups and scaling
- [ ] Step 4: Design value/time/asset embeddings
- [ ] Step 5: Specify masks to block leakage
- [ ] Step 6: Plan ablations and sanity checks
```

### Step 1: Define objective first

- Specify task type: direction classification, return regression, or volatility regime.
- Specify horizon (for example, next 1/5/20 bars).
- Explicitly document which timestamp is the latest input and which timestamp is the label.

### Step 2: Choose tokenization unit

Default choice: one token per bar.

Alternative choices:
- Patch tokenization (merge several consecutive bars).
- Multi-resolution tokens (for example, 5m and 1h in parallel).

Record trade-offs:
- Bar tokens preserve detail but increase sequence length.
- Patch tokens reduce length but may smooth out reversal signals.

### Step 3: Build feature groups

Recommended grouping:
- Raw price-volume: open, high, low, close, volume, turnover.
- Returns and ranges: log-return, high-low range, gap features.
- Technical indicators: MA/EMA, RSI, MACD, ATR, Bollinger features.
- Calendar/time: minute-of-day, day-of-week, session segment.

Normalize per group and document method:
- Rolling z-score for non-stationary scales.
- Log transform for heavy-tailed positive values.
- Keep all transforms causal (no future window usage).

### Step 4: Design embedding components

Use additive composition:

`x_t = E_value(f_t) + E_time(t) + E_asset(a) + E_pos(t)`

Where:
- `E_value(f_t)`: projection/MLP embedding of numeric feature vector at time t.
- `E_time(t)`: learned embedding for calendar/session buckets.
- `E_asset(a)`: asset ID embedding for multi-asset training.
- `E_pos(t)`: positional embedding (absolute or relative).

Practical defaults:
- Start with linear projection for `E_value`, then test 2-layer MLP.
- Prefer relative position bias when sequence length varies.
- If single-asset only, drop `E_asset` and document why.

### Step 5: Prevent leakage in embedding stage

- Any rolling indicator must use past-only windows.
- For patched tokens, verify patch boundary does not include target horizon.
- Ensure normalization statistics are computed on train split only.
- For cross-asset batches, avoid using future-listed assets in past periods if survivorship is a concern.

### Step 6: Plan embedding ablations

Minimum ablation set:
- Remove `E_time` to test pure value features.
- Compare linear vs MLP `E_value`.
- Absolute vs relative positional encoding.
- With and without indicator subset (raw OHLCV only baseline).

## Decision Heuristics

Use these defaults unless evidence shows otherwise:

- Short-horizon intraday: stronger time/session embeddings, shorter context windows.
- Swing/longer horizon: richer positional bias, lower reliance on minute-level calendar features.
- Small dataset: simpler embedding (linear projection + small d_model) to reduce overfit.
- Multi-asset dataset: include asset embedding but regularize heavily.

## Common Failure Modes

- Indicator leakage from centered windows or global normalization.
- Over-parameterized value embedding dominating temporal signal.
- Excessive feature stacking without ablation, causing unstable conclusions.
- Mixing assets/timeframes without explicit embedding tags.

## Quick Prompt Template

When asked to continue notes, use this prompt scaffold:

```text
Continue the Transformer investment note.
Task: [classification/regression], Horizon: [N bars], Universe: [assets].
Current embedding baseline: [formula].
Need: [new ablation or redesign].
Return: chosen design, leakage checks, and minimal experiment plan.
```
