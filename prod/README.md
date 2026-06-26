# Production releases

本目录存放可部署/可复现的生产固化包。

| 版本 | 策略 | 说明 |
|---|---|---|
| [v0.0.0](v0.0.0/README.md) | v016 tuned2 + 0062e | test OOS：9.66% / -0.58% MDD |
| [v0.1.0](v0.1.0/README.md) | v020 trend-segment tuned + 0065a | test OOS：13.24% / -2.86% MDD |
| [v1.0.0](v1.0.0/README.md) | v021 trend-bias full_bias + 0062e | 8.25% / -0.93% MDD |
| [v1.1.0](v1.1.0/README.md) | **v022 trend-quality + 0062e** | **当前正式候选**：9.31% / -0.85% MDD |
| [v1.1.1](v1.1.1/README.md) | **v024 B0 research baseline + c1_pw20** | 024/025 研究基线：9.01% / 26.7% cov（非生产） |

每个子目录为独立快照，包含代码副本、配置、checkpoint 与参考回测指标。
