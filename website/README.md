# wildrose 交易监控网站

面向 `trading_system` 后台的实时监控、统计分析与 K 线交易标注展示。

## 文档

- [001 立项说明书](doc/001_交易监控网站立项说明书.md) — 功能、目标、技术路径、里程碑

## 目录（规划）

```text
website/
├── doc/          # 设计文档
├── src/
│   ├── backend/  # FastAPI 服务
│   ├── adapters/ # 多版本后台适配层
│   ├── models/   # 数据模型
│   ├── db/       # 存储
│   └── frontend/ # React 前端
├── configs/      # 后台注册配置
└── scripts/      # 启动与数据导入脚本
```

## 状态

立项阶段（M0）。`src/` 待 M1 起按立项文档实施。
