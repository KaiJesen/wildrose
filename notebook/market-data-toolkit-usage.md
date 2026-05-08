# market_data 套件使用手册

> 多数据源 K 线抓取套件：统一 OHLCV schema + 可插拔 Provider + 通用绘图。
>
> 目标：让 A 股、加密合约等异构数据源在同一套接口下使用，给出的 DataFrame 列名、时序、绘图方式完全一致，方便研究阶段快速切换数据源做对比。

## 套件总览

```
market_data/
├── base.py            # KlineProvider 抽象基类（统一 fetch_kline 接口）
├── schema.py          # 标准列名常量 + 中文列规范化函数
├── registry.py        # Provider 工厂：get_kline_provider("xxx", **kwargs)
├── http_utils.py      # 通用网络重试、东方财富友好 header
├── plotting.py        # 蜡烛图 / 量价图（仅依赖 matplotlib）
└── sources/
    ├── akshare_eastmoney.py   # A 股分钟/日线（AkShare → 东方财富）
    ├── binance_futures.py     # Binance USDT-M 永续合约（fapi REST 实时）
    └── binance_vision.py      # Binance Vision 历史归档（公开 ZIP，免代理）
```

依赖关系：

- 核心运行依赖：`pandas>=2.0`、`requests>=2.28`
- A 股 Provider：额外 `akshare>=1.14`
- 绘图：额外 `matplotlib>=3.8`
- 全部依赖：`pip install -e ".[all]"`（在项目根 `02_python/` 下）

## 三层抽象

### 1) `KlineProvider`：统一接口

每个数据源都实现这一接口，研究脚本里只跟接口打交道：

```python
class KlineProvider(ABC):
    id: str
    description: str = ""

    @abstractmethod
    def fetch_kline(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str = "",
    ) -> pd.DataFrame: ...

    @property
    def supported_intervals(self) -> frozenset[str]: ...
```

返回的 DataFrame 一定包含标准列 `time / open / high / low / close / volume`，加密源还会附带 `quote_volume / trades / taker_buy_*` 等扩展列。

### 2) `schema`：标准列名

| 列名 | 含义 | 类型 |
|---|---|---|
| `time` | K 线开始时间（datetime，加密源 UTC） | datetime |
| `open / high / low / close` | OHLC | float |
| `volume` | 成交量（基础币 / 股数） | float |
| `amount` | 成交额（计价币 / 元） | float |
| `quote_volume` | 计价币成交额（加密专属，等价 amount） | float |
| `trades` | 成交笔数（加密专属） | Int64 |
| `taker_buy_base / taker_buy_quote` | 主动买入量 / 额（加密专属） | float |
| `close_time` | 收盘时间（加密专属） | datetime |
| `amplitude / pct_change / change / turnover` | 振幅、涨跌幅、涨跌额、换手率（A 股专属） | float |

工具函数：

- `normalize_ohlcv_df(raw)`：把中文列（"开盘/收盘/..."）规范化成标准列，并按时间排序。所有 Provider 内部都已调用，使用方一般不用直接调。

### 3) `registry`：工厂 + 别名

```python
from market_data import get_kline_provider, list_kline_providers

# 列出已注册的全部数据源
for pid, desc, intervals in list_kline_providers():
    print(pid, sorted(intervals)[:5], desc)

# 实例化（支持别名）
p = get_kline_provider("akshare_em", retries=8)        # A 股
p = get_kline_provider("binance",      retries=5)       # binance_futures
p = get_kline_provider("vision")                        # binance_vision
```

内置别名表：

| 别名 | 实际 id |
|---|---|
| `eastmoney`、`akshare`、`em` | `akshare_em` |
| `binance`、`binance_um`、`binance_usdm`、`binance_perp`、`fapi` | `binance_futures` |
| `vision`、`binance_archive`、`binance_history` | `binance_vision` |

## 数据源 1：`akshare_em` (A 股 / AkShare → 东方财富)

适用：A 股股票分钟/日线行情、可选复权。

| 项 | 说明 |
|---|---|
| `id` | `akshare_em`（别名 `eastmoney`、`em`、`akshare`） |
| 支持周期 | `1m, 5m, 15m, 30m, 60m, 1d` |
| 复权 | `''`（不复权）、`qfq`（前复权）、`hfq`（后复权） |
| `symbol` 含义 | 6 位股票代码，如 `600519`、`000001` |
| 主要构造参数 | `retries=5, retry_base_sleep_s=2.0, verbose_retry=True` |

最小用法：

```python
from datetime import datetime, timedelta
from market_data import get_kline_provider

p = get_kline_provider("eastmoney", retries=8)
end   = datetime.now()
start = end - timedelta(days=30)

df = p.fetch_kline("600519", "60m", start, end, adjust="qfq")
print(df.head())
```

注意点：

- AkShare 抓东方财富会带上友好 header（`User-Agent + Referer`），并对瞬时网络错误指数退避重试。
- 分钟级别不支持复权字段全设——具体支持情况见 AkShare 文档。
- 时间是本地时区（A 股交易时间），不要硬把它当 UTC 用。

## 数据源 2：`binance_futures` (Binance USDT-M 实时合约)

适用：DOGEUSDT / BTCUSDT 等 USDT 永续合约的**实时**行情，需直连 `fapi.binance.com`。

| 项 | 说明 |
|---|---|
| `id` | `binance_futures`（别名 `binance`、`fapi`） |
| 支持周期 | `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M`，外加 A 股别名 `60m → 1h` 等 |
| `symbol` | 大写合约代码：`DOGEUSDT`、`BTCUSDT`、`ETHUSDT_240927`（交割） |
| 复权 | 不支持（加密货币没有），传入会报错 |
| 主要构造参数 | `base_url, retries, request_timeout, proxies, trust_env=True, page_sleep_s` |
| 分页 | 单次最大 1500 根 K 线，长区间自动按 cursor 翻页 |

最小用法：

```python
from datetime import datetime, timedelta, timezone
from market_data import get_kline_provider

p = get_kline_provider("binance", retries=5)  # 默认 trust_env=True，自动认 HTTPS_PROXY
end   = datetime.now(tz=timezone.utc)
start = end - timedelta(days=7)

df = p.fetch_kline("DOGEUSDT", "1h", start, end)
```

走代理的几种方式（任选其一）：

```python
# 方式 A：环境变量（trust_env=True 时自动生效）
#   export HTTPS_PROXY=http://127.0.0.1:7890

# 方式 B：构造时显式传
p = get_kline_provider("binance", proxies={
    "http":  "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
})

# 方式 C：SOCKS5（需先 pip install 'requests[socks]'）
p = get_kline_provider("binance", proxies={
    "http":  "socks5h://127.0.0.1:1080",
    "https": "socks5h://127.0.0.1:1080",
})
```

国内 / WSL2 注意：

- `fapi.binance.com` 在国内通常不可达，必须代理。
- WSL2 默认 NAT 模式下 `127.0.0.1:7890` 指 WSL 自己，**不是 Windows 主机**。要么：
  1. 在 Windows 代理软件里勾选"允许局域网连接"，然后用 Windows 主机 IP（`ip route show default` 看到的 gateway，例如 `172.26.192.1:7890`）；
  2. 或在 `%USERPROFILE%\.wslconfig` 里启用镜像网络模式：
     ```ini
     [wsl2]
     networkingMode=mirrored
     ```
     然后 `wsl --shutdown` 重启，之后 `127.0.0.1:7890` 与 Windows 主机一致。
- 都不通时 → 改用 `binance_vision` 拿历史数据。

## 数据源 3：`binance_vision` (Binance 公开历史归档)

适用：DOGEUSDT 等合约的**历史 K 线**（覆盖 2020 年至今），完全公开 ZIP，**国内多数网络可直连免代理**。

| 项 | 说明 |
|---|---|
| `id` | `binance_vision`（别名 `vision`、`binance_archive`、`binance_history`） |
| 数据范围 | 月度归档（完整月） + 日度归档（不完整月份用），自动拼装 |
| 支持周期 | `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo`（注意月线写作 `1mo` 不是 `1M`） |
| 时延 | daily 一般 1~2 天后才有；最新一两天数据需要 `binance_futures` |
| `contract_type` | `"um"`（USDT-M 默认）、`"cm"`（币本位）、`"spot"`（现货） |
| 限制澄清 | **不存在"非会员只能下 4h"的说法**——所有 interval 对所有人公开下载 |

最小用法：

```python
from datetime import datetime, timedelta, timezone
from market_data import get_kline_provider

p = get_kline_provider("vision")  # 默认 USDT-M 合约
end   = datetime.now(tz=timezone.utc)
start = end - timedelta(days=60)

df = p.fetch_kline("DOGEUSDT", "1h", start, end)
print(len(df), df.columns.tolist())
```

工作原理：

1. 把 `[start, end]` 拆成 `(完整月, 区间两端的不完整月按日)` 两组键；
2. 拼出 URL：
   - monthly: `data/futures/um/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip`
   - daily:   `data/futures/um/daily/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`
3. 下载 → 解压 → CSV 解析（自动识别有无表头）→ 时区/类型规整 → 时间过滤；
4. 缺失日期（404）默认跳过，避免最新两天没归档时整体失败。

可选参数：

```python
get_kline_provider(
    "vision",
    contract_type="um",       # um/cm/spot
    prefer_monthly=True,       # 长区间优先用月度归档（更省请求）
    proxies=None,              # 可选；环境变量也能识别
    retries=5, request_timeout=30.0, verbose=True,
)
```

## 绘图模块 `market_data.plotting`

只要拿到符合标准 schema 的 DataFrame，就能直接画蜡烛图。三个函数自顶向下分层：

```python
from market_data.plotting import plot_candlestick, plot_volume, plot_candlestick_volume
```

| 函数 | 作用 |
|---|---|
| `plot_candlestick(df, *, ax=None, color_style="crypto", title=None, ...)` | 在一个 Axes 上画蜡烛 |
| `plot_volume(df, *, ax=None, color_style="crypto", ...)` | 单独画量柱（按当根涨跌着色） |
| `plot_candlestick_volume(df, *, title=None, save_path=None, color_style="crypto", figsize=(11,6), height_ratios=(3,1), price_label="Price", volume_label="Volume", date_format="%m-%d\n%H:%M", show=True, tight_layout=True)` | 上下双子图（蜡烛 + 成交量），最常用 |

`color_style` 接受三种形式：

- `"crypto"`：绿涨红跌（默认，加密圈惯例）
- `"ashare"`：红涨绿跌（A 股惯例）
- `(up_hex, down_hex)`：自定义二元组，例如 `("#ff8800", "#0088ff")`

最小用法：

```python
from market_data.plotting import plot_candlestick_volume

# 一行画完，存图不弹窗
plot_candlestick_volume(df, title="DOGEUSDT 1h", save_path="doge.png", show=False)
```

嵌入到自己的 figure 里：

```python
import matplotlib.pyplot as plt
from market_data.plotting import plot_candlestick, plot_volume

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8),
                         gridspec_kw={"height_ratios": [3, 1, 1]})

plot_candlestick(df, ax=axes[0], title="DOGEUSDT 1h")
axes[0].plot(df["time"], df["close"].rolling(20).mean(), label="MA20")
axes[0].legend()

plot_volume(df, ax=axes[1])

# 第三幅是用户自己的指标
axes[2].plot(df["time"], df["close"].pct_change())
axes[2].set_ylabel("ret")
fig.savefig("custom_panel.png")
```

绘图注意：

- 函数都是 `*` 后纯关键字参数，不要用 positional——这样后续加参数不会破坏调用方。
- matplotlib 是局部 import：没装也能 `import market_data`，只是触到绘图函数时才报错。
- 输入空 DataFrame / 缺列时会主动 `ValueError` 给出清晰提示，先用 `normalize_ohlcv_df` 规范列名再画。

## 端到端示例

### 示例 1：DOGEUSDT 历史 K 线 + 出图（无需代理）

```python
from datetime import datetime, timedelta, timezone
from market_data import get_kline_provider
from market_data.plotting import plot_candlestick_volume

p = get_kline_provider("vision")
end   = datetime.now(tz=timezone.utc)
start = end - timedelta(days=30)

df = p.fetch_kline("DOGEUSDT", "1h", start, end)
plot_candlestick_volume(
    df,
    title="DOGEUSDT 1h — Binance Vision",
    save_path="doge_1h.png",
    color_style="crypto",
    price_label="Price (USDT)",
    volume_label="Volume (DOGE)",
    show=False,
)
```

对应命令行 demo：`examples/binance_doge_futures_demo.py`（默认走 vision）。

### 示例 2：A 股股票 + 红涨绿跌

```python
from datetime import datetime, timedelta
from market_data import get_kline_provider
from market_data.plotting import plot_candlestick_volume

p = get_kline_provider("eastmoney")
end   = datetime.now()
start = end - timedelta(days=60)

df = p.fetch_kline("600519", "60m", start, end, adjust="qfq")
plot_candlestick_volume(df, title="600519 60m qfq", color_style="ashare",
                       price_label="Price (OHLC)", show=False, save_path="600519_60m.png")
```

对应命令行 demo：`examples/akshare_1h_kline_demo.py`。

### 示例 3：多源切换跑同一份策略

```python
from datetime import datetime, timedelta, timezone
from market_data import get_kline_provider

end   = datetime.now(tz=timezone.utc)
start = end - timedelta(days=10)

for src in ("vision", "akshare_em"):
    p = get_kline_provider(src)
    if src == "akshare_em":
        df = p.fetch_kline("600519", "60m", start.replace(tzinfo=None), end.replace(tzinfo=None))
    else:
        df = p.fetch_kline("DOGEUSDT", "1h", start, end)
    print(src, len(df), df["close"].mean())
```

## 注册自定义 Provider

新增数据源只需三件事：

1. 继承 `KlineProvider`，实现 `id / description / supported_intervals / fetch_kline`；
2. 返回的 DataFrame 用 `normalize_ohlcv_df` 走一遍（保证列序、时间类型、时间排序）；
3. 调 `register_provider(MyProvider)` 把它放进工厂。

骨架：

```python
import pandas as pd
from datetime import datetime
from market_data.base import KlineProvider
from market_data.schema import normalize_ohlcv_df
from market_data.registry import register_provider


class MyExchangeProvider(KlineProvider):
    id = "my_exchange"
    description = "My private feed"

    @property
    def supported_intervals(self):
        return frozenset(("1m", "5m", "1h", "1d"))

    def fetch_kline(self, symbol, interval, start, end, *, adjust=""):
        # 1. 调你的接口拿原始 df
        raw = ...
        # 2. 列名映射到 schema 标准
        # 3. 标准化 + 排序
        return normalize_ohlcv_df(raw)


register_provider(MyExchangeProvider)
```

之后 `get_kline_provider("my_exchange")` 就能拿到。也可以在 `registry._ALIASES` 里加别名，但通常用户层 `register_provider` 后直接用 id 即可。

## 故障排查 / FAQ

### Q1：跑 `binance_futures` 报 `Network is unreachable / Connection timed out`

99% 是没走代理（GFW）。检查清单：

```bash
# 1) 看环境变量
echo $HTTPS_PROXY $HTTP_PROXY

# 2) 看 fapi 是否可达
curl -m 8 https://fapi.binance.com/fapi/v1/ping
curl -m 8 -x http://127.0.0.1:7890 https://fapi.binance.com/fapi/v1/ping

# 3) 看代理端口是否监听（WSL2 → Windows 主机）
HOST_IP=$(ip route show default | awk '/default/ {print $3}')
nc -zv $HOST_IP 7890
```

如果代理软件只监听 `127.0.0.1:7890`，WSL2 这边连不进去——参考前面"国内 / WSL2 注意"部分。

实在搞不定就改用 `binance_vision`：直连可达、覆盖全量历史，缺点是最新一两天数据要等。

### Q2：`binance_vision` 提示 "404 跳过"

正常现象：最近 1~2 天的 daily zip 还没生成。如果你拉的区间右端就是今天，能拿到的最新一根大概率是昨天的。

### Q3：`KeyError: '时间'` / `缺少时间列`

直接把别处拿来的 DataFrame 喂进了 `normalize_ohlcv_df`，但既没有"时间/日期"中文列，也没有 `time` 标准列。先在调用前手动 `df.rename(columns={"我的时间列": "time"})`。

### Q4：返回的 `time` 是 UTC 还是本地时间？

- `akshare_em`：本地时间（A 股交易时区）
- `binance_futures` / `binance_vision`：**UTC**（带时区）
- 跨源对比时记得显式转换，不要让 pandas 隐式比较 tz-naive 与 tz-aware 而报错。

### Q5：长区间分钟级数据要拉多久？

- `binance_futures`：1500 根 / 请求，1m 大概 1 天 / 请求；半年 1m 数据约 180 个请求 + 退避，偶发限速。可加 `page_sleep_s=0.05` 缓和。
- `binance_vision`：每月一个 zip（解压后 30 天 1m 数据），再快也要等 zip 下载，但请求次数少很多，大区间更快。

### Q6：怎么知道自己装了哪些 Provider？

```python
from market_data import list_kline_providers

for pid, desc, intervals in list_kline_providers():
    print(f"{pid:20s}  {sorted(intervals)}  {desc}")
```

或命令行：`python examples/akshare_1h_kline_demo.py --list-sources`。

## 套件演进 TODO

- [ ] 加 OKX / Bybit / Gate 等海外节点 Provider 作为 Binance 备选
- [ ] 加资金费率、持仓量、清算事件等衍生数据接口
- [ ] 给 `binance_vision` 加本地缓存（zip 下完后落盘，避免重复下载）
- [ ] 提供 `to_qlib_format(df)` 之类导出器，方便对接 qlib / backtrader
