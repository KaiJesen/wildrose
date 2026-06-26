const API = "";
let backendId = "";
let candleSeries = null;
let equitySeries = null;
let candleChart = null;
let equityChart = null;
let markers = [];

async function api(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function fmtPct(x) {
  if (x == null || Number.isNaN(x)) return "-";
  return (x * 100).toFixed(2) + "%";
}

function fmtTs(ts) {
  if (!ts) return "-";
  return ts.replace("T", " ").replace("Z", " UTC");
}

async function loadBackends() {
  const list = await api("/api/backends");
  const sel = document.getElementById("backendSelect");
  sel.innerHTML = "";
  list.forEach((b) => {
    const o = document.createElement("option");
    o.value = b.id;
    o.textContent = `${b.display_name} (${b.mode})`;
    sel.appendChild(o);
  });
  if (list.length) {
    backendId = list[0].id;
    sel.value = backendId;
  }
  sel.onchange = () => {
    backendId = sel.value;
    refreshAll();
    connectWs();
  };
}

function renderCards(d) {
  const el = document.getElementById("cards");
  const sig = d.signals || {};
  const pnl = d.pnl || {};
  const pos = d.position || {};
  const h = d.health || {};
  el.innerHTML = `
    <div class="card"><h3>信号</h3><div class="val">${fmtPct(sig.avg_edge)}</div>
      <div class="sub">edge · block ${fmtPct(sig.block_rate)} · n=${sig.count || 0}</div></div>
    <div class="card"><h3>收益</h3><div class="val">${fmtPct(pnl.total_return)}</div>
      <div class="sub">胜率 ${fmtPct(pnl.win_rate)} · 交易 ${pnl.trade_count || 0}</div></div>
    <div class="card"><h3>持仓</h3><div class="val">${pos.state || "FLAT"}</div>
      <div class="sub">仓位 ${fmtPct(pos.position_ratio)} · 杠杆 ${pos.leverage || 20}x</div></div>
    <div class="card"><h3>运行态</h3><div class="val">${h.runner_alive ? "RUNNING" : "IDLE"}</div>
      <div class="sub">lag ${h.lag_seconds != null ? Math.round(h.lag_seconds) + "s" : "-"}</div></div>
  `;
  const badge = document.getElementById("liveBadge");
  badge.classList.toggle("off", d.mode !== "live");
  badge.textContent = d.mode === "live" ? "LIVE" : "ARCHIVE";
  document.getElementById("latestBar").textContent =
    "最新 bar: " + fmtTs(h.latest_bar_ts);
}

function renderBlocks(d) {
  const ul = document.getElementById("blockList");
  ul.innerHTML = "";
  (d.signals?.block_topn || []).forEach((x) => {
    const li = document.createElement("li");
    li.textContent = `${x.reason}: ${x.count}`;
    ul.appendChild(li);
  });
}

function renderTrades(rows) {
  const tb = document.querySelector("#tradesTable tbody");
  tb.innerHTML = "";
  rows.slice(0, 20).forEach((t) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.side}</td>
      <td>${fmtTs(t.entry_ts)}</td>
      <td>${fmtTs(t.exit_ts)}</td>
      <td>${Number(t.entry_price).toFixed(2)}</td>
      <td>${Number(t.exit_price).toFixed(2)}</td>
      <td>${fmtPct(t.net_pnl)}</td>`;
    tb.appendChild(tr);
  });
}

function tsToChart(ts) {
  return Math.floor(new Date(ts.replace("Z", "Z")).getTime() / 1000);
}

function ensureCharts() {
  if (!candleChart) {
    candleChart = LightweightCharts.createChart(document.getElementById("candleChart"), {
      layout: { background: { color: "#1a2332" }, textColor: "#c5d0dc" },
      grid: { vertLines: { color: "#2a3545" }, horzLines: { color: "#2a3545" } },
      timeScale: { timeVisible: true },
    });
    candleSeries = candleChart.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350",
      borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
  }
  if (!equityChart) {
    equityChart = LightweightCharts.createChart(document.getElementById("equityChart"), {
      layout: { background: { color: "#1a2332" }, textColor: "#c5d0dc" },
      grid: { vertLines: { color: "#2a3545" }, horzLines: { color: "#2a3545" } },
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.1 } },
    });
    equitySeries = equityChart.addLineSeries({ color: "#4fc3f7", lineWidth: 2 });
  }
}

async function loadOhlcvAndTrades() {
  ensureCharts();
  const [ohlcv, trades] = await Promise.all([
    api(`/api/backends/${backendId}/ohlcv?limit=500`),
    api(`/api/backends/${backendId}/trades?limit=200`),
  ]);
  const data = ohlcv.map((b) => ({
    time: tsToChart(b.ts),
    open: b.open, high: b.high, low: b.low, close: b.close,
  }));
  candleSeries.setData(data);
  markers = [];
  trades.forEach((t) => {
    if (!t.entry_ts) return;
    const isLong = (t.side || "").toUpperCase() === "LONG";
    markers.push({
      time: tsToChart(t.entry_ts),
      position: isLong ? "belowBar" : "aboveBar",
      color: isLong ? "#1565c0" : "#f9a825",
      shape: isLong ? "arrowUp" : "arrowDown",
      text: isLong ? "L" : "S",
    });
    if (t.exit_ts) {
      markers.push({
        time: tsToChart(t.exit_ts),
        position: "aboveBar",
        color: "#111",
        shape: "circle",
        text: "X",
      });
    }
  });
  candleSeries.setMarkers(markers.sort((a, b) => a.time - b.time));
  candleChart.timeScale().fitContent();
  renderTrades(trades);
}

async function loadEquity() {
  ensureCharts();
  const eq = await api(`/api/backends/${backendId}/equity`);
  equitySeries.setData(eq.map((r) => ({ time: tsToChart(r.ts), value: r.equity })));
  equityChart.timeScale().fitContent();
}

async function refreshAll() {
  if (!backendId) return;
  const dash = await api(`/api/backends/${backendId}/dashboard?period=30d`);
  renderCards(dash);
  renderBlocks(dash);
  await Promise.all([loadOhlcvAndTrades(), loadEquity()]);
}

let ws = null;
function connectWs() {
  if (ws) { ws.close(); ws = null; }
  if (!backendId) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/backends/${backendId}`);
  ws.onmessage = () => refreshAll();
}

document.getElementById("refreshBtn").onclick = refreshAll;

(async () => {
  await loadBackends();
  await refreshAll();
  connectWs();
  setInterval(refreshAll, 60000);
})();
