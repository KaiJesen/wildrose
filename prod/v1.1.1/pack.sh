#!/usr/bin/env bash
# Pack 024 B0 research baseline (c1_pw20 + TEQ wp=0.35) for cross-machine reproduction.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROD="$(cd "$(dirname "$0")" && pwd)"
export ROOT PROD

rm -rf "$PROD/code"
mkdir -p "$PROD/code" "$PROD/config" "$PROD/checkpoint" "$PROD/calibration" \
  "$PROD/data/kline" "$PROD/metrics" "$PROD/scripts" "$PROD/docs"

for pkg in trading_system transformer_kit market_data best_point; do
  rsync -a --exclude '__pycache__' "$ROOT/$pkg/" "$PROD/code/$pkg/"
done

CKPT_SRC="$ROOT/checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt"
KLINE_SRC="$ROOT/data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"
CALIB_SRC="$ROOT/backtest/v024_constrained/teq_edge_calibration.json"

for f in "$CKPT_SRC" "$KLINE_SRC" "$CALIB_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "error: missing required artifact: $f" >&2
    exit 1
  fi
done

cp "$CKPT_SRC" "$PROD/checkpoint/market_state_best.pt"
cp "$KLINE_SRC" "$PROD/data/kline/$(basename "$KLINE_SRC")"
cp "$CALIB_SRC" "$PROD/calibration/teq_edge_calibration.json"

python3 <<'PY'
import json
from pathlib import Path
import os

root = Path(os.environ["ROOT"])
prod = Path(os.environ["PROD"])
src = root / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["teq_edge"]["calibration_path"] = "calibration/teq_edge_calibration.json"
cfg["teq_edge"]["model_checkpoint"] = "checkpoint/market_state_best.pt"
cfg.setdefault("_024_meta", {})["checkpoint"] = "checkpoint/market_state_best.pt"
cfg["_prod_meta"] = {
    "version": "v1.1.1",
    "kind": "research_baseline_b0",
    "frozen_kline": "data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv",
    "checkpoint_hash_prefix": "82ca51cf637a258c",
    "phase0_return": 0.0901,
    "phase0_coverage": 0.267,
    "phase0_teq_open": 3,
}
(prod / "config/trading_rule.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

cp "$ROOT/examples/backtest_trading_system_v014.py" "$PROD/scripts/backtest.py"
cp "$ROOT/examples/_train_common.py" "$PROD/scripts/_train_common.py"
cp "$ROOT/prod/v1.1.0/scripts/run_backtest.sh" "$PROD/scripts/run_backtest.sh"
chmod +x "$PROD/scripts/run_backtest.sh"

for script in backtest.py _train_common.py; do
  if grep -q 'parents\[1\]' "$PROD/scripts/$script"; then
    sed -i 's/parents\[1\]/parents[2]/' "$PROD/scripts/$script" 2>/dev/null || \
      sed -i '' 's/parents\[1\]/parents[2]/' "$PROD/scripts/$script"
  fi
done

cat > "$PROD/scripts/verify_phase0.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
PROD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROD_ROOT/../.." && pwd)"
export PYTHONPATH="$PROD_ROOT/code:${PYTHONPATH:-}"
OUT="${1:-$REPO_ROOT/backtest/prod_v1.1.1_smoke}"
cd "$REPO_ROOT"

python3 "$PROD_ROOT/scripts/backtest.py" \
  --config "$PROD_ROOT/config/trading_rule.json" \
  --checkpoint "$PROD_ROOT/checkpoint/market_state_best.pt" \
  --split test \
  --output-dir "$OUT" \
  --csv "$PROD_ROOT/data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"

python3 "$REPO_ROOT/examples/eval_participation.py" \
  --backtest-dir "$OUT" \
  --output "$OUT/participation.json"

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
m = json.loads((out / "metrics.json").read_text())
p = json.loads((out / "participation.json").read_text())
cov = p["test"]["participation_metrics"]["leg_count_coverage_ratio"]
ret = m["total_return"]
teq = int(m["trend_qualified_open_count"])
ok = abs(ret - 0.0901) <= 0.002 and abs(cov - 0.267) <= 0.01 and teq == 3
print(f"return={ret*100:.2f}% coverage={cov*100:.2f}% teq={teq} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
SH
chmod +x "$PROD/scripts/verify_phase0.sh"

if [[ -f "$ROOT/backtest/v025_phase0/b0_test/metrics.json" ]]; then
  cp "$ROOT/backtest/v025_phase0/b0_test/metrics.json" "$PROD/metrics/backtest_test_oos.json"
fi
if [[ -f "$ROOT/backtest/v025_phase0/phase0_summary.json" ]]; then
  cp "$ROOT/backtest/v025_phase0/phase0_summary.json" "$PROD/metrics/phase0_summary.json"
fi

for doc in \
  "document/025_多通道参与信号规则扩展/架构师-025-项目裁定与下一步指导.md" \
  "backtest/v025_phase0/REPORT_025_PHASE0.md" \
  "backtest/v025_ab/REPORT_025_AB.md" \
  "backtest/v025_ab_a3b/REPORT_025_A3B.md"
do
  if [[ -f "$ROOT/$doc" ]]; then
    cp "$ROOT/$doc" "$PROD/docs/$(basename "$doc")"
  fi
done

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DATE="$(git -C "$ROOT" log -1 --format=%ci 2>/dev/null || echo unknown)"
CKPT_HASH="$(python3 -c "import hashlib; p=open('$PROD/checkpoint/market_state_best.pt','rb'); h=hashlib.sha256(); 
import sys
while True:
    c=p.read(1<<20)
    if not c: break
    h.update(c)
print(h.hexdigest()[:16])")"
export CKPT_HASH GIT_SHA GIT_DATE

python3 <<'PY'
import json
import os
from datetime import date

prod = os.environ["PROD"]
manifest = {
    "product": "wildrose-trading-system",
    "version": "v1.1.1",
    "release_date": date.today().isoformat(),
    "kind": "research_baseline_b0",
    "git_commit": os.environ["GIT_SHA"],
    "git_commit_date": os.environ["GIT_DATE"],
    "strategy": "v024 phase1c TEQ c1_pw20 wp=0.35 (B0)",
    "model_checkpoint": "checkpoint/market_state_best.pt",
    "model_source": "0065a_leg_align_c1_pw20",
    "checkpoint_hash_prefix": os.environ["CKPT_HASH"],
    "config": "config/trading_rule.json",
    "calibration": "calibration/teq_edge_calibration.json",
    "frozen_kline": "data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv",
    "code_packages": ["trading_system", "transformer_kit", "market_data", "best_point"],
    "production_note": "prod trading remains v1.1.0 (0062e); v1.1.1 is 024/025 research baseline only",
    "backtest_reference": {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "days": 365,
        "split": "test",
        "metrics_file": "metrics/backtest_test_oos.json",
        "expected_return": 0.0901,
        "expected_coverage": 0.267,
        "expected_teq_open": 3,
    },
    "supplemental_references": {
        "architect_ruling": "docs/架构师-025-项目裁定与下一步指导.md",
        "phase0_report": "docs/REPORT_025_PHASE0.md",
        "a3a_report": "docs/REPORT_025_AB.md",
        "a3b_report": "docs/REPORT_025_A3B.md",
        "project_log": "docs/PROJECT_LOG.md",
    },
}
with open(f"{prod}/MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

echo "packed prod/v1.1.1 at $PROD (checkpoint $CKPT_HASH)"
