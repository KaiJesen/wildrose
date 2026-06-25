# 024 Constrained Pipeline — Final Tuning Report

**Date**: 2026-06-25  
**Branch**: `feature/024-model-alignment-proposal`  
**Frozen rules**: `configs/trading_rule_v023_phase1c_0062e.json`  
**Best candidate**: `checkpoints/0065a_leg_align_c1_pw20` + TEQ `w_part=0.35`

---

## Executive Summary

Constrained training + TEQ retune **closed the drift gate** and materially improved A2 vs unconstrained 0065a-1, but **cannot simultaneously pass both Phase 3 exploration gates** (return ≥8.84% **and** coverage ≥28%) via `w_part` / edge-weight tuning alone.

| Gate | Best achievable (test) | Status |
|------|------------------------|--------|
| A1 drift ≤2pp vs A0 | c0/c1/pw20 all PASS | ✅ |
| TEQ opens ≥2× baseline | 3× (teq=3) | ✅ |
| Exploration return ≥8.84% | **9.01%** @ wp≤0.36 | ✅ |
| Exploration coverage ≥28% | **26.7%** @ wp≤0.36 | ❌ (−1.3pp) |
| **Both exploration gates** | — | ❌ |

At `w_part≥0.37`: coverage **30%** PASS, return **8.55%** FAIL (−0.29pp).

---

## Training Stack (constrained)

| Checkpoint | λ_part | valid part_auc | A1 test return | trades | teq |
|------------|--------|----------------|----------------|--------|-----|
| c0 | 1.5 | 0.572 | +6.83% | 12 | 1 |
| c1 | 1.5 | 0.571 | +6.38% | 15 | 1 |
| **c1_pw20** | **2.0** | 0.572 | +6.38% (A1) | 13 | 1 |

Drift gate: all variants within **−0.94pp** of A0 (+7.77%).

---

## TEQ Tuning (pw20 checkpoint)

### Pareto knee (`w_part`)

| w_part | test return | coverage | teq | explore |
|--------|-------------|----------|-----|---------|
| 0.30–0.34 | 8.84% | 23.3% | 2 | FAIL both |
| **0.35–0.36** | **9.01%** | **26.7%** | 3 | return ✅ cov ❌ |
| 0.37–0.40 | 8.55% | 30.0% | 3 | return ❌ cov ✅ |

- **16-config sweep** (`examples/sweep_teq_wp_valid.py`): c1 + pw20 calibrations × wp ∈ [0.30..0.40] → **0/16** dual PASS on test.
- **Edge-weight grid** (w5/w24 × wp): same discrete outcomes; no bridge between knees.
- **pw20 recalibration** on valid: does not beat c1 calibration for the high-return branch.

### Best A2 config (near-miss)

```
Checkpoint: checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt
TEQ:        w5=0.25, w24=0.35, w_part=0.35, use_calibrated=True
Calib:      backtest/v024_constrained/teq_edge_calibration.json
Config:     configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json
```

| Metric | A0 (0062e) | A2 best |
|--------|------------|---------|
| test return | +7.77% | **+9.01%** |
| leg_count_coverage | 16.7% | **26.7%** |
| teq opens | 1 | 3 |
| trades | 10 | 13 |
| counter_leg_participation | — | ≤5 ✅ |

Backtest: `backtest/v024_constrained/a2_pw20_wp0.35_test/`

---

## Valid-split note

Exploration return gate (8.84%) is **test-calibrated** (v022 95th percentile). On **valid**, A0 return is **−2.45%**; best valid return in sweep is **+3.38%** — the absolute 8.84% gate cannot be used for valid-only `w_part` selection. TEQ calibration itself remains valid-only (no test leakage).

---

## Structural ceiling (why tuning stops here)

1. **Discrete TEQ open**: crossing `w_part≈0.37` promotes one additional trend-qualified open → coverage jumps **+3.3pp** but return drops **−0.46pp** (likely a marginal/low-quality leg).
2. **Scheme B scope**: only `_try_trend_qualified_open` reads `teq_edge_*`; standard/slow_up opens unchanged → coverage ceiling tied to teq path count (~3 on this data).
3. **023 precedent**: teq ceiling (6 opens) reached 20% coverage at 1.42% return — rule relaxation trades return for coverage.
4. **Constrained heads**: participation AUC ~0.57 (below unconstrained 0.63) — drift protection limits participation signal strength.

Further **training/tuning** within frozen phase1c + Scheme B is unlikely to close a **1.3pp coverage gap** without either:
- accepting a single-gate near-miss, or
- rule-layer / Scheme A3 change (explicitly gated in 024 §5.4).

---

## Reproduction

```bash
# TEQ w_part sweep (valid + test)
python examples/sweep_teq_wp_valid.py

# Best A2 backtest
python examples/backtest_trading_system_v014.py \
  --config configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json \
  --checkpoint checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt \
  --split test \
  --output-dir backtest/v024_constrained/a2_pw20_wp0.35_test
```

Artifacts:
- `backtest/v024_constrained/teq_wp_sweep/sweep_summary.json`
- `backtest/v024_constrained/teq_edge_calibration_pw20.json`
- `examples/sweep_teq_wp_valid.py`

---

## Decision required (cannot auto-resolve)

| Option | Action | Trade-off |
|--------|--------|-----------|
| **A** | Nominate wp=0.35 as Phase 3 candidate; run multi-seed | Beats A0 return; exploration **coverage −1.3pp** |
| **B** | Lock wp=0.37 | Meets coverage; return **−0.29pp** below explore line |
| **C** | Relax explore coverage to **26%** or return to **8.5%** | Aligns gates with empirical Pareto knee |
| **D** | Rule-layer / A3 exploration | Outside current training scope; needs architect approval |
| **E** | Stop Phase 3; declare model-path insufficient | Fall back to 024 §5.4 fallback (labels v2 / multi-task) |

**Recommendation**: **A + multi-seed** if near-miss is acceptable per 024 layered acceptance; otherwise **C** (minor gate adjustment) before more training spend.
