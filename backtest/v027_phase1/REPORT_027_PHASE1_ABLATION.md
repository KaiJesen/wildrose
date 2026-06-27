# 027 Phase 1 Core Ablation (valid)

| baseline return | -6.21% |
| baseline coverage | 25.00% |
| gate return ≥ | -4.19% |
| gate coverage ≥ | 20.25% |

| tag | return | coverage | slow_up | crash | gate |
|-----|--------|----------|---------|-------|------|
| slow_up_on | -6.21% | 25.00% | 0 | 4 | FAIL |
| crash_regime_repeat | -6.21% | 25.00% | 0 | 4 | FAIL |
| crash_hold_30 | -6.21% | 25.00% | 0 | 4 | FAIL |
| crash_hold_48 | -6.21% | 25.00% | 0 | 4 | FAIL |
| crash_hold_72 | -6.21% | 25.00% | 0 | 4 | FAIL |
| trend_hold_p50 | -12.25% | 18.75% | 0 | 3 | FAIL |

**slow_up 裁定**: no_opens_on_valid
**best valid**: none
**Phase 1 ablation gate**: FAIL

## 复现
```bash
python examples/run_v027_core_ablation.py
```
