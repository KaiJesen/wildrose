# 027 Phase 2 报告

| 臂 | return | trades | sat trades | Sharpe |
|----|--------|--------|------------|--------|
| core_only | -4.12% | 22 | — | — |
| satellite_only | -4.32% | 13 | 13 | -1.35 |
| core+sat | -11.47% | 33 | 11 | -3.65 |

**Satellite valid gate**: FAIL
**Combined ≥ core**: FAIL
**Phase 2**: FAIL

```bash
python examples/run_v027_phase2.py --split valid
```
