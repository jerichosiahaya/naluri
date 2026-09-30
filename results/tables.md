| Eval set | Random | Naluri M0 | Naluri M1 | Naluri M2 | Naluri M3 | Naluri M4 | Laya (zero-shot) | Fair vs Laya? |
|---|---|---|---|---|---|---|---|---|
| MASSIVE intent, trained languages | 12.5% | 82.5% | 90.5% | 91.3% | **91.7%** | 90.9% | 62.5% | no: task trained |
| MASSIVE intent, unseen languages | 12.5% | 65.5% | 70.4% | **75.7%** | 72.4% | 75.0% | 37.6% | no: task trained |
| XNLI logic, trained languages | 33.3% | 33.7% | 62.8% | 64.5% | 63.3% | **64.9%** | 54.5% | no: task trained |
| XNLI logic, unseen languages | 33.3% | 36.2% | 57.7% | 56.0% | **58.9%** | 57.4% | 46.1% | no: task trained |
| Amazon stars (1-5, exact) | 20.0% | 19.7% (MAE 1.26) | **56.9%** (MAE 0.60) | 52.0% (MAE 0.62) | 52.1% (MAE 0.64) | 54.0% (MAE 0.60) | 29.0% (MAE 1.45) | no: task trained |
| Sentiment, new domain + languages | 33.3% | 34.5% | 57.2% | 57.4% | 62.3% | **63.4%** | 52.7% | partly: similar skill |
| XCOPA cause/effect (unseen task) | 50.0% | 49.7% | 54.8% | **58.6%** | 57.2% | 57.2% | 54.7% | yes |
| SIB-200 topic (unseen task**) | 14.3% | 61.1% | 55.4% | 67.5% | 62.4% | **74.4%** | 70.8% | yes** |
| Belebele reading (unseen task*) | 25.0% | 28.6% | 28.9% | **34.8%** | 34.2% | 33.7% | 33.1% | yes* |

| Metric (range over eval sets) | Naluri M0 | Naluri M1 | Naluri M2 | Naluri M3 | Naluri M4 | Laya (zero-shot) |
|---|---|---|---|---|---|---|
| Calibration error (ECE, lower is better) | 0.01–0.22 | 0.02–0.11 | 0.03–0.07 | 0.03–0.07 | 0.03–0.11 | 0.12–0.56 |
| Order consistency (higher is better) | 0.39–0.94 | 0.64–0.96 | 0.72–0.97 | 0.70–0.96 | 0.68–0.97 | 0.68–0.94 |
| Latency p50, ms (RTX 5050) | 5.60–9.70 | 6.40–14.60 | 4.60–5.20 | 4.50–5.10 | 4.65–5.55 | 9.30–18.50 |
