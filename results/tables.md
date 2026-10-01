| Eval set | Random | Naluri M0 | Naluri M1 | Naluri M2 | Naluri M3 | Naluri M4 | Naluri M5 | Naluri M6 | Naluri M6-pairs | Naluri M7 | Laya (zero-shot) | Fair vs Laya? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASSIVE intent, trained languages | 12.5% | 82.5% | 90.5% | 91.3% | 91.7% | 90.9% | 91.9% | 91.3% | **92.1%** | **92.1%** | 62.5% | no: task trained |
| MASSIVE intent, unseen languages | 12.5% | 65.5% | 70.4% | 75.7% | 72.4% | 75.0% | 75.7% | 75.2% | **75.8%** | 75.2% | 37.6% | no: task trained |
| XNLI logic, trained languages | 33.3% | 33.7% | 62.8% | 64.5% | 63.3% | 64.9% | 64.6% | 63.1% | **65.3%** | 63.1% | 54.5% | no: task trained |
| XNLI logic, unseen languages | 33.3% | 36.2% | 57.7% | 56.0% | 58.9% | 57.4% | 56.5% | **59.4%** | 58.1% | 56.8% | 46.1% | no: task trained |
| Amazon stars (1-5, exact) | 20.0% | 19.7% (MAE 1.26) | **56.9%** (MAE 0.60) | 52.0% (MAE 0.62) | 52.1% (MAE 0.64) | 54.0% (MAE 0.60) | 56.6% (MAE 0.58) | 55.8% (MAE 0.58) | 55.7% (MAE 0.58) | 55.4% (MAE 0.58) | 29.0% (MAE 1.45) | no: task trained |
| Sentiment, new domain + languages | 33.3% | 34.5% | 57.2% | 57.4% | 62.3% | 63.4% | 62.7% | 63.3% | 61.1% | **63.6%** | 52.7% | partly: similar skill |
| XCOPA cause/effect (unseen task) | 50.0% | 49.7% | 54.8% | **58.6%** | 57.2% | 57.2% | 58.0% | 57.9% | 57.2% | 57.8% | 54.7% | yes |
| SIB-200 topic (unseen task**) | 14.3% | 61.1% | 55.4% | 67.5% | 62.4% | **74.4%** | 73.7% | 71.0% | 73.2% | 73.0% | 70.8% | yes** |
| Belebele reading (unseen task*) | 25.0% | 28.6% | 28.9% | 34.8% | 34.2% | 33.7% | 35.2% | **36.2%** | **36.2%** | 34.5% | 33.1% | yes* |

| Metric (range over eval sets) | Naluri M0 | Naluri M1 | Naluri M2 | Naluri M3 | Naluri M4 | Naluri M5 | Naluri M6 | Naluri M6-pairs | Naluri M7 | Laya (zero-shot) |
|---|---|---|---|---|---|---|---|---|---|---|
| Calibration error (ECE, lower is better) | 0.01–0.22 | 0.02–0.11 | 0.03–0.07 | 0.03–0.07 | 0.03–0.11 | 0.03–0.13 | 0.02–0.12 | 0.03–0.14 | 0.02–0.17 | 0.12–0.56 |
| Order consistency (higher is better) | 0.39–0.94 | 0.64–0.96 | 0.72–0.97 | 0.70–0.96 | 0.68–0.97 | 0.65–0.97 | 0.70–0.97 | 0.67–0.97 | 0.71–0.96 | 0.68–0.94 |
| Latency p50, ms (RTX 5050) | 5.60–9.70 | 6.40–14.60 | 4.60–5.20 | 4.50–5.10 | 4.65–5.55 | 4.60–5.20 | 4.60–5.20 | 4.80–5.30 | 4.80–5.30 | 9.30–18.50 |
