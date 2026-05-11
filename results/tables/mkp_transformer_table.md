# MKP-Transformer Reproduction Table

- Metric: average objective (higher is better)
- Instances per completed row: see `instances` column
- Device: cpu

| scale | method | status | instances | duration_s | T=1 | T=5 | T=10 | T=20 | T=50 | gap_vs_aco_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | DeepACO-Transformer | ok | 100 | 1278.61 | 62.8857 | 63.1614 | 63.2035 | 63.2644 | 63.3233 | +2.40% |
| 300 | ACO | ok | 100 | 1114.54 | 46.6903 | 55.493 | 58.2231 | 60.2641 | 61.8418 |  |
| 500 | DeepACO-Transformer | ok | 100 | 2001.11 | 101.667 | 102.494 | 102.763 | 102.968 | 103.297 | +2.27% |
| 500 | ACO | ok | 100 | 1747.18 | 75.3242 | 89.3762 | 94.0226 | 97.944 | 101.005 |  |
