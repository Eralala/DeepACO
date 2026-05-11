# MKP Reproduction Table

- Metric: average objective (higher is better)
- Instances per completed row: see `instances` column
- Device: cpu

| scale | method | status | instances | duration_s | T=1 | T=10 | T=20 | T=30 | T=40 | T=50 | T=100 | gap_vs_aco_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | DeepACO-GNN | ok | 100 | 2400.75 | 57.8002 | 59.0085 | 59.4802 | 59.8163 | 59.995 | 60.1422 | 60.7821 | +3.46% |
| 300 | ACO | ok | 100 | 2017.85 | 44.2943 | 47.2897 | 48.8579 | 50.9827 | 53.367 | 55.4772 | 58.7482 |  |
| 500 | DeepACO-GNN | ok | 100 | 3664.66 | 98.5138 | 99.7143 | 100.166 | 100.649 | 100.938 | 101.18 | 101.778 | +5.85% |
| 500 | ACO | ok | 100 | 2686.56 | 70.3979 | 74.1686 | 76.2566 | 79.3048 | 83.7807 | 87.7628 | 96.1524 |  |
