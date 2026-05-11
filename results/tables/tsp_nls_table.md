# TSP-NLS Reproduction Table

- Metric: average cost (lower is better)
- Instances per completed row: see `instances` column
- Device: cuda:0

| scale | method | status | instances | duration_s | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=7 | T=8 | T=9 | T=10 | gap_vs_aco_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | DeepACO-NLS | ok | 1280 | 3012.27 | 7.77659 | 7.76974 | 7.76686 | 7.76497 | 7.76379 | 7.76293 | 7.7623 | 7.76179 | 7.76142 | 7.76104 |  |
| 500 | DeepACO-NLS | ok | 128 | 3951.18 | 16.978 | 16.9286 | 16.9075 | 16.8896 | 16.8752 | 16.8671 | 16.8576 | 16.8528 | 16.8493 | 16.8426 |  |
| 1000 | DeepACO-NLS | ok | 128 | 17957.85 | 24.0247 | 23.9429 | 23.9136 | 23.8879 | 23.8635 | 23.8504 | 23.8425 | 23.8323 | 23.8263 | 23.8211 |  |
