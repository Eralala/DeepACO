# CVRP-NLS Reproduction Table

- Metric: average route length (lower is better)
- Instances per completed row: see `instances` column
- Device: cpu

| scale | method | status | instances | duration_s | T=1 | gap_vs_aco_last |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | DeepACO-NLS | error | 1 |  |  |  |

## Errors

- 100 DeepACO-NLS: `OSError: [WinError 193] %1 不是有效的 Win32 应用程序。`
