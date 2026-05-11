# CVRP-NLS Reproduction Table

- Metric: average route length (lower is better)
- Instances per completed row: see `instances` column
- Device: cuda:0

| scale | method | status | instances | duration_s | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=7 | T=8 | T=9 | T=10 | gap_vs_aco_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | DeepACO-NLS | error | 100 |  |  |  |  |  |  |  |  |  |  |  |  |
| 100 | ACO-NLS | error | 100 |  |  |  |  |  |  |  |  |  |  |  |  |
| 500 | DeepACO-NLS | error | 100 |  |  |  |  |  |  |  |  |  |  |  |  |
| 500 | ACO-NLS | error | 100 |  |  |  |  |  |  |  |  |  |  |  |  |

## Errors

- 100 DeepACO-NLS: `OSError: [WinError 193] %1 不是有效的 Win32 应用程序。`
- 100 ACO-NLS: `OSError: [WinError 193] %1 不是有效的 Win32 应用程序。`
- 500 DeepACO-NLS: `OSError: [WinError 193] %1 不是有效的 Win32 应用程序。`
- 500 ACO-NLS: `OSError: [WinError 193] %1 不是有效的 Win32 应用程序。`
