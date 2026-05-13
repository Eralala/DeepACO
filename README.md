# DeepACO 复现实验

本仓库用于复现 NeurIPS 2023 论文 **DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization** 的预训练模型测试实验。当前版本在原源码基础上补充了统一复现脚本、12 个问题目录下的 `reproduce.ipynb`、结构化结果输出，以及论文风格曲线图生成单元。

## 复现内容

每个问题目录都提供一个可直接运行的 notebook，内部调用 `scripts/reproduce_problem.py`：

| 问题 | notebook | 默认规模 |
| --- | --- | --- |
| TSP | `tsp/reproduce.ipynb` | 20, 100, 500 |
| TSP-NLS | `tsp_nls/reproduce.ipynb` | 100, 500, 1000 |
| CVRP | `cvrp/reproduce.ipynb` | 20, 100, 500 |
| CVRP-NLS | `cvrp_nls/reproduce.ipynb` | 100, 500 |
| OP | `op/reproduce.ipynb` | 100, 200, 300 |
| PCTSP | `pctsp/reproduce.ipynb` | 20, 100, 500 |
| SOP | `sop/reproduce.ipynb` | 20, 50, 100 |
| SMTWTP | `smtwtp/reproduce.ipynb` | 50, 100, 500 |
| RCPSP | `rcpsp/reproduce.ipynb` | j30, j60, j120 |
| MKP | `mkp/reproduce.ipynb` | 300, 500 |
| MKP-Transformer | `mkp_transformer/reproduce.ipynb` | 300, 500 |
| BPP | `bpp/reproduce.ipynb` | 120 |

运行结果统一写入：

```text
logs/                 # 每个问题的运行日志
results/raw/          # 原始 JSON，含环境、参数、错误堆栈和逐行结果
results/tables/       # Markdown 结果表
results/figures/      # DeepACO 与 ACO 对比曲线
results/paper_data/   # 论文风格曲线的缓存数据
results/paper_figures/ # 论文风格曲线图
```

## 快速运行

首次下载或解压仓库后，请先在项目根目录用 `uv` 根据 `pyproject.toml` 和 `uv.lock` 重建环境：

```powershell
uv sync
```

同步完成后，使用新生成的虚拟环境运行单个问题：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py tsp
```

运行全部问题：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py --all
```

快速 smoke test：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py tsp --sizes 20 --t-aco 10 --limit-instances 32 --methods deepaco --no-baseline
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--sizes 20,100` | 只跑指定规模 |
| `--t-aco 1,10,50` | 指定 ACO 迭代检查点，耗时主要取决于最大值 |
| `--methods deepaco,aco` | 指定 DeepACO 或 ACO baseline |
| `--limit-instances 32` | 限制测试实例数，用于快速检查 |
| `--device auto/cpu/cuda` | 控制推理设备 |
| `--no-baseline` | 跳过 ACO baseline |

Notebook 中也可以通过顶部参数缩短运行时间：

```python
LIMIT_INSTANCES = 32
SIZES = [20]
T_ACO = [10]
METHODS = ['deepaco']
INCLUDE_BASELINE = False
```

## 当前结果

以下结果来自 `results/raw/*_results.json` 和 `results/tables/*_table.md`，复现环境为 Windows 10/11 `10.0.22621`、Python `3.8.20`、PyTorch `1.7.0+cu110`、NVIDIA GeForce RTX 2060。

| 问题 | 规模 | 方法 | 状态 | 设备 | 已完成耗时 | 结果表 |
| --- | --- | --- | --- | --- | ---: | --- |
| TSP | 20, 100, 500 | deepaco, aco | 完成 6/6 | cuda:0 | 8.86h | [tsp_table.md](results/tables/tsp_table.md) |
| TSP-NLS | 100, 500, 1000 | deepaco | 完成 3/3 | cuda:0 | 6.92h | [tsp_nls_table.md](results/tables/tsp_nls_table.md) |
| CVRP | 20, 100, 500 | deepaco, aco | 完成 6/6 | cpu | 2.41h | [cvrp_table.md](results/tables/cvrp_table.md) |
| CVRP-NLS | 100, 500 | deepaco, aco | 失败 4/4 | cuda:0 | 0.00h | [cvrp_nls_table.md](results/tables/cvrp_nls_table.md) |
| OP | 100, 200, 300 | deepaco, aco | 完成 6/6 | cpu | 3.81h | [op_table.md](results/tables/op_table.md) |
| PCTSP | 20, 100, 500 | deepaco, aco | 完成 6/6 | cpu | 2.80h | [pctsp_table.md](results/tables/pctsp_table.md) |
| SOP | 20, 50, 100 | deepaco, aco | 完成 6/6 | cpu | 0.75h | [sop_table.md](results/tables/sop_table.md) |
| SMTWTP | 50, 100, 500 | deepaco, aco | 完成 6/6 | cpu | 1.25h | [smtwtp_table.md](results/tables/smtwtp_table.md) |
| RCPSP | 30, 60, 120 | deepaco, aco | 完成 6/6 | cpu | 2.55h | [rcpsp_table.md](results/tables/rcpsp_table.md) |
| MKP | 300, 500 | deepaco, aco | 完成 4/4 | cpu | 2.99h | [mkp_table.md](results/tables/mkp_table.md) |
| MKP-Transformer | 300, 500 | deepaco, aco | 完成 4/4 | cpu | 1.71h | [mkp_transformer_table.md](results/tables/mkp_transformer_table.md) |
| BPP | 120 | deepaco, aco | 完成 2/2 | cpu | 0.11h | [bpp_table.md](results/tables/bpp_table.md) |

CVRP-NLS 在 Windows 原生环境下失败，原因是 `cvrp_nls/HGS-CVRP-main/build/libhgscvrp.so` 是 Linux 共享库，加载时报 `WinError 193`。完整复现 CVRP-NLS 需要在 Linux/WSL 中运行，或重新编译与当前平台兼容的 HGS 动态库。

## TSP 复现摘要

TSP 是当前最耗时的复现实验。默认配置会跑 3 个规模、DeepACO 与 ACO baseline、`T=[1,10,20,30,40,50,100]`，完整耗时约 8.86 小时。

| 规模 | 实例数 | DeepACO T=100 | ACO T=100 | gap_vs_aco_last | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP20 | 1280 | 3.84848 | 3.87339 | -0.64% | 1.80h |
| TSP100 | 1280 | 8.30295 | 9.56064 | -13.15% | 5.55h |
| TSP500 | 128 | 18.8419 | 24.9488 | -24.48% | 1.52h |

`nvidia-smi` 中显存占用不高是预期现象：当前实现按实例串行推理，ACO 内部又按迭代和城市数循环，GPU 上主要是大量小张量随机采样和信息素更新。最大 TSP500 的矩阵也只有 500x500 量级，因此显存不是瓶颈，主要耗时来自小 kernel 调度、Python 循环和同步开销。

## 论文风格曲线

部分问题已经生成论文风格曲线图和缓存数据：

```text
results/paper_data/
results/paper_figures/
```

已生成曲线覆盖 TSP、CVRP、OP、PCTSP、SOP、SMTWTP、RCPSP 和 MKP。对应 notebook 的最后一个单元不是单纯绘图，如果某个规模没有缓存，会重新运行多个 ACO 变体；运行前建议先设置 `PAPER_SIZES`、`PAPER_LIMIT_INSTANCES` 和 `PAPER_T_ACO`。

## 数据与权重

预训练权重位于 `pretrained/`，测试数据位于 `data/`。部分缺失测试数据会在运行时按固定随机种子生成，例如：

- `data/cvrp_nls/testDataset-*.pt`
- `data/mkp_transformer/testDataset-*.pt`

依赖由 `pyproject.toml` 和 `uv.lock` 记录。由于 `.venv` 不随压缩包上传，用户需要自行执行 `uv sync` 安装依赖。当前项目要求 Python `>=3.8,<3.9`，核心依赖包括 `torch==1.7.0+cu110`、`torch-geometric==2.0.4`、`torch-scatter==2.0.7` 和 `torch-sparse==0.6.9`。

## 原论文引用

```bibtex
@inproceedings{ye2023deepaco,
  title={DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization},
  author={Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Liang, Helan and Li, Yong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
