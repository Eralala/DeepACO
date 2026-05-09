# DeepACO 论文实验复现

本仓库用于复现 NeurIPS 2023 论文 **DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization** 的预训练模型测试实验。当前版本在原源码基础上补充了统一复现入口、12 个问题目录下的 `reproduce.ipynb`、结构化结果输出和实验说明文档。

## 复现入口

每个问题目录下都有一个可直接运行的 notebook：

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

Notebook 内部调用统一脚本 `scripts/reproduce_problem.py`，并把输出保存到：

```text
logs/                 # 每个问题的运行日志
results/raw/          # 原始 JSON，含环境、参数、错误堆栈和逐行结果
results/tables/       # Markdown 表格
results/figures/      # PNG 曲线图
```

## 命令行运行

使用项目自带虚拟环境：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py tsp
```

运行全部 12 个问题：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py --all
```

快速检查单个问题：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_problem.py cvrp --sizes 20 --t-aco 1 --limit-instances 1 --methods deepaco --device cpu
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--sizes 20,100` | 只跑指定规模 |
| `--t-aco 1,10,50` | 指定 ACO 迭代检查点 |
| `--methods deepaco,aco` | 指定 DeepACO 或 ACO baseline |
| `--limit-instances 5` | 只跑前 5 个实例，用于 smoke test |
| `--device auto/cpu/cuda` | 控制设备 |
| `--no-baseline` | 跳过 ACO baseline |

## 环境

当前复现环境：

| 项目 | 当前值 |
| --- | --- |
| OS | Windows 10/11 |
| Python | 3.8.20 |
| PyTorch | 1.7.0+cu110 |
| CUDA | 可用 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |

依赖由 `pyproject.toml` 和 `uv.lock` 记录。大多数原始实现默认在 CPU 上测试；TSP、TSP-NLS、CVRP-NLS 的神经网络推理在 `--device auto` 下会优先使用 CUDA，NLS 的局部搜索仍在 CPU 上执行。

## 数据与权重

预训练权重位于 `pretrained/`。测试数据位于 `data/`，缺失的数据会在运行时按固定随机种子生成：

- `data/cvrp_nls/testDataset-*.pt`
- `data/mkp_transformer/testDataset-*.pt`

CVRP-NLS 依赖 `cvrp_nls/HGS-CVRP-main/build/libhgscvrp.so` 执行 SWAP* 局部搜索。该文件是 Linux 共享库，在 Windows 原生 Python 下会报 `WinError 193`。要完整复现 CVRP-NLS，请在 Linux/WSL 中运行，或按 `cvrp_nls/README.md` 重新编译与当前平台兼容的 HGS 动态库。

## 实验说明

更详细的实验配置、数据规模、输出格式、当前 smoke test 结果和平台注意事项见：

```text
EXPERIMENT_REPORT.md
```

## 原论文引用

```bibtex
@inproceedings{ye2023deepaco,
  title={DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization},
  author={Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Liang, Helan and Li, Yong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
