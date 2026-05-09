from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = [
    "tsp",
    "tsp_nls",
    "cvrp",
    "cvrp_nls",
    "op",
    "pctsp",
    "sop",
    "smtwtp",
    "rcpsp",
    "mkp",
    "mkp_transformer",
    "bpp",
]

DISPLAY_NAMES = {
    "tsp": "TSP",
    "tsp_nls": "TSP-NLS",
    "cvrp": "CVRP",
    "cvrp_nls": "CVRP-NLS",
    "op": "OP",
    "pctsp": "PCTSP",
    "sop": "SOP",
    "smtwtp": "SMTWTP",
    "rcpsp": "RCPSP",
    "mkp": "MKP",
    "mkp_transformer": "MKP-Transformer",
    "bpp": "BPP",
}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(problem: str) -> dict:
    display = DISPLAY_NAMES[problem]
    cells = [
        markdown_cell(
            f"# {display} pretrained reproduction\n\n"
            "This notebook runs the unified DeepACO reproduction entry point for this problem. "
            "It writes logs, raw JSON, Markdown tables, and PNG figures to the repository-level output folders."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "cwd = Path.cwd().resolve()\n"
            "ROOT = cwd if (cwd / 'pretrained').exists() else cwd.parent\n"
            "sys.path.insert(0, str(ROOT))\n\n"
            "from scripts.reproduce_problem import run_problem\n\n"
            f"PROBLEM = '{problem}'\n"
            "DEVICE = 'auto'          # 'auto', 'cpu', 'cuda', or 'cuda:0'\n"
            "SEED = 12345\n"
            "LIMIT_INSTANCES = None   # set to a small integer for a quick smoke test\n"
            "SIZES = None             # e.g. [20] or None for all configured scales\n"
            "T_ACO = None             # e.g. [1, 10] or None for paper-style checkpoints\n"
            "METHODS = None           # e.g. ['deepaco'] or ['deepaco', 'aco']\n"
            "INCLUDE_BASELINE = True\n"
        ),
        code_cell(
            "raw = run_problem(\n"
            "    PROBLEM,\n"
            "    sizes=SIZES,\n"
            "    t_aco=T_ACO,\n"
            "    methods=METHODS,\n"
            "    limit_instances=LIMIT_INSTANCES,\n"
            "    device=DEVICE,\n"
            "    seed=SEED,\n"
            "    include_baseline=INCLUDE_BASELINE,\n"
            ")\n"
            "raw['outputs']\n"
        ),
        code_cell(
            "from IPython.display import Image, Markdown, display\n\n"
            "table_path = ROOT / raw['outputs']['table']\n"
            "figure_path = ROOT / raw['outputs']['figure']\n\n"
            "display(Markdown(table_path.read_text(encoding='utf-8')))\n"
            "display(Image(filename=str(figure_path)))\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    for problem in PROBLEMS:
        path = ROOT / problem / "reproduce.ipynb"
        path.write_text(json.dumps(build_notebook(problem), indent=1, ensure_ascii=False), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
