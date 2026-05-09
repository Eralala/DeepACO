from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

EPS = 1e-10
PROBLEM_MODULES = ["net", "aco", "utils", "rcpsp_inst", "two_opt", "swapstar"]


PROBLEMS: Dict[str, Dict[str, Any]] = {
    "tsp": {
        "display": "TSP",
        "sizes": [20, 100, 500],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average cost",
        "cwd": "problem",
        "device": "auto_cuda",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "tsp_nls": {
        "display": "TSP-NLS",
        "sizes": [100, 500, 1000],
        "t_aco": list(range(1, 11)),
        "n_ants": 48,
        "direction": "min",
        "metric": "average cost",
        "cwd": "problem",
        "device": "auto_cuda",
        "methods": ["deepaco"],
        "labels": {"deepaco": "DeepACO-NLS", "aco": "ACO-NLS"},
    },
    "cvrp": {
        "display": "CVRP",
        "sizes": [20, 100, 500],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average objective",
        "cwd": "root",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "cvrp_nls": {
        "display": "CVRP-NLS",
        "sizes": [100, 500],
        "t_aco": list(range(1, 11)),
        "n_ants": 20,
        "direction": "min",
        "metric": "average route length",
        "cwd": "problem",
        "device": "auto_cuda",
        "methods": ["deepaco"],
        "labels": {"deepaco": "DeepACO-NLS", "aco": "ACO-NLS"},
    },
    "op": {
        "display": "OP",
        "sizes": [100, 200, 300],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "max",
        "metric": "average objective",
        "cwd": "root",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "pctsp": {
        "display": "PCTSP",
        "sizes": [20, 100, 500],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average objective",
        "cwd": "root",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "sop": {
        "display": "SOP",
        "sizes": [20, 50, 100],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average cost",
        "cwd": "problem",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "smtwtp": {
        "display": "SMTWTP",
        "sizes": [50, 100, 500],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average weighted tardiness",
        "cwd": "problem",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "rcpsp": {
        "display": "RCPSP",
        "sizes": [30, 60, 120],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "min",
        "metric": "average makespan",
        "cwd": "problem",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
    "mkp": {
        "display": "MKP",
        "sizes": [300, 500],
        "t_aco": [1, 10, 20, 30, 40, 50, 100],
        "n_ants": 20,
        "direction": "max",
        "metric": "average objective",
        "cwd": "root",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO-GNN", "aco": "ACO"},
    },
    "mkp_transformer": {
        "display": "MKP-Transformer",
        "sizes": [300, 500],
        "t_aco": [1, 5, 10, 20, 50],
        "n_ants": 20,
        "direction": "max",
        "metric": "average objective",
        "cwd": "root",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO-Transformer", "aco": "ACO"},
    },
    "bpp": {
        "display": "BPP",
        "sizes": [120],
        "t_aco": [1, 5, 10, 20],
        "n_ants": 20,
        "direction": "max",
        "metric": "average fitness",
        "cwd": "problem",
        "device": "cpu",
        "methods": ["deepaco", "aco"],
        "labels": {"deepaco": "DeepACO", "aco": "ACO"},
    },
}


def ensure_output_dirs() -> None:
    for path in [LOG_DIR, RAW_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def collect_environment(device_used: str) -> Dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
        "device_used": device_used,
        "cwd": str(ROOT),
    }


def resolve_device(requested: str, spec_default: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        if spec_default == "auto_cuda" and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
    return requested


@contextlib.contextmanager
def problem_context(problem: str, cwd_mode: str):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in PROBLEM_MODULES}
    for name in PROBLEM_MODULES:
        sys.modules.pop(name, None)

    problem_dir = ROOT / problem
    sys.path.insert(0, str(problem_dir))
    os.chdir(ROOT if cwd_mode == "root" else problem_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name in PROBLEM_MODULES:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diff_iterations(t_aco: Sequence[int]) -> List[int]:
    points = [0] + list(t_aco)
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def limited_dataset(dataset: Any, limit_instances: Optional[int]) -> Any:
    if limit_instances is None:
        return dataset
    return dataset[:limit_instances]


def values_from_tensor(tensor: torch.Tensor) -> List[float]:
    return [float(x) for x in tensor.detach().cpu().double().tolist()]


@torch.no_grad()
def test_dataset(
    dataset: Iterable[Any],
    infer_instance: Callable[[Any, Sequence[int]], torch.Tensor],
    t_aco: Sequence[int],
) -> Dict[str, Any]:
    t_aco_diff = diff_iterations(t_aco)
    sum_results = torch.zeros(size=(len(t_aco_diff),), dtype=torch.float64)
    start = time.time()
    count = 0
    for instance in dataset:
        results = infer_instance(instance, t_aco_diff)
        if not isinstance(results, torch.Tensor):
            results = torch.as_tensor(results)
        sum_results += results.detach().cpu().double()
        count += 1
    duration = time.time() - start
    if count == 0:
        raise ValueError("Dataset is empty after applying the instance limit.")
    return {
        "values": values_from_tensor(sum_results / count),
        "duration_seconds": duration,
        "instances": count,
    }


def load_model(model_cls: Callable[[], torch.nn.Module], checkpoint: Path, device: str) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = model_cls().to(device)
    model.load_state_dict(torch.load(str(checkpoint), map_location=device))
    model.eval()
    return model


def base_run(
    problem: str,
    spec: Dict[str, Any],
    size: int,
    method: str,
    t_aco: Sequence[int],
    dataset_size: int,
    used_instances: int,
    device: str,
    checkpoint: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "problem": problem,
        "problem_display": spec["display"],
        "scale": size,
        "scale_label": f"j{size}" if problem == "rcpsp" else str(size),
        "method": method,
        "method_label": spec["labels"].get(method, method),
        "direction": spec["direction"],
        "metric": spec["metric"],
        "t_aco": list(t_aco),
        "dataset_instances": dataset_size,
        "instances": used_instances,
        "n_ants": spec["n_ants"],
        "device": device,
        "checkpoint": rel(checkpoint) if checkpoint else None,
        "dataset_path": rel(dataset_path) if dataset_path else None,
    }


def finish_ok(run: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    run.update(result)
    run["status"] = "ok"
    return run


def finish_error(run: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    run["status"] = "error"
    run["error"] = f"{type(exc).__name__}: {exc}"
    run["traceback"] = traceback.format_exc()
    return run


def append_evaluation(
    runs: List[Dict[str, Any]],
    run: Dict[str, Any],
    dataset: Iterable[Any],
    infer_instance: Callable[[Any, Sequence[int]], torch.Tensor],
    t_aco: Sequence[int],
    log: Callable[[str], None],
) -> None:
    try:
        result = test_dataset(dataset, infer_instance, t_aco)
    except Exception as exc:
        log(f"Error in {run['problem_display']} {run['scale_label']} {run['method_label']}: {type(exc).__name__}: {exc}")
        runs.append(finish_error(run, exc))
    else:
        log(
            f"Finished {run['problem_display']} {run['scale_label']} {run['method_label']} "
            f"in {result['duration_seconds']:.2f}s"
        )
        runs.append(finish_ok(run, result))


def _dataset_len(dataset: Any) -> int:
    return len(dataset)


def _run_tsp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        k_sparse = 10 if n_node == 20 else n_node // 10
        dataset_path = ROOT / "data" / "tsp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, k_sparse, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"TSP{n_node}: {used}/{dataset_size} instances, k_sparse={k_sparse}")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "tsp" / f"tsp{n_node}.pt" if method == "deepaco" else None
            run = base_run("tsp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                pyg_data, distances = instance
                if method == "deepaco":
                    assert model is not None
                    heu_vec = model(pyg_data)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO(n_ants=spec["n_ants"], heuristic=heu_mat, distances=distances, device=device)
                else:
                    aco = ACO(n_ants=spec["n_ants"], distances=distances, device=device)
                    aco.sparsify(k_sparse)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_tsp_nls(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        k_sparse = n_node // 10
        dataset_path = ROOT / "data" / "tsp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, k_sparse, device, start_node=0)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"TSP-NLS{n_node}: {used}/{dataset_size} instances, k_sparse={k_sparse}")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "tsp_nls" / f"tsp{n_node}.pt" if method == "deepaco" else None
            run = base_run("tsp_nls", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                pyg_data, distances = instance
                if method == "deepaco":
                    assert model is not None
                    heu_vec = model(pyg_data)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO(
                        n_ants=spec["n_ants"],
                        heuristic=heu_mat.cpu(),
                        distances=distances.cpu(),
                        device="cpu",
                        local_search="nls",
                    )
                else:
                    aco = ACO(n_ants=spec["n_ants"], distances=distances.cpu(), device="cpu", local_search="nls")
                    aco.sparsify(k_sparse)
                results = torch.zeros(size=(len(t_aco_diff),))
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t, inference=True)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_cvrp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "cvrp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"CVRP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "cvrp" / f"cvrp{n_node}.pt" if method == "deepaco" else None
            run = base_run("cvrp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                demands, distances = instance
                if method == "deepaco":
                    assert model is not None
                    pyg_data = utils.gen_pyg_data(demands, distances, device)
                    heu_vec = model(pyg_data)
                    heu_mat = heu_vec.reshape((n_node + 1, n_node + 1)) + EPS
                    aco = ACO(distances=distances, demand=demands, n_ants=spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(distances=distances, demand=demands, n_ants=spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def ensure_cvrp_nls_dataset(n_node: int, log: Callable[[str], None]) -> Path:
    path = ROOT / "data" / "cvrp_nls" / f"testDataset-{n_node}.pt"
    if path.is_file():
        return path
    import utils

    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Generating missing CVRP-NLS dataset: {rel(path)}")
    set_reproducibility(123456)
    inst_list = []
    for _ in range(100):
        demand, dist, position = utils.gen_instance(n_node, "cpu", True)
        inst_list.append(torch.vstack([demand, position.T, dist]))
    torch.save(torch.stack(inst_list), str(path))
    return path


def _validate_cvrp_route(distance: torch.Tensor, demands: torch.Tensor, routes: List[torch.Tensor]) -> float:
    length = 0.0
    visited = {0}
    for route in routes:
        if demands[route].sum().item() > 1.000001:
            raise ValueError("Invalid CVRP route: capacity exceeded.")
        length += distance[route[:-1], route[1:]].sum().item()
        for node in route:
            idx = int(node.item())
            if idx < 0 or idx >= distance.size(0):
                raise ValueError("Invalid CVRP route: node index out of range.")
            visited.add(idx)
    if len(visited) != distance.size(0):
        raise ValueError("Invalid CVRP route: not all customers are visited.")
    return length


def _run_cvrp_nls(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO, get_subroutes
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ensure_cvrp_nls_dataset(n_node, log)
        k_sparse = n_node // 10
        dataset = utils.load_test_dataset(n_node, k_sparse, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"CVRP-NLS{n_node}: {used}/{dataset_size} instances, k_sparse={k_sparse}")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "cvrp_nls" / f"cvrp{n_node}.pt" if method == "deepaco" else None
            run = base_run("cvrp_nls", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                pyg_data, demands, distances, positions = instance
                if method == "deepaco":
                    assert model is not None
                    heu_vec = model(pyg_data)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO(
                        n_ants=spec["n_ants"],
                        heuristic=heu_mat.cpu(),
                        demand=demands.cpu(),
                        distances=distances.cpu(),
                        device="cpu",
                        swapstar=True,
                        positions=positions.cpu(),
                        inference=True,
                    )
                else:
                    aco = ACO(
                        n_ants=spec["n_ants"],
                        demand=demands.cpu(),
                        distances=distances.cpu(),
                        device="cpu",
                        swapstar=True,
                        positions=positions.cpu(),
                        inference=True,
                    )
                results = torch.zeros(size=(len(t_aco_diff),))
                for i, t in enumerate(t_aco_diff):
                    aco.run(t, inference=True)
                    routes = get_subroutes(aco.shortest_path)
                    results[i] = _validate_cvrp_route(distances, demands, routes)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_op(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    max_len = {100: 4, 200: 5, 300: 6}
    sparse_table = {100: 20, 200: 50, 300: 50}
    runs = []
    for n_node in sizes:
        k_sparse = sparse_table[n_node]
        dataset_path = ROOT / "data" / "op" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, k_sparse, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"OP{n_node}: {used}/{dataset_size} instances, k_sparse={k_sparse}")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "op" / f"op{n_node}.pt" if method == "deepaco" else None
            run = base_run("op", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                pyg_data, distances, prizes = instance
                if method == "deepaco":
                    assert model is not None
                    heu_mat = model.reshape(pyg_data, model(pyg_data)) + EPS
                    aco = ACO(distances, prizes, max_len[len(prizes)], spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(distances, prizes, max_len[len(prizes)], spec["n_ants"], device=device, k_sparse=k_sparse)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    best_obj, _ = aco.run(t)
                    results[i] = best_obj
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_pctsp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "pctsp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"PCTSP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "pctsp" / f"pctsp{n_node}.pt" if method == "deepaco" else None
            run = base_run("pctsp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                dist_mat, prizes, penalties = instance
                if method == "deepaco":
                    assert model is not None
                    pyg_data = utils.gen_pyg_data(prizes, penalties, dist_mat)
                    heu_mat = model(pyg_data)
                    heu_mat = (heu_mat / (heu_mat.min() + EPS) + EPS).reshape(prizes.size(0), prizes.size(0))
                    aco = ACO(dist_mat, prizes, penalties, spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(dist_mat, prizes, penalties, spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    best_cost, _ = aco.run(t)
                    results[i] = best_cost
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_sop(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "sop" / f"test{n_node}.pkl"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"SOP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "sop" / f"sop{n_node}.pt" if method == "deepaco" else None
            run = base_run("sop", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                distances, adj_mat, prec_cons = instance
                if method == "deepaco":
                    assert model is not None
                    pyg_data = utils.gen_pyg_data(distances, adj_mat, device)
                    heu_vec = model(pyg_data)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO(distances=distances, prec_cons=prec_cons, n_ants=spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(distances=distances, prec_cons=prec_cons, n_ants=spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_smtwtp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "smtwtp" / f"test{n_node}.pkl"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"SMTWTP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "smtwtp" / f"smtwtp{n_node}.pt" if method == "deepaco" else None
            run = base_run("smtwtp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                pyg_data, due_time, weights, processing_time = instance
                if method == "deepaco":
                    assert model is not None
                    heu_vec = model(pyg_data)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO(
                        due_time=due_time,
                        weights=weights,
                        processing_time=processing_time,
                        n_ants=spec["n_ants"],
                        heuristic=heu_mat,
                        device=device,
                    )
                else:
                    aco = ACO(
                        due_time=due_time,
                        weights=weights,
                        processing_time=processing_time,
                        n_ants=spec["n_ants"],
                        device=device,
                    )
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_rcpsp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    from aco import ACO_RCPSP
    from net import Net
    from rcpsp_inst import load_dataset

    acoparam = {"elitist": True, "min_max": True}
    runs = []
    for n_node in sizes:
        dataset_dir = ROOT / "data" / "rcpsp" / f"j{n_node}rcp"
        _, dataset = load_dataset(str(dataset_dir))
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"RCPSP j{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "rcpsp" / f"rcpsp{n_node}-5.pt" if method == "deepaco" else None
            run = base_run("rcpsp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_dir)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                if method == "deepaco":
                    assert model is not None
                    pyg_data = instance.to_pyg_data(device=device)
                    _, heu_vec = model(pyg_data, require_phe=True, require_heu=True)
                    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
                    aco = ACO_RCPSP(
                        instance,
                        n_ants=spec["n_ants"],
                        pheromone=None,
                        heuristic=heu_mat,
                        device=device,
                        **acoparam,
                    )
                else:
                    aco = ACO_RCPSP(instance, n_ants=spec["n_ants"], device=device, **acoparam)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t).cost
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_mkp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "mkp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"MKP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "mkp" / f"mkp{n_node}.pt" if method == "deepaco" else None
            run = base_run("mkp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                prize, weight = instance
                if method == "deepaco":
                    assert model is not None
                    src = utils.gen_pyg_data(prize, weight)
                    heu_mat = model(src).reshape((prize.size(0), prize.size(0)))
                    heu_mat = heu_mat / (heu_mat.min() + EPS) + EPS
                    aco = ACO(prize=prize, weight=weight, n_ants=spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(prize=prize, weight=weight, n_ants=spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    best_obj, _ = aco.run(t)
                    results[i] = best_obj
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def ensure_mkp_transformer_dataset(n_node: int, log: Callable[[str], None]) -> Path:
    path = ROOT / "data" / "mkp_transformer" / f"testDataset-{n_node}.pt"
    if path.is_file():
        return path
    import utils

    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Generating missing MKP-Transformer dataset: {rel(path)}")
    set_reproducibility(123456)
    instances = []
    for _ in range(100):
        price, weight = utils.gen_instance(n_node, 5)
        instances.append(torch.cat((price.unsqueeze(0), weight), dim=0))
    torch.save(torch.stack(instances), str(path))
    return path


def _run_mkp_transformer(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import TransformerModel

    runs = []
    for n_node in sizes:
        dataset_path = ensure_mkp_transformer_dataset(n_node, log)
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"MKP-Transformer{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "mkp_transformer" / f"mkp{n_node}.pt" if method == "deepaco" else None
            run = base_run("mkp_transformer", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                price, weight = instance
                if method == "deepaco":
                    assert model is not None
                    src = utils.reformat(price, weight)
                    heu_vec = model(src) + EPS
                    aco = ACO(price=price, weight=weight, n_ants=spec["n_ants"], heuristic=heu_vec, device=device)
                else:
                    aco = ACO(price=price, weight=weight, n_ants=spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    best_obj, _ = aco.run(t)
                    results[i] = best_obj
                return results

            model = load_model(TransformerModel, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


def _run_bpp(
    spec: Dict[str, Any],
    sizes: Sequence[int],
    methods: Sequence[str],
    t_aco: Sequence[int],
    limit_instances: Optional[int],
    device: str,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    import utils
    from aco import ACO
    from net import Net

    runs = []
    for n_node in sizes:
        dataset_path = ROOT / "data" / "bpp" / f"testDataset-{n_node}.pt"
        dataset = utils.load_test_dataset(n_node, device)
        dataset_size = _dataset_len(dataset)
        dataset = limited_dataset(dataset, limit_instances)
        used = _dataset_len(dataset)
        log(f"BPP{n_node}: {used}/{dataset_size} instances")

        for method in methods:
            checkpoint = ROOT / "pretrained" / "bpp" / f"bpp{n_node}.pt" if method == "deepaco" else None
            run = base_run("bpp", spec, n_node, method, t_aco, dataset_size, used, device, checkpoint, dataset_path)

            def infer(instance: Any, t_aco_diff: Sequence[int]) -> torch.Tensor:
                demands = instance
                if method == "deepaco":
                    assert model is not None
                    pyg_data = utils.gen_pyg_data(demands, device)
                    heu_vec = model(pyg_data)
                    heu_mat = heu_vec.reshape((n_node + 1, n_node + 1)) + EPS
                    aco = ACO(demand=demands, n_ants=spec["n_ants"], heuristic=heu_mat, device=device)
                else:
                    aco = ACO(demand=demands, n_ants=spec["n_ants"], device=device)
                results = torch.zeros(size=(len(t_aco_diff),), device=device)
                for i, t in enumerate(t_aco_diff):
                    results[i] = aco.run(t)
                return results

            model = load_model(Net, checkpoint, device) if method == "deepaco" else None
            append_evaluation(runs, run, dataset, infer, t_aco, log)
    return runs


RUNNERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "tsp": _run_tsp,
    "tsp_nls": _run_tsp_nls,
    "cvrp": _run_cvrp,
    "cvrp_nls": _run_cvrp_nls,
    "op": _run_op,
    "pctsp": _run_pctsp,
    "sop": _run_sop,
    "smtwtp": _run_smtwtp,
    "rcpsp": _run_rcpsp,
    "mkp": _run_mkp,
    "mkp_transformer": _run_mkp_transformer,
    "bpp": _run_bpp,
}


def normalize_int_sequence(value: Any, default: Sequence[int], name: str) -> List[int]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, Integral):
        return [int(value)]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
        return [int(item) for item in items] or list(default)
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an int or an iterable of ints, got {type(value).__name__}") from exc
    return [int(item) for item in items] or list(default)


def normalize_method_sequence(methods: Any) -> Optional[List[str]]:
    if methods is None or methods == "":
        return None
    if isinstance(methods, str):
        return [part.strip().lower() for part in methods.split(",") if part.strip()]
    return [str(method).strip().lower() for method in methods if str(method).strip()]


def select_methods(spec: Dict[str, Any], methods: Optional[Sequence[str]], include_baseline: bool) -> List[str]:
    normalized_methods = normalize_method_sequence(methods)
    if normalized_methods:
        selected = normalized_methods
    else:
        selected = list(spec["methods"])
    if not include_baseline:
        selected = [method for method in selected if method != "aco"]
    invalid = [method for method in selected if method not in spec["labels"]]
    if invalid:
        raise ValueError(f"Unknown methods for {spec['display']}: {invalid}")
    return selected


def markdown_table(raw: Dict[str, Any]) -> str:
    spec = PROBLEMS[raw["problem"]]
    t_aco = raw["t_aco"]
    direction_text = "lower is better" if spec["direction"] == "min" else "higher is better"
    lines = [
        f"# {spec['display']} Reproduction Table",
        "",
        f"- Metric: {spec['metric']} ({direction_text})",
        f"- Instances per completed row: see `instances` column",
        f"- Device: {raw['environment']['device_used']}",
        "",
    ]
    header = ["scale", "method", "status", "instances", "duration_s"] + [f"T={t}" for t in t_aco] + ["gap_vs_aco_last"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    baseline_by_scale: Dict[Any, Dict[str, Any]] = {}
    for run in raw["runs"]:
        if run["method"] == "aco" and run["status"] == "ok":
            baseline_by_scale[run["scale"]] = run

    for run in raw["runs"]:
        if run["status"] == "ok":
            values = [f"{value:.6g}" for value in run["values"]]
            duration = f"{run['duration_seconds']:.2f}"
            gap = ""
            baseline = baseline_by_scale.get(run["scale"])
            if run["method"] != "aco" and baseline and baseline.get("values"):
                deep = run["values"][-1]
                base = baseline["values"][-1]
                if base != 0:
                    gap_value = (deep - base) / abs(base) * 100.0
                    gap = f"{gap_value:+.2f}%"
            row = [
                run["scale_label"],
                run["method_label"],
                "ok",
                str(run["instances"]),
                duration,
                *values,
                gap,
            ]
        else:
            row = [
                run["scale_label"],
                run["method_label"],
                "error",
                str(run.get("instances", "")),
                "",
                *["" for _ in t_aco],
                "",
            ]
        lines.append("| " + " | ".join(row) + " |")

    error_runs = [run for run in raw["runs"] if run["status"] == "error"]
    if error_runs:
        lines.extend(["", "## Errors", ""])
        for run in error_runs:
            lines.append(f"- {run['scale_label']} {run['method_label']}: `{run['error']}`")
    lines.append("")
    return "\n".join(lines)


def write_figure(raw: Dict[str, Any], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = PROBLEMS[raw["problem"]]
    ok_runs = [run for run in raw["runs"] if run["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(10, 6))
    if ok_runs:
        for run in ok_runs:
            label = f"{run['scale_label']} {run['method_label']}"
            ax.plot(run["t_aco"], run["values"], marker="o", linewidth=1.8, markersize=4, label=label)
        ax.set_xlabel("ACO iterations (T)")
        better = "lower is better" if spec["direction"] == "min" else "higher is better"
        ax.set_ylabel(f"{spec['metric']} ({better})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "No successful runs. See the log and raw JSON.", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(f"{spec['display']} reproduction")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def write_outputs(raw: Dict[str, Any]) -> Dict[str, str]:
    problem = raw["problem"]
    raw_path = RAW_DIR / f"{problem}_results.json"
    table_path = TABLE_DIR / f"{problem}_table.md"
    figure_path = FIGURE_DIR / f"{problem}_comparison.png"

    table_path.write_text(markdown_table(raw), encoding="utf-8")
    write_figure(raw, figure_path)

    outputs = {
        "raw": rel(raw_path),
        "table": rel(table_path),
        "figure": rel(figure_path),
        "log": rel(LOG_DIR / f"{problem}.log"),
    }
    raw["outputs"] = outputs
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return outputs


def run_problem(
    problem: str,
    sizes: Optional[Sequence[int]] = None,
    t_aco: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    limit_instances: Optional[int] = None,
    device: str = "auto",
    seed: int = 12345,
    include_baseline: bool = True,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    ensure_output_dirs()
    problem = problem.lower()
    if problem not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem}'. Available: {', '.join(PROBLEMS)}")

    spec = PROBLEMS[problem]
    selected_sizes = normalize_int_sequence(sizes, spec["sizes"], "sizes")
    selected_t_aco = normalize_int_sequence(t_aco, spec["t_aco"], "t_aco")
    selected_methods = select_methods(spec, methods, include_baseline)
    resolved_device = resolve_device(device, spec["device"])

    raw: Dict[str, Any] = {
        "problem": problem,
        "problem_display": spec["display"],
        "metric": spec["metric"],
        "direction": spec["direction"],
        "sizes": selected_sizes,
        "t_aco": selected_t_aco,
        "methods": selected_methods,
        "seed": seed,
        "limit_instances": limit_instances,
        "environment": collect_environment(resolved_device),
        "runs": [],
    }

    log_path = LOG_DIR / f"{problem}.log"

    def log(message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        print(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log_path.write_text("", encoding="utf-8")
    log(f"Starting {spec['display']} reproduction")
    log(f"sizes={selected_sizes}, t_aco={selected_t_aco}, methods={selected_methods}, device={resolved_device}")

    set_reproducibility(seed)
    try:
        with problem_context(problem, spec["cwd"]):
            raw["runs"] = RUNNERS[problem](
                spec,
                selected_sizes,
                selected_methods,
                selected_t_aco,
                limit_instances,
                resolved_device,
                log,
            )
    except Exception as exc:
        if not continue_on_error:
            raise
        log(f"Problem-level error: {type(exc).__name__}: {exc}")
        for size in selected_sizes:
            for method in selected_methods:
                run = base_run(problem, spec, size, method, selected_t_aco, 0, 0, resolved_device)
                raw["runs"].append(finish_error(run, exc))

    if not continue_on_error:
        error_runs = [run for run in raw["runs"] if run.get("status") == "error"]
        if error_runs:
            raise RuntimeError(error_runs[0]["error"])

    outputs = write_outputs(raw)
    log(f"Wrote raw results: {outputs['raw']}")
    log(f"Wrote table: {outputs['table']}")
    log(f"Wrote figure: {outputs['figure']}")
    return raw


def run_all(
    limit_instances: Optional[int] = None,
    device: str = "auto",
    seed: int = 12345,
    include_baseline: bool = True,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    ensure_output_dirs()
    all_runs = []
    by_problem = {}
    for problem in PROBLEMS:
        raw = run_problem(
            problem,
            limit_instances=limit_instances,
            device=device,
            seed=seed,
            include_baseline=include_baseline,
            continue_on_error=continue_on_error,
        )
        by_problem[problem] = raw["outputs"]
        all_runs.extend(raw["runs"])

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "problems": by_problem,
        "runs": all_runs,
    }
    manifest_path = RAW_DIR / "all_results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# DeepACO Reproduction Summary",
        "",
        "| problem | table | figure | ok_runs | error_runs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for problem, outputs in by_problem.items():
        raw_path = ROOT / outputs["raw"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        ok = sum(1 for run in raw["runs"] if run["status"] == "ok")
        err = sum(1 for run in raw["runs"] if run["status"] == "error")
        summary_lines.append(
            f"| {raw['problem_display']} | [{outputs['table']}](../../{outputs['table']}) | "
            f"[{outputs['figure']}](../../{outputs['figure']}) | {ok} | {err} |"
        )
    summary_path = TABLE_DIR / "all_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest["summary_table"] = rel(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",")]


def parse_method_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if not value.strip():
        return None
    return [part.strip().lower() for part in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepACO pretrained reproduction experiments.")
    parser.add_argument("problem", nargs="?", choices=list(PROBLEMS), help="Problem folder to reproduce.")
    parser.add_argument("--all", action="store_true", help="Run every supported problem.")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated problem sizes, e.g. 20,100.")
    parser.add_argument("--t-aco", type=str, default=None, help="Comma-separated ACO iteration checkpoints.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods: deepaco,aco.")
    parser.add_argument("--limit-instances", type=int, default=None, help="Use only the first N test instances.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-baseline", action="store_true", help="Skip ACO baseline rows.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise the first error instead of writing error rows.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.problem:
        parser.error("Provide a problem name or --all.")

    if args.all:
        manifest = run_all(
            limit_instances=args.limit_instances,
            device=args.device,
            seed=args.seed,
            include_baseline=not args.no_baseline,
            continue_on_error=not args.fail_fast,
        )
        print(json.dumps({"raw": "results/raw/all_results.json", "summary": manifest["summary_table"]}, ensure_ascii=False))
        return 0

    raw = run_problem(
        args.problem,
        sizes=parse_int_list(args.sizes),
        t_aco=parse_int_list(args.t_aco),
        methods=parse_method_list(args.methods),
        limit_instances=args.limit_instances,
        device=args.device,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        continue_on_error=not args.fail_fast,
    )
    print(json.dumps(raw["outputs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
