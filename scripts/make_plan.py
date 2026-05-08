#!/usr/bin/env python3
"""
scripts/make_plan.py

Balanced two-phase task planner.

Key change:
  compute_units != number_of_shards.

For 16 available compute units, the default is now:
  phase1_compute_units = 16
  phase2_compute_units = 16
  phase1_num_tasks     = 8
  phase2_num_tasks     = 8
  threads_per_task     = 2

This gives fewer, stronger shards instead of many underpowered tiny shards.

Environment overrides:
  PHASE1_COMPUTE_UNITS=16
  PHASE2_COMPUTE_UNITS=16
  PHASE1_NUM_TASKS=8
  PHASE2_NUM_TASKS=8
  PHASE1_THREADS_PER_TASK=2
  PHASE2_THREADS_PER_TASK=2

Config keys supported:
  phase1_compute_units, phase2_compute_units
  phase1_num_tasks, phase2_num_tasks
  phase1_threads_per_task, phase2_threads_per_task

Backward compatibility:
  If none of the above are set, num_tasks_total is split half/half as before.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PHASE1_METHODS = [
    "greedy",
    "rens",
    "local_search_lp",
    "dive_fix",
    "feasibility_pump",
]

PHASE2_METHODS = [
    "rins_warmstart",
    "local_branching_hamming",
    "lns_fix_optimize",
    "objective_bound_search",
    "polishing_mip",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_int(x: Any, default: int) -> int:
    try:
        val = int(x)
    except Exception:
        return int(default)
    return val


def positive_int(x: Any, default: int, lo: int = 1) -> int:
    return max(lo, as_int(x, default))


def env_int(name: str, default: int, lo: int = 0) -> int:
    if name not in os.environ:
        return max(lo, int(default))
    return max(lo, as_int(os.environ.get(name), default))


def first_present(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def objective_sense_from_features(features: Dict[str, Any]) -> str:
    for key in ("objective_sense", "sense", "model_sense"):
        if key in features:
            v = features[key]
            if isinstance(v, str):
                s = v.lower()
                if s.startswith("max"):
                    return "max"
                if s.startswith("min"):
                    return "min"
            if isinstance(v, (int, float)):
                return "max" if int(v) == -1 else "min"

    for parent in ("model", "features", "metadata"):
        sub = features.get(parent)
        if isinstance(sub, dict):
            s = objective_sense_from_features(sub)
            if s in ("min", "max"):
                return s

    return "min"


def feature_number(features: Dict[str, Any], names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in features:
            try:
                return float(features[name])
            except Exception:
                pass

    for parent in ("model", "features", "metadata"):
        sub = features.get(parent)
        if isinstance(sub, dict):
            val = feature_number(sub, names, default=None)
            if val is not None:
                return float(val)

    return float(default)


def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def feature_profile(features: Dict[str, Any]) -> Dict[str, float]:
    n_vars = feature_number(features, ("num_vars", "num_vars_est", "n_vars", "NumVars"), 0)
    n_bin = feature_number(features, ("num_bin", "num_bin_est", "num_binary", "num_binary_vars", "n_bin"), 0)
    n_int_nonbin = feature_number(features, ("num_int", "num_int_est", "num_integer_nonbinary", "n_int_nonbin"), 0)
    n_cons = feature_number(features, ("num_constrs", "num_constr_est", "num_constraints", "n_cons", "NumConstrs"), 0)
    n_nz = feature_number(features, ("num_nz", "num_nz_est", "nnz", "NumNZs"), 0)

    integrality_ratio = feature_number(features, ("integrality_ratio", "integrality_ratio_est"), -1)
    if integrality_ratio < 0:
        integrality_ratio = (n_bin + n_int_nonbin) / n_vars if n_vars > 0 else 0.0

    bin_ratio = feature_number(features, ("bin_ratio", "bin_ratio_est", "binary_ratio"), -1)
    if bin_ratio < 0:
        bin_ratio = n_bin / n_vars if n_vars > 0 else 0.0

    integer_total = max(0.0, n_bin + n_int_nonbin)
    integer_binary_share = n_bin / integer_total if integer_total > 0 else 0.0
    continuous_ratio = clamp_float(1.0 - integrality_ratio, 0.0, 1.0)

    density = feature_number(features, ("density", "density_est", "matrix_density"), -1)
    if density < 0:
        density = n_nz / (n_vars * n_cons) if n_vars > 0 and n_cons > 0 else 0.0

    avg_row_nz = n_nz / n_cons if n_cons > 0 else 0.0
    avg_col_nz = n_nz / n_vars if n_vars > 0 else 0.0
    size_score = clamp_float(math.log10(max(10.0, n_vars + n_cons)) / 6.0, 0.0, 1.0)

    return {
        "num_vars": n_vars,
        "num_binary": n_bin,
        "num_integer_nonbinary": n_int_nonbin,
        "num_integer_total": integer_total,
        "num_constraints": n_cons,
        "num_nonzeros": n_nz,
        "density": density,
        "avg_row_nonzeros": avg_row_nz,
        "avg_col_nonzeros": avg_col_nz,
        "bin_ratio": clamp_float(bin_ratio, 0.0, 1.0),
        "integrality_ratio": clamp_float(integrality_ratio, 0.0, 1.0),
        "integer_binary_share": clamp_float(integer_binary_share, 0.0, 1.0),
        "continuous_ratio": continuous_ratio,
        "size_score": size_score,
    }


def add_reason(reasons: Dict[str, List[str]], method: str, text: str) -> None:
    reasons.setdefault(method, []).append(text)


def weight_result(
    config: Dict[str, Any],
    methods: List[str],
    defaults: Dict[str, float],
    override_keys: Tuple[str, ...],
    reasons: Dict[str, List[str]],
) -> Dict[str, Any]:
    override = first_present(config, override_keys, None)
    weights = normalize_weights(override, methods, defaults)
    return {
        "weights": weights,
        "overridden": isinstance(override, dict),
        "reasons": {m: reasons.get(m, []) for m in methods},
    }


def cartesian_grid(param_grid: Any) -> List[Dict[str, Any]]:
    if not param_grid:
        return [{}]

    if isinstance(param_grid, list):
        return [dict(x) for x in param_grid if isinstance(x, dict)] or [{}]

    if not isinstance(param_grid, dict):
        return [{}]

    keys = sorted(param_grid.keys())
    vals = []
    for k in keys:
        v = param_grid[k]
        vals.append(v if isinstance(v, list) else [v])

    out = []
    for combo in itertools.product(*vals):
        out.append({k: combo[i] for i, k in enumerate(keys)})
    return out or [{}]


def method_param_grids(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grids: Dict[str, List[Dict[str, Any]]] = {}

    sources = []
    for key in ("method_params", "heuristic_params", "params_by_method", "heuristics", "grids"):
        if isinstance(config.get(key), dict):
            sources.append(config[key])

    if isinstance(config.get("phase1_params"), dict):
        sources.append(config["phase1_params"])
    if isinstance(config.get("phase2_params"), dict):
        sources.append(config["phase2_params"])

    if isinstance(config.get("methods"), dict):
        src = {}
        for m, spec in config["methods"].items():
            if isinstance(spec, dict):
                src[m] = spec.get("params", spec.get("grid", spec))
        sources.append(src)

    for src in sources:
        for method, grid in src.items():
            if method in PHASE1_METHODS or method in PHASE2_METHODS:
                grids[method] = cartesian_grid(grid)

    for m in PHASE1_METHODS + PHASE2_METHODS:
        grids.setdefault(m, [{}])

    return grids


def normalize_weights(raw: Any, methods: List[str], defaults: Dict[str, float]) -> Dict[str, float]:
    if not isinstance(raw, dict):
        raw = {}

    weights = {}
    for m in methods:
        try:
            w = float(raw.get(m, defaults.get(m, 1.0)))
        except Exception:
            w = defaults.get(m, 1.0)
        weights[m] = max(0.0, w)

    if sum(weights.values()) <= 0:
        weights = {m: defaults.get(m, 1.0) for m in methods}

    return weights


def allocate_counts(total_tasks: int, methods: List[str], weights: Dict[str, float], minimum_active: bool = True) -> Dict[str, int]:
    total_tasks = int(max(0, total_tasks))
    if total_tasks == 0:
        return {m: 0 for m in methods}

    active = [m for m in methods if weights.get(m, 0.0) > 0.0]
    if not active:
        active = list(methods)
        weights = {m: 1.0 for m in active}

    counts = {m: 0 for m in methods}

    if minimum_active:
        for m in active[:total_tasks]:
            counts[m] = 1
        remaining = total_tasks - sum(counts.values())
    else:
        remaining = total_tasks

    if remaining <= 0:
        return counts

    total_w = sum(weights[m] for m in active)
    fractional: List[Tuple[float, str]] = []
    assigned = 0

    for m in active:
        exact = remaining * weights[m] / total_w if total_w > 0 else remaining / len(active)
        base = int(exact)
        counts[m] += base
        assigned += base
        fractional.append((exact - base, m))

    leftover = remaining - assigned
    fractional.sort(reverse=True)
    for _, m in fractional[:leftover]:
        counts[m] += 1

    return counts


def structure_weights_phase1(config: Dict[str, Any], profile: Dict[str, float]) -> Dict[str, Any]:
    n_vars = profile["num_vars"]
    n_bin = profile["num_binary"]
    n_int_nonbin = profile["num_integer_nonbinary"]
    n_int_total = profile["num_integer_total"]
    n_cons = profile["num_constraints"]
    density = profile["density"]
    bin_ratio = profile["bin_ratio"]
    int_ratio = profile["integrality_ratio"]
    binary_share = profile["integer_binary_share"]
    continuous_ratio = profile["continuous_ratio"]
    avg_row_nz = profile["avg_row_nonzeros"]
    size_score = profile["size_score"]

    w = {
        "greedy": 1.0,
        "rens": 2.0,
        "local_search_lp": 2.0,
        "dive_fix": 1.0,
        "feasibility_pump": 2.0,
    }
    reasons: Dict[str, List[str]] = {}

    if n_vars >= 50000 or n_cons >= 50000:
        boost = 0.75 + size_score
        w["feasibility_pump"] += boost
        w["greedy"] += 0.50
        w["rens"] -= 0.25
        w["local_search_lp"] -= 0.25
        add_reason(reasons, "feasibility_pump", "large model favors cheap randomized feasibility attempts")
        add_reason(reasons, "greedy", "large model favors inexpensive LP-guided starts")

    if bin_ratio >= 0.55 or binary_share >= 0.75:
        w["rens"] += 0.75
        w["local_search_lp"] += 0.50
        w["feasibility_pump"] += 0.75
        add_reason(reasons, "rens", "binary-heavy model makes LP agreement neighborhoods attractive")
        add_reason(reasons, "local_search_lp", "binary-heavy model gives useful local branching/fix-top neighborhoods")
        add_reason(reasons, "feasibility_pump", "binary-heavy model benefits from diversified rounding/projection starts")

    if n_int_nonbin > 0 and binary_share < 0.25:
        w["dive_fix"] += 1.00
        w["greedy"] += 0.50
        w["feasibility_pump"] -= 0.25
        add_reason(reasons, "dive_fix", "general-integer model benefits from progressive LP-guided fixing")
        add_reason(reasons, "greedy", "general-integer model benefits from rounded LP starts")

    if 0 < density < 0.005:
        w["greedy"] += 0.50
        w["dive_fix"] += 0.25
        add_reason(reasons, "greedy", "sparse matrix makes cheap ranking and repair attractive")
        add_reason(reasons, "dive_fix", "sparse matrix supports incremental fixing")

    if density >= 0.02 or avg_row_nz >= 200:
        w["rens"] += 0.50
        w["local_search_lp"] += 0.50
        w["dive_fix"] -= 0.25
        add_reason(reasons, "rens", "denser coupling favors restricted sub-MIP neighborhoods")
        add_reason(reasons, "local_search_lp", "denser coupling favors bounded neighborhood search")

    if int_ratio < 0.35 and continuous_ratio > 0.65:
        w["rens"] += 0.50
        w["local_search_lp"] += 0.75
        w["feasibility_pump"] -= 0.25
        add_reason(reasons, "rens", "mostly continuous model can exploit LP relaxation agreement")
        add_reason(reasons, "local_search_lp", "mostly continuous model makes LP-seeded neighborhoods useful")

    if n_int_total <= 2000 and n_int_total > 0:
        w["rens"] += 0.50
        w["local_search_lp"] += 0.50
        add_reason(reasons, "rens", "small integer core makes restricted neighborhoods affordable")
        add_reason(reasons, "local_search_lp", "small integer core makes local sub-MIPs affordable")

    return weight_result(config, PHASE1_METHODS, w, ("phase1_method_weights", "phase1_weights"), reasons)


def structure_weights_phase2(config: Dict[str, Any], profile: Dict[str, float]) -> Dict[str, Any]:
    n_vars = profile["num_vars"]
    n_bin = profile["num_binary"]
    n_int_total = profile["num_integer_total"]
    density = profile["density"]
    bin_ratio = profile["bin_ratio"]
    binary_share = profile["integer_binary_share"]
    avg_row_nz = profile["avg_row_nonzeros"]
    size_score = profile["size_score"]

    w = {
        "rins_warmstart": 2.0,
        "local_branching_hamming": 3.0,
        "lns_fix_optimize": 3.0,
        "objective_bound_search": 1.0,
        "polishing_mip": 2.0,
    }
    reasons: Dict[str, List[str]] = {}

    if n_vars >= 50000:
        w["lns_fix_optimize"] += 0.75 + size_score
        w["polishing_mip"] += 0.5
        add_reason(reasons, "lns_fix_optimize", "large model favors block neighborhoods")
        add_reason(reasons, "polishing_mip", "large model benefits from incumbent improvement settings")

    if n_bin <= 0:
        w["local_branching_hamming"] -= 1.25
        w["rins_warmstart"] += 0.50
        w["lns_fix_optimize"] += 0.50
        w["polishing_mip"] += 0.50
        add_reason(reasons, "local_branching_hamming", "no binary variables means no Hamming-ball constraint can be added")
        add_reason(reasons, "rins_warmstart", "general-integer incumbent is better served by agreement neighborhoods")
        add_reason(reasons, "lns_fix_optimize", "general-integer incumbent is better served by block fixing")
        add_reason(reasons, "polishing_mip", "general-integer incumbent can still benefit from global improvement settings")

    if bin_ratio >= 0.55 or binary_share >= 0.75:
        w["local_branching_hamming"] += 1.0
        w["rins_warmstart"] += 0.5
        add_reason(reasons, "local_branching_hamming", "binary-heavy incumbent is naturally searched by Hamming balls")
        add_reason(reasons, "rins_warmstart", "binary-heavy LP/incumbent agreement can define compact neighborhoods")

    if n_int_total <= 2000 and n_int_total > 0:
        w["rins_warmstart"] += 0.75
        w["objective_bound_search"] += 0.50
        add_reason(reasons, "rins_warmstart", "small integer core makes RINS-style sub-MIPs cheaper")
        add_reason(reasons, "objective_bound_search", "small integer core can afford improvement-threshold neighborhoods")

    if density >= 0.02 or avg_row_nz >= 200:
        w["polishing_mip"] += 0.50
        w["objective_bound_search"] += 0.50
        w["local_branching_hamming"] -= 0.25
        add_reason(reasons, "polishing_mip", "dense coupling favors global incumbent-improvement search")
        add_reason(reasons, "objective_bound_search", "dense coupling benefits from objective-bounded neighborhoods")

    if 0 < density < 0.005:
        w["lns_fix_optimize"] += 0.50
        add_reason(reasons, "lns_fix_optimize", "sparse matrix favors fixing most variables and freeing blocks")

    return weight_result(config, PHASE2_METHODS, w, ("phase2_method_weights", "phase2_weights"), reasons)


def compute_phase_resources(config: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    """
    Returns:
      p1_compute_units, p2_compute_units, p1_num_tasks, p2_num_tasks,
      p1_threads_per_task, p2_threads_per_task
    """
    # Backward-compatible old total task behavior.
    old_total = positive_int(config.get("num_tasks_total", config.get("array_tasks", 16)), 16, lo=1)

    p1_compute = env_int(
        "PHASE1_COMPUTE_UNITS",
        positive_int(first_present(config, ("phase1_compute_units", "phase1_cores"), 16), 16),
        lo=1,
    )
    p2_compute = env_int(
        "PHASE2_COMPUTE_UNITS",
        positive_int(first_present(config, ("phase2_compute_units", "phase2_cores"), 16), 16),
        lo=1,
    )

    # Crucial change: default task count is fewer than compute units.
    # If user only has old num_tasks_total, preserve old split.
    explicit_p1_tasks = first_present(config, ("phase1_num_tasks", "phase1_shards", "phase1_tasks_target"), None)
    explicit_p2_tasks = first_present(config, ("phase2_num_tasks", "phase2_shards", "phase2_tasks_target"), None)

    if explicit_p1_tasks is None and explicit_p2_tasks is None and "PHASE1_NUM_TASKS" not in os.environ and "PHASE2_NUM_TASKS" not in os.environ:
        # If config explicitly requested per-phase compute units, use 8-task default.
        if ("phase1_compute_units" in config or "phase2_compute_units" in config or
            "PHASE1_COMPUTE_UNITS" in os.environ or "PHASE2_COMPUTE_UNITS" in os.environ):
            p1_tasks_default = min(8, p1_compute)
            p2_tasks_default = min(8, p2_compute)
        else:
            # Old behavior: num_tasks_total means total planned tasks.
            p1_tasks_default = old_total // 2
            p2_tasks_default = old_total - p1_tasks_default
    else:
        p1_tasks_default = min(8, p1_compute)
        p2_tasks_default = min(8, p2_compute)

    p1_tasks = env_int("PHASE1_NUM_TASKS", positive_int(explicit_p1_tasks, p1_tasks_default), lo=0)
    p2_tasks = env_int("PHASE2_NUM_TASKS", positive_int(explicit_p2_tasks, p2_tasks_default), lo=0)

    p1_threads_default = max(1, p1_compute // max(1, p1_tasks))
    p2_threads_default = max(1, p2_compute // max(1, p2_tasks))

    p1_threads = env_int(
        "PHASE1_THREADS_PER_TASK",
        positive_int(first_present(config, ("phase1_threads_per_task", "phase1_task_threads"), p1_threads_default), p1_threads_default),
        lo=1,
    )
    p2_threads = env_int(
        "PHASE2_THREADS_PER_TASK",
        positive_int(first_present(config, ("phase2_threads_per_task", "phase2_task_threads"), p2_threads_default), p2_threads_default),
        lo=1,
    )

    return p1_compute, p2_compute, p1_tasks, p2_tasks, p1_threads, p2_threads


def task_params_for(method: str, grids: Dict[str, List[Dict[str, Any]]], shard_idx: int, threads_per_task: int) -> Dict[str, Any]:
    grid = grids.get(method) or [{}]
    params = dict(grid[shard_idx % len(grid)])
    params.setdefault("seed", 1000 + shard_idx)
    params.setdefault("shard_id", shard_idx)

    # Encourage existing heuristics/common wrappers to use the intended thread count
    # where they support it.
    params.setdefault("threads", threads_per_task)
    params.setdefault("mip_threads", threads_per_task)

    # LP-heavy methods should not starve their LPs.
    if method in ("rens", "local_search_lp", "dive_fix"):
        params.setdefault("lp_threads", threads_per_task)
        params.setdefault("lp_cap_max", 10.0)
        params.setdefault("lp_fraction", 0.30)

    return params


def build_tasks(
    phase: str,
    methods: List[str],
    counts: Dict[str, int],
    grids: Dict[str, List[Dict[str, Any]]],
    start_global_id: int,
    threads_per_task: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    local_id = 0
    global_id = start_global_id

    for method in methods:
        for shard_idx in range(counts.get(method, 0)):
            params = task_params_for(method, grids, shard_idx, threads_per_task)
            tasks.append({
                "task_id": global_id,
                "global_task_id": global_id,
                "phase_task_id": local_id,
                "local_task_id": local_id,
                "phase": phase,
                "method": method,
                "shard_id": shard_idx,
                "threads_per_task": threads_per_task,
                "params": params,
            })
            local_id += 1
            global_id += 1

    return tasks


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: python3 -m scripts.make_plan CONFIG_JSON FEATURES_JSON INSTANCE_PATH OUT_PLAN_JSON", file=sys.stderr)
        return 2

    config_path, features_path, instance_path, out_path = sys.argv[1:5]
    config = load_json(config_path)
    features = load_json(features_path)

    sense = objective_sense_from_features(features)

    p1_compute, p2_compute, p1_tasks_n, p2_tasks_n, p1_threads, p2_threads = compute_phase_resources(config)

    grids = method_param_grids(config)
    profile = feature_profile(features)
    p1_weight_info = structure_weights_phase1(config, profile)
    p2_weight_info = structure_weights_phase2(config, profile)
    p1_weights = p1_weight_info["weights"]
    p2_weights = p2_weight_info["weights"]

    p1_counts = allocate_counts(p1_tasks_n, PHASE1_METHODS, p1_weights, minimum_active=True)
    p2_counts = allocate_counts(p2_tasks_n, PHASE2_METHODS, p2_weights, minimum_active=True)

    phase1_tasks = build_tasks("phase1", PHASE1_METHODS, p1_counts, grids, 0, p1_threads)
    phase2_tasks = build_tasks("phase2", PHASE2_METHODS, p2_counts, grids, len(phase1_tasks), p2_threads)
    tasks = phase1_tasks + phase2_tasks

    plan = {
        "instance_path": instance_path,
        "features_path": features_path,
        "config_path": config_path,
        "objective_sense": sense,
        "sense": sense,
        "num_tasks_total": len(tasks),

        "phase1_compute_units": p1_compute,
        "phase2_compute_units": p2_compute,
        "phase1_num_tasks": len(phase1_tasks),
        "phase2_num_tasks": len(phase2_tasks),
        "phase1_threads_per_task": p1_threads,
        "phase2_threads_per_task": p2_threads,

        "phase1": p1_counts,
        "phase2": p2_counts,
        "phase1_method_weights": p1_weights,
        "phase2_method_weights": p2_weights,
        "structure_profile": profile,
        "allocation_diagnostics": {
            "phase1": p1_weight_info,
            "phase2": p2_weight_info,
        },
        "phase1_tasks": phase1_tasks,
        "phase2_tasks": phase2_tasks,
        "tasks": tasks,

        "planner": {
            "name": "balanced_two_phase_make_plan",
            "structure_aware": True,
            "rl_allocator": False,
            "notes": [
                "compute_units are cores/slots available per wave.",
                "num_tasks/shards can be smaller than compute_units to give each shard more time/threads.",
                "method weights are adjusted from lightweight instance structure unless phase-specific weights are overridden in config.",
                "Future RL allocator should preserve this plan schema.",
            ],
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, allow_nan=False)

    print(f"Wrote plan to {out}")
    print(f"Objective sense: {sense}")
    print(f"Total tasks: {len(tasks)}")
    print(json.dumps({"phase1": p1_counts, "phase2": p2_counts}, indent=2))
    print(f"Phase 1 tasks: {len(phase1_tasks)}")
    print(f"Phase 2 tasks: {len(phase2_tasks)}")
    print(f"Phase 1 compute units: {p1_compute}")
    print(f"Phase 2 compute units: {p2_compute}")
    print(f"Phase 1 threads/task: {p1_threads}")
    print(f"Phase 2 threads/task: {p2_threads}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
