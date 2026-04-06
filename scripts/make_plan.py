#!/usr/bin/env python3
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

METHODS = ["greedy", "rens", "local_search_lp", "dive_fix"]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def cartesian_grid(grid_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def split_evenly(items: List[Any], num_buckets: int) -> List[List[Any]]:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    buckets = [[] for _ in range(num_buckets)]
    for i, item in enumerate(items):
        buckets[i % num_buckets].append(item)
    return buckets


def choose_local_search_heavy_allocation(total_tasks: int) -> Dict[str, int]:
    """
    For the current research goal, prioritize local_search_lp so that with
    farm_seconds=160 and total_tasks=16, each local-search candidate gets ~5 sec.

    Default target at 16 tasks:
      greedy: 1
      rens: 2
      local_search_lp: 12
      dive_fix: 1
    """
    if total_tasks == 16:
        return {
            "greedy": 1,
            "rens": 2,
            "local_search_lp": 12,
            "dive_fix": 1,
        }

    # Fallback for other task counts:
    # reserve 75% for local search, distribute the rest minimally.
    ls = max(1, int(round(0.75 * total_tasks)))
    rem = total_tasks - ls

    alloc = {
        "greedy": 1,
        "rens": 1,
        "local_search_lp": ls,
        "dive_fix": 1,
    }

    # Adjust if we overshot
    while sum(alloc.values()) > total_tasks:
        # trim from local_search first if needed
        if alloc["local_search_lp"] > 1:
            alloc["local_search_lp"] -= 1
        elif alloc["rens"] > 1:
            alloc["rens"] -= 1
        elif alloc["greedy"] > 1:
            alloc["greedy"] -= 1
        elif alloc["dive_fix"] > 1:
            alloc["dive_fix"] -= 1
        else:
            break

    # Adjust if short
    while sum(alloc.values()) < total_tasks:
        alloc["local_search_lp"] += 1

    return alloc


def score_local_search_combo(p: Dict[str, Any]) -> tuple:
    """
    Rank local_search_lp parameter settings so we keep the strongest-looking
    ones when we trim the grid.

    We deliberately prefer:
    - move_policy == '1flip'
    - neighborhood_size around 16
    - larger submip_nodes
    - stable deterministic ordering by seed
    """
    move_policy = str(p.get("move_policy", ""))
    neighborhood_size = int(p.get("neighborhood_size", 0))
    submip_nodes = int(p.get("submip_nodes", 0))
    seed = int(p.get("seed", 0))

    move_bonus = 2 if move_policy == "1flip" else 0
    neigh_bonus = -abs(neighborhood_size - 16)  # best when close to 16
    node_bonus = submip_nodes
    # deterministic tie-break on lower seed first
    return (move_bonus, neigh_bonus, node_bonus, -seed)


def trim_local_search_grid(full_grid: List[Dict[str, Any]], alloc_ls_tasks: int) -> List[Dict[str, Any]]:
    """
    Under farm_seconds=160 and total_tasks=16, each task gets ~10 sec.
    To mimic local_search_lp 5 sec per parameter point, each shard should have 2 candidates.
    So we cap local_search candidates to 2 * (# local_search tasks).
    """
    target_candidates = 2 * alloc_ls_tasks
    if len(full_grid) <= target_candidates:
        return full_grid

    ranked = sorted(full_grid, key=score_local_search_combo, reverse=True)
    return ranked[:target_candidates]


def build_plan(config: Dict[str, Any], features: Dict[str, Any], instance_path: str) -> Dict[str, Any]:
    total_tasks = int(config["allocation"]["total_tasks"])
    grids = dict(config["grids"])

    alloc = choose_local_search_heavy_allocation(total_tasks)

    plan = {
        "instance_path": instance_path,
        "features": features,
        "global": config["global"],
        "objective_sense": features.get("objective_sense", "min"),
        "allocation": alloc,
        "tasks": [],
        "planning_notes": [
            "Local-search-heavy plan to match ~5 sec per local_search candidate when farm_seconds=160 and total_tasks=16.",
            "This deliberately trims the local_search grid and gives it most of the farm tasks.",
        ],
    }

    task_id = 0
    for method in METHODS:
        full_grid = cartesian_grid(grids[method])

        if method == "local_search_lp":
            full_grid = trim_local_search_grid(full_grid, alloc["local_search_lp"])

        num_shards = int(alloc[method])
        shards = split_evenly(full_grid, num_shards)

        for shard_idx, shard in enumerate(shards):
            plan["tasks"].append({
                "task_id": task_id,
                "method": method,
                "method_task_index": shard_idx,
                "num_method_tasks": num_shards,
                "grid_size_total": len(full_grid),
                "grid_size_local": len(shard),
                "params_list": shard,
            })
            task_id += 1

    return plan


def main():
    if len(sys.argv) != 5:
        print("Usage: make_plan.py <config.json> <features.json> <instance.mps> <plan.json>")
        sys.exit(1)

    config_path, features_path, instance_path, out_plan_path = sys.argv[1:5]

    config = load_json(config_path)
    features = load_json(features_path)

    plan = build_plan(config, features, instance_path)
    dump_json(plan, out_plan_path)

    print(f"Wrote plan to {out_plan_path}")
    print(f"Objective sense: {plan['objective_sense']}")
    print(f"Total tasks: {len(plan['tasks'])}")
    print("Allocation:")
    for k, v in plan["allocation"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()