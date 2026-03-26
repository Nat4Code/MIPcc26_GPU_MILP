#!/usr/bin/env python3
import itertools
import json
import math
import os
import sys
from typing import Dict, List, Any

METHODS = ["greedy", "rens", "local_search_lp", "dive_fix"]

def cartesian_grid(grid_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out

def split_evenly(items: List[Any], num_buckets: int) -> List[List[Any]]:
    buckets = [[] for _ in range(num_buckets)]
    for i, item in enumerate(items):
        buckets[i % num_buckets].append(item)
    return buckets

def heuristic_task_allocator(features: Dict[str, Any], total_tasks: int, default_split: Dict[str, int]) -> Dict[str, int]:
    """
    Hand-coded placeholder.
    Later replace this with ML or learned allocation.
    """
    alloc = default_split.copy()

    # Simple example rules:
    bin_ratio = features.get("bin_ratio_est", 0.25)
    nbin = features.get("num_bin_est", 100)
    ncon = features.get("num_constr_est", 100)

    # Start from default.
    # Shift emphasis based on rough structure.
    if bin_ratio > 0.4:
        alloc["rens"] += 2
        alloc["greedy"] -= 1
        alloc["local_search_lp"] -= 1

    if nbin > 500:
        alloc["dive_fix"] += 2
        alloc["greedy"] -= 1
        alloc["rens"] -= 1

    if ncon < 200:
        alloc["greedy"] += 1
        alloc["local_search_lp"] += 1
        alloc["dive_fix"] -= 1
        alloc["rens"] -= 1

    # Repair negatives.
    for k in alloc:
        if alloc[k] < 1:
            alloc[k] = 1

    # Normalize to total_tasks.
    s = sum(alloc.values())
    if s != total_tasks:
        scale = total_tasks / s
        scaled = {k: max(1, int(round(v * scale))) for k, v in alloc.items()}
        # Fix rounding drift
        while sum(scaled.values()) < total_tasks:
            k = min(scaled, key=lambda x: scaled[x])
            scaled[k] += 1
        while sum(scaled.values()) > total_tasks:
            k = max(scaled, key=lambda x: scaled[x])
            if scaled[k] > 1:
                scaled[k] -= 1
            else:
                break
        alloc = scaled

    return alloc

def build_plan(config: Dict[str, Any], features: Dict[str, Any], instance_path: str) -> Dict[str, Any]:
    total_tasks = config["allocation"]["total_tasks"]
    default_split = config["allocation"]["default_split"]
    grids = config["grids"]

    alloc = heuristic_task_allocator(features, total_tasks, default_split)

    plan = {
        "instance_path": instance_path,
        "features": features,
        "global": config["global"],
        "allocation": alloc,
        "tasks": []
    }

    task_id = 0
    for method in METHODS:
        full_grid = cartesian_grid(grids[method])
        num_shards = alloc[method]
        shards = split_evenly(full_grid, num_shards)

        for shard_idx, shard in enumerate(shards):
            plan["tasks"].append({
                "task_id": task_id,
                "method": method,
                "method_task_index": shard_idx,
                "num_method_tasks": num_shards,
                "grid_size_total": len(full_grid),
                "grid_size_local": len(shard),
                "params_list": shard
            })
            task_id += 1

    return plan

def main():
    if len(sys.argv) != 5:
        print("Usage: make_plan.py <config.json> <features.json> <instance.mps> <plan.json>")
        sys.exit(1)

    config_path, features_path, instance_path, out_plan_path = sys.argv[1:5]

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(features_path, "r") as f:
        features = json.load(f)

    plan = build_plan(config, features, instance_path)

    with open(out_plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"Wrote plan to {out_plan_path}")
    print(f"Total tasks: {len(plan['tasks'])}")
    print("Allocation:")
    for k, v in plan["allocation"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()