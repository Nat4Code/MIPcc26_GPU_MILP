#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def import_heuristic_runner(method: str):
    module = importlib.import_module(f"heuristics.{method}")
    if not hasattr(module, "run_heuristic"):
        raise RuntimeError(f"heuristics.{method} does not define run_heuristic()")
    return module.run_heuristic


def extract_candidate_objective(cand: Dict[str, Any]) -> Optional[float]:
    # 1) direct top-level objective
    obj = cand.get("objective")
    if obj is not None:
        try:
            return float(obj)
        except Exception:
            pass

    # 2) incumbent.objective
    incumbent = cand.get("incumbent")
    if isinstance(incumbent, dict):
        obj = incumbent.get("objective")
        if obj is not None:
            try:
                return float(obj)
            except Exception:
                pass

    # 3) diagnostics.incumbent_obj
    diagnostics = cand.get("diagnostics")
    if isinstance(diagnostics, dict):
        obj = diagnostics.get("incumbent_obj")
        if obj is not None:
            try:
                return float(obj)
            except Exception:
                pass

    return None


def summarize_candidate(idx: int, cand: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = cand.get("diagnostics") if isinstance(cand.get("diagnostics"), dict) else {}
    incumbent = cand.get("incumbent") if isinstance(cand.get("incumbent"), dict) else {}

    return {
        "index": idx,
        "method": cand.get("method"),
        "status": cand.get("status"),
        "feasible": cand.get("feasible"),
        "reported_objective": cand.get("objective"),
        "effective_objective": extract_candidate_objective(cand),
        "runtime_sec": cand.get("runtime_sec"),
        "incumbent_runtime_sec": incumbent.get("runtime_sec"),
        "incumbent_solution_count": incumbent.get("solution_count"),
        "incumbent_status_code": incumbent.get("status_code"),
        "diagnostic_incumbent_obj": diagnostics.get("incumbent_obj"),
        "params": cand.get("params"),
    }


def objective_sort_key(summary: Dict[str, Any], sense: str):
    obj = summary.get("effective_objective")
    if obj is None:
        return float("inf") if sense == "min" else float("-inf")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_path")
    ap.add_argument("plan_json")
    ap.add_argument("task_id", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-limit", type=float, default=None)
    args = ap.parse_args()

    plan = load_json(args.plan_json)
    tasks = plan.get("tasks", [])
    if args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit(f"task_id {args.task_id} out of range; plan has {len(tasks)} tasks")

    task = tasks[args.task_id]
    method = task["method"]
    params_list = task.get("params_list", [])
    global_cfg = plan.get("global", {})
    features = plan.get("features", {})

    instance_path = args.instance_path or plan.get("instance_path")
    if not instance_path:
        raise SystemExit("No instance path provided")

    task_time_limit = args.time_limit
    if task_time_limit is None:
        task_time_limit = float(global_cfg.get("time_limit_sec", 5.0))

    sense = str(plan.get("objective_sense", "min")).lower()
    if sense not in {"min", "max"}:
        sense = "min"

    runner = import_heuristic_runner(method)

    t0 = time.time()
    candidates = []
    errors = []

    # Honest-ish shard budgeting:
    # split the task budget across the parameter points in this shard.
    # Keep a tiny floor so very fine shards are still runnable.
    per_candidate_time = max(0.05, float(task_time_limit) / max(1, len(params_list)))

    for i, params in enumerate(params_list):
        try:
            out = runner(instance_path, params, per_candidate_time, features=features)
            if not isinstance(out, dict):
                raise RuntimeError(f"heuristic returned non-dict result for shard candidate {i}")

            # Normalize the top-level objective so downstream code sees the best value consistently.
            eff_obj = extract_candidate_objective(out)
            if eff_obj is not None:
                out["objective"] = eff_obj

            candidates.append(out)

        except Exception as exc:
            errors.append({
                "index": i,
                "params": params,
                "error": repr(exc),
            })

    best = None
    best_obj = None

    for cand in candidates:
        if not cand.get("feasible", False):
            continue

        obj = extract_candidate_objective(cand)
        if obj is None:
            continue

        if best is None:
            best = cand
            best_obj = obj
        else:
            if (sense == "min" and obj < best_obj) or (sense == "max" and obj > best_obj):
                best = cand
                best_obj = obj

    candidate_summaries = [summarize_candidate(i, cand) for i, cand in enumerate(candidates)]
    candidate_summaries_sorted = sorted(
        candidate_summaries,
        key=lambda s: objective_sort_key(s, sense),
        reverse=(sense == "max"),
    )

    result = {
        "task_id": int(args.task_id),
        "method": method,
        "method_task_index": task.get("method_task_index"),
        "num_method_tasks": task.get("num_method_tasks"),
        "grid_size_total": task.get("grid_size_total"),
        "grid_size_local": task.get("grid_size_local"),
        "instance_path": instance_path,
        "objective_sense": sense,
        "task_time_limit_sec": float(task_time_limit),
        "per_candidate_time_limit_sec": float(per_candidate_time),
        "status": "ok" if candidates else "error",
        "elapsed_sec": time.time() - t0,
        "num_candidates": len(candidates),
        "num_errors": len(errors),
        "best_effective_objective": best_obj,
        "best_candidate": best,
        "candidate_summaries": candidate_summaries_sorted,
        "top_candidate_summaries": candidate_summaries_sorted[:5],
        "candidates": candidates,
        "errors": errors,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = args.out
    if out_path is None:
        output_dir = global_cfg.get("output_dir", "results")
        out_path = os.path.join(output_dir, f"task_{args.task_id:03d}.json")

    dump_json(result, out_path)

    print(json.dumps({
        "task_id": result["task_id"],
        "method": result["method"],
        "objective_sense": result["objective_sense"],
        "status": result["status"],
        "num_candidates": result["num_candidates"],
        "num_errors": result["num_errors"],
        "task_time_limit_sec": result["task_time_limit_sec"],
        "per_candidate_time_limit_sec": result["per_candidate_time_limit_sec"],
        "best_objective": best_obj,
        "out": out_path,
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())