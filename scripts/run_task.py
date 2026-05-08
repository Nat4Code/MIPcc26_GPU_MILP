#!/usr/bin/env python3
"""
scripts/run_task.py

Run exactly one planned heuristic shard.

This replacement avoids the old behavior where one Slurm task expanded into a
large internal grid and gave each candidate a tiny fraction of the time budget.

It supports:
  python3 -m scripts.run_task INSTANCE PLAN TASK_ID --out OUT --time-limit T
  python3 -m scripts.run_task INSTANCE PLAN LOCAL_TASK_ID --phase phase1 --out OUT --time-limit T
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def safe_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y) or math.isinf(y):
        return None
    return y


def extract_tasks(plan: Dict[str, Any], phase: Optional[str]) -> List[Dict[str, Any]]:
    if phase:
        key = f"{phase}_tasks"
        if isinstance(plan.get(key), list):
            return list(plan[key])
        if isinstance(plan.get("tasks"), list):
            return [t for t in plan["tasks"] if str(t.get("phase", "")).lower() == phase.lower()]
        return []
    if isinstance(plan.get("tasks"), list):
        return list(plan["tasks"])
    out = []
    for key in ("phase1_tasks", "phase2_tasks"):
        if isinstance(plan.get(key), list):
            out.extend(plan[key])
    return out


def obj_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for key in ("objective", "best_objective", "best_obj", "incumbent_obj", "obj"):
        v = safe_float(d.get(key))
        if v is not None:
            return v
    for key in ("incumbent", "solution", "best"):
        sub = d.get(key)
        if isinstance(sub, dict):
            v = obj_from_dict(sub)
            if v is not None:
                return v
    return None


def extract_incumbent(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("incumbent", "solution", "best", "best_incumbent"):
        sub = d.get(key)
        if isinstance(sub, dict):
            if "values" in sub or "objective" in sub:
                return dict(sub)
    if isinstance(d.get("repair"), dict) and isinstance(d["repair"].get("incumbent"), dict):
        return dict(d["repair"]["incumbent"])
    return None


def load_warmstart(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    d = load_json(str(p))
    inc = extract_incumbent(d)
    obj = obj_from_dict(d)
    if inc is not None:
        if "objective" not in inc and obj is not None:
            inc["objective"] = obj
        return inc
    return None


def normalize_result(result: Dict[str, Any], task: Dict[str, Any], runtime: float, warmstart: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {}

    method = task.get("method", result.get("method", "unknown"))
    phase = task.get("phase", result.get("phase", "unknown"))
    params = task.get("params", {})

    inc = extract_incumbent(result)
    obj = obj_from_dict(result)
    found = bool(result.get("found", result.get("found_incumbent", inc is not None or obj is not None)))

    # Preserve incumbent if method returns one.
    if inc is not None and obj is None:
        obj = obj_from_dict(inc)
    if inc is not None and "objective" not in inc and obj is not None:
        inc["objective"] = obj

    # If phase2 method failed but a warmstart was supplied, emit the warmstart as
    # the candidate. This prevents phase2 from erasing the incumbent path.
    if not found and warmstart is not None:
        inc = dict(warmstart)
        obj = obj_from_dict(inc)
        found = obj is not None
        result.setdefault("notes", [])
        if isinstance(result["notes"], list):
            result["notes"].append("Method did not improve; carrying warmstart incumbent forward.")

    out = dict(result)
    out.update({
        "task_id": task.get("task_id"),
        "phase_task_id": task.get("phase_task_id", task.get("local_task_id")),
        "phase": phase,
        "method": method,
        "params": params,
        "runtime": result.get("runtime", runtime),
        "runtime_sec": result.get("runtime_sec", runtime),
        "found": bool(found),
        "found_incumbent": bool(found),
        "objective": obj,
        "best_obj": obj,
        "incumbent_obj": obj,
        "incumbent": inc,
        "solution": inc,
        "warmstart_loaded": warmstart is not None,
        "run_task_note": "one planned shard received the full task time limit",
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_path")
    ap.add_argument("plan_json")
    ap.add_argument("task_id", type=int)
    ap.add_argument("--phase", default=None, choices=["phase1", "phase2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmstart-json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    plan = load_json(args.plan_json)
    tasks = extract_tasks(plan, args.phase)

    if args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit(f"task_id {args.task_id} out of range for phase={args.phase}; num_tasks={len(tasks)}")

    task = dict(tasks[args.task_id])
    params = dict(task.get("params") or {})
    task["params"] = params

    threads = args.threads
    if threads is None:
        threads = int(task.get("threads_per_task", params.get("threads", params.get("mip_threads", 1))))

    params.setdefault("threads", threads)
    params.setdefault("mip_threads", threads)
    params.setdefault("lp_threads", threads)

    warmstart = load_warmstart(args.warmstart_json)
    if warmstart is not None:
        params["warmstart_incumbent"] = warmstart
        params["warmstart_values"] = warmstart.get("values", {})
        params["warmstart_objective"] = obj_from_dict(warmstart)

    method = str(task.get("method"))
    module = importlib.import_module(f"heuristics.{method}")

    if not hasattr(module, "run_heuristic"):
        raise SystemExit(f"heuristics.{method} has no run_heuristic(...)")

    result = module.run_heuristic(
        args.instance_path,
        params,
        float(args.time_limit),
        features=plan,
    )

    runtime = time.time() - t0
    out = normalize_result(result, task, runtime, warmstart)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    print(json.dumps({
        "out": str(p),
        "task_id": task.get("task_id"),
        "phase_task_id": task.get("phase_task_id", task.get("local_task_id")),
        "phase": task.get("phase"),
        "method": method,
        "time_limit_sec": float(args.time_limit),
        "threads": threads,
        "warmstart_loaded": warmstart is not None,
        "found": out.get("found"),
        "objective": out.get("objective"),
        "runtime": out.get("runtime"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())