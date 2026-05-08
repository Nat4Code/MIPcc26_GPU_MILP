#!/usr/bin/env python3
"""
scripts/final_gurobi_solve.py

Robust final Gurobi exact solve with MIP-start loading from merged heuristic JSON.

Usage:
  python3 -m scripts.final_gurobi_solve INSTANCE.mps MERGED.json \
    --out final_gurobi.json \
    --time-limit 150 \
    --threads 16 \
    --log-to-console

This version is deliberately tolerant of merged result schemas:
  - {"incumbent": {"objective": ..., "values": {...}}}
  - {"solution": {"objective": ..., "values": {...}}}
  - {"best": ...}
  - nested repair/incumbent objects
  - raw variable dictionaries under x / values / vars / solution
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB


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


def status_name(st: int) -> str:
    return {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(st, f"STATUS_{st}")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def obj_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for k in ("objective", "best_objective", "best_obj", "incumbent_obj", "obj"):
        v = safe_float(d.get(k))
        if v is not None:
            return v
    return None


def looks_like_var_dict(d: Dict[str, Any]) -> bool:
    if not d:
        return False
    checked = 0
    numeric = 0
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        checked += 1
        if safe_float(v) is not None:
            numeric += 1
        if checked >= 20:
            break
    return checked > 0 and numeric == checked


def extract_values_recursive(obj: Any) -> Optional[Dict[str, float]]:
    """
    Try hard to find a variable assignment dictionary.
    """
    if isinstance(obj, dict):
        # Common explicit value containers.
        for key in ("values", "x", "vars", "variables", "start_values", "solution_values"):
            sub = obj.get(key)
            if isinstance(sub, dict) and looks_like_var_dict(sub):
                return {str(k): float(v) for k, v in sub.items() if safe_float(v) is not None}

        # Sometimes the object itself is the var dict.
        if looks_like_var_dict(obj):
            return {str(k): float(v) for k, v in obj.items() if safe_float(v) is not None}

        # Prefer incumbent-like structures first.
        for key in ("incumbent", "solution", "best", "best_incumbent"):
            sub = obj.get(key)
            vals = extract_values_recursive(sub)
            if vals:
                return vals

        # LP seed repair nested structure.
        repair = obj.get("repair")
        if isinstance(repair, dict):
            vals = extract_values_recursive(repair.get("incumbent"))
            if vals:
                return vals

        diag = obj.get("diagnostics")
        if isinstance(diag, dict):
            vals = extract_values_recursive(diag)
            if vals:
                return vals

        # Last resort recursive search.
        for v in obj.values():
            vals = extract_values_recursive(v)
            if vals:
                return vals

    elif isinstance(obj, list):
        for v in obj:
            vals = extract_values_recursive(v)
            if vals:
                return vals

    return None


def extract_incumbent(merged: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict[str, float]], str]:
    """
    Returns objective, values, source.
    """
    # Direct merged format.
    for key in ("incumbent", "solution", "best", "best_incumbent"):
        sub = merged.get(key)
        if isinstance(sub, dict):
            vals = extract_values_recursive(sub)
            obj = obj_from_dict(sub) or obj_from_dict(merged)
            if vals:
                return obj, vals, key

    # Candidate raw fallback.
    candidates = merged.get("candidates")
    if isinstance(candidates, list):
        # Usually summaries only, no values. But tolerate richer versions.
        best = None
        for c in candidates:
            if isinstance(c, dict):
                obj = obj_from_dict(c)
                vals = extract_values_recursive(c)
                if vals and obj is not None:
                    if best is None or obj < best[0]:
                        best = (obj, vals, "candidates")
        if best:
            return best

    vals = extract_values_recursive(merged)
    obj = obj_from_dict(merged)
    if vals:
        return obj, vals, "recursive"

    return obj, None, "none"


def apply_mip_start(model: gp.Model, values: Dict[str, float]) -> Dict[str, Any]:
    applied = 0
    missing = 0
    clipped = 0
    integer_rounded = 0

    for v in model.getVars():
        if v.VarName not in values:
            missing += 1
            continue

        val = float(values[v.VarName])
        if val < v.LB:
            val = float(v.LB)
            clipped += 1
        if val > v.UB:
            val = float(v.UB)
            clipped += 1

        if v.VType in (GRB.BINARY, GRB.INTEGER, GRB.SEMIINT):
            rv = round(val)
            if abs(rv - val) > 1e-7:
                integer_rounded += 1
            val = float(rv)
            if v.VType == GRB.BINARY:
                val = 1.0 if val >= 0.5 else 0.0

        v.Start = val
        applied += 1

    model.update()

    return {
        "applied_start_values": applied,
        "missing_model_vars": missing,
        "clipped_values": clipped,
        "integer_rounded_values": integer_rounded,
        "model_var_count": model.NumVars,
    }


def set_optional_param(model: gp.Model, name: str, value: Any) -> bool:
    if value is None:
        return False
    try:
        setattr(model.Params, name, value)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_path")
    ap.add_argument("merged_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=150.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus-mode", choices=["balanced", "incumbent", "prove"], default="incumbent")
    ap.add_argument("--mip-focus", type=int, default=None)
    ap.add_argument("--heuristics", type=float, default=0.05)
    ap.add_argument("--cuts", type=int, default=None)
    ap.add_argument("--cut-passes", type=int, default=None)
    ap.add_argument("--presolve", type=int, default=None)
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--mip-gap-abs", type=float, default=None)
    ap.add_argument("--start-node-limit", type=int, default=500)
    ap.add_argument("--start-time-limit", type=float, default=2.0)
    ap.add_argument("--event-log", default=None,
                    help="Optional CSV path for incumbent events found during this solve.")
    ap.add_argument("--log-to-console", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    merged = load_json(args.merged_json)
    start_obj, start_values, start_source = extract_incumbent(merged)

    m = gp.read(args.instance_path)
    m.Params.OutputFlag = 1 if args.log_to_console else 0
    m.Params.TimeLimit = float(args.time_limit)
    m.Params.Threads = int(args.threads)
    m.Params.Seed = int(args.seed)

    if args.mip_focus is not None:
        mip_focus = int(args.mip_focus)
    elif args.focus_mode == "prove":
        mip_focus = 3
    elif args.focus_mode == "balanced":
        mip_focus = 0
    else:
        mip_focus = 1

    m.Params.MIPFocus = int(mip_focus)
    m.Params.Heuristics = float(args.heuristics)
    m.Params.StartNodeLimit = int(args.start_node_limit)
    m.Params.StartTimeLimit = float(args.start_time_limit)

    optional_params = {
        "Cuts": args.cuts,
        "CutPasses": args.cut_passes,
        "Presolve": args.presolve,
        "MIPGap": args.mip_gap,
        "MIPGapAbs": args.mip_gap_abs,
    }
    applied_params = {
        "OutputFlag": 1 if args.log_to_console else 0,
        "TimeLimit": float(args.time_limit),
        "Threads": int(args.threads),
        "Seed": int(args.seed),
        "FocusMode": args.focus_mode,
        "MIPFocus": int(mip_focus),
        "Heuristics": float(args.heuristics),
        "StartNodeLimit": int(args.start_node_limit),
        "StartTimeLimit": float(args.start_time_limit),
    }
    skipped_optional_params = []
    for name, value in optional_params.items():
        if value is None:
            continue
        if set_optional_param(m, name, value):
            applied_params[name] = value
        else:
            skipped_optional_params.append(name)

    start_info: Dict[str, Any] = {
        "warmstart_loaded": False,
        "warmstart_source": start_source,
        "warmstart_objective_from_json": start_obj,
    }

    if start_values:
        start_info.update(apply_mip_start(m, start_values))
        start_info["warmstart_loaded"] = start_info.get("applied_start_values", 0) > 0
        print(json.dumps({
            "phase": "final_start_load",
            "warmstart_loaded": start_info["warmstart_loaded"],
            "source": start_source,
            "objective_from_json": start_obj,
            "applied_start_values": start_info.get("applied_start_values"),
            "model_var_count": start_info.get("model_var_count"),
        }))
    else:
        print(json.dumps({
            "phase": "final_start_load",
            "warmstart_loaded": False,
            "source": start_source,
            "objective_from_json": start_obj,
            "reason": "no variable values found in merged JSON",
        }))

    event_file = None
    event_writer = None
    incumbent_events = 0
    best_callback_obj: Optional[float] = None
    if args.event_log:
        event_path = Path(args.event_log)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_file = event_path.open("w", encoding="utf-8", newline="")
        event_writer = csv.DictWriter(
            event_file,
            fieldnames=["time_sec", "objective", "incumbent_objective", "method", "phase", "source"],
        )
        event_writer.writeheader()
        event_file.flush()

    sense = "min" if int(m.ModelSense) == 1 else "max"

    def incumbent_callback(model: gp.Model, where: int) -> None:
        nonlocal best_callback_obj, incumbent_events
        if where != GRB.Callback.MIPSOL or event_writer is None or event_file is None:
            return
        try:
            obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        except Exception:
            return
        if best_callback_obj is not None:
            if sense == "min" and obj >= best_callback_obj - 1e-9:
                return
            if sense == "max" and obj <= best_callback_obj + 1e-9:
                return
        best_callback_obj = obj
        incumbent_events += 1
        event_writer.writerow({
            "time_sec": runtime,
            "objective": obj,
            "incumbent_objective": obj,
            "method": "final_gurobi",
            "phase": "final_gurobi",
            "source": "mipsol_callback",
        })
        event_file.flush()

    try:
        if event_writer is not None:
            m.optimize(incumbent_callback)
        else:
            m.optimize()
    finally:
        if event_file is not None:
            event_file.close()

    found = m.SolCount > 0
    obj = float(m.ObjVal) if found else None
    bound = None
    gap = None
    try:
        bound = float(m.ObjBound)
    except Exception:
        pass
    try:
        gap = float(m.MIPGap)
    except Exception:
        pass

    out = {
        "phase": "final_gurobi",
        "method": "final_gurobi",
        "found": bool(found),
        "found_incumbent": bool(found),
        "objective": obj,
        "best_obj": obj,
        "best_objective": obj,
        "best_bound": bound,
        "mip_gap": gap,
        "status": int(m.Status),
        "status_name": status_name(int(m.Status)),
        "runtime": float(m.Runtime),
        "wall_runtime": time.time() - t0,
        "time_limit": float(args.time_limit),
        "threads": int(args.threads),
        "gurobi_params": applied_params,
        "skipped_optional_gurobi_params": skipped_optional_params,
        "warmstart": start_info,
        "event_log": args.event_log,
        "incumbent_event_count": int(incumbent_events),
        "incumbent": None,
    }

    if found:
        out["incumbent"] = {
            "objective": obj,
            "values": {v.VarName: float(v.X) for v in m.getVars()},
        }
        out["solution"] = out["incumbent"]

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    print(json.dumps({
        "out": str(path),
        "found": found,
        "objective": obj,
        "best_bound": bound,
        "gap": gap,
        "status": out["status_name"],
        "warmstart_loaded": start_info["warmstart_loaded"],
        "applied_start_values": start_info.get("applied_start_values", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
