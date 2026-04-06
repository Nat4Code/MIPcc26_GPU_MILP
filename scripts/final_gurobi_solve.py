#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


INTEGER_VTYPES = {"B", "I"}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def apply_mip_start(model, values):
    used = 0
    for v in model.getVars():
        if v.VarName in values:
            val = values[v.VarName]
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            try:
                vv = float(val)
            except Exception:
                continue
            vv = max(v.LB, min(v.UB, vv))
            v.Start = vv
            used += 1
        else:
            v.Start = GRB.UNDEFINED
    model.update()
    return used


def collect_solution(model):
    if int(model.SolCount) <= 0:
        return None

    values = {}
    for v in model.getVars():
        if v.VType in INTEGER_VTYPES:
            try:
                values[v.VarName] = float(v.X)
            except Exception:
                pass

    out = {
        "objective": None,
        "solution_count": int(model.SolCount),
        "runtime_sec": float(model.Runtime),
        "status_code": int(model.Status),
        "node_count": float(getattr(model, "NodeCount", 0.0)),
        "mip_gap": None,
        "bound": None,
        "values": values,
    }
    try:
        out["objective"] = float(model.ObjVal)
    except Exception:
        pass
    try:
        out["mip_gap"] = float(model.MIPGap)
    except Exception:
        pass
    try:
        out["bound"] = float(model.ObjBound)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_path")
    ap.add_argument("merged_results_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=50.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mip-focus", type=int, default=1)
    ap.add_argument("--heuristics", type=float, default=0.05)
    ap.add_argument("--presolve", type=int, default=-1)
    ap.add_argument("--cuts", type=int, default=-1)
    ap.add_argument("--symmetry", type=int, default=-1)
    ap.add_argument("--start-node-limit", type=int, default=500)
    ap.add_argument("--start-time-limit", type=float, default=2.0)
    ap.add_argument("--log-to-console", action="store_true")
    args = ap.parse_args()

    merged = load_json(args.merged_results_json)

    best = merged.get("best_result")
    if not best:
        dump_json(
            {
                "status": "no_best_result",
                "message": "Merged results did not contain a best_result.",
            },
            args.out,
        )
        return 0

    incumbent = best.get("incumbent")
    if not incumbent:
        dump_json(
            {
                "status": "no_incumbent",
                "message": "Best result did not contain an incumbent block.",
                "best_result": best,
            },
            args.out,
        )
        return 0

    start_values = incumbent.get("values", {})
    if not start_values:
        dump_json(
            {
                "status": "no_start_values",
                "message": "Best incumbent had no variable assignments.",
                "best_result": best,
            },
            args.out,
        )
        return 0

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if args.log_to_console else 0)
    env.start()

    model = gp.read(args.instance_path, env=env)
    model.Params.TimeLimit = max(0.01, float(args.time_limit))
    model.Params.Threads = max(1, int(args.threads))
    model.Params.Seed = int(args.seed)
    model.Params.MIPFocus = int(args.mip_focus)
    model.Params.Heuristics = float(args.heuristics)
    model.Params.StartNodeLimit = int(args.start_node_limit)
    model.Params.StartTimeLimit = float(args.start_time_limit)
    model.Params.Presolve = int(args.presolve)
    model.Params.Cuts = int(args.cuts)
    model.Params.Symmetry = int(args.symmetry)

    used_start_values = apply_mip_start(model, start_values)

    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0

    final_sol = collect_solution(model)

    out = {
        "status": "ok",
        "instance_path": args.instance_path,
        "merged_results_json": args.merged_results_json,
        "used_start_values": used_start_values,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "seed": args.seed,
        "mip_focus": args.mip_focus,
        "heuristics": args.heuristics,
        "start_node_limit": args.start_node_limit,
        "start_time_limit": args.start_time_limit,
        "best_heuristic_method": best.get("method"),
        "best_heuristic_objective": best.get("objective"),
        "solver_status": int(model.Status),
        "runtime_sec": elapsed,
        "final_solution": final_sol,
    }

    dump_json(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())