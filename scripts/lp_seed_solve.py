#!/usr/bin/env python3
"""
scripts/lp_seed_solve.py

Shared LP seed + optional LP basis export, emitted in the same result schema as
the heuristic methods so scripts.merge_results.py can recognize the incumbent.

Usage:
  python3 -m scripts.lp_seed_solve INSTANCE.mps OUT.json \
    --time-limit 30 \
    --threads 16 \
    --method 1 \
    --lp-warm-start 2 \
    --repair-seconds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

try:
    from scripts.common import make_result
except Exception:
    make_result = None


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


def is_int_type(vtype: str) -> bool:
    return vtype in (GRB.BINARY, GRB.INTEGER, GRB.SEMIINT)


def clamp(x: float, lb: float, ub: float) -> float:
    return max(lb, min(ub, x))


def rounded_for_original_type(vtype: str, lb: float, ub: float, x: float) -> float:
    if vtype == GRB.BINARY or (lb >= -1e-9 and ub <= 1.0 + 1e-9):
        return 1.0 if x >= 0.5 else 0.0
    if vtype in (GRB.INTEGER, GRB.SEMIINT):
        return float(round(clamp(x, lb, ub)))
    return float(x)


def build_lp_model(instance_path: str):
    m = gp.read(instance_path)
    original_types: Dict[str, str] = {}
    original_bounds: Dict[str, List[float]] = {}
    int_names: List[str] = []
    for v in m.getVars():
        original_types[v.VarName] = v.VType
        original_bounds[v.VarName] = [float(v.LB), float(v.UB)]
        if is_int_type(v.VType):
            int_names.append(v.VarName)
            v.VType = GRB.CONTINUOUS
    m.update()
    return m, original_types, original_bounds, int_names


def try_get_basis(model: gp.Model) -> Dict[str, Any]:
    try:
        vars_ = model.getVars()
        constrs = model.getConstrs()
        return {
            "available": True,
            "var_names": [v.VarName for v in vars_],
            "constr_names": [c.ConstrName for c in constrs],
            "vbasis": [int(x) for x in model.getAttr(GRB.Attr.VBasis, vars_)],
            "cbasis": [int(x) for x in model.getAttr(GRB.Attr.CBasis, constrs)],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def repair_seed(instance_path: str, seed: Dict[str, float], seconds: float, threads: int) -> Dict[str, Any]:
    if seconds <= 0.0 or not seed:
        return {"repair_found": False, "reason": "repair disabled or empty seed"}

    m = gp.read(instance_path)
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = float(seconds)
    m.Params.Threads = int(threads)
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.8
    m.Params.PumpPasses = 10
    m.Params.RINS = 10
    m.Params.StartNodeLimit = 500
    m.Params.StartTimeLimit = min(max(0.5, 0.5 * seconds), seconds)

    applied = 0
    for v in m.getVars():
        if v.VarName not in seed:
            continue
        val = clamp(float(seed[v.VarName]), float(v.LB), float(v.UB))
        if is_int_type(v.VType):
            val = rounded_for_original_type(v.VType, float(v.LB), float(v.UB), val)
        v.Start = val
        applied += 1

    m.optimize()

    out: Dict[str, Any] = {
        "repair_status": int(m.Status),
        "repair_status_name": status_name(int(m.Status)),
        "repair_runtime": float(m.Runtime),
        "applied_start_values": int(applied),
        "repair_found": bool(m.SolCount > 0),
    }

    if m.SolCount > 0:
        vals = {v.VarName: float(v.X) for v in m.getVars()}
        obj = float(m.ObjVal)
        out["objective"] = obj
        out["incumbent"] = {
            "objective": obj,
            "values": vals,
        }
    return out


def fallback_result(params: Dict[str, Any], runtime: float, found: bool,
                    incumbent: Optional[Dict[str, Any]], notes: List[str],
                    diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    # Include multiple common aliases so old/new merge_results variants can parse it.
    obj = None if not incumbent else incumbent.get("objective")
    return {
        "method": "lp_seed",
        "phase": "phase1",
        "params": params,
        "runtime": runtime,
        "runtime_sec": runtime,
        "found": bool(found),
        "found_incumbent": bool(found),
        "objective": obj,
        "best_obj": obj,
        "incumbent_obj": obj,
        "incumbent": incumbent,
        "solution": incumbent,
        "notes": notes,
        "diagnostics": diagnostics,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_path")
    ap.add_argument("out_json")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--method", type=int, default=1, help="1 dual simplex gives reusable basis.")
    ap.add_argument("--crossover", type=int, default=-1)
    ap.add_argument("--lp-warm-start", type=int, default=2)
    ap.add_argument("--repair-seconds", type=float, default=5.0)
    ap.add_argument("--repair-threads", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    params = {
        "lp_time_limit": args.time_limit,
        "lp_threads": args.threads,
        "lp_method": args.method,
        "lp_warm_start": args.lp_warm_start,
        "repair_seconds": args.repair_seconds,
        "repair_threads": args.repair_threads,
    }

    lp, original_types, original_bounds, int_names = build_lp_model(args.instance_path)

    lp.Params.OutputFlag = 1
    lp.Params.TimeLimit = float(args.time_limit)
    lp.Params.Threads = int(args.threads)
    lp.Params.Method = int(args.method)
    if int(args.crossover) >= 0:
        lp.Params.Crossover = int(args.crossover)
    lp.Params.LPWarmStart = int(args.lp_warm_start)
    lp.Params.DualReductions = 0

    lp.optimize()

    lp_obj = None
    x_lp_int: Dict[str, float] = {}
    reduced_costs_int: Dict[str, float] = {}
    rounded_seed: Dict[str, float] = {}
    frac_vals: List[float] = []
    basis = {"available": False, "reason": "no LP solution"}

    if lp.SolCount > 0:
        try:
            lp_obj = float(lp.ObjVal)
        except Exception:
            lp_obj = None

        for name in int_names:
            v = lp.getVarByName(name)
            if v is None:
                continue
            x = float(v.X)
            lb, ub = original_bounds[name]
            vt = original_types[name]
            x_lp_int[name] = x
            try:
                reduced_costs_int[name] = float(v.RC)
            except Exception:
                pass
            rounded_seed[name] = rounded_for_original_type(vt, lb, ub, x)
            frac_vals.append(abs(x - round(x)))

        basis = try_get_basis(lp)

    repair = repair_seed(args.instance_path, rounded_seed, float(args.repair_seconds), int(args.repair_threads))

    incumbent = repair.get("incumbent") if repair.get("repair_found") else None
    found = incumbent is not None
    runtime = time.time() - t0

    diagnostics: Dict[str, Any] = {
        "lp_meta": {
            "status": int(lp.Status),
            "status_name": status_name(int(lp.Status)),
            "runtime": float(lp.Runtime),
            "threads": int(args.threads),
            "method": int(args.method),
            "crossover": int(args.crossover),
            "lp_warm_start": int(args.lp_warm_start),
            "time_limit": float(args.time_limit),
            "num_vars": int(lp.NumVars),
            "num_constrs": int(lp.NumConstrs),
            "model_sense": int(lp.ModelSense),
        },
        "lp_obj": lp_obj,
        "basis_available": bool(basis.get("available")),
        "basis": basis,
        "fractionality": {
            "fractional_count": int(sum(1 for f in frac_vals if f > 1e-6)) if frac_vals else 0,
            "max_fractionality": float(max(frac_vals)) if frac_vals else None,
            "mean_fractionality": float(sum(frac_vals) / len(frac_vals)) if frac_vals else None,
        },
        "integer_var_count": len(int_names),
        "repair": repair,
        # Keep these available for LP-dependent heuristics or later analysis.
        "x_lp_int": x_lp_int,
        "reduced_costs_int": reduced_costs_int,
        "rounded_seed": rounded_seed,
        "original_types": {k: original_types[k] for k in int_names},
        "original_bounds": {k: original_bounds[k] for k in int_names},
    }

    notes = [
        "Shared LP seed solved before phase-1 heuristic farm.",
        "LP basis is exported for LP reuse; final MIP should use incumbent/MIP start.",
    ]

    if make_result is not None:
        result = make_result("lp_seed", params, runtime, found, incumbent, notes, diagnostics)
        # Add aliases for stricter/older merge parsers.
        if found and incumbent:
            result["objective"] = incumbent.get("objective")
            result["best_obj"] = incumbent.get("objective")
            result["incumbent_obj"] = incumbent.get("objective")
            result["solution"] = incumbent
        result["phase"] = "phase1"
    else:
        result = fallback_result(params, runtime, found, incumbent, notes, diagnostics)

    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    print(json.dumps({
        "out": str(p),
        "lp_status": diagnostics["lp_meta"]["status_name"],
        "lp_obj": lp_obj,
        "basis_available": bool(basis.get("available")),
        "repair_found": repair.get("repair_found"),
        "objective": None if not incumbent else incumbent.get("objective"),
        "runtime": runtime,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())