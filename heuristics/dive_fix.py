import time
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    integer_vars,
    make_result,
    read_model,
    seed_from_lp_values,
    try_root_seed,
)

from scripts.lp_basis_utils import apply_basis_if_valid, capture_basis


def _rank_for_diving(model, x_lp: Dict[str, float], rc: Dict[str, float], mode: str) -> List[Any]:
    scored = []
    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        if x is None:
            continue
        frac = abs(x - round(x))
        red = abs(rc.get(v.VarName, 0.0))
        obj = abs(float(v.Obj))

        if mode == "fractionality":
            score = frac
        elif mode == "reduced_cost":
            score = red
        elif mode == "objective":
            score = obj
        else:
            score = 0.6 * frac + 0.25 * red + 0.15 * obj

        scored.append((score, v, x))
    scored.sort(key=lambda t: (-t[0], t[1].VarName))
    return scored


def _status_name(st: int) -> str:
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


def _build_lp_model(instance_path: str) -> gp.Model:
    lp = gp.read(instance_path)
    for v in lp.getVars():
        if v.VType in (GRB.BINARY, GRB.INTEGER, GRB.SEMIINT):
            v.VType = GRB.CONTINUOUS
    lp.update()
    return lp


def _sync_bounds_from_mip_to_lp(mip: gp.Model, lp: gp.Model, names: List[str]) -> None:
    for name in names:
        mv = mip.getVarByName(name)
        lv = lp.getVarByName(name)
        if mv is None or lv is None:
            continue
        lv.LB = mv.LB
        lv.UB = mv.UB
    lp.update()


def _solve_lp_with_basis(
    lp: gp.Model,
    int_names: List[str],
    time_limit: float,
    threads: int,
    seed: int,
    basis: Dict[str, Any] | None,
    lp_warm_start: int,
) -> Tuple[Any, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any], Dict[str, Any] | None]:
    lp.Params.OutputFlag = 0
    lp.Params.TimeLimit = float(max(0.01, time_limit))
    lp.Params.Threads = int(max(1, threads))
    lp.Params.Method = 1
    lp.Params.LPWarmStart = int(lp_warm_start)
    lp.Params.Seed = int(seed)
    lp.Params.DualReductions = 0

    basis_applied = apply_basis_if_valid(lp, basis, lp_warm_start=lp_warm_start) if basis else False

    t0 = time.time()
    lp.optimize()
    wall = time.time() - t0

    meta = {
        "status": int(lp.Status),
        "status_name": _status_name(int(lp.Status)),
        "runtime": float(lp.Runtime),
        "wall_runtime": wall,
        "solcount": int(lp.SolCount),
        "method": 1,
        "threads": int(max(1, threads)),
        "basis_applied": bool(basis_applied),
    }

    x_lp: Dict[str, float] = {}
    rc: Dict[str, float] = {}
    fracs: Dict[str, float] = {}

    if lp.SolCount > 0:
        for name in int_names:
            v = lp.getVarByName(name)
            if v is None:
                continue
            x = float(v.X)
            x_lp[name] = x
            fracs[name] = abs(x - round(x))
            try:
                rc[name] = float(v.RC)
            except Exception:
                rc[name] = 0.0

    lp_obj = None
    if lp.SolCount > 0:
        try:
            lp_obj = float(lp.ObjVal)
        except Exception:
            pass

    new_basis = capture_basis(lp) if lp.SolCount > 0 else basis
    return lp_obj, x_lp, rc, fracs, meta, new_basis


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get("seed", 0))
    branch_score = str(params.get("branch_score", "fractionality"))
    fix_batch_size = int(params.get("fix_batch_size", 8))
    max_iters = int(params.get("max_iters", 4))
    backtrack_depth = int(params.get("backtrack_depth", 1))

    lp_cap_max = float(params.get("lp_cap_max", 8.0))
    lp_fraction = float(params.get("lp_fraction", 0.25))
    lp_threads = int(params.get("lp_threads", 1))
    lp_warm_start = int(params.get("lp_warm_start", 2))

    notes = [
        "Dive-and-fix based on repeated LP guidance.",
        "Reuses a valid simplex basis between repeated LP relaxations when possible.",
    ]

    base = read_model(instance_path)
    model_stats = describe_model(base)
    int_names = [v.VarName for v in integer_vars(base)]
    lp_model = _build_lp_model(instance_path)
    lp_basis = None

    iterations: List[Dict[str, Any]] = []
    incumbent = None
    original_bounds = {v.VarName: (float(v.LB), float(v.UB)) for v in integer_vars(base)}

    for it in range(max_iters):
        spent = time.time() - t0
        remain = time_limit_sec - spent
        if remain <= 0.05:
            break

        _sync_bounds_from_mip_to_lp(base, lp_model, int_names)

        lp_cap = max(0.05, min(lp_cap_max, lp_fraction * remain))
        lp_obj, x_lp, rc, fracs, lp_meta, lp_basis = _solve_lp_with_basis(
            lp_model,
            int_names=int_names,
            time_limit=lp_cap,
            threads=lp_threads,
            seed=seed + it,
            basis=lp_basis,
            lp_warm_start=lp_warm_start,
        )

        if not x_lp:
            seed_vals, seed_meta = try_root_seed(
                base,
                time_limit=min(0.75, max(0.05, lp_cap)),
                seed=seed + it,
            )
            if seed_vals:
                apply_partial_start(base, seed_vals)
                iterations.append({"iter": it, "event": "root_seed_fallback", "seed_meta": seed_meta, "lp_meta": lp_meta})
            else:
                iterations.append({"iter": it, "event": "lp_failed", "lp_meta": lp_meta, "root_seed_meta": seed_meta})
            break

        ranked = _rank_for_diving(base, x_lp, rc, branch_score)
        batch = ranked[:fix_batch_size]

        fixed_now = []
        for _, v, x in batch:
            val = round(x)
            val = max(v.LB, min(v.UB, val))
            fixed_now.append((v.VarName, float(val), float(x)))
            v.LB = val
            v.UB = val
        base.update()

        start_values = seed_from_lp_values(base, x_lp, rc)
        apply_partial_start(base, start_values)

        sub_time = max(0.05, min(1.50, 0.60 * (time_limit_sec - (time.time() - t0))))
        configure_model_for_heuristic(
            base,
            time_limit=sub_time,
            seed=seed + it,
            threads=1,
            mip_focus=int(params.get("mip_focus", 1)),
            heuristics=float(params.get("heuristics", 0.35)),
            submip_nodes=int(params.get("submip_nodes", 300)),
            start_node_limit=int(params.get("start_node_limit", 200)),
            start_time_limit=float(params.get("start_time_limit", min(0.50, max(0.05, sub_time / 2.0)))),
            improve_start_time=float(params.get("improve_start_time", min(0.30, max(0.05, sub_time / 3.0)))),
            rins=int(params.get("rins", 10)),
            presolve=int(params.get("presolve", 2)),
            cuts=int(params.get("cuts", 0)),
            symmetry=int(params.get("symmetry", -1)),
        )
        base.optimize()
        cand = collect_solution(base)

        iterations.append({
            "iter": it,
            "event": "submip",
            "lp_obj": lp_obj,
            "lp_meta": lp_meta,
            "fixed_count": len(fixed_now),
            "fixed_preview": fixed_now[: min(8, len(fixed_now))],
            "solver_status": int(base.Status),
            "found_solution": cand is not None,
            "objective": None if cand is None else cand.get("objective"),
            "basis_available_after_lp": bool((lp_basis or {}).get("available")),
        })

        if cand is not None:
            incumbent = cand
            break

        if backtrack_depth > 0:
            for name, _, _ in fixed_now[-backtrack_depth:]:
                v = base.getVarByName(name)
                if v is None:
                    continue
                lo, hi = original_bounds.get(name, (v.LB, v.UB))
                if v.VType == "B":
                    v.LB = max(0.0, lo)
                    v.UB = min(1.0, hi)
                else:
                    v.LB = lo
                    v.UB = hi
            base.update()

    if incumbent is None:
        final_remain = max(0.05, time_limit_sec - (time.time() - t0))
        if final_remain > 0.05:
            configure_model_for_heuristic(
                base,
                time_limit=final_remain,
                seed=seed + 97,
                threads=1,
                mip_focus=1,
                heuristics=0.50,
                submip_nodes=200,
                start_node_limit=150,
                start_time_limit=min(0.50, max(0.05, final_remain / 2.0)),
                improve_start_time=min(0.30, max(0.05, final_remain / 3.0)),
                rins=10,
                presolve=2,
                cuts=0,
                symmetry=-1,
            )
            base.optimize()
            incumbent = collect_solution(base)

    diag = {
        "model": model_stats,
        "iterations": iterations,
        "branch_score": branch_score,
        "fix_batch_size": fix_batch_size,
        "max_iters": max_iters,
        "backtrack_depth": backtrack_depth,
        "lp_basis_warmstart_enabled": True,
        "lp_threads": lp_threads,
        "lp_cap_max": lp_cap_max,
        "lp_warm_start": lp_warm_start,
        "solver_status": int(base.Status),
    }
    if incumbent is not None:
        diag["incumbent_obj"] = incumbent["objective"]

    return make_result("dive_fix", params, time.time() - t0, incumbent is not None, incumbent, notes, diag)
