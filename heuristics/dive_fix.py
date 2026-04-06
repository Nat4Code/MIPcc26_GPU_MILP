import time
from typing import Any, Dict, List

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    integer_vars,
    make_result,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


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


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get("seed", 0))
    branch_score = str(params.get("branch_score", "fractionality"))
    fix_batch_size = int(params.get("fix_batch_size", 8))
    max_iters = int(params.get("max_iters", 4))
    backtrack_depth = int(params.get("backtrack_depth", 1))

    notes = [
        "Dive-and-fix based on repeated LP guidance.",
        "Fixes a small batch of variables per iteration and runs a short capped sub-MIP.",
    ]

    base = read_model(instance_path)
    model_stats = describe_model(base)
    iterations: List[Dict[str, Any]] = []
    incumbent = None

    for it in range(max_iters):
        spent = time.time() - t0
        remain = time_limit_sec - spent
        if remain <= 0.05:
            break

        lp_cap = max(0.03, min(0.20, 0.35 * remain))
        lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(base, time_limit=lp_cap)

        if not x_lp:
            seed_vals, seed_meta = try_root_seed(
                base,
                time_limit=min(0.20, max(0.03, lp_cap)),
                seed=seed + it,
            )
            if seed_vals:
                apply_partial_start(base, seed_vals)
                iterations.append({
                    "iter": it,
                    "event": "root_seed_fallback",
                    "seed_meta": seed_meta,
                    "lp_meta": lp_meta,
                })
            else:
                iterations.append({
                    "iter": it,
                    "event": "lp_failed",
                    "lp_meta": lp_meta,
                    "root_seed_meta": seed_meta,
                })
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

        sub_time = max(0.03, min(0.30, 0.60 * (time_limit_sec - (time.time() - t0))))
        configure_model_for_heuristic(
            base,
            time_limit=sub_time,
            seed=seed + it,
            threads=1,
            mip_focus=int(params.get("mip_focus", 1)),
            heuristics=float(params.get("heuristics", 0.35)),
            submip_nodes=int(params.get("submip_nodes", 300)),
            start_node_limit=int(params.get("start_node_limit", 200)),
            start_time_limit=float(params.get("start_time_limit", min(0.25, max(0.03, sub_time / 2.0)))),
            improve_start_time=float(params.get("improve_start_time", min(0.20, max(0.03, sub_time / 3.0)))),
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
        })

        if cand is not None:
            incumbent = cand
            break

        # crude shallow backtracking:
        if backtrack_depth > 0:
            for name, _, _ in fixed_now[-backtrack_depth:]:
                v = base.getVarByName(name)
                if v is None:
                    continue
                # Cannot restore original bounds exactly unless tracked from start.
                # So only relax binaries to [0,1] and general ints back to original Var attrs if possible.
                if v.VType == "B":
                    v.LB = 0.0
                    v.UB = 1.0
            base.update()

    if incumbent is None:
        # One last try if we at least have a start sitting on the model.
        final_remain = max(0.03, time_limit_sec - (time.time() - t0))
        if final_remain > 0.03:
            configure_model_for_heuristic(
                base,
                time_limit=final_remain,
                seed=seed + 97,
                threads=1,
                mip_focus=1,
                heuristics=0.50,
                submip_nodes=200,
                start_node_limit=150,
                start_time_limit=min(0.20, max(0.03, final_remain / 2.0)),
                improve_start_time=min(0.15, max(0.03, final_remain / 3.0)),
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
        "solver_status": int(base.Status),
    }
    if incumbent is not None:
        diag["incumbent_obj"] = incumbent["objective"]

    return make_result(
        "dive_fix",
        params,
        time.time() - t0,
        incumbent is not None,
        incumbent,
        notes,
        diag,
    )