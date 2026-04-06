import time
from typing import Any, Dict

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


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get("seed", 0))
    notes = [
        "RENS-style neighborhood from LP relaxation.",
        "Fixes near-integral variables and restricts the remaining integer neighborhood.",
    ]

    model = read_model(instance_path)
    model_stats = describe_model(model)

    lp_obj, x_lp, rc, fracs, lp_meta = solve_lp_relaxation(
        model,
        time_limit=max(0.05, min(0.60, 0.25 * time_limit_sec)),
    )

    if not x_lp:
        seed_vals, seed_meta = try_root_seed(
            model,
            time_limit=min(0.25, max(0.03, 0.25 * time_limit_sec)),
            seed=seed,
        )
        if not seed_vals:
            return make_result(
                "rens",
                params,
                time.time() - t0,
                False,
                None,
                notes,
                {
                    "error": "No LP seed and no root seed",
                    "model": model_stats,
                    "lp_meta": lp_meta,
                    "root_seed_meta": seed_meta,
                },
            )

        apply_partial_start(model, seed_vals)
        remain = max(0.05, time_limit_sec - (time.time() - t0))
        configure_model_for_heuristic(
            model,
            time_limit=remain,
            seed=seed,
            threads=1,
            mip_focus=1,
            heuristics=float(params.get("heuristics", 0.55)),
            submip_nodes=int(params.get("submip_nodes", 400)),
            start_node_limit=int(params.get("start_node_limit", 300)),
            start_time_limit=float(params.get("start_time_limit", min(0.50, max(0.05, remain / 2.0)))),
            improve_start_time=float(params.get("improve_start_time", min(0.35, max(0.05, remain / 3.0)))),
            rins=int(params.get("rins", 10)),
            presolve=2,
            cuts=-1,
            symmetry=-1,
        )
        model.optimize()
        incumbent = collect_solution(model)
        diag = {
            "model": model_stats,
            "lp_meta": lp_meta,
            "seed_source": "root_seed_fallback",
            "root_seed_meta": seed_meta,
            "solver_status": int(model.Status),
        }
        if incumbent is not None:
            diag["incumbent_obj"] = incumbent["objective"]
        return make_result("rens", params, time.time() - t0, incumbent is not None, incumbent, notes, diag)

    int_tol = float(params.get("integral_tolerance", 0.10))
    radius = int(params.get("free_radius", 24))
    restrict_band = float(params.get("restrict_band", 0.20))

    start_values = seed_from_lp_values(model, x_lp, rc)

    fixed = 0
    restricted = 0
    free = 0

    frac_rank = sorted(
        [(fracs.get(v.VarName, 0.0), v) for v in integer_vars(model)],
        key=lambda t: (t[0], t[1].VarName)
    )

    free_names = {v.VarName for _, v in frac_rank[-radius:]} if radius > 0 else set()

    for v in integer_vars(model):
        name = v.VarName
        x = x_lp.get(name)
        if x is None:
            continue

        if name in free_names:
            free += 1
            continue

        frac = fracs.get(name, abs(x - round(x)))
        if frac <= int_tol:
            val = round(x)
            val = max(v.LB, min(v.UB, val))
            v.LB = val
            v.UB = val
            fixed += 1
        else:
            lo = max(v.LB, round(x - restrict_band))
            hi = min(v.UB, round(x + restrict_band))
            if lo <= hi and (lo > v.LB or hi < v.UB):
                v.LB = lo
                v.UB = hi
                restricted += 1

    apply_partial_start(model, start_values)

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(
        model,
        time_limit=remain,
        seed=seed,
        threads=1,
        mip_focus=int(params.get("mip_focus", 1)),
        heuristics=float(params.get("heuristics", 0.45)),
        submip_nodes=int(params.get("submip_nodes", 600)),
        start_node_limit=int(params.get("start_node_limit", 300)),
        start_time_limit=float(params.get("start_time_limit", min(0.40, max(0.05, remain / 2.0)))),
        improve_start_time=float(params.get("improve_start_time", min(0.30, max(0.05, remain / 3.0)))),
        rins=int(params.get("rins", 20)),
        presolve=int(params.get("presolve", 2)),
        cuts=int(params.get("cuts", -1)),
        symmetry=int(params.get("symmetry", -1)),
    )
    model.optimize()
    incumbent = collect_solution(model)

    diag = {
        "model": model_stats,
        "lp_obj": lp_obj,
        "lp_meta": lp_meta,
        "num_fixed": fixed,
        "num_restricted": restricted,
        "num_free": free,
        "radius": radius,
        "int_tol": int_tol,
        "restrict_band": restrict_band,
        "solver_status": int(model.Status),
    }
    if incumbent is not None:
        diag["incumbent_obj"] = incumbent["objective"]

    return make_result(
        "rens",
        params,
        time.time() - t0,
        incumbent is not None,
        incumbent,
        notes,
        diag,
    )