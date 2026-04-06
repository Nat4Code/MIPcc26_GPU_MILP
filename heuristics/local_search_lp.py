import time
from typing import Any, Dict, List, Tuple

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    integer_vars,
    is_binary_var,
    make_result,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def _build_start_and_rank(model, x_lp: Dict[str, float], move_policy: str) -> Tuple[Dict[str, float], List[Any]]:
    start_values: Dict[str, float] = {}
    ranking: List[Tuple[float, Any]] = []

    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        if x is None:
            continue

        if v.VType == "B":
            val = 1.0 if x >= 0.5 else 0.0
        else:
            val = round(x)
            val = max(v.LB, min(v.UB, val))
        start_values[v.VarName] = float(val)

        frac = abs(x - round(x))
        if move_policy == "fractionality":
            score = frac
        elif move_policy == "objective":
            score = abs(float(v.Obj))
        else:
            score = 0.7 * frac + 0.3 * abs(float(v.Obj))
        ranking.append((score, v))

    ranking.sort(key=lambda t: (-t[0], t[1].VarName))
    ranked_vars = [v for _, v in ranking]
    return start_values, ranked_vars


def _add_local_branching_neighborhood(model, start_values: Dict[str, float], vars_ranked: List[Any], k: int) -> Dict[str, Any]:
    expr = 0.0
    used = 0

    for v in vars_ranked:
        if not is_binary_var(v):
            continue
        if v.VarName not in start_values:
            continue
        s = int(round(start_values[v.VarName]))
        expr += (1 - v) if s == 1 else v
        used += 1

    details = {"neighborhood_type": "local_branching", "lb_binary_vars_used": used, "lb_k": int(k)}
    if used > 0:
        model.addConstr(expr <= int(k), name=f"local_branching_k_{int(k)}")
        model.update()
    return details


def _restrict_top_variables(model, start_values: Dict[str, float], ranked_vars: List[Any], free_count: int) -> Dict[str, Any]:
    free_names = {v.VarName for v in ranked_vars[:free_count]}
    fixed = 0

    for v in integer_vars(model):
        if v.VarName in free_names:
            continue
        if v.VarName not in start_values:
            continue
        val = start_values[v.VarName]
        v.LB = val
        v.UB = val
        fixed += 1

    model.update()
    return {
        "neighborhood_type": "fix_all_but_top_ranked",
        "free_count": int(free_count),
        "fixed_count": fixed,
    }


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get("seed", 0))
    move_policy = str(params.get("move_policy", "hybrid"))
    neighborhood = str(params.get("neighborhood", "local_branching"))
    neighborhood_size = int(params.get("neighborhood_size", 16))

    notes = [
        "LP-seeded local search using a rounded incumbent candidate.",
        "Uses either a local-branching neighborhood or a restricted free-variable neighborhood.",
    ]

    model = read_model(instance_path)
    model_stats = describe_model(model)

    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(
        model,
        time_limit=max(0.05, min(0.60, 0.25 * time_limit_sec)),
    )

    if x_lp:
        start_values, ranked_vars = _build_start_and_rank(model, x_lp, move_policy)
        start_values.update(seed_from_lp_values(model, x_lp, rc))
    else:
        seed_vals, seed_meta = try_root_seed(
            model,
            time_limit=min(0.25, max(0.03, 0.25 * time_limit_sec)),
            seed=seed,
        )
        if not seed_vals:
            return make_result(
                "local_search_lp",
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

        start_values = seed_vals
        ranked_vars = sorted(integer_vars(model), key=lambda v: (-abs(float(v.Obj)), v.VarName))
        lp_meta = {**lp_meta, "fallback_seed": seed_meta}

    apply_partial_start(model, start_values)

    if neighborhood == "local_branching":
        neighborhood_details = _add_local_branching_neighborhood(model, start_values, ranked_vars, neighborhood_size)
    else:
        neighborhood_details = _restrict_top_variables(model, start_values, ranked_vars, neighborhood_size)

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(
        model,
        time_limit=remain,
        seed=seed,
        threads=1,
        mip_focus=int(params.get("mip_focus", 1)),
        heuristics=float(params.get("heuristics", 0.40)),
        submip_nodes=int(params.get("submip_nodes", 500)),
        start_node_limit=int(params.get("start_node_limit", 250)),
        start_time_limit=float(params.get("start_time_limit", min(0.35, max(0.05, remain / 2.0)))),
        improve_start_time=float(params.get("improve_start_time", min(0.30, max(0.05, remain / 3.0)))),
        rins=int(params.get("rins", 10)),
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
        "solver_status": int(model.Status),
        "move_policy": move_policy,
        "neighborhood_size": neighborhood_size,
        **neighborhood_details,
    }
    if incumbent is not None:
        diag["incumbent_obj"] = incumbent["objective"]

    return make_result(
        "local_search_lp",
        params,
        time.time() - t0,
        incumbent is not None,
        incumbent,
        notes,
        diag,
    )