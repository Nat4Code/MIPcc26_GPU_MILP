import math
import time
from typing import Any, Dict, List, Tuple

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    extract_lock_counts,
    integer_vars,
    make_result,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def _score_var(v, x: float, rc: Dict[str, float], locks: Dict[str, Dict[str, int]], params: Dict[str, Any]) -> float:
    frac = abs(x - round(x))
    red_cost = abs(rc.get(v.VarName, 0.0))
    obj_mag = abs(float(v.Obj))
    up_locks = locks.get(v.VarName, {}).get("up", 0)
    down_locks = locks.get(v.VarName, {}).get("down", 0)
    lock_score = up_locks + down_locks

    w_frac = float(params.get("w_fractionality", 1.0))
    w_rc = float(params.get("w_reduced_cost", 0.25))
    w_obj = float(params.get("w_objective", 0.15))
    w_lock = float(params.get("w_locks", 0.10))

    return (
        w_frac * frac
        + w_rc * red_cost
        + w_obj * obj_mag
        + w_lock * lock_score
    )


def _choose_value(v, x: float, rc_val: float, locks: Dict[str, int], sense: int) -> float:
    lo = math.floor(x)
    hi = math.ceil(x)

    if v.VType == "B":
        lo, hi = 0.0, 1.0

    lo = max(v.LB, lo)
    hi = min(v.UB, hi)

    if abs(x - round(x)) <= 1e-9:
        return float(max(v.LB, min(v.UB, round(x))))

    up_locks = locks.get("up", 0)
    down_locks = locks.get("down", 0)

    # Lower is better for minimization when sense=1.
    # Bias direction using reduced cost, objective sign, and locks.
    score_hi = abs(hi - x)
    score_lo = abs(x - lo)

    score_hi += 0.05 * max(0.0, rc_val)
    score_lo += 0.05 * max(0.0, -rc_val)

    score_hi += 0.02 * sense * float(v.Obj) * hi
    score_lo += 0.02 * sense * float(v.Obj) * lo

    score_hi += 0.01 * up_locks
    score_lo += 0.01 * down_locks

    return float(hi if score_hi <= score_lo else lo)


def _construct_start(model, x_lp: Dict[str, float], rc: Dict[str, float], locks: Dict[str, Dict[str, int]], params: Dict[str, Any]):
    sense = int(getattr(model, "ModelSense", 1))
    ranked: List[Tuple[float, Any, float]] = []

    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        if x is None:
            continue
        ranked.append((_score_var(v, x, rc, locks, params), v, x))

    ranked.sort(key=lambda t: (-t[0], t[1].VarName))

    start_values: Dict[str, float] = {}
    rounded_integral = 0
    fractional_handled = 0

    for _, v, x in ranked:
        val = _choose_value(v, x, rc.get(v.VarName, 0.0), locks.get(v.VarName, {}), sense)
        val = max(v.LB, min(v.UB, val))
        start_values[v.VarName] = float(val)
        if abs(x - round(x)) <= 1e-9:
            rounded_integral += 1
        else:
            fractional_handled += 1

    diag = {
        "rounded_integral": rounded_integral,
        "fractional_handled": fractional_handled,
        "num_seeded_int_vars": len(start_values),
    }
    return start_values, diag


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get("seed", 0))
    notes = [
        "LP-guided greedy construction using reduced costs, objective coefficients, and lock counts.",
        "Hands a rounded partial MIP start to a short capped Gurobi improvement phase.",
    ]

    model = read_model(instance_path)
    model_stats = describe_model(model)

    lp_time = max(0.05, min(0.50, 0.20 * time_limit_sec))
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=lp_time)

    if x_lp:
        locks = extract_lock_counts(model)
        start_values, diag = _construct_start(model, x_lp, rc, locks, params)
        # Blend in common LP seed logic too.
        start_values.update(seed_from_lp_values(model, x_lp, rc, locks))
    else:
        seed_vals, seed_meta = try_root_seed(
            model,
            time_limit=min(0.25, max(0.03, 0.25 * time_limit_sec)),
            seed=seed,
        )
        if not seed_vals:
            return make_result(
                "greedy",
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
        diag = {
            "seed_source": "root_seed_fallback",
            "root_seed_meta": seed_meta,
            "num_seeded_int_vars": len(start_values),
        }

    apply_partial_start(model, start_values)

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(
        model,
        time_limit=remain,
        seed=seed,
        threads=1,
        mip_focus=int(params.get("mip_focus", 1)),
        heuristics=float(params.get("heuristics", 0.60)),
        submip_nodes=int(params.get("submip_nodes", 300)),
        start_node_limit=int(params.get("start_node_limit", 200)),
        start_time_limit=float(params.get("start_time_limit", min(0.50, max(0.05, remain / 2.0)))),
        improve_start_time=float(params.get("improve_start_time", min(0.35, max(0.05, remain / 3.0)))),
        no_rel_heur_time=float(params.get("norel_heur_time", min(0.15, max(0.02, remain / 5.0)))),
        pump_passes=int(params.get("pump_passes", 2)),
        rins=int(params.get("rins", 10)),
        presolve=int(params.get("presolve", 2)),
        cuts=int(params.get("cuts", 0)),
        symmetry=int(params.get("symmetry", -1)),
    )
    model.optimize()
    incumbent = collect_solution(model)

    diag["model"] = model_stats
    diag["lp_obj"] = lp_obj
    diag["lp_meta"] = lp_meta
    diag["solver_status"] = int(model.Status)
    if incumbent is not None:
        diag["incumbent_obj"] = incumbent["objective"]

    return make_result(
        "greedy",
        params,
        time.time() - t0,
        incumbent is not None,
        incumbent,
        notes,
        diag,
    )
